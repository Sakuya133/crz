from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import os
from copy import deepcopy
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cxr_steganalysis.config import DEFAULT_CONFIG
from cxr_steganalysis.data.dataset import PairedPatchDataset
from cxr_steganalysis.data.manifest import build_pair_manifest, generate_stego_records
from cxr_steganalysis.models.residual_cnn import HighPassResidualCNN
from cxr_steganalysis.reproducibility import seed_everything
from cxr_steganalysis.training.checkpoint import load_checkpoint
from cxr_steganalysis.training.trainer import Trainer
from cxr_steganalysis.training.groupdro import BalancedViewBatchSampler


def _manifest(tmp_path):
    rows = []
    rng = np.random.default_rng(9)
    for index, split in enumerate(["train", "train", "validation", "validation"]):
        path = tmp_path / f"{index:08d}_000.png"
        Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8), mode="L").save(path)
        rows.append({
            "image_id": path.name,
            "patient_id": str(index),
            "view_position": "PA",
            "finding_labels": "No Finding",
            "split": split,
            "cover_path": str(path),
        })
    split_frame = pd.DataFrame(rows)
    generated = generate_stego_records(split_frame, tmp_path / "stego", 0.5, 22)
    return build_pair_manifest(split_frame, generated)


def test_one_batch_forward_backward_and_resume(tmp_path):
    manifest = _manifest(tmp_path)
    train = PairedPatchDataset(manifest, split="train", patch_size=16)
    validation = PairedPatchDataset(manifest, split="validation", patch_size=16)
    train_loader = DataLoader(train, batch_size=2)
    validation_loader = DataLoader(validation, batch_size=2)
    config = dict(DEFAULT_CONFIG)
    config.update({
        "seed": 22,
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "mixed_precision": False,
        "early_stopping_patience": 3,
    })
    model = HighPassResidualCNN(base_channels=4)
    trainer = Trainer(model, config, tmp_path / "run", device="cpu")
    first = trainer.fit(train_loader, validation_loader)
    checkpoint = load_checkpoint(first["last_checkpoint"])
    for key in (
        "epoch", "model_state", "optimizer_state", "scheduler_state",
        "best_metric", "config", "seed", "scaler_state", "rng_state",
        "model_name", "model_version",
        "initial_model_sha256",
    ):
        assert key in checkpoint
    assert checkpoint["epoch"] == 0
    assert len(checkpoint["initial_model_sha256"]) == 64

    resumed_model = HighPassResidualCNN(base_channels=4)
    resumed = Trainer(resumed_model, config, tmp_path / "run", device="cpu")
    summary = resumed.fit(
        train_loader, validation_loader, resume=first["last_checkpoint"], max_epochs=2
    )
    assert summary["last_epoch"] == 1
    assert summary["initial_model_sha256"] == checkpoint["initial_model_sha256"]
    assert (tmp_path / "run" / "checkpoints" / "best.pt").is_file()
    assert torch.isfinite(next(resumed_model.parameters())).all()


def test_new_run_refuses_to_overwrite_training_artifacts(tmp_path):
    manifest = _manifest(tmp_path)
    train = PairedPatchDataset(manifest, split="train", patch_size=16)
    validation = PairedPatchDataset(manifest, split="validation", patch_size=16)
    config = dict(DEFAULT_CONFIG)
    config.update({"epochs": 1, "mixed_precision": False})
    run = tmp_path / "protected-run"
    trainer = Trainer(HighPassResidualCNN(base_channels=4), config, run, device="cpu")
    trainer.fit(DataLoader(train, batch_size=2), DataLoader(validation, batch_size=2))
    second = Trainer(HighPassResidualCNN(base_channels=4), config, run, device="cpu")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        second.fit(DataLoader(train, batch_size=2), DataLoader(validation, batch_size=2))


def test_epoch_boundary_resume_matches_continuous_training(tmp_path):
    manifest = _manifest(tmp_path)
    train = PairedPatchDataset(manifest, split="train", patch_size=16)
    validation = PairedPatchDataset(manifest, split="validation", patch_size=16)
    config = dict(DEFAULT_CONFIG)
    config.update({
        "seed": 77,
        "epochs": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "mixed_precision": False,
        "early_stopping_patience": 5,
    })

    def loaders():
        return (
            DataLoader(
                train,
                batch_size=1,
                shuffle=True,
                generator=torch.Generator().manual_seed(77),
            ),
            DataLoader(
                validation,
                batch_size=1,
                generator=torch.Generator().manual_seed(78),
            ),
        )

    seed_everything(77)
    continuous_model = HighPassResidualCNN(base_channels=4)
    continuous = Trainer(continuous_model, config, tmp_path / "continuous", device="cpu")
    continuous.fit(*loaders())

    seed_everything(77)
    resumed_model_epoch0 = HighPassResidualCNN(base_channels=4)
    first = Trainer(resumed_model_epoch0, config, tmp_path / "resumed", device="cpu")
    train_loader, validation_loader = loaders()
    first.fit(train_loader, validation_loader, max_epochs=1)
    resumed_model = HighPassResidualCNN(base_channels=4)
    resumed = Trainer(resumed_model, config, tmp_path / "resumed", device="cpu")
    train_loader, validation_loader = loaders()
    resumed.fit(
        train_loader,
        validation_loader,
        resume=tmp_path / "resumed" / "checkpoints" / "last.pt",
        max_epochs=2,
    )
    for key, value in continuous_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[key]), key


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_groupdro_resume_exact_model_adversary_optimizer_rng(tmp_path, device):
    if device == "cuda" and (os.environ.get("CXR_TEST_CUDA") != "1" or not torch.cuda.is_available()):
        pytest.skip("Explicit CXR_TEST_CUDA=1 required to avoid concurrent GPU jobs")
    frame = _manifest(tmp_path)
    frame.loc[[0, 2], "view_position"] = "AP"
    train = PairedPatchDataset(frame, split="train", patch_size=16)
    val = PairedPatchDataset(frame, split="validation", patch_size=16)
    c = deepcopy(DEFAULT_CONFIG)
    c.update(seed=77, epochs=2, mixed_precision=device == "cuda", early_stopping_patience=5)
    c["training"].update(train_view="ALL", objective="groupdro", groupdro_step_size=.01)
    def loaders():
        generator = torch.Generator().manual_seed(77)
        return DataLoader(train, batch_sampler=BalancedViewBatchSampler(train.frame.view_position.tolist(),2,generator), generator=generator), DataLoader(val,batch_size=2,generator=torch.Generator().manual_seed(78))
    seed_everything(77)
    full = Trainer(HighPassResidualCNN(4), c, tmp_path/"full",device)
    full.fit(*loaders())
    seed_everything(77)
    first = Trainer(HighPassResidualCNN(4), c, tmp_path/"partial",device)
    first.fit(*loaders(), max_epochs=1)
    resumed = Trainer(HighPassResidualCNN(4), c, tmp_path/"partial",device)
    resumed.fit(*loaders(), resume=tmp_path/"partial/checkpoints/last.pt")
    for key,value in full.model.state_dict().items():
        assert torch.equal(value,resumed.model.state_dict()[key]),key
    assert torch.equal(full.groupdro.log_q, resumed.groupdro.log_q)
    assert torch.equal(full.groupdro.updates, resumed.groupdro.updates)
    assert full.scaler.state_dict() == resumed.scaler.state_dict()
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
