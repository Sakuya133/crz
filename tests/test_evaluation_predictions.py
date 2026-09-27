from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from cxr_steganalysis.data.dataset import PairedPatchDataset
from cxr_steganalysis.data.manifest import (
    build_pair_manifest,
    generate_stego_records,
    save_manifest,
)
from cxr_steganalysis.evaluation import cross_view
from cxr_steganalysis.evaluation.cross_view import (
    PREDICTION_KEY_COLUMNS,
    RAW_PREDICTION_COLUMNS,
    evaluate_cross_view,
    predict_paired,
    save_evaluation_results,
    validate_prediction_frame,
)
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.training.metrics import binary_metrics, select_threshold


class MeanModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=(1, 2, 3)) * 4.0 - 2.0


class TwoLogitMeanModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        margin = inputs.mean(dim=(1, 2, 3)) * 4.0 - 2.0
        return torch.stack((-margin / 2.0, margin / 2.0), dim=1)


def _evaluation_fixture(tmp_path: Path) -> dict[str, Path]:
    rows = []
    rng = np.random.default_rng(17)
    splits = ["validation", "validation", "test", "test", "test"]
    for index, split in enumerate(splits):
        cover_path = tmp_path / "covers" / f"{index:08d}_000.png"
        cover_path.parent.mkdir(parents=True, exist_ok=True)
        image = rng.integers(0, 256, (24, 24), dtype=np.uint8)
        Image.fromarray(image, mode="L").save(cover_path)
        rows.append({
            "image_id": cover_path.name,
            "patient_id": str(index + 1),
            "view_position": "PA",
            "finding_labels": "No Finding",
            "split": split,
            "cover_path": str(cover_path),
        })
    split_frame = pd.DataFrame(rows)
    generation = generate_stego_records(
        split_frame, tmp_path / "stego", payload_bpp=0.5, global_seed=1337
    )
    manifest = build_pair_manifest(split_frame, generation)
    manifest_path = tmp_path / "cover_stego.csv"
    save_manifest(manifest, manifest_path)
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"epoch": 3, "source_view": "PA"}, checkpoint_path)
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text("seed: 1337\npatch_size: 12\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "checkpoint": checkpoint_path,
        "config": config_path,
    }


def _run_evaluation(tmp_path: Path, monkeypatch=None):
    paths = _evaluation_fixture(tmp_path)
    threshold_calls: list[int] = []
    if monkeypatch is not None:
        original = cross_view.select_threshold

        def threshold_spy(labels, scores):
            threshold_calls.append(len(labels))
            return original(labels, scores)

        monkeypatch.setattr(cross_view, "select_threshold", threshold_spy)
    results = evaluate_cross_view(
        MeanModel(),
        paths["manifest"],
        "PA",
        ["PA"],
        paths["checkpoint"],
        1337,
        12,
        1,
        2,
        0,
        0.5,
        "lsb_matching",
        torch.device("cpu"),
        "synthetic_prediction_test",
        config_path=paths["config"],
        checkpoint_epoch=3,
        stored_source_view="PA",
    )
    output = tmp_path / "metrics" / "pa_to_pa"
    save_evaluation_results(results, output)
    return paths, output, results, threshold_calls


def test_dataset_metadata_and_predict_api_remain_backward_compatible(tmp_path):
    paths = _evaluation_fixture(tmp_path)
    dataset = PairedPatchDataset(
        paths["manifest"], split="test", view_position="PA", patch_size=12
    )
    sample = dataset[0]
    assert {"patient_id", "image_id", "pair_id"} <= set(sample["metadata"])
    assert sample["metadata"]["crop_top"] == dataset[0]["metadata"]["crop_top"]
    assert sample["metadata"]["crop_left"] == dataset[0]["metadata"]["crop_left"]
    predictions = predict_paired(
        MeanModel(), DataLoader(dataset, batch_size=2), torch.device("cpu")
    )
    assert isinstance(predictions, tuple) and len(predictions) == 2
    two_logit_labels, two_logit_scores = predict_paired(
        TwoLogitMeanModel(), DataLoader(dataset, batch_size=2), torch.device("cpu")
    )
    assert np.array_equal(two_logit_labels, predictions[0])
    assert np.allclose(two_logit_scores, predictions[1])


def test_raw_prediction_export_and_metric_reconstruction(tmp_path, monkeypatch):
    paths, output, results, threshold_calls = _run_evaluation(tmp_path, monkeypatch)
    test_path = output.with_name(output.name + "_predictions.csv")
    validation_path = output.with_name(output.name + "_validation_predictions.csv")
    test = pd.read_csv(test_path, dtype={"patient_id": str})
    validation = pd.read_csv(validation_path, dtype={"patient_id": str})
    assert set(RAW_PREDICTION_COLUMNS) <= set(test.columns)
    assert len(test) == 6
    assert len(validation) == 4
    assert not test[PREDICTION_KEY_COLUMNS].duplicated().any()
    assert test.groupby(["pair_id", "patch_index"])["crop_top"].nunique().eq(1).all()
    assert test.groupby(["pair_id", "patch_index"])["crop_left"].nunique().eq(1).all()
    assert set(test.loc[test["class_name"] == "cover", "label"]) == {0}
    assert set(test.loc[test["class_name"] == "stego", "label"]) == {1}
    assert threshold_calls == [4, 4]
    assert float(test["threshold"].iloc[0]) == pytest.approx(
        select_threshold(validation["label"].to_numpy(), validation["score"].to_numpy())
    )

    reconstructed = binary_metrics(
        test["label"].to_numpy(),
        test["score"].to_numpy(),
        float(test["threshold"].iloc[0]),
    )
    for metric in (
        "roc_auc", "balanced_accuracy", "precision", "recall", "specificity", "f1"
    ):
        assert reconstructed[metric] == pytest.approx(results[0]["metrics"][metric])
    assert reconstructed["confusion_matrix"] == results[0]["metrics"]["confusion_matrix"]

    assert set(test["checkpoint_sha256"]) == {sha256_file(paths["checkpoint"])}
    assert set(test["config_sha256"]) == {sha256_file(paths["config"])}
    assert set(test["manifest_sha256"]) == {sha256_file(paths["manifest"])}
    aggregate = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))[0]
    assert aggregate["best_checkpoint_epoch"] == 3
    assert aggregate["stored_source_view"] == "PA"
    assert aggregate["dataset_size"]["patients"] == 3
    assert aggregate["validation_size"]["patients"] == 2
    assert aggregate["test_prediction_file"] == str(test_path.resolve())
    assert aggregate["validation_prediction_file"] == str(validation_path.resolve())


def test_prediction_validation_rejects_duplicate_and_missing_patient(tmp_path):
    _paths, output, _results, _calls = _run_evaluation(tmp_path)
    frame = pd.read_csv(output.with_name(output.name + "_predictions.csv"))
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="composite key"):
        validate_prediction_frame(duplicate)
    missing = frame.copy()
    missing.loc[0, "patient_id"] = None
    with pytest.raises(ValueError, match="patient_id"):
        validate_prediction_frame(missing)


def test_prediction_validation_accepts_explicit_svm_decision_margins(tmp_path):
    _paths, output, _results, _calls = _run_evaluation(tmp_path)
    frame = pd.read_csv(output.with_name(output.name + "_predictions.csv"))
    margins = np.linspace(-2.0, 3.0, len(frame))
    frame["score_type"] = "decision_margin"
    frame["model_name"] = "srm_svm"
    frame["model_version"] = "test"
    frame["logit"] = margins
    frame["score"] = margins
    frame["threshold"] = 1.5
    frame["predicted_label"] = (margins >= 1.5).astype(int)
    validate_prediction_frame(frame)


def test_legacy_pair_id_derivation_is_deterministic_and_non_mutating():
    legacy = pd.DataFrame([{
        "image_id": "00000001_000.png",
        "algorithm": "lsb_matching",
        "payload_bpp": 0.2,
    }])
    first = cross_view._ensure_pair_ids(legacy)
    second = cross_view._ensure_pair_ids(legacy)
    assert "pair_id" not in legacy.columns
    assert first.loc[0, "pair_id"] == second.loc[0, "pair_id"]
    assert len(first.loc[0, "pair_id"]) == 24


def test_confirmatory_filter_never_changes_source_validation(tmp_path):
    paths=_evaluation_fixture(tmp_path)
    frame=pd.read_csv(paths["manifest"],dtype={"patient_id":str,"seed":str})
    frame["test_cohort"]=["not_test","not_test","pilot_exposed","confirmatory_unseen_patient","confirmatory_unseen_patient"]
    frame.to_csv(paths["manifest"],index=False)
    arguments=(MeanModel(),paths["manifest"],"PA",["PA"],paths["checkpoint"],1337,12,1,2,0,.5,"lsb_matching",torch.device("cpu"),"synthetic_cohort_filter")
    full=evaluate_cross_view(*arguments,config_path=paths["config"])[0]
    confirmed=evaluate_cross_view(*arguments,config_path=paths["config"],test_cohort="confirmatory_unseen_patient")[0]
    assert full["threshold"]==confirmed["threshold"]
    assert confirmed["dataset_size"]["pairs"]==2
    assert confirmed["validation_size"]==full["validation_size"]
    assert set(confirmed["_test_predictions"].patient_id)=={"4","5"}
    for column in ("pair_id","score","predicted_label","threshold"):
        assert confirmed["_validation_predictions"][column].equals(full["_validation_predictions"][column])
