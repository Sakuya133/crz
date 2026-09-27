#!/usr/bin/env python3
"""Train an explicitly selected CNN steganalysis model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.data.dataset import PairedPatchDataset  # noqa: E402
from cxr_steganalysis.data.manifest import validate_pair_manifest  # noqa: E402
from cxr_steganalysis.models.factory import create_model, model_identity  # noqa: E402
from cxr_steganalysis.provenance import sha256_file  # noqa: E402
from cxr_steganalysis.reproducibility import seed_everything, seed_worker  # noqa: E402
from cxr_steganalysis.training.trainer import Trainer  # noqa: E402
from cxr_steganalysis.training.groupdro import BalancedViewBatchSampler
from cxr_steganalysis.experiment import effective_seeds, manifest_identity, select_mixed_training, save_run_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a selected CNN for cover=0 versus stego=1."
    )
    parser.add_argument("--config", type=Path, required=True, help="Pilot YAML configuration")
    parser.add_argument("--manifest", type=Path, help="Override pair manifest CSV")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument(
        "--model",
        choices=["highpass", "srnet", "hsmnet"],
        help="Explicit model; default is config model.name (highpass in pilot config)",
    )
    parser.add_argument("--train-view", choices=["ALL", "PA", "AP"], help="Source view filter")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--resume", type=Path, help="Resume from a last.pt checkpoint")
    parser.add_argument("--epochs", type=int, help="Override total number of epochs")
    parser.add_argument(
        "--num-workers", type=int, help="Override DataLoader workers (use 0 in restricted runtimes)"
    )
    parser.add_argument(
        "--overfit-pairs",
        type=int,
        help="Debug only: limit training to the first N pairs (recommended 50-100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["epochs"] = args.epochs
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        config["num_workers"] = args.num_workers
    train_view = args.train_view or str(config["training"]["train_view"]).upper()
    config["training"]["train_view"] = train_view
    model_name = args.model or str(config["model"].get("name", "highpass"))
    config["model"]["name"] = model_name
    manifest_path = args.manifest or resolve_config_path(config, "pair_manifest")
    output_dir = args.output_dir or resolve_config_path(config, "output_dir")
    resume = args.resume or config["training"].get("resume")
    if not resume and Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise FileExistsError("Training output is not empty; use explicit --resume or a new directory")
    if config.get("dataset_profile") != "synthetic_smoke" and (args.device or config["training"]["device"]) == "auto" and not torch.cuda.is_available():
        raise RuntimeError("NIH training cannot silently fall back to CPU; use the synthetic smoke preset")
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str, "seed": str})
    validate_pair_manifest(frame, check_files=False)
    original_identity = manifest_identity(frame)
    seeds = effective_seeds(config)
    if config["training"].get("mixed_pairs_per_view") is not None:
        if train_view != "ALL":
            raise ValueError("Mixed protocol requires --train-view ALL")
        frame = select_mixed_training(frame, int(config["training"]["mixed_pairs_per_view"]), seeds["sampling"])
    if args.overfit_pairs is not None:
        if args.overfit_pairs <= 0:
            raise ValueError("--overfit-pairs must be positive")
        eligible = frame[frame["split"] == "train"]
        if train_view != "ALL":
            eligible = eligible[eligible["view_position"] == train_view]
        keep_ids = set(eligible.head(args.overfit_pairs)["pair_id"])
        frame = pd.concat([
            frame[(frame["split"] == "train") & frame["pair_id"].isin(keep_ids)],
            frame[frame["split"] == "validation"],
        ], ignore_index=True)
        print(f"Overfit debug mode: retained {len(keep_ids)} training pairs")

    seed = seeds["training"]
    config["seed"] = seed
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    seed_everything(seed)
    dataset_args = {
        "patch_size": int(config["patch_size"]),
        "patches_per_image": int(config["patches_per_image"]),
        "seed": seeds["crop"],
        "normalize": config["data"]["normalize"],
    }
    train_dataset = PairedPatchDataset(
        frame, split="train", view_position=train_view, **dataset_args
    )
    validation_dataset = PairedPatchDataset(
        frame, split="validation", view_position=train_view, **dataset_args
    )
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    loader_args = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if config["training"].get("mixed_pairs_per_view") is not None:
        mixed_loader_args = {k: v for k, v in loader_args.items() if k != "batch_size"}
        views = train_dataset.frame.view_position.repeat(int(config["patches_per_image"])).tolist()
        sampler = BalancedViewBatchSampler(views, int(config["batch_size"]), train_generator)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, generator=train_generator, **mixed_loader_args)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator, **loader_args)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, generator=validation_generator, **loader_args
    )
    model = create_model(model_name, config)
    device = args.device or config["training"]["device"]
    binding = {
        **model_identity(model_name),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_identity": original_identity,
        "protocol_id": config.get("protocol_id", "legacy_pilot"),
        "seeds": seeds,
        "objective": config["training"].get("objective", "erm"),
        "train_pair_ids": sorted(train_dataset.frame.pair_id.astype(str)),
        "validation_pair_ids": sorted(validation_dataset.frame.pair_id.astype(str)),
        "source_view": train_view,
        "seed": seed,
        "patch_size": int(config["patch_size"]),
        "patches_per_image": int(config["patches_per_image"]),
        "payload_bpp": float(config["payload_bpp"]),
        "algorithm": str(config["algorithm"]),
        "normalization": str(config["data"]["normalize"]),
        "debug_overfit_pair_ids": sorted(keep_ids) if args.overfit_pairs is not None else None,
    }
    trainer = Trainer(
        model,
        config,
        output_dir=output_dir,
        device=device,
        model_name=model_name,
        training_binding=binding,
    )
    resume = args.resume or config["training"].get("resume")
    if not resume and (Path(output_dir) / "checkpoints/last.pt").exists():
        raise FileExistsError("Existing run: use --resume or a new output directory")
    save_run_provenance(output_dir, config, binding)
    summary = trainer.fit(train_loader, validation_loader, resume=resume)
    print(
        f"Training {model_name} complete on {summary['device']}; "
        f"best metric={summary['best_metric']:.4f}"
    )
    print(f"Best checkpoint: {summary['best_checkpoint']}")


if __name__ == "__main__":
    main()
