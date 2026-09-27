#!/usr/bin/env python3
"""Evaluate a trained baseline within and across acquisition views."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.evaluation.cross_view import (  # noqa: E402
    evaluate_cross_view,
    save_evaluation_results,
)
from cxr_steganalysis.models.factory import create_model, model_identity  # noqa: E402
from cxr_steganalysis.provenance import sha256_file  # noqa: E402
from cxr_steganalysis.reproducibility import seed_everything  # noqa: E402
from cxr_steganalysis.training.checkpoint import load_checkpoint  # noqa: E402
from cxr_steganalysis.training.trainer import resolve_device  # noqa: E402
from cxr_steganalysis.experiment import effective_seeds, manifest_identity
from cxr_steganalysis.protocol_lock import validate_test_access
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a PA- or AP-trained checkpoint on PA/AP test pairs. "
            "The decision threshold is selected on validation data from the training view."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Pilot YAML configuration")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt checkpoint")
    parser.add_argument(
        "--model",
        choices=["highpass", "srnet", "hsmnet"],
        help="Optional assertion; model identity is otherwise read from the checkpoint",
    )
    parser.add_argument("--manifest", type=Path, help="Override pair manifest CSV")
    parser.add_argument("--train-view", choices=["PA", "AP", "ALL"], help="Override/infer source view")
    parser.add_argument("--test-views", nargs="+", choices=["PA", "AP"], help="Target views")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--num-workers", type=int, help="Override DataLoader workers (use 0 in restricted runtimes)"
    )
    parser.add_argument("--output", type=Path, help="Output stem for JSON and CSV")
    parser.add_argument("--experiment-name", default="pilot")
    parser.add_argument("--test-cohort", choices=["pilot_exposed", "confirmatory_unseen_patient"], help="Full NIH: explicitly separate previously exposed and new patients")
    parser.add_argument("--protocol-lock", type=Path, help="Required for additional confirmatory patients; see freeze_protocol.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        config["num_workers"] = args.num_workers
    device = resolve_device(args.device or config["training"]["device"])
    # Evaluation needs only model weights; keep optimizer/scheduler tensors on CPU.
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    checkpoint_config = checkpoint.get("config", config)
    checkpoint_model = str(
        checkpoint.get(
            "model_name",
            checkpoint_config.get("model", {}).get("name", "highpass"),
        )
    ).lower()
    if args.model is not None and args.model != checkpoint_model:
        raise ValueError(
            f"--model {args.model} conflicts with checkpoint model {checkpoint_model}"
        )
    model_name = args.model or checkpoint_model
    checkpoint_view = str(
        checkpoint_config.get("training", {}).get("train_view", "ALL")
    ).upper()
    if args.train_view is not None and checkpoint_view in {"PA", "AP", "ALL"}:
        if args.train_view != checkpoint_view:
            raise ValueError(
                f"--train-view {args.train_view} conflicts with checkpoint source view "
                f"{checkpoint_view}"
            )
    train_view = args.train_view or checkpoint_view
    if train_view not in {"PA", "AP", "ALL"}:
        raise ValueError(
            "Cross-view evaluation requires a source-specific checkpoint. "
            "Train with --train-view PA or --train-view AP, or pass --train-view explicitly."
        )
    test_views = args.test_views or config["evaluation"]["test_views"]
    seed = int(checkpoint.get("seed", config["seed"]))
    seed_everything(seed)
    checkpoint_config.setdefault("model", {})["name"] = model_name
    model = create_model(model_name, checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    manifest_path = args.manifest or resolve_config_path(config, "pair_manifest")
    training_binding = checkpoint.get("training_binding")
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str, "seed": str})
    identity = manifest_identity(frame)
    protocol_lock_hash = validate_test_access(checkpoint_config, frame, args.test_cohort, args.protocol_lock)
    if training_binding is not None:
        expected_manifest_hash = training_binding.get("manifest_sha256")
        actual_manifest_hash = sha256_file(manifest_path)
        if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash and training_binding.get("manifest_identity") != identity:
            raise ValueError(
                "Evaluation manifest differs from the manifest bound to this checkpoint"
            )
    output = args.output or (
        resolve_config_path(config, "output_dir") / "metrics" / f"evaluation_{train_view.lower()}"
    )
    results = evaluate_cross_view(
        model=model,
        manifest=manifest_path,
        train_view=train_view,
        test_views=test_views,
        checkpoint_path=args.checkpoint,
        seed=seed,
        patch_size=int(checkpoint_config["patch_size"]),
        patches_per_image=int(checkpoint_config["patches_per_image"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        payload_bpp=float(checkpoint_config["payload_bpp"]),
        algorithm=str(checkpoint_config["algorithm"]),
        device=device,
        experiment_name=args.experiment_name,
        normalize=str(checkpoint_config["data"]["normalize"]),
        config_path=args.config,
        checkpoint_epoch=int(checkpoint["epoch"]),
        stored_source_view=checkpoint_view,
        crop_seed=effective_seeds(checkpoint_config)["crop"],
        protocol_id=checkpoint_config.get("protocol_id", "legacy_pilot"),
        objective=checkpoint.get("objective", "erm"),
        manifest_identity=identity,
        test_cohort=args.test_cohort,
        **model_identity(model_name),
    )
    if protocol_lock_hash:
        for result in results:
            result["protocol_lock_sha256"] = protocol_lock_hash
            result["_test_predictions"]["protocol_lock_sha256"] = protocol_lock_hash
            result["_validation_predictions"]["protocol_lock_sha256"] = protocol_lock_hash
    save_evaluation_results(results, output)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['train_view']} -> {result['test_view']}: "
            f"AUC={metrics['roc_auc']:.4f}, balanced_accuracy="
            f"{metrics['balanced_accuracy']:.4f}, threshold={result['threshold']:.4f}"
        )
    print(f"Evaluation JSON/CSV written with stem: {output}")


if __name__ == "__main__":
    main()
