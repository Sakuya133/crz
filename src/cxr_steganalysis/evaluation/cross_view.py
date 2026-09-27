"""Within-view and cross-view evaluation with auditable raw predictions."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cxr_steganalysis.data.dataset import PairedPatchDataset, flatten_paired_batch
from cxr_steganalysis.data.manifest import make_pair_id, validate_pair_manifest
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.reproducibility import runtime_environment
from cxr_steganalysis.training.metrics import binary_metrics, select_threshold


RAW_PREDICTION_COLUMNS = [
    "pair_id",
    "image_id",
    "patient_id",
    "split",
    "view_position",
    "source_view",
    "target_view",
    "class_name",
    "label",
    "logit",
    "score",
    "predicted_label",
    "threshold",
    "patch_index",
    "crop_top",
    "crop_left",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_sha256",
    "manifest_sha256",
    "seed",
    "payload_bpp",
    "algorithm",
]

OPTIONAL_PREDICTION_COLUMNS = ["model_name", "model_version", "score_type"]

PREDICTION_KEY_COLUMNS = [
    "source_view",
    "target_view",
    "split",
    "pair_id",
    "patch_index",
    "class_name",
]

PAIRED_SAMPLE_KEY_COLUMNS = [
    "target_view",
    "split",
    "pair_id",
    "patch_index",
    "class_name",
]

_IDENTIFIER_COLUMNS = [
    "pair_id",
    "image_id",
    "patient_id",
    "split",
    "view_position",
    "source_view",
    "target_view",
    "class_name",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_sha256",
    "manifest_sha256",
    "algorithm",
]


def _ensure_pair_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic pair IDs to a copy of a legacy manifest when needed."""
    result = frame.copy()
    if "pair_id" in result.columns:
        return result
    required = {"image_id", "algorithm", "payload_bpp"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(
            "A manifest without pair_id requires stable identity columns: "
            + ", ".join(sorted(required))
        )
    result["pair_id"] = result.apply(
        lambda row: make_pair_id(row["image_id"], row["algorithm"], row["payload_bpp"]),
        axis=1,
    )
    return result


def _batch_values(value: Any, expected: int, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"Batched metadata {name!r} has unsupported type {type(value)!r}")
    if len(values) != expected:
        raise ValueError(
            f"Batched metadata {name!r} has {len(values)} values; expected {expected}"
        )
    return values


def _raw_rows_for_batch(
    batch: dict[str, Any],
    labels: np.ndarray,
    logits: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Evaluation dataset must return a metadata dictionary")
    batch_size = int(batch["cover"].shape[0])
    required = [
        "pair_id", "image_id", "patient_id", "view_position", "split",
        "patch_index", "crop_top", "crop_left",
    ]
    missing = sorted(set(required) - set(metadata))
    if missing:
        raise ValueError(f"Evaluation batch metadata missing column(s): {', '.join(missing)}")
    values = {
        name: _batch_values(metadata[name], batch_size, name)
        for name in required
    }
    rows: list[dict[str, Any]] = []
    class_names = ("cover", "stego")
    for class_index, class_name in enumerate(class_names):
        offset = class_index * batch_size
        for item_index in range(batch_size):
            prediction_index = offset + item_index
            rows.append({
                "pair_id": str(values["pair_id"][item_index]),
                "image_id": str(values["image_id"][item_index]),
                "patient_id": str(values["patient_id"][item_index]),
                "split": str(values["split"][item_index]),
                "view_position": str(values["view_position"][item_index]),
                "class_name": class_name,
                "label": int(labels[prediction_index]),
                "logit": float(logits[prediction_index]),
                "score": float(scores[prediction_index]),
                "patch_index": int(values["patch_index"][item_index]),
                "crop_top": int(values["crop_top"][item_index]),
                "crop_left": int(values["crop_left"][item_index]),
            })
    return rows


@torch.no_grad()
def predict_paired(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Predict paired batches; the default two-array API remains backward compatible."""
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images, labels = flatten_paired_batch(batch)
        model_output = model(images.to(device, non_blocking=True))
        if model_output.ndim == 1:
            logits_tensor = model_output
            scores_tensor = torch.sigmoid(logits_tensor)
        elif model_output.ndim == 2 and model_output.shape[1] == 2:
            # The logit margin has sigmoid(margin) == softmax(logits)[stego].
            logits_tensor = model_output[:, 1] - model_output[:, 0]
            scores_tensor = torch.softmax(model_output, dim=1)[:, 1]
        else:
            raise ValueError(
                "Model must return one binary logit or two cover/stego logits per "
                f"sample; got {tuple(model_output.shape)}"
            )
        labels_array = labels.numpy()
        logits_array = logits_tensor.detach().cpu().numpy()
        scores_array = scores_tensor.detach().cpu().numpy()
        labels_all.append(labels_array)
        scores_all.append(scores_array)
        if return_metadata:
            rows.extend(
                _raw_rows_for_batch(
                    batch, labels_array, logits_array, scores_array
                )
            )
    if not labels_all:
        raise ValueError("Evaluation loader produced no batches")
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    if return_metadata:
        predictions = pd.DataFrame(rows)
        if len(predictions) != len(labels):
            raise AssertionError("Raw prediction rows do not match evaluated samples")
        return labels, scores, predictions
    return labels, scores


def _finalize_predictions(
    predictions: pd.DataFrame,
    *,
    source_view: str,
    target_view: str,
    threshold: float,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    config_sha256: str,
    manifest_sha256: str,
    seed: int,
    payload_bpp: float,
    algorithm: str,
    model_name: str | None = None,
    model_version: str | None = None,
    score_type: str = "sigmoid_probability",
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["source_view"] = source_view
    frame["target_view"] = frame["view_position"] if target_view == "ALL" else target_view
    frame["predicted_label"] = (
        frame["score"].to_numpy(dtype=np.float64) >= float(threshold)
    ).astype(np.int64)
    frame["threshold"] = float(threshold)
    frame["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    frame["checkpoint_sha256"] = checkpoint_sha256
    frame["config_sha256"] = config_sha256
    frame["manifest_sha256"] = manifest_sha256
    frame["seed"] = int(seed)
    frame["payload_bpp"] = float(payload_bpp)
    frame["algorithm"] = algorithm
    frame["model_name"] = model_name or "unspecified"
    frame["model_version"] = model_version or "unspecified"
    frame["score_type"] = score_type
    frame = frame[RAW_PREDICTION_COLUMNS + OPTIONAL_PREDICTION_COLUMNS]
    validate_prediction_frame(frame)
    return frame


def _is_blank(series: pd.Series) -> pd.Series:
    strings = series.astype("string")
    return strings.isna() | strings.str.strip().isin({"", "nan", "None", "<NA>"})


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_split: str | None = None,
    expected_target_view: str | None = None,
) -> None:
    """Hard validation for one-row-per-classification-sample predictions."""
    missing = sorted(set(RAW_PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file missing column(s): {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Prediction file is empty")
    for column in _IDENTIFIER_COLUMNS:
        if _is_blank(frame[column]).any():
            raise ValueError(f"Prediction column {column!r} contains missing/blank values")
    for column in OPTIONAL_PREDICTION_COLUMNS:
        if column in frame.columns and _is_blank(frame[column]).any():
            raise ValueError(f"Prediction column {column!r} contains missing/blank values")
    if frame[PREDICTION_KEY_COLUMNS].duplicated().any():
        duplicates = frame.loc[
            frame[PREDICTION_KEY_COLUMNS].duplicated(keep=False), PREDICTION_KEY_COLUMNS
        ].head(5)
        raise ValueError(f"Prediction composite key is not unique: {duplicates.to_dict('records')}")
    if expected_split is not None and set(frame["split"].astype(str)) != {expected_split}:
        raise ValueError(f"Prediction rows must all use split={expected_split!r}")
    if expected_target_view is not None:
        targets = set(frame["target_view"].astype(str))
        if targets != {expected_target_view}:
            raise ValueError(
                f"Prediction target view mismatch: expected {expected_target_view}, got {targets}"
            )
    if not set(frame["view_position"].astype(str)).issubset({"AP", "PA"}):
        raise ValueError("Prediction rows contain a view other than AP/PA")
    if not set(frame["source_view"].astype(str)).issubset({"AP", "PA", "ALL"}):
        raise ValueError("Prediction rows contain a source view other than AP/PA")
    if not set(frame["target_view"].astype(str)).issubset({"AP", "PA"}):
        raise ValueError("Prediction rows contain a target view other than AP/PA")
    if not frame["view_position"].astype(str).eq(
        frame["target_view"].astype(str)
    ).all():
        raise ValueError("Prediction view_position must equal target_view")
    if expected_split == "validation" and not (
        frame["source_view"].astype(str).eq(frame["target_view"].astype(str))
        | frame["source_view"].eq("ALL")
    ).all():
        raise ValueError("Validation predictions must use the source validation view")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    expected_labels = frame["class_name"].map({"cover": 0, "stego": 1})
    if expected_labels.isna().any() or not np.array_equal(labels, expected_labels.to_numpy()):
        raise ValueError("Prediction labels must be cover=0 and stego=1")
    numeric_columns = [
        "logit", "score", "threshold", "patch_index", "crop_top", "crop_left",
        "seed", "payload_bpp",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Prediction column {column!r} contains non-finite values")
    for column in ("patch_index", "crop_top", "crop_left"):
        if (pd.to_numeric(frame[column], errors="raise") < 0).any():
            raise ValueError(f"Prediction column {column!r} cannot be negative")
    score_types = (
        set(frame["score_type"].astype(str))
        if "score_type" in frame.columns
        else {"sigmoid_probability"}
    )
    if len(score_types) != 1 or not score_types <= {
        "sigmoid_probability", "decision_margin"
    }:
        raise ValueError(f"Unknown or mixed prediction score_type: {score_types}")
    score_type = next(iter(score_types))
    if score_type == "sigmoid_probability":
        if not frame["score"].between(0.0, 1.0, inclusive="both").all():
            raise ValueError("Probability scores must be in [0, 1]")
    if not frame["payload_bpp"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Prediction payload_bpp must be in [0, 1]")
    if score_type == "sigmoid_probability":
        expected_scores = 1.0 / (
            1.0 + np.exp(-frame["logit"].to_numpy(dtype=np.float64))
        )
        if not np.allclose(
            frame["score"].to_numpy(dtype=np.float64),
            expected_scores,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("Prediction score is inconsistent with sigmoid(logit)")
    elif not np.allclose(
        frame["score"].to_numpy(dtype=np.float64),
        frame["logit"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Decision-margin score must equal the stored raw logit field")
    expected_predictions = (
        frame["score"].to_numpy(dtype=np.float64)
        >= frame["threshold"].to_numpy(dtype=np.float64)
    ).astype(np.int64)
    if not np.array_equal(
        pd.to_numeric(frame["predicted_label"], errors="raise").to_numpy(dtype=np.int64),
        expected_predictions,
    ):
        raise ValueError("predicted_label is inconsistent with score and threshold")
    for hash_column in ("checkpoint_sha256", "config_sha256", "manifest_sha256"):
        if not frame[hash_column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Prediction column {hash_column!r} contains an invalid SHA-256")

    pair_patch = ["pair_id", "patch_index"]
    counts = frame.groupby(pair_patch, dropna=False).size()
    if not counts.eq(2).all():
        raise ValueError("Each pair/patch must have exactly one cover and one stego row")
    for column in (
        "image_id", "patient_id", "split", "view_position", "source_view",
        "target_view", "crop_top", "crop_left",
    ):
        if frame.groupby(pair_patch, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(
                f"Cover/stego rows for a pair/patch disagree on {column!r}"
            )


def _assert_metric_consistency(
    predictions: pd.DataFrame,
    aggregate: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> None:
    thresholds = predictions["threshold"].unique()
    if len(thresholds) != 1:
        raise ValueError("Prediction threshold must be constant within one evaluation")
    reconstructed = binary_metrics(
        predictions["label"].to_numpy(dtype=np.int64),
        predictions["score"].to_numpy(dtype=np.float64),
        float(thresholds[0]),
    )
    scalar_metrics = [
        "roc_auc", "balanced_accuracy", "accuracy", "precision", "recall",
        "sensitivity", "specificity", "f1", "false_positive_rate",
        "false_negative_rate", "error_probability", "threshold",
    ]
    for name in scalar_metrics:
        expected = float(aggregate[name])
        actual = float(reconstructed[name])
        if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(
                f"Raw predictions reconstruct {name}={actual}, aggregate stores {expected}"
            )
    if reconstructed["confusion_matrix"] != aggregate["confusion_matrix"]:
        raise AssertionError(
            "Raw prediction confusion matrix does not match aggregate [[TN, FP], [FN, TP]]"
        )
    for name in ("tn", "fp", "fn", "tp"):
        if int(reconstructed[name]) != int(aggregate[name]):
            raise AssertionError(f"Raw predictions do not reconstruct aggregate {name}")


def evaluate_cross_view(
    model: torch.nn.Module,
    manifest: str | Path | pd.DataFrame,
    train_view: str,
    test_views: Iterable[str],
    checkpoint_path: str | Path,
    seed: int,
    patch_size: int,
    patches_per_image: int,
    batch_size: int,
    num_workers: int,
    payload_bpp: float,
    algorithm: str,
    device: torch.device,
    experiment_name: str,
    normalize: str = "unit_range",
    *,
    config_path: str | Path | None = None,
    checkpoint_epoch: int | None = None,
    stored_source_view: str | None = None,
    model_name: str = "highpass",
    model_version: str = "HighPassResidualCNN-v1",
    crop_seed: int | None = None,
    protocol_id: str = "legacy_pilot",
    objective: str = "erm",
    manifest_identity: str | None = None,
    test_cohort: str | None = None,
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest) if isinstance(manifest, (str, Path)) else None
    frame = (
        pd.read_csv(manifest_path, dtype={"patient_id": str, "seed": str})
        if manifest_path is not None
        else manifest.copy()
    )
    frame = _ensure_pair_ids(frame)
    validate_pair_manifest(frame, check_files=False)
    if test_cohort is not None:
        if "test_cohort" not in frame or test_cohort not in set(frame.test_cohort):
            raise ValueError("Requested test cohort is absent; no fallback to exposed patients")
        frame = frame[(frame.split != "test") | frame.test_cohort.eq(test_cohort)]
    if manifest_path is None:
        raise ValueError(
            "Auditable evaluation requires a manifest file path so its SHA-256 can be recorded"
        )
    if config_path is None:
        raise ValueError(
            "Auditable evaluation requires config_path so the YAML SHA-256 can be recorded"
        )
    provenance = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    source_view = train_view.upper()
    if source_view not in {"AP", "PA", "ALL"}:
        raise ValueError("Unknown source view")
    crop_seed = seed if crop_seed is None else crop_seed
    stored_view = (stored_source_view or source_view).upper()
    if stored_view in {"AP", "PA"} and stored_view != source_view:
        raise ValueError(
            f"Stored checkpoint source view {stored_view} conflicts with evaluation source "
            f"view {source_view}"
        )
    validation_dataset = PairedPatchDataset(
        frame,
        split="validation",
        view_position=source_view,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        seed=crop_seed,
        normalize=normalize,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    validation_labels, validation_scores, validation_raw = predict_paired(
        model, validation_loader, device, return_metadata=True
    )
    threshold = select_threshold(validation_labels, validation_scores)
    validation_predictions = _finalize_predictions(
        validation_raw,
        source_view=source_view,
        target_view=source_view,
        threshold=threshold,
        checkpoint_path=checkpoint_path,
        seed=seed,
        payload_bpp=payload_bpp,
        algorithm=algorithm,
        model_name=model_name,
        model_version=model_version,
        **provenance,
    )
    if source_view == "ALL":
        validation_predictions["target_view"] = validation_predictions["view_position"]
    extra = {"protocol_id": protocol_id, "objective": objective, "crop_seed": crop_seed,
             "patch_size": patch_size, "normalization": normalize,
             "test_cohort": test_cohort or "all_enrolled_test",
             "manifest_identity": manifest_identity or provenance["manifest_sha256"]}
    for key, value in extra.items():
        validation_predictions[key] = value

    results: list[dict[str, Any]] = []
    environment = runtime_environment()
    for test_view in test_views:
        target_view = test_view.upper()
        started = time.perf_counter()
        test_dataset = PairedPatchDataset(
            frame,
            split="test",
            view_position=target_view,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            seed=crop_seed,
            normalize=normalize,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        labels, scores, test_raw = predict_paired(
            model, test_loader, device, return_metadata=True
        )
        test_predictions = _finalize_predictions(
            test_raw,
            source_view=source_view,
            target_view=target_view,
            threshold=threshold,
            checkpoint_path=checkpoint_path,
            seed=seed,
            payload_bpp=payload_bpp,
            algorithm=algorithm,
            model_name=model_name,
            model_version=model_version,
            **provenance,
        )
        for key, value in extra.items():
            test_predictions[key] = value
        metrics = binary_metrics(labels, scores, threshold)
        _assert_metric_consistency(test_predictions, metrics)
        results.append({
            "experiment_name": experiment_name,
            "model_name": model_name,
            "model_version": model_version,
            **extra,
            "train_view": source_view,
            "test_view": target_view,
            "stored_source_view": stored_view,
            "seed": int(seed),
            "evaluation_seed": int(seed),
            "best_checkpoint_epoch": (
                int(checkpoint_epoch) if checkpoint_epoch is not None else None
            ),
            "dataset_size": {
                "patients": int(test_dataset.frame["patient_id"].astype(str).nunique()),
                "pairs": len(test_dataset.frame),
                "patches_per_class": len(test_dataset),
                "classification_samples": len(labels),
            },
            "validation_size": {
                "patients": int(
                    validation_dataset.frame["patient_id"].astype(str).nunique()
                ),
                "pairs": len(validation_dataset.frame),
                "patches_per_class": len(validation_dataset),
                "classification_samples": len(validation_predictions),
            },
            "validation_pairs_for_threshold": len(validation_dataset.frame),
            "payload_bpp": float(payload_bpp),
            "algorithm": algorithm,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            **provenance,
            "provenance_scope": "evaluation_time",
            "threshold": threshold,
            "metrics": metrics,
            "runtime_seconds": time.perf_counter() - started,
            "runtime_environment": environment,
            "_validation_predictions": validation_predictions,
            "_test_predictions": test_predictions,
        })
    return results


def _prediction_path(stem: Path, suffix: str) -> Path:
    return stem.with_name(stem.name + suffix)


def _read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={
            "pair_id": str,
            "image_id": str,
            "patient_id": str,
            "source_view": str,
            "target_view": str,
            "checkpoint_sha256": str,
            "config_sha256": str,
            "manifest_sha256": str,
        },
    )


def save_evaluation_results(results: list[dict[str, Any]], output_stem: str | Path) -> None:
    if not results:
        raise ValueError("No evaluation results to save")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    test_prediction_path = _prediction_path(stem, "_predictions.csv")
    validation_prediction_path = _prediction_path(
        stem, "_validation_predictions.csv"
    )
    existing = [
        path for path in (
            json_path,
            csv_path,
            test_prediction_path,
            validation_prediction_path,
        ) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation artifacts; choose a new "
            f"output stem: {existing}"
        )

    test_predictions = pd.concat(
        [result["_test_predictions"] for result in results], ignore_index=True
    )
    validation_predictions = results[0]["_validation_predictions"].copy()
    for result in results[1:]:
        candidate = result["_validation_predictions"]
        if not validation_predictions.equals(candidate):
            raise AssertionError("Validation predictions changed between target evaluations")
    validate_prediction_frame(test_predictions, expected_split="test")
    validate_prediction_frame(validation_predictions, expected_split="validation")
    test_predictions.to_csv(test_prediction_path, index=False)
    validation_predictions.to_csv(validation_prediction_path, index=False)

    persisted_test = _read_predictions(test_prediction_path)
    persisted_validation = _read_predictions(validation_prediction_path)
    validate_prediction_frame(persisted_test, expected_split="test")
    validate_prediction_frame(persisted_validation, expected_split="validation")
    persisted_thresholds = persisted_validation["threshold"].unique()
    if len(persisted_thresholds) != 1:
        raise AssertionError("Persisted validation predictions contain multiple thresholds")
    reconstructed_threshold = select_threshold(
        persisted_validation["label"].to_numpy(dtype=np.int64),
        persisted_validation["score"].to_numpy(dtype=np.float64),
    )
    if not math.isclose(
        reconstructed_threshold,
        float(persisted_thresholds[0]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("Persisted threshold was not selected from validation predictions")

    public_results: list[dict[str, Any]] = []
    test_file = str(test_prediction_path.resolve())
    validation_file = str(validation_prediction_path.resolve())
    for result in results:
        target_view = str(result["test_view"])
        target_rows = persisted_test[persisted_test["target_view"] == target_view].copy()
        validate_prediction_frame(
            target_rows, expected_split="test", expected_target_view=target_view
        )
        _assert_metric_consistency(target_rows, result["metrics"])
        public = {
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        }
        public["validation_prediction_file"] = validation_file
        public["test_prediction_file"] = test_file
        public_results.append(public)

    json_path.write_text(
        json.dumps(public_results, indent=2, allow_nan=False), encoding="utf-8"
    )
    flat_rows = []
    for result in public_results:
        row = {key: value for key, value in result.items() if not isinstance(value, dict)}
        row.update({f"dataset_{key}": value for key, value in result["dataset_size"].items()})
        row.update({
            f"validation_{key}": value for key, value in result["validation_size"].items()
        })
        row.update(result["metrics"])
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        row["runtime_environment"] = json.dumps(
            result["runtime_environment"], sort_keys=True
        )
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
