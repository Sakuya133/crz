"""Patient-cluster paired bootstrap for fixed-target cross-view comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cxr_steganalysis.evaluation.cross_view import (
    PAIRED_SAMPLE_KEY_COLUMNS,
    validate_prediction_frame,
)
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.training.metrics import binary_metrics


BOOTSTRAP_METRICS = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "false_positive_rate")


def load_prediction_file(path: str | Path) -> pd.DataFrame:
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    return pd.read_csv(
        prediction_path,
        float_precision="round_trip",
        dtype={
            "pair_id": str,
            "image_id": str,
            "patient_id": str,
            "split": str,
            "view_position": str,
            "source_view": str,
            "target_view": str,
            "class_name": str,
            "checkpoint_sha256": str,
            "config_sha256": str,
            "manifest_sha256": str,
        },
    )


def resample_patient_clusters(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row positions for a patient bootstrap, retaining complete clusters."""
    patients = np.asarray(sorted(frame["patient_id"].astype(str).unique()), dtype=object)
    if len(patients) == 0:
        raise ValueError("Cannot bootstrap an empty patient set")
    selected = rng.choice(patients, size=len(patients), replace=True)
    patient_values = frame["patient_id"].astype(str).to_numpy()
    positions = np.concatenate([
        np.flatnonzero(patient_values == patient) for patient in selected
    ])
    return positions, selected


def _constant_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Prediction column {column!r} must be constant within a file")
    return values.iloc[0]


def _align_prediction_frames(
    matched: pd.DataFrame,
    mismatched: pd.DataFrame,
    target_view: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, float, float]:
    target = target_view.upper()
    validate_prediction_frame(
        matched, expected_split="test", expected_target_view=target
    )
    validate_prediction_frame(
        mismatched, expected_split="test", expected_target_view=target
    )
    matched_source = str(_constant_value(matched, "source_view")).upper()
    mismatched_source = str(_constant_value(mismatched, "source_view")).upper()
    if {matched_source, mismatched_source} != {"AP", "PA"}:
        raise ValueError("Cross-view mode requires AP and PA source-specific models")
    if matched_source != target:
        raise ValueError(
            f"Matched predictions must use source_view={target}, got {matched_source}"
        )
    if mismatched_source == target:
        raise ValueError(
            f"Mismatched predictions must use a source view different from {target}"
        )
    matched_threshold = float(_constant_value(matched, "threshold"))
    mismatched_threshold = float(_constant_value(mismatched, "threshold"))
    for column in (
        "checkpoint_sha256", "config_sha256", "manifest_sha256",
        "payload_bpp", "algorithm",
    ):
        _constant_value(matched, column)
        _constant_value(mismatched, column)
    for column in ("model_name", "model_version", "score_type"):
        if column in matched.columns or column in mismatched.columns:
            if column not in matched.columns or column not in mismatched.columns:
                raise ValueError(f"Only one prediction file contains {column!r}")
            left_value = _constant_value(matched, column)
            right_value = _constant_value(mismatched, column)
            if left_value != right_value:
                raise ValueError(
                    f"Matched and mismatched predictions use different {column}: "
                    f"{left_value!r} vs {right_value!r}"
                )
    for column in ("seed", "protocol_id", "crop_seed", "patch_size", "normalization", "objective"):
        if column in matched or column in mismatched:
            if column not in matched or column not in mismatched:
                raise ValueError(f"Only one prediction file has protocol column {column}")
            if _constant_value(matched, column) != _constant_value(mismatched, column):
                raise ValueError(f"Cross-view protocol mismatch: {column}")

    matched_sorted = matched.sort_values(PAIRED_SAMPLE_KEY_COLUMNS).reset_index(drop=True)
    mismatched_sorted = mismatched.sort_values(PAIRED_SAMPLE_KEY_COLUMNS).reset_index(drop=True)
    if len(matched_sorted) != len(mismatched_sorted):
        raise ValueError(
            f"Prediction sample counts differ: {len(matched_sorted)} vs "
            f"{len(mismatched_sorted)}"
        )
    if not matched_sorted[PAIRED_SAMPLE_KEY_COLUMNS].equals(
        mismatched_sorted[PAIRED_SAMPLE_KEY_COLUMNS]
    ):
        raise ValueError("Matched and mismatched prediction composite keys are not aligned")
    for column in (
        "image_id", "patient_id", "label", "view_position", "crop_top", "crop_left",
        "payload_bpp", "algorithm", "manifest_sha256",
    ):
        left = matched_sorted[column].to_numpy()
        right = mismatched_sorted[column].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(
                f"Matched and mismatched predictions disagree on aligned {column!r}"
            )
    return (
        matched_sorted,
        mismatched_sorted,
        matched_source,
        mismatched_source,
        matched_threshold,
        mismatched_threshold,
    )


def _align_method_frames(baseline, candidate, target):
    """Explicit method/objective comparison; never weakens cross-view guards."""
    for frame in (baseline, candidate):
        validate_prediction_frame(frame, expected_split="test", expected_target_view=target)
    required = ("protocol_id", "manifest_identity", "crop_seed", "patch_size", "seed", "source_view", "payload_bpp", "algorithm")
    for column in required:
        if column not in baseline or column not in candidate:
            raise ValueError(f"Method comparison requires protocol binding {column}")
        if _constant_value(baseline, column) != _constant_value(candidate, column):
            raise ValueError(f"Method comparison protocol mismatch: {column}")
    for column in ("test_cohort",):
        values = [_constant_value(f,column) if column in f else "all_enrolled_test" for f in (baseline,candidate)]
        if values[0] != values[1]:
            raise ValueError(f"Method comparison protocol mismatch: {column}")
    left = baseline.sort_values(PAIRED_SAMPLE_KEY_COLUMNS).reset_index(drop=True)
    right = candidate.sort_values(PAIRED_SAMPLE_KEY_COLUMNS).reset_index(drop=True)
    if not left[PAIRED_SAMPLE_KEY_COLUMNS].equals(right[PAIRED_SAMPLE_KEY_COLUMNS]):
        raise ValueError("Method prediction sample keys are not aligned")
    for column in ("patient_id", "image_id", "label", "view_position", "crop_top", "crop_left"):
        if not np.array_equal(left[column], right[column]):
            raise ValueError(f"Method comparison aligned {column} differs")
    # Different score types/normalization are permitted, raw pixels/crops are not.
    return left, right, str(_constant_value(left, "source_view")), str(_constant_value(right, "source_view")), float(_constant_value(left, "threshold")), float(_constant_value(right, "threshold"))


def _selected_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    metrics = binary_metrics(labels, scores, threshold)
    return {name: float(metrics[name]) for name in BOOTSTRAP_METRICS}


def patient_cluster_paired_bootstrap(
    matched_predictions: pd.DataFrame,
    mismatched_predictions: pd.DataFrame,
    *,
    target_view: str,
    replicates: int = 10_000,
    seed: int = 1337,
    matched_prediction_sha256: str | None = None,
    mismatched_prediction_sha256: str | None = None,
    comparison_mode: str = "cross_view",
) -> dict[str, Any]:
    """Estimate fixed-target performance and mismatched-minus-matched deltas."""
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    if comparison_mode not in {"cross_view", "method"}:
        raise ValueError("comparison_mode must be cross_view or method")
    aligner = _align_prediction_frames if comparison_mode == "cross_view" else _align_method_frames
    (
        matched,
        mismatched,
        matched_source,
        mismatched_source,
        matched_threshold,
        mismatched_threshold,
    ) = aligner(
        matched_predictions.copy(), mismatched_predictions.copy(), target_view
    )
    labels = matched["label"].to_numpy(dtype=np.int64)
    matched_scores = matched["score"].to_numpy(dtype=np.float64)
    mismatched_scores = mismatched["score"].to_numpy(dtype=np.float64)
    matched_point = _selected_metrics(labels, matched_scores, matched_threshold)
    mismatched_point = _selected_metrics(labels, mismatched_scores, mismatched_threshold)

    distributions: dict[str, dict[str, list[float]]] = {
        metric: {"matched": [], "mismatched": [], "delta": []}
        for metric in BOOTSTRAP_METRICS
    }
    rng = np.random.default_rng(seed)
    invalid = 0
    patients = np.asarray(sorted(matched.patient_id.astype(str).unique()), dtype=object)
    clusters = [np.flatnonzero(matched.patient_id.astype(str).to_numpy() == p) for p in patients]
    for _ in range(replicates):
        selected = rng.choice(len(clusters), size=len(clusters), replace=True)
        positions = np.concatenate([clusters[i] for i in selected])
        replicate_labels = labels[positions]
        replicate_matched = _selected_metrics(
            replicate_labels, matched_scores[positions], matched_threshold
        )
        replicate_mismatched = _selected_metrics(
            replicate_labels, mismatched_scores[positions], mismatched_threshold
        )
        values = [
            *replicate_matched.values(),
            *replicate_mismatched.values(),
        ]
        if not all(math.isfinite(value) for value in values):
            invalid += 1
            continue
        for metric in BOOTSTRAP_METRICS:
            matched_value = replicate_matched[metric]
            mismatched_value = replicate_mismatched[metric]
            distributions[metric]["matched"].append(matched_value)
            distributions[metric]["mismatched"].append(mismatched_value)
            distributions[metric]["delta"].append(mismatched_value - matched_value)
    valid = replicates - invalid
    if valid == 0:
        raise ValueError("All bootstrap replicates were invalid")

    estimates: dict[str, dict[str, dict[str, float]]] = {}
    for metric in BOOTSTRAP_METRICS:
        estimates[metric] = {}
        points = {
            "matched": matched_point[metric],
            "mismatched": mismatched_point[metric],
            "delta": mismatched_point[metric] - matched_point[metric],
        }
        for comparison, point in points.items():
            values = np.asarray(distributions[metric][comparison], dtype=np.float64)
            lower, upper = np.percentile(values, [2.5, 97.5])
            estimates[metric][comparison] = {
                "point_estimate": float(point),
                "lower_2_5": float(lower),
                "upper_97_5": float(upper),
            }

    return {
        "target_view": target_view.upper(),
        "matched_source_view": matched_source,
        "mismatched_source_view": mismatched_source,
        "patients": int(matched["patient_id"].astype(str).nunique()),
        "pairs": int(matched["pair_id"].astype(str).nunique()),
        "samples": int(len(matched)),
        "replicates_requested": int(replicates),
        "replicates_valid": int(valid),
        "replicates_invalid": int(invalid),
        "bootstrap_seed": int(seed),
        "matched_threshold": matched_threshold,
        "mismatched_threshold": mismatched_threshold,
        "matched_prediction_sha256": matched_prediction_sha256,
        "mismatched_prediction_sha256": mismatched_prediction_sha256,
        "comparison_mode": comparison_mode,
        "delta_definition": "mismatched - matched" if comparison_mode == "cross_view" else "candidate - baseline",
        "slot_semantics": "matched=baseline; mismatched=candidate" if comparison_mode == "method" else "source view matched/mismatched",
        "training_seed_uncertainty": "not captured; fixed fitted models",
        "confidence_interval": "95% percentile patient-cluster paired bootstrap",
        "estimates": estimates,
    }


def bootstrap_prediction_files(
    matched_path: str | Path,
    mismatched_path: str | Path,
    *,
    target_view: str,
    replicates: int = 10_000,
    seed: int = 1337,
) -> dict[str, Any]:
    # Lab prediction files hold both targets; compare only the requested target.
    matched, mismatched = (
        frame[frame["target_view"] == target_view.upper()]
        for frame in (load_prediction_file(matched_path), load_prediction_file(mismatched_path))
    )
    return patient_cluster_paired_bootstrap(
        matched,
        mismatched,
        target_view=target_view,
        replicates=replicates,
        seed=seed,
        matched_prediction_sha256=sha256_file(matched_path),
        mismatched_prediction_sha256=sha256_file(mismatched_path),
    )


def save_bootstrap_results(results: dict[str, Any], output_stem: str | Path) -> None:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    existing = [
        path for path in (stem.with_suffix(".json"), stem.with_suffix(".csv"))
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite bootstrap artifacts; use a new stem: {existing}"
        )
    stem.with_suffix(".json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    metadata = {
        key: value for key, value in results.items() if key != "estimates"
    }
    rows: list[dict[str, Any]] = []
    for metric, comparisons in results["estimates"].items():
        for comparison, values in comparisons.items():
            rows.append({
                **metadata,
                "metric": metric,
                "comparison": comparison,
                **values,
            })
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
