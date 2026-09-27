"""Patient-disjoint split creation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from PIL import Image


SPLIT_NAMES = ("train", "validation", "test")


def validate_official_patient_partition(
    metadata: pd.DataFrame,
    official_train_ids: Iterable[str],
    official_test_ids: Iterable[str],
) -> tuple[set[str], set[str]]:
    """Validate official lists against full metadata before local filtering."""
    train_ids = {str(value) for value in official_train_ids}
    test_ids = {str(value) for value in official_test_ids}
    if not train_ids or not test_ids:
        raise ValueError("Both official image lists must be non-empty")
    if train_ids & test_ids:
        raise ValueError("Official train/test lists overlap by image ID")
    metadata_ids = set(metadata["image_id"].astype(str))
    missing_train = train_ids - metadata_ids
    missing_test = test_ids - metadata_ids
    if missing_train or missing_test:
        raise ValueError(
            "Official lists contain image IDs absent from full metadata: "
            f"train={sorted(missing_train)[:5]}, test={sorted(missing_test)[:5]}"
        )
    train_patients = set(
        metadata.loc[metadata["image_id"].isin(train_ids), "patient_id"].astype(str)
    )
    test_patients = set(
        metadata.loc[metadata["image_id"].isin(test_ids), "patient_id"].astype(str)
    )
    overlap = train_patients & test_patients
    if overlap:
        raise AssertionError(
            "Official train/test lists overlap by patient in full metadata: "
            f"{sorted(overlap)[:10]}"
        )
    return train_patients, test_patients


def _patient_view_group(rows: pd.DataFrame) -> str:
    views = set(rows["view_position"].astype(str)) & {"AP", "PA"}
    if views == {"AP"}:
        return "AP"
    if views == {"PA"}:
        return "PA"
    if views == {"AP", "PA"}:
        return "MIXED"
    return "OTHER"


def _choose_patients(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
    minimum_if_possible: bool = True,
) -> set[str]:
    """Select a deterministic, approximately view-stratified patient subset."""
    patients = list(frame["patient_id"].drop_duplicates())
    if not patients or fraction <= 0:
        return set()
    target = int(round(len(patients) * fraction))
    if minimum_if_possible and len(patients) > 1:
        target = max(1, min(target, len(patients) - 1))
    else:
        target = max(0, min(target, len(patients)))

    grouped = frame.groupby("patient_id", sort=False)[["view_position"]].apply(
        _patient_view_group
    )
    rng = np.random.default_rng(seed)
    ordered: list[str] = []
    buckets: dict[str, list[str]] = {}
    for group in ("AP", "PA", "MIXED", "OTHER"):
        bucket = grouped[grouped == group].index.astype(str).tolist()
        rng.shuffle(bucket)
        buckets[group] = bucket
    # Round-robin avoids exhausting the largest source group first.
    while any(buckets.values()):
        for group in ("AP", "PA", "MIXED", "OTHER"):
            if buckets[group]:
                ordered.append(buckets[group].pop())
    return set(ordered[:target])


def assert_patient_disjoint(frame: pd.DataFrame) -> None:
    if "split" not in frame.columns:
        raise ValueError("Split manifest is missing required column: split")
    patients = {
        split: set(frame.loc[frame["split"] == split, "patient_id"].astype(str))
        for split in SPLIT_NAMES
    }
    overlaps = {
        "train/validation": patients["train"] & patients["validation"],
        "train/test": patients["train"] & patients["test"],
        "validation/test": patients["validation"] & patients["test"],
    }
    nonempty = {name: sorted(values) for name, values in overlaps.items() if values}
    if nonempty:
        raise AssertionError(f"Patient overlap detected across splits: {nonempty}")


def scope_official_image_lists(
    candidate_image_ids: Iterable[str],
    official_train_ids: Iterable[str],
    official_test_ids: Iterable[str],
) -> tuple[set[str], set[str], dict[str, int]]:
    """Restrict full official lists to an explicitly discovered local candidate pool."""
    candidates = {str(value) for value in candidate_image_ids}
    full_train = {str(value) for value in official_train_ids}
    full_test = {str(value) for value in official_test_ids}
    if full_train & full_test:
        raise ValueError("Official train/test lists contain duplicate image IDs")
    if full_train or full_test:
        unlisted = candidates - (full_train | full_test)
        if unlisted:
            raise ValueError(
                f"Candidate pool contains {len(unlisted)} image(s) absent from official lists: "
                f"{sorted(unlisted)[:5]}"
            )
    scoped_train = full_train & candidates
    scoped_test = full_test & candidates
    report = {
        "train_entries_total": len(full_train),
        "train_entries_retained": len(scoped_train),
        "train_entries_excluded": len(full_train - candidates),
        "test_entries_total": len(full_test),
        "test_entries_retained": len(scoped_test),
        "test_entries_excluded": len(full_test - candidates),
    }
    return scoped_train, scoped_test, report


def create_patient_split(
    frame: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.20,
    official_train_ids: Iterable[str] | None = None,
    official_test_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Assign images to patient-disjoint train/validation/test partitions."""
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Cannot split duplicate image_id values: {duplicates[:5]}")
    result = frame.sort_values(["patient_id", "image_id"], kind="mergesort").reset_index(
        drop=True
    ).copy()
    result["patient_id"] = result["patient_id"].astype(str)
    result["split"] = ""

    official_test = set(official_test_ids or [])
    official_train = set(official_train_ids or [])
    use_official = official_test_ids is not None or official_train_ids is not None
    if use_official:
        if not official_test:
            raise ValueError("No locally available official test images; refusing random-test fallback")
        unknown_train = official_train - set(result["image_id"])
        unknown_test = official_test - set(result["image_id"])
        if unknown_train:
            raise ValueError(
                f"Official train list contains {len(unknown_train)} image(s) absent from metadata"
            )
        if unknown_test:
            raise ValueError(
                f"Official test list contains {len(unknown_test)} image(s) absent from metadata"
            )
        test_patients = set(result.loc[result["image_id"].isin(official_test), "patient_id"])
        if official_train:
            train_patients = set(
                result.loc[result["image_id"].isin(official_train), "patient_id"]
            )
            overlap = train_patients & test_patients
            if overlap:
                raise AssertionError(
                    f"Official train/test lists overlap by patient: {sorted(overlap)[:10]}"
                )
        result.loc[result["patient_id"].isin(test_patients), "split"] = "test"
        development = result.loc[~result["patient_id"].isin(test_patients)]
        development_patients = set(development["patient_id"])
        if official_train:
            listed_development = set(
                result.loc[result["image_id"].isin(official_train), "patient_id"]
            )
            unlisted = development_patients - listed_development
            if unlisted:
                raise ValueError(
                    f"Metadata has {len(unlisted)} patient(s) absent from official lists"
                )
        dev_fraction = validation_fraction / (train_fraction + validation_fraction)
        validation_patients = _choose_patients(development, dev_fraction, seed)
        result.loc[result["patient_id"].isin(validation_patients), "split"] = "validation"
        result.loc[result["split"].eq(""), "split"] = "train"
    else:
        test_patients = _choose_patients(result, test_fraction, seed)
        result.loc[result["patient_id"].isin(test_patients), "split"] = "test"
        development = result.loc[~result["patient_id"].isin(test_patients)]
        relative_validation = validation_fraction / (train_fraction + validation_fraction)
        validation_patients = _choose_patients(development, relative_validation, seed + 1)
        result.loc[result["patient_id"].isin(validation_patients), "split"] = "validation"
        result.loc[result["split"].eq(""), "split"] = "train"

    if result["split"].eq("").any():
        raise AssertionError("Some rows were not assigned to a split")
    assert_patient_disjoint(result)
    return result


def build_patient_assignments(
    assigned_frame: pd.DataFrame,
    official_test_patient_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create one auditable row for every eligible patient before image sampling."""
    assert_patient_disjoint(assigned_frame)
    official_test_patients = {str(value) for value in (official_test_patient_ids or [])}
    rows: list[dict[str, object]] = []
    for patient_id, group in assigned_frame.groupby("patient_id", sort=True):
        patient_splits = set(group["split"].astype(str))
        if len(patient_splits) != 1:
            raise AssertionError(f"Patient {patient_id} has multiple split assignments")
        view_counts = group["view_position"].value_counts().to_dict()
        rows.append({
            "patient_id": str(patient_id),
            "split": next(iter(patient_splits)),
            "view_group": _patient_view_group(group),
            "images": int(len(group)),
            "ap_images": int(view_counts.get("AP", 0)),
            "pa_images": int(view_counts.get("PA", 0)),
            "official_test_patient": str(patient_id) in official_test_patients,
        })
    assignments = pd.DataFrame(rows)
    validate_patient_assignments(assignments, official_test_patients)
    return assignments


def validate_patient_assignments(
    assignments: pd.DataFrame,
    official_test_patient_ids: Iterable[str] | None = None,
) -> None:
    required = {
        "patient_id", "split", "view_group", "images", "ap_images", "pa_images",
        "official_test_patient",
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Patient assignments missing column(s): {', '.join(missing)}")
    if assignments["patient_id"].duplicated().any():
        raise ValueError("Patient assignments contain duplicate patient_id values")
    unknown_splits = sorted(set(assignments["split"].astype(str)) - set(SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(f"Patient assignments contain unknown split(s): {unknown_splits}")
    official_test_patients = {str(value) for value in (official_test_patient_ids or [])}
    leaked = assignments.loc[
        assignments["patient_id"].astype(str).isin(official_test_patients)
        & ~assignments["split"].eq("test"),
        "patient_id",
    ].astype(str).tolist()
    if leaked:
        raise AssertionError(f"Official test patient leakage detected: {leaked[:10]}")


def _sampling_seed(seed: int, split: str, view: str) -> int:
    payload = f"{int(seed)}\x1f{split}\x1f{view}\x1fpilot_image_sampling"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_split_by_view(
    assigned_frame: pd.DataFrame,
    quotas_per_view: Mapping[str, int],
    seed: int,
    allowed_views: Iterable[str] = ("AP", "PA"),
) -> pd.DataFrame:
    """Sample exact image quotas only after all eligible patients have a split."""
    assert_patient_disjoint(assigned_frame)
    if assigned_frame["image_id"].duplicated().any():
        raise ValueError("Cannot sample duplicate image_id values")
    views = tuple(dict.fromkeys(str(view).upper() for view in allowed_views))
    unknown_views = sorted(set(assigned_frame["view_position"].astype(str)) - set(views))
    if unknown_views:
        raise ValueError(f"Assigned pool contains disallowed view(s): {unknown_views}")
    if set(quotas_per_view) != set(SPLIT_NAMES):
        raise ValueError("quotas_per_view must contain train, validation, and test")

    selections: list[pd.DataFrame] = []
    for split in SPLIT_NAMES:
        quota = int(quotas_per_view[split])
        if quota <= 0:
            raise ValueError(f"Quota for {split} must be positive")
        for view in views:
            eligible = assigned_frame.loc[
                assigned_frame["split"].eq(split)
                & assigned_frame["view_position"].eq(view)
            ].sort_values("image_id", kind="mergesort")
            available = len(eligible)
            if available < quota:
                raise ValueError(
                    f"Insufficient quota for split={split}, view={view}: "
                    f"required={quota}, available={available}"
                )
            rng = np.random.default_rng(_sampling_seed(seed, split, view))
            selected_positions = np.sort(rng.choice(available, size=quota, replace=False))
            selections.append(eligible.iloc[selected_positions].copy())

    sampled = pd.concat(selections, ignore_index=True)
    split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    view_order = {name: index for index, name in enumerate(views)}
    sampled["_split_order"] = sampled["split"].map(split_order)
    sampled["_view_order"] = sampled["view_position"].map(view_order)
    sampled = sampled.sort_values(
        ["_split_order", "_view_order", "image_id"], kind="mergesort"
    ).drop(columns=["_split_order", "_view_order"]).reset_index(drop=True)
    assert_patient_disjoint(sampled)
    return sampled


def validate_pilot_split(
    frame: pd.DataFrame,
    quotas_per_view: Mapping[str, int],
    allowed_views: Iterable[str] = ("AP", "PA"),
    required_image_mode: str = "L",
    official_test_patient_ids: Iterable[str] | None = None,
    check_files: bool = True,
    minimum_image_size: int | None = None,
) -> None:
    """Hard validation for a final balanced, patient-disjoint pilot split."""
    required = {"image_id", "patient_id", "view_position", "split", "cover_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pilot split missing required column(s): {', '.join(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Pilot split contains duplicate image_id values")

    views = tuple(dict.fromkeys(str(view).upper() for view in allowed_views))
    unknown_views = sorted(set(frame["view_position"].astype(str)) - set(views))
    if unknown_views:
        raise ValueError(f"Pilot split contains disallowed view(s): {unknown_views}")
    unknown_splits = sorted(set(frame["split"].astype(str)) - set(SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(f"Pilot split contains unknown split(s): {unknown_splits}")
    if set(quotas_per_view) != set(SPLIT_NAMES):
        raise ValueError("quotas_per_view must contain train, validation, and test")

    expected_total = len(views) * sum(int(quotas_per_view[name]) for name in SPLIT_NAMES)
    if len(frame) != expected_total:
        raise ValueError(f"Pilot split must contain {expected_total} rows, got {len(frame)}")
    for split in SPLIT_NAMES:
        for view in views:
            actual = int((frame["split"].eq(split) & frame["view_position"].eq(view)).sum())
            expected = int(quotas_per_view[split])
            if actual != expected:
                raise ValueError(
                    f"Pilot quota mismatch for split={split}, view={view}: "
                    f"expected={expected}, actual={actual}"
                )
    assert_patient_disjoint(frame)

    official_test_patients = {str(value) for value in (official_test_patient_ids or [])}
    leaked = frame.loc[
        frame["patient_id"].astype(str).isin(official_test_patients)
        & ~frame["split"].eq("test"),
        "patient_id",
    ].astype(str).unique().tolist()
    if leaked:
        raise AssertionError(f"Official test patient leakage detected: {leaked[:10]}")

    if "image_mode" in frame.columns:
        wrong_recorded_modes = frame.loc[
            ~frame["image_mode"].eq(required_image_mode), ["image_id", "image_mode"]
        ]
        if not wrong_recorded_modes.empty:
            raise ValueError(
                "Pilot split contains non-required recorded image modes: "
                f"{wrong_recorded_modes.head().to_dict('records')}"
            )
    if check_files:
        for row in frame.itertuples(index=False):
            cover_path = Path(row.cover_path)
            if not cover_path.is_file():
                raise FileNotFoundError(f"Pilot cover is missing for {row.image_id}: {cover_path}")
            with Image.open(cover_path) as image:
                if image.mode != required_image_mode:
                    raise ValueError(
                        f"Pilot cover must use mode {required_image_mode}, got "
                        f"{image.mode}: {cover_path}"
                    )
                if minimum_image_size is not None and min(image.size) < int(minimum_image_size):
                    raise ValueError(
                        f"Pilot cover {image.size} is smaller than required patch "
                        f"{minimum_image_size}: {cover_path}"
                    )


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in SPLIT_NAMES:
        subset = frame[frame["split"] == split]
        views = subset["view_position"].value_counts().to_dict()
        rows.append({
            "split": split,
            "images": len(subset),
            "patients": subset["patient_id"].nunique(),
            "ap": int(views.get("AP", 0)),
            "pa": int(views.get("PA", 0)),
            "other_view": int(len(subset) - views.get("AP", 0) - views.get("PA", 0)),
        })
    return pd.DataFrame(rows)


def save_split(frame: pd.DataFrame, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert_patient_disjoint(frame)
    frame.to_csv(output_path, index=False)


def save_patient_assignments(frame: pd.DataFrame, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_patient_assignments(frame)
    frame.to_csv(output_path, index=False)
