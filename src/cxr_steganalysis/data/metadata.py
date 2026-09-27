"""NIH ChestX-ray14 metadata normalization and inspection."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image


NIH_COLUMN_MAP = {
    "Image Index": "image_id",
    "Finding Labels": "finding_labels",
    "Patient ID": "patient_id",
    "View Position": "view_position",
    "Patient Age": "patient_age",
    "Patient Gender": "patient_gender",
}
REQUIRED_COLUMNS = {"Image Index"}


def infer_patient_id(image_id: str) -> str:
    """Infer NIH patient ID from a filename, with a deterministic fallback."""
    name = Path(str(image_id)).name
    match = re.match(r"^(\d+)(?:_|\.)", name)
    if match:
        return match.group(1)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"inferred-{digest}"


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load NIH CSV and normalize available fields to stable internal names."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"NIH metadata CSV not found: {metadata_path}")
    frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing_required = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_required:
        raise ValueError(
            "Metadata is missing required column(s): " + ", ".join(missing_required)
        )
    frame = frame.rename(columns={
        source: target for source, target in NIH_COLUMN_MAP.items() if source in frame.columns
    })
    source_missing_values = {
        column: int((
            frame[column].isna()
            | frame[column].fillna("").astype(str).str.strip().eq("")
        ).sum())
        for column in frame.columns
    }
    frame["image_id"] = frame["image_id"].astype(str).str.strip()
    if "patient_id" not in frame.columns:
        frame["patient_id"] = frame["image_id"].map(infer_patient_id)
    else:
        ids = frame["patient_id"].astype(str).str.strip()
        missing = ids.eq("")
        ids.loc[missing] = frame.loc[missing, "image_id"].map(infer_patient_id)
        frame["patient_id"] = ids
    # Keep legacy formatting when consistent, but never let explicit "1" and
    # inferred "00000001" become different patients. This changes only ambiguous
    # numeric aliases, not the locked pilot's already consistent IDs.
    ids = frame["patient_id"].astype(str)
    numeric = ids.str.fullmatch(r"\d+")
    aliases = {}
    for value in ids[numeric].unique():
        aliases.setdefault(str(int(value)), []).append(value)
    preferred = {key:min(values,key=lambda v:(len(v),v)) for key,values in aliases.items()}
    frame.loc[numeric,"patient_id"] = ids[numeric].map(lambda value:preferred[str(int(value))])
    if "view_position" not in frame.columns:
        frame["view_position"] = "UNKNOWN"
    else:
        view = frame["view_position"].astype(str).str.strip().str.upper()
        frame["view_position"] = view.mask(view.eq(""), "UNKNOWN")
    if "finding_labels" not in frame.columns:
        frame["finding_labels"] = ""
    else:
        frame["finding_labels"] = frame["finding_labels"].astype(str).str.strip()
    frame.attrs["source_missing_values"] = source_missing_values
    return frame


def discover_local_pngs(raw_dir: str | Path) -> pd.DataFrame:
    """Discover decodable local PNG files with stable ordering and unique basenames."""
    raw_path = Path(raw_dir).expanduser().resolve()
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw PNG directory not found: {raw_path}")
    paths = sorted(
        (
            path.resolve()
            for path in raw_path.rglob("*")
            if path.is_file() and path.suffix.lower() == ".png"
        ),
        key=lambda path: path.as_posix(),
    )
    basename_counts = Counter(path.name for path in paths)
    duplicates = sorted(name for name, count in basename_counts.items() if count > 1)
    if duplicates:
        examples = {
            name: [str(path) for path in paths if path.name == name]
            for name in duplicates[:5]
        }
        raise ValueError(f"Duplicate PNG basename(s) found under {raw_path}: {examples}")

    records: list[dict[str, str]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image_mode = image.mode
                width, height = image.size
                # Reading only the header can let a truncated PNG enter the split and
                # fail much later during training. Decode it during the data audit.
                image.load()
        except Exception as error:
            raise ValueError(f"Cannot fully decode PNG: {path}: {error}") from error
        records.append({
            "image_id": path.name,
            "cover_path": str(path),
            "image_mode": image_mode,
            "image_width": int(width),
            "image_height": int(height),
        })
    return pd.DataFrame(records, columns=[
        "image_id", "cover_path", "image_mode", "image_width", "image_height",
    ])


def filter_metadata_to_local_images(
    frame: pd.DataFrame,
    raw_dir: str | Path,
    allowed_views: Iterable[str] = ("AP", "PA"),
    required_image_mode: str | None = "L",
    minimum_image_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join metadata to local PNGs and return an explicitly filtered pilot pool."""
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[
            frame["image_id"].duplicated(keep=False), "image_id"
        ].astype(str).tolist()
        raise ValueError(f"Metadata contains duplicate image_id values: {duplicates[:5]}")

    local = discover_local_pngs(raw_dir)
    metadata_ids = set(frame["image_id"].astype(str))
    local_ids = set(local["image_id"].astype(str))
    matched_ids = metadata_ids & local_ids
    files_without_metadata = sorted(local_ids - metadata_ids)
    metadata_without_files = metadata_ids - local_ids

    matched = frame.merge(local, on="image_id", how="inner", validate="one_to_one")
    matched = matched.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    views = tuple(dict.fromkeys(str(view).strip().upper() for view in allowed_views))
    if not views:
        raise ValueError("allowed_views must contain at least one view")
    view_eligible = matched[matched["view_position"].isin(views)].copy()
    excluded_views = matched.loc[~matched["view_position"].isin(views), "view_position"]

    required_mode = None if required_image_mode is None else str(required_image_mode)
    mode_mask = (
        pd.Series(True, index=view_eligible.index)
        if required_mode is None
        else view_eligible["image_mode"].eq(required_mode)
    )
    excluded_modes = view_eligible.loc[~mode_mask].copy()
    mode_eligible = view_eligible.loc[mode_mask].copy()
    if minimum_image_size is None:
        size_mask = pd.Series(True, index=mode_eligible.index)
    else:
        minimum_image_size = int(minimum_image_size)
        if minimum_image_size <= 0:
            raise ValueError("minimum_image_size must be positive when provided")
        size_mask = (
            mode_eligible["image_width"].ge(minimum_image_size)
            & mode_eligible["image_height"].ge(minimum_image_size)
        )
    excluded_sizes = mode_eligible.loc[~size_mask].copy()
    candidates = mode_eligible.loc[size_mask].copy()
    candidates = candidates.sort_values("image_id", kind="mergesort").reset_index(drop=True)

    excluded_mode_view_counts = {
        str(mode): {
            str(view): int(count)
            for view, count in group["view_position"].value_counts().sort_index().items()
        }
        for mode, group in excluded_modes.groupby("image_mode", sort=True)
    }
    excluded_images = [
        {
            "image_id": str(row.image_id),
            "view_position": str(row.view_position),
            "reason": f"unsupported_mode:{row.image_mode}",
        }
        for row in excluded_modes.itertuples(index=False)
    ]
    excluded_images.extend(
        {
            "image_id": str(row.image_id),
            "view_position": str(row.view_position),
            "reason": (
                f"smaller_than_patch:{row.image_width}x{row.image_height}"
            ),
        }
        for row in excluded_sizes.itertuples(index=False)
    )
    report: dict[str, Any] = {
        "num_png_files_found": int(len(local)),
        "num_metadata_matches": int(len(matched_ids)),
        "num_metadata_without_file": int(len(metadata_without_files)),
        "metadata_without_file_examples": sorted(metadata_without_files)[:20],
        "num_files_without_metadata": int(len(files_without_metadata)),
        "files_without_metadata": files_without_metadata,
        "image_mode_counts": {
            str(mode): int(count)
            for mode, count in local["image_mode"].value_counts().sort_index().items()
        },
        "allowed_views": list(views),
        "num_excluded_by_view": int(len(excluded_views)),
        "excluded_view_counts": {
            str(view): int(count)
            for view, count in excluded_views.value_counts().sort_index().items()
        },
        "required_image_mode": required_mode,
        "num_excluded_by_mode": int(len(excluded_modes)),
        "excluded_mode_counts": {
            str(mode): int(count)
            for mode, count in excluded_modes["image_mode"].value_counts().sort_index().items()
        },
        "excluded_mode_view_counts": excluded_mode_view_counts,
        "minimum_image_size": minimum_image_size,
        "num_excluded_by_size": int(len(excluded_sizes)),
        "excluded_size_view_counts": {
            str(view): int(count)
            for view, count in excluded_sizes["view_position"].value_counts().sort_index().items()
        },
        "excluded_images": excluded_images,
        "num_candidate_images": int(len(candidates)),
        "candidate_view_counts": {
            str(view): int(count)
            for view, count in candidates["view_position"].value_counts().sort_index().items()
        },
        "num_candidate_patients": int(candidates["patient_id"].nunique()),
    }
    return candidates, report


def read_image_list(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    list_path = Path(path)
    if not list_path.is_file():
        return set()
    return {
        line.strip() for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _label_distribution(values: Iterable[str]) -> dict[str, int]:
    labels: Counter[str] = Counter()
    for value in values:
        labels.update(label.strip() for label in str(value).split("|") if label.strip())
    return dict(sorted(labels.items(), key=lambda item: (-item[1], item[0])))


def inspect_metadata(
    frame: pd.DataFrame,
    raw_dir: str | Path | None = None,
    official_train: set[str] | None = None,
    official_test: set[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable quality and distribution report."""
    official_train = official_train or set()
    official_test = official_test or set()
    train_rows = frame[frame["image_id"].isin(official_train)]
    test_rows = frame[frame["image_id"].isin(official_test)]
    overlap = sorted(set(train_rows["patient_id"]) & set(test_rows["patient_id"]))

    raw_path = Path(raw_dir) if raw_dir is not None else None
    missing_files: list[str] | None = None
    if "cover_path" in frame.columns:
        missing_files = [
            str(path) for path in frame["cover_path"] if not Path(path).is_file()
        ]
    elif raw_path is not None and raw_path.is_dir():
        missing_files = [
            image_id for image_id in frame["image_id"]
            if not (raw_path / image_id).is_file()
        ]

    missing_values = frame.attrs.get("source_missing_values") or {
        column: int((
            frame[column].isna()
            | frame[column].fillna("").astype(str).str.strip().eq("")
        ).sum())
        for column in frame.columns
    }
    per_patient = frame.groupby("patient_id", sort=False).size()
    duplicates = frame.loc[frame["image_id"].duplicated(keep=False), "image_id"].tolist()
    view_counts = frame["view_position"].value_counts(dropna=False).to_dict()
    report: dict[str, Any] = {
        "num_images": int(len(frame)),
        "num_unique_patients": int(frame["patient_id"].nunique()),
        "view_counts": {str(key): int(value) for key, value in view_counts.items()},
        "num_ap": int(view_counts.get("AP", 0)),
        "num_pa": int(view_counts.get("PA", 0)),
        "images_per_patient": {
            "min": int(per_patient.min()) if len(per_patient) else 0,
            "max": int(per_patient.max()) if len(per_patient) else 0,
            "mean": float(per_patient.mean()) if len(per_patient) else 0.0,
            "median": float(per_patient.median()) if len(per_patient) else 0.0,
            "distribution": {
                str(key): int(value)
                for key, value in per_patient.value_counts().sort_index().items()
            },
        },
        "num_no_finding": int(frame["finding_labels"].str.split("|").map(
            lambda labels: "No Finding" in labels
        ).sum()),
        "pathology_distribution": _label_distribution(frame["finding_labels"]),
        "duplicate_image_ids": duplicates,
        "num_duplicate_rows": len(duplicates),
        "missing_values": missing_values,
        "official_lists": {
            "train_images_in_metadata": int(len(train_rows)),
            "test_images_in_metadata": int(len(test_rows)),
            "patient_overlap_count": len(overlap),
            "patient_overlap": overlap,
        },
        "missing_image_files": missing_files,
        "num_missing_image_files": len(missing_files) if missing_files is not None else None,
    }
    return report


def save_metadata_report(report: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    elif output_path.suffix.lower() == ".csv":
        rows = [
            {"section": "summary", "name": key, "value": value}
            for key, value in report.items()
            if not isinstance(value, (dict, list))
        ]
        rows.extend(
            {"section": "view_counts", "name": key, "value": value}
            for key, value in report["view_counts"].items()
        )
        rows.extend(
            {"section": "pathology_distribution", "name": key, "value": value}
            for key, value in report["pathology_distribution"].items()
        )
        pd.DataFrame(rows).to_csv(output_path, index=False)
    else:
        raise ValueError("Metadata report output must use .json or .csv")


def human_summary(report: dict[str, Any]) -> str:
    overlap = report["official_lists"]["patient_overlap_count"]
    missing = report["num_missing_image_files"]
    missing_text = "not checked" if missing is None else str(missing)
    lines = [
        f"Images: {report['num_images']}",
        f"Unique patients: {report['num_unique_patients']}",
        f"Views: AP={report['num_ap']}, PA={report['num_pa']}",
        f"No Finding: {report['num_no_finding']}",
        f"Duplicate image rows: {report['num_duplicate_rows']}",
        f"Official train/test patient overlap: {overlap}",
        f"Missing image files: {missing_text}",
    ]
    discovery = report.get("local_discovery")
    if discovery:
        lines.extend([
            f"PNG files found: {discovery['num_png_files_found']}",
            f"Matched local metadata: {discovery['num_metadata_matches']}",
            f"Excluded by mode: {discovery['num_excluded_by_mode']} "
            f"{discovery['excluded_mode_counts']}",
            f"Eligible local images: {discovery['num_candidate_images']}",
            f"Eligible views: {discovery['candidate_view_counts']}",
        ])
    return "\n".join(lines)
