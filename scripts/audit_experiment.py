#!/usr/bin/env python3
"""Audit persisted pilot artifacts without training or changing experiment data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.data.manifest import validate_pair_manifest  # noqa: E402
from cxr_steganalysis.data.metadata import load_metadata, read_image_list  # noqa: E402
from cxr_steganalysis.data.split import (  # noqa: E402
    SPLIT_NAMES,
    assert_patient_disjoint,
    validate_patient_assignments,
    validate_pilot_split,
)
from cxr_steganalysis.stego.lsb_matching import (  # noqa: E402
    embed_lsb_matching,
    load_grayscale_png,
)


def _counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = frame.groupby(["split", "view_position"], sort=True)
    return [
        {
            "split": str(split),
            "view_position": str(view),
            "images": int(len(group)),
            "patients": int(group["patient_id"].astype(str).nunique()),
        }
        for (split, view), group in grouped
    ]


def _assert_same_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
    left_name: str,
    right_name: str,
) -> None:
    a = left.sort_values("image_id").reset_index(drop=True)
    b = right.sort_values("image_id").reset_index(drop=True)
    if len(a) != len(b) or not a[columns].equals(b[columns]):
        raise ValueError(f"{left_name} and {right_name} disagree on {columns}")


def audit(
    config_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    split_path: str | Path | None = None,
    assignments_path: str | Path | None = None,
    generation_path: str | Path | None = None,
    verify_hashes: bool = False,
    verify_pixels: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    manifest_path = Path(manifest_path or resolve_config_path(config, "pair_manifest"))
    split_path = Path(split_path or resolve_config_path(config, "split_manifest"))
    assignments_path = Path(
        assignments_path or resolve_config_path(config, "patient_assignments")
    )
    generation_path = Path(
        generation_path or resolve_config_path(config, "generation_manifest")
    )
    split = pd.read_csv(split_path, dtype={"patient_id": str})
    assignments = pd.read_csv(assignments_path, dtype={"patient_id": str})
    generation = pd.read_csv(generation_path, dtype={"seed": str})
    manifest = pd.read_csv(
        manifest_path,
        dtype={"patient_id": str, "seed": str},
    )
    assert_patient_disjoint(split)
    validate_patient_assignments(assignments)
    quotas = config["pilot"].get("quotas_per_view")
    if quotas is None:
        raise ValueError("Audit expects explicit pilot.quotas_per_view")
    validate_pilot_split(
        split,
        quotas,
        allowed_views=config["pilot"]["allowed_views"],
        required_image_mode=str(config["pilot"]["required_image_mode"]),
        minimum_image_size=int(config["patch_size"]),
        check_files=True,
    )
    validate_pair_manifest(
        manifest,
        check_files=True,
        verify_hashes=verify_hashes,
    )
    if set(split["image_id"]) != set(generation["image_id"]):
        raise ValueError("Generation records do not cover exactly the selected split images")
    _assert_same_columns(
        split,
        manifest,
        ["image_id", "patient_id", "view_position", "split"],
        "split manifest",
        "pair manifest",
    )
    assignment_map = assignments.set_index("patient_id")["split"].astype(str)
    selected_assignment = split["patient_id"].astype(str).map(assignment_map)
    if selected_assignment.isna().any() or not selected_assignment.equals(
        split["split"].astype(str)
    ):
        raise ValueError("Selected images disagree with the pre-sampling patient assignment")

    official_report: dict[str, Any] = {"enabled": False}
    if config["split"]["use_official_test"]:
        train_ids = read_image_list(resolve_config_path(config, "official_train_list"))
        test_ids = read_image_list(resolve_config_path(config, "official_test_list"))
        if not train_ids or not test_ids:
            raise FileNotFoundError("Configured official list is missing or empty")
        metadata = load_metadata(resolve_config_path(config, "metadata"))
        official_train_patients = set(
            metadata.loc[metadata["image_id"].isin(train_ids), "patient_id"].astype(str)
        )
        official_test_patients = set(
            metadata.loc[metadata["image_id"].isin(test_ids), "patient_id"].astype(str)
        )
        if official_train_patients & official_test_patients:
            raise AssertionError("Official lists overlap by patient in full metadata")
        selected_train_validation = set(
            split.loc[split["split"].isin(["train", "validation"]), "image_id"]
        )
        selected_test = set(split.loc[split["split"].eq("test"), "image_id"])
        if not selected_train_validation <= train_ids or not selected_test <= test_ids:
            raise AssertionError("Selected image violates official train/test membership")
        assigned_official_test = assignments["patient_id"].astype(str).isin(
            official_test_patients
        )
        if not assignments.loc[assigned_official_test, "split"].eq("test").all():
            raise AssertionError("Official test patient appears outside test assignment")
        if assignments.loc[~assigned_official_test, "split"].eq("test").any():
            raise AssertionError("Non-official-test patient appears in test assignment")
        official_report = {
            "enabled": True,
            "official_train_images": len(train_ids),
            "official_test_images": len(test_ids),
            "official_train_patients": len(official_train_patients),
            "official_test_patients": len(official_test_patients),
            "selected_train_validation_membership_valid": True,
            "selected_test_membership_valid": True,
            "full_metadata_patient_overlap": 0,
        }

    embedded_bit_mismatches: list[str] = []
    changed_count_mismatches: list[str] = []
    exact_stego_mismatches: list[str] = []
    total_pixels = 0
    for row in manifest.itertuples(index=False):
        with Image.open(row.cover_path) as cover_image:
            width, height = cover_image.size
        total_pixels += width * height
        expected_bits = int(np.floor(float(row.payload_bpp) * width * height))
        if int(row.embedded_bits) != expected_bits:
            embedded_bit_mismatches.append(str(row.image_id))
        if int(row.changed_pixels) > int(row.embedded_bits):
            changed_count_mismatches.append(str(row.image_id))
        if verify_pixels:
            cover = load_grayscale_png(row.cover_path)
            stored_stego = load_grayscale_png(row.stego_path)
            regenerated = embed_lsb_matching(cover, float(row.payload_bpp), int(row.seed))
            if not np.array_equal(stored_stego, regenerated.image):
                exact_stego_mismatches.append(str(row.image_id))
            if int(row.changed_pixels) != int(np.count_nonzero(cover != stored_stego)):
                changed_count_mismatches.append(str(row.image_id))
    if embedded_bit_mismatches:
        raise AssertionError(
            f"embedded_bits disagrees with floor(payload*pixel_count): "
            f"{embedded_bit_mismatches[:10]}"
        )
    if changed_count_mismatches:
        raise AssertionError(
            f"changed_pixels audit failed: {sorted(set(changed_count_mismatches))[:10]}"
        )
    if exact_stego_mismatches:
        raise AssertionError(
            f"Stored stego cannot be regenerated exactly: {exact_stego_mismatches[:10]}"
        )

    patient_sets = {
        name: set(split.loc[split["split"].eq(name), "patient_id"].astype(str))
        for name in SPLIT_NAMES
    }
    return {
        "status": "passed",
        "config": str(Path(config_path).resolve()),
        "artifacts": {
            "split_manifest": str(split_path.resolve()),
            "patient_assignments": str(assignments_path.resolve()),
            "generation_manifest": str(generation_path.resolve()),
            "pair_manifest": str(manifest_path.resolve()),
        },
        "checks": {
            "patient_disjoint": True,
            "patient_assignment_precedes_sampling_and_agrees": True,
            "exact_split_view_quotas": True,
            "cover_stego_share_one_pair_row_and_split": True,
            "all_png_mode_and_size_valid": True,
            "embedded_bits_use_floor_bpp_definition": True,
            "changed_pixels_not_payload_bpp": True,
            "file_sha256_verified": bool(verify_hashes),
            "exact_stego_regeneration_verified": bool(verify_pixels),
        },
        "data": {
            "pairs": int(len(manifest)),
            "classification_samples_per_patch": int(2 * len(manifest)),
            "assigned_patients_before_sampling": int(len(assignments)),
            "selected_patients": int(split["patient_id"].astype(str).nunique()),
            "patients_by_split": {name: len(values) for name, values in patient_sets.items()},
            "split_view_counts": _counts(split),
            "payload_bpp": sorted(manifest["payload_bpp"].astype(float).unique().tolist()),
            "changed_pixel_fraction": float(
                manifest["changed_pixels"].astype(int).sum() / total_pixels
            ),
        },
        "official_split": official_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--patient-assignments", type=Path)
    parser.add_argument("--generation-manifest", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--verify-pixels", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {args.output}")
    report = audit(
        args.config,
        manifest_path=args.manifest,
        split_path=args.split_manifest,
        assignments_path=args.patient_assignments,
        generation_path=args.generation_manifest,
        verify_hashes=args.verify_hashes,
        verify_pixels=args.verify_pixels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["data"], indent=2, sort_keys=True))
    print(f"Audit passed: {args.output}")


if __name__ == "__main__":
    main()
