#!/usr/bin/env python3
"""Create patient-disjoint train/validation/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.experiment import effective_seeds
from cxr_steganalysis.data.metadata import (  # noqa: E402
    filter_metadata_to_local_images,
    load_metadata,
    read_image_list,
)
from cxr_steganalysis.data.split import (  # noqa: E402
    build_patient_assignments,
    create_patient_split,
    sample_split_by_view,
    save_patient_assignments,
    save_split,
    scope_official_image_lists,
    split_summary,
    validate_official_patient_partition,
    validate_patient_assignments,
    validate_pilot_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic patient-disjoint NIH splits."
    )
    parser.add_argument("--metadata", type=Path, help="Override metadata CSV")
    parser.add_argument("--config", type=Path, required=True, help="Pilot YAML configuration")
    parser.add_argument("--output", type=Path, required=True, help="Output split CSV")
    parser.add_argument("--raw-dir", type=Path, help="Override raw PNG directory")
    parser.add_argument(
        "--patient-assignments-output",
        type=Path,
        help="One-row-per-patient assignment CSV (pilot available-only mode)",
    )
    parser.add_argument(
        "--available-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Discover local files and enable post-split balanced pilot sampling",
    )
    parser.add_argument("--allowed-views", nargs="+", help="Override pilot allowed views")
    parser.add_argument("--required-image-mode", help="Override required Pillow mode")
    parser.add_argument("--sampling-seed", type=int, help="Override pilot image sampling seed")
    parser.add_argument("--quota-train", type=int, help="Images per view in train")
    parser.add_argument("--quota-validation", type=int, help="Images per view in validation")
    parser.add_argument("--quota-test", type=int, help="Images per view in test")
    parser.add_argument(
        "--ignore-official-test",
        action="store_true",
        help="Create a fresh 70/10/20 patient split even when official lists exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.get("dataset_profile") in {"pilot_existing", "full_all_eligible"}:
        raise ValueError("Protected enrollment: import the audited pilot with scripts/data.py or expand locked patients with scripts/full_data.py. This generic splitter cannot redraw these protocols.")
    metadata_path = args.metadata or resolve_config_path(config, "metadata")
    frame = load_metadata(metadata_path)
    raw_dir = args.raw_dir or resolve_config_path(config, "raw_dir")

    split_config = config["split"]
    use_official = split_config["use_official_test"] and not args.ignore_official_test
    train_list_path = resolve_config_path(config, "official_train_list")
    test_list_path = resolve_config_path(config, "official_test_list")
    pilot_config = config.get("pilot", {})
    available_only = (
        args.available_only
        if args.available_only is not None
        else bool(pilot_config.get("available_only", False))
    )
    if use_official:
        missing_lists = [
            str(path) for path in (train_list_path, test_list_path) if not path.is_file()
        ]
        if missing_lists:
            raise FileNotFoundError(
                f"Official split requires both list files; missing: {missing_lists}. "
                "Set split.use_official_test=false or pass --ignore-official-test "
                "to create an explicitly non-official split."
            )
    official_train = read_image_list(train_list_path) if use_official else set()
    official_test = read_image_list(test_list_path) if use_official else set()
    if use_official and (bool(official_train) != bool(official_test)):
        raise FileNotFoundError(
            "Use both official list files or neither. Expected: "
            f"{train_list_path} and {test_list_path}"
        )
    if use_official:
        validate_official_patient_partition(frame, official_train, official_test)

    discovery_report = None
    scope_report = None
    official_test_patients: set[str] = set()
    if available_only:
        allowed_views = args.allowed_views or pilot_config.get("allowed_views", ["AP", "PA"])
        required_mode = args.required_image_mode or pilot_config.get("required_image_mode")
        if required_mode is None:
            raise ValueError("Pilot available-only split requires required_image_mode")
        frame, discovery_report = filter_metadata_to_local_images(
            frame,
            raw_dir=raw_dir,
            allowed_views=allowed_views,
            required_image_mode=str(required_mode),
            minimum_image_size=int(config["patch_size"]),
        )
        if use_official:
            official_train, official_test, scope_report = scope_official_image_lists(
                frame["image_id"], official_train, official_test
            )
            official_test_patients = set(
                frame.loc[frame["image_id"].isin(official_test), "patient_id"].astype(str)
            )
    else:
        frame["cover_path"] = frame["image_id"].map(lambda value: str(raw_dir / value))

    assigned = create_patient_split(
        frame,
        seed=effective_seeds(config)["split"],
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        official_train_ids=official_train if use_official else None,
        official_test_ids=official_test if use_official else None,
    )

    if available_only:
        assignments = build_patient_assignments(assigned, official_test_patients)
        validate_patient_assignments(assignments, official_test_patients)
        assignments_output = args.patient_assignments_output or resolve_config_path(
            config, "patient_assignments"
        )
        quotas = dict(pilot_config.get("quotas_per_view") or {})
        quota_overrides = {
            "train": args.quota_train,
            "validation": args.quota_validation,
            "test": args.quota_test,
        }
        for split, override in quota_overrides.items():
            if override is not None:
                quotas[split] = override
        if set(quotas) != {"train", "validation", "test"}:
            raise ValueError(
                "Pilot available-only split requires quotas_per_view for train, "
                "validation, and test"
            )
        sampling_seed = (
            args.sampling_seed
            if args.sampling_seed is not None
            else int(pilot_config.get("sampling_seed", config["seed"]))
        )
        result = sample_split_by_view(
            assigned,
            quotas_per_view=quotas,
            seed=sampling_seed,
            allowed_views=allowed_views,
        )
        validate_pilot_split(
            result,
            quotas_per_view=quotas,
            allowed_views=allowed_views,
            required_image_mode=str(required_mode),
            official_test_patient_ids=official_test_patients,
            minimum_image_size=int(config["patch_size"]),
            check_files=True,
        )
        save_patient_assignments(assignments, assignments_output)
        print(f"Patient assignments written to: {assignments_output}")
        print(
            "Local discovery: "
            f"files={discovery_report['num_png_files_found']}, "
            f"matched={discovery_report['num_metadata_matches']}, "
            f"excluded_mode={discovery_report['num_excluded_by_mode']}, "
            f"candidates={discovery_report['num_candidate_images']}, "
            f"views={discovery_report['candidate_view_counts']}"
        )
        if scope_report is not None:
            print(f"Official list scoping: {scope_report}")
    else:
        result = assigned

    save_split(result, args.output)
    print(split_summary(result).to_string(index=False))
    print("Patient overlap across all split pairs: 0")
    print(f"Split manifest written to: {args.output}")


if __name__ == "__main__":
    main()
