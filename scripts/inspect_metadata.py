#!/usr/bin/env python3
"""Inspect NIH ChestX-ray14 metadata without downloading the dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.data.metadata import (  # noqa: E402
    filter_metadata_to_local_images,
    human_summary,
    inspect_metadata,
    load_metadata,
    read_image_list,
    save_metadata_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect NIH metadata, source views, patients, labels, and file integrity."
    )
    parser.add_argument("--config", type=Path, help="Optional pilot YAML configuration")
    parser.add_argument("--metadata", type=Path, help="Override metadata CSV")
    parser.add_argument("--output", type=Path, required=True, help="Report path (.json or .csv)")
    parser.add_argument("--raw-dir", type=Path, help="Optional directory containing NIH PNG files")
    parser.add_argument("--train-list", type=Path, help="Optional official train_val_list.txt")
    parser.add_argument("--test-list", type=Path, help="Optional official test_list.txt")
    parser.add_argument(
        "--available-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Join metadata to local PNG files before reporting",
    )
    parser.add_argument(
        "--allowed-views",
        nargs="+",
        help="Views retained by available-only inspection (for example AP PA)",
    )
    parser.add_argument(
        "--required-image-mode",
        help="Pillow image mode required by available-only inspection (for example L)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config is not None else None
    metadata_path = args.metadata or (
        resolve_config_path(config, "metadata") if config is not None else None
    )
    if metadata_path is None:
        raise ValueError("Pass --metadata or provide --config with paths.metadata")
    raw_dir = args.raw_dir or (
        resolve_config_path(config, "raw_dir") if config is not None else None
    )
    train_list_path = args.train_list or (
        resolve_config_path(config, "official_train_list") if config is not None else None
    )
    test_list_path = args.test_list or (
        resolve_config_path(config, "official_test_list") if config is not None else None
    )
    pilot_config = config.get("pilot", {}) if config is not None else {}
    available_only = (
        args.available_only
        if args.available_only is not None
        else bool(pilot_config.get("available_only", False))
    )
    allowed_views = args.allowed_views or pilot_config.get("allowed_views", ["AP", "PA"])
    required_mode = args.required_image_mode or pilot_config.get("required_image_mode") or "L"

    frame = load_metadata(metadata_path)
    official_train = read_image_list(train_list_path)
    official_test = read_image_list(test_list_path)
    discovery_report = None
    official_scope = None
    if available_only:
        if raw_dir is None:
            raise ValueError("available-only inspection requires --raw-dir or config paths.raw_dir")
        frame, discovery_report = filter_metadata_to_local_images(
            frame,
            raw_dir=raw_dir,
            allowed_views=allowed_views,
            required_image_mode=required_mode,
            minimum_image_size=int(config["patch_size"]) if config is not None else None,
        )
        candidate_ids = set(frame["image_id"])
        scoped_train = official_train & candidate_ids
        scoped_test = official_test & candidate_ids
        official_scope = {
            "train_entries_total": len(official_train),
            "train_entries_retained": len(scoped_train),
            "train_entries_excluded": len(official_train - candidate_ids),
            "test_entries_total": len(official_test),
            "test_entries_retained": len(scoped_test),
            "test_entries_excluded": len(official_test - candidate_ids),
        }
        official_train = scoped_train
        official_test = scoped_test
    report = inspect_metadata(
        frame,
        raw_dir=None if available_only else raw_dir,
        official_train=official_train,
        official_test=official_test,
    )
    if discovery_report is not None:
        report["local_discovery"] = discovery_report
        report["official_list_scope"] = official_scope
    save_metadata_report(report, args.output)
    print(human_summary(report))
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
