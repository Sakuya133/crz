#!/usr/bin/env python3
"""Build and validate the one-row-per-pair cover-stego manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.data.manifest import build_pair_manifest, save_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge split and generation records into a validated pair manifest."
    )
    parser.add_argument("--split-manifest", type=Path, required=True, help="Patient split CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output cover-stego CSV")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_existing.yaml"))
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        help="Generation record CSV; defaults to config paths.generation_manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    generation_path = args.generation_manifest or resolve_config_path(
        config, "generation_manifest"
    )
    split_frame = pd.read_csv(args.split_manifest, dtype={"patient_id": str})
    generation_frame = pd.read_csv(generation_path, dtype={"seed": str})
    manifest = build_pair_manifest(split_frame, generation_frame)
    save_manifest(manifest, args.output)
    print(f"Validated {len(manifest)} unique cover-stego pairs")
    print("Patient overlap across all split pairs: 0")
    print(f"Pair manifest written to: {args.output}")


if __name__ == "__main__":
    main()
