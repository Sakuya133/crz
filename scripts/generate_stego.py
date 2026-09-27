#!/usr/bin/env python3
"""Generate deterministic LSB Matching stego PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.data.manifest import generate_stego_records  # noqa: E402
from cxr_steganalysis.experiment import effective_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grayscale PNG stego images using deterministic LSB Matching."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Patient split CSV")
    parser.add_argument("--config", type=Path, required=True, help="Pilot YAML configuration")
    parser.add_argument("--output-dir", type=Path, help="Override stego image directory")
    parser.add_argument("--output-manifest", type=Path, help="Override generation record CSV")
    parser.add_argument(
        "--allow-grayscale-conversion",
        action="store_true",
        help="Explicitly convert non-L images to grayscale (disabled by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_frame = pd.read_csv(args.manifest, dtype={"patient_id": str})
    output_dir = args.output_dir or resolve_config_path(config, "stego_dir")
    output_manifest = args.output_manifest or resolve_config_path(config, "generation_manifest")
    if output_manifest.exists() or (output_dir.exists() and any(output_dir.iterdir())):
        raise FileExistsError("Generation output exists. For resumable audited pilot import use scripts/data.py; never overwrite stego.")
    allow_conversion = bool(
        args.allow_grayscale_conversion or config["data"]["allow_grayscale_conversion"]
    )
    records = generate_stego_records(
        split_frame,
        output_dir=output_dir,
        payload_bpp=float(config["payload_bpp"]),
        global_seed=effective_seeds(config)["embedding"],
        allow_conversion=allow_conversion,
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_manifest, index=False)
    print(
        f"Generated {len(records)} stego PNG files; "
        f"changed pixels={int(records['changed_pixels'].sum())}"
    )
    print(f"Generation manifest written to: {output_manifest}")


if __name__ == "__main__":
    main()
