#!/usr/bin/env python3
"""Extract optional full-SRM features on the same patches used by the CNNs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.models.srm_svm import extract_srm_features  # noqa: E402
from cxr_steganalysis.experiment import effective_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sealwatch full SRM (34,671 features) from shared uint8 crops."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="CPU processes; default config srm_workers or 1; budget RAM")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = args.manifest or resolve_config_path(config, "pair_manifest")
    path = extract_srm_features(
        manifest,
        args.output_dir,
        patch_size=int(config["patch_size"]),
        patches_per_image=int(config["patches_per_image"]),
        crop_seed=effective_seeds(config)["crop"],
        workers=args.workers if args.workers is not None else int(config.get("srm_workers", 1)),
    )
    print(f"Full SRM feature matrix: {path}")


if __name__ == "__main__":
    main()
