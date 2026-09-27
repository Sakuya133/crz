#!/usr/bin/env python3
"""Train the optional full-SRM + StandardScaler + LinearSVC comparator."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.models.srm_svm import train_srm_svm  # noqa: E402
from cxr_steganalysis.experiment import save_run_provenance, effective_seeds
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit full SRM features with a train-only StandardScaler and LinearSVC. "
            "The classifier is not the original SRM FLD ensemble."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-view", choices=["AP", "PA"], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seeds"] = effective_seeds(config)
        config["seed"] = args.seed
        config["seeds"]["training"] = args.seed
    manifest = args.manifest or resolve_config_path(config, "pair_manifest")
    path = train_srm_svm(
        manifest,
        args.feature_dir,
        args.output_dir,
        source_view=args.train_view,
        seed=args.seed if args.seed is not None else int(config["seed"]),
        c_grid=args.c_grid,
    )
    binding = json.loads((args.output_dir/"model.json").read_text())["binding"]
    config["training"]["train_view"] = args.train_view
    config["classifier"] = {"name":"full_srm_standardscaler_linearsvc", "c_grid":args.c_grid, "seed":args.seed if args.seed is not None else int(config["seed"])}
    save_run_provenance(args.output_dir, config, binding)
    print(f"SRM+LinearSVC checkpoint: {path}")


if __name__ == "__main__":
    main()
