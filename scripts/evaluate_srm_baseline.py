#!/usr/bin/env python3
"""Evaluate an optional SRM+LinearSVC run with its fixed validation margin."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.config import load_config, resolve_config_path  # noqa: E402
from cxr_steganalysis.models.srm_svm import evaluate_srm_svm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SRM+LinearSVC; exported scores are margins, not probabilities."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output file stem")
    parser.add_argument("--test-views", nargs="+", choices=["AP", "PA"], default=["AP", "PA"])
    parser.add_argument("--test-cohort", choices=["pilot_exposed", "confirmatory_unseen_patient"])
    parser.add_argument("--protocol-lock", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = args.manifest or resolve_config_path(config, "pair_manifest")
    results = evaluate_srm_svm(
        manifest,
        args.feature_dir,
        args.run_dir,
        args.output,
        config_path=args.config,
        target_views=args.test_views,
        test_cohort=args.test_cohort,
        protocol_lock=args.protocol_lock,
    )
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['train_view']} -> {result['test_view']}: "
            f"AUC={metrics['roc_auc']:.4f}, BA={metrics['balanced_accuracy']:.4f}, "
            f"margin_threshold={result['threshold']:.6g}"
        )


if __name__ == "__main__":
    main()
