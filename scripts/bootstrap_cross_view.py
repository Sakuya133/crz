#!/usr/bin/env python3
"""Run a fixed-target patient-cluster paired bootstrap comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cxr_steganalysis.evaluation.cluster_bootstrap import (  # noqa: E402
    bootstrap_prediction_files,
    save_bootstrap_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched and mismatched models on the same target samples using "
            "a patient-cluster paired percentile bootstrap. Delta is mismatched - matched."
        )
    )
    parser.add_argument("--matched-predictions", type=Path, required=True)
    parser.add_argument("--mismatched-predictions", type=Path, required=True)
    parser.add_argument("--target-view", choices=["AP", "PA"], required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, required=True, help="Output stem for JSON/CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = bootstrap_prediction_files(
        args.matched_predictions,
        args.mismatched_predictions,
        target_view=args.target_view,
        replicates=args.replicates,
        seed=args.seed,
    )
    save_bootstrap_results(results, args.output)
    print(
        f"Bootstrap target={results['target_view']}: "
        f"valid={results['replicates_valid']}/"
        f"{results['replicates_requested']}; outputs={args.output}.json/.csv"
    )


if __name__ == "__main__":
    main()
