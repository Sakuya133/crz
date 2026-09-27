#!/usr/bin/env python3
"""Explicit mode comparison of two prediction files, filtering one fixed target."""
import argparse
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from cxr_steganalysis.evaluation.cluster_bootstrap import load_prediction_file, patient_cluster_paired_bootstrap, save_bootstrap_results
from cxr_steganalysis.provenance import sha256_file

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="Matched source for cross_view mode")
    p.add_argument("--candidate", type=Path, required=True, help="Mismatched source for cross_view mode")
    p.add_argument("--mode", choices=["cross_view", "method"], required=True)
    p.add_argument("--target-view", choices=["AP", "PA"], required=True)
    p.add_argument("--replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frames = [load_prediction_file(path) for path in (a.baseline,a.candidate)]
    frames = [frame[frame.target_view == a.target_view] for frame in frames]
    result = patient_cluster_paired_bootstrap(*frames,target_view=a.target_view,replicates=a.replicates,seed=a.seed,comparison_mode=a.mode,
               matched_prediction_sha256=sha256_file(a.baseline),mismatched_prediction_sha256=sha256_file(a.candidate))
    save_bootstrap_results(result,a.output)
    print(result["delta_definition"],result["estimates"]["roc_auc"]["delta"])
