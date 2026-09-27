"""Joint AP/PA patient bootstrap for a controlled mixed-view objective ablation."""
import numpy as np
import pandas as pd
from cxr_steganalysis.evaluation.cluster_bootstrap import _align_method_frames, _constant_value
from cxr_steganalysis.training.metrics import binary_metrics


METRICS = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "false_positive_rate")


def mixed_patient_bootstrap(baseline, candidate, replicates=10000, seed=1337):
    """Retain BOTH views of each resampled patient and the sampling multiplicity.

    Macro AUC is the mean of within-view AUCs, never AUC of pooled raw scores.
    The worst-view identity is recomputed within each replicate.
    """
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    for frame in (baseline, candidate):
        if set(frame.target_view) != {"AP", "PA"} or _constant_value(frame, "source_view") != "ALL":
            raise ValueError("Joint mitigation summary requires both targets and mixed-view source ALL")
        for column in ("threshold", "checkpoint_sha256", "model_name", "model_version", "protocol_id", "manifest_identity", "seed", "crop_seed", "patch_size"):
            _constant_value(frame, column)
    aligned = [_align_method_frames(baseline[baseline.target_view == v], candidate[candidate.target_view == v], v) for v in ("AP", "PA")]
    left = pd.concat([a[0] for a in aligned], ignore_index=True)
    right = pd.concat([a[1] for a in aligned], ignore_index=True)
    for column in ("model_name", "model_version", "normalization"):
        if _constant_value(left, column) != _constant_value(right, column):
            raise ValueError("Objective ablation requires identical backbone and normalization")
    thresholds = [float(_constant_value(f, "threshold")) for f in (left, right)]
    labels = left.label.to_numpy()
    views = left.target_view.to_numpy()
    scores = [f.score.to_numpy() for f in (left, right)]
    patients = sorted(left.patient_id.unique())
    clusters = [np.flatnonzero(left.patient_id.to_numpy() == p) for p in patients]

    def summaries(index):
        masks = [index[views[index] == v] for v in ("AP", "PA")]
        if any(len(i) == 0 for i in masks):
            return None
        result = {}
        for slot, values, threshold in zip(("matched", "mismatched"), scores, thresholds):
            per_view = [binary_metrics(labels[i], values[i], threshold) for i in masks]
            for aggregate in ("macro", "worst_view"):
                for metric in METRICS:
                    m = [r[metric] for r in per_view]
                    value = np.mean(m) if aggregate == "macro" else max(m) if metric == "false_positive_rate" else min(m)
                    result[aggregate, metric, slot] = float(value)
        for aggregate in ("macro", "worst_view"):
            for metric in METRICS:
                result[aggregate, metric, "delta"] = result[aggregate, metric, "mismatched"] - result[aggregate, metric, "matched"]
        return result

    point = summaries(np.arange(len(left)))
    distributions = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    invalid = 0
    for _ in range(replicates):
        selected = rng.choice(len(clusters), len(clusters), replace=True)
        values = summaries(np.concatenate([clusters[i] for i in selected]))
        if values is None or not all(np.isfinite(list(values.values()))):
            invalid += 1
            continue
        for key, value in values.items():
            distributions[key].append(value)
    if invalid == replicates:
        raise ValueError("No valid joint bootstrap replicates")
    output = []
    for aggregate in ("macro", "worst_view"):
        estimates = {}
        for metric in METRICS:
            estimates[metric] = {}
            for slot in ("matched", "mismatched", "delta"):
                key = aggregate, metric, slot
                lo, hi = np.percentile(distributions[key], [2.5, 97.5])
                estimates[metric][slot] = dict(point_estimate=point[key], lower_2_5=float(lo), upper_97_5=float(hi))
        output.append(dict(target_view=aggregate, patients=len(patients), pairs=int(left.pair_id.nunique()), samples=len(left),
                           replicates_requested=replicates, replicates_valid=replicates-invalid, replicates_invalid=invalid,
                           bootstrap_seed=seed, delta_definition="candidate - baseline", comparison_mode="mixed_objective_summary",
                           slot_semantics="matched=ERM baseline; mismatched=GroupDRO candidate",
                           cluster_scope="joint patients across AP and PA; all rows and multiplicity retained",
                           confidence_interval="95% percentile paired patient-cluster bootstrap; worst-view recomputed each replicate",
                           training_seed_uncertainty="not captured; fixed fitted models", estimates=estimates))
    return output
