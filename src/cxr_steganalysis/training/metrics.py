"""Dependency-light binary classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_ranks[inverse]
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional arrays of equal shape")
    if len(labels) == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be a non-empty binary array containing only 0/1")
    if not np.isfinite(scores).all() or not np.isfinite(float(threshold)):
        raise ValueError("scores and threshold must be finite")
    predictions = (scores >= threshold).astype(np.int64)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "roc_auc": roc_auc_score(labels, scores),
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
        "precision": precision,
        "recall": recall,
        "sensitivity": tpr,
        "specificity": tnr,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "error_probability": (fpr + fnr) / 2.0,
        "threshold": float(threshold),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Maximize validation balanced accuracy; largest threshold wins ties."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both classes in validation data")
    if not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires finite validation scores")
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("Threshold labels/scores must be equal-length 1D arrays")
    order = np.argsort(scores, kind="mergesort")
    unique, starts = np.unique(scores[order], return_index=True)
    if len(unique) == 1:
        # This finite threshold is the all-negative operating point. Both constant
        # predictions have BA=.5; the documented largest-threshold tie rule applies.
        return float(np.nextafter(unique[0], np.inf))
    all_negative = np.nextafter(unique[-1], np.inf)
    candidates = np.concatenate((unique, [all_negative]))
    positives = np.r_[0, np.cumsum(labels[order] == 1)]
    negatives = np.r_[0, np.cumsum(labels[order] == 0)]
    ba = ((positives[-1] - positives[starts]) / positives[-1] + negatives[starts] / negatives[-1]) / 2
    balanced = np.r_[ba, .5]
    winners = np.isclose(balanced, balanced.max(), rtol=0.0, atol=1e-12)
    return float(candidates[winners].max())
