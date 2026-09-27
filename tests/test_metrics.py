from __future__ import annotations


def test_fast_metrics_and_threshold_equal_sklearn_and_exhaustive():
    import numpy as np
    from sklearn.metrics import roc_auc_score as reference_auc
    from cxr_steganalysis.training.metrics import binary_metrics, select_threshold, roc_auc_score
    rng = np.random.default_rng(19)
    for n in (20, 100, 400):
        labels = rng.integers(0, 2, n)
        scores = rng.integers(-10, 11, n)/10
        assert np.isclose(roc_auc_score(labels, scores), reference_auc(labels, scores), atol=1e-15)
        choices = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
        ba = np.array([binary_metrics(labels,scores,t)["balanced_accuracy"] for t in choices])
        expected = choices[np.isclose(ba,ba.max(),atol=1e-12,rtol=0)].max()
        assert select_threshold(labels, scores) == expected

import numpy as np

from cxr_steganalysis.training.metrics import binary_metrics, select_threshold


def test_metrics_and_validation_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    assert metrics["roc_auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["error_probability"] == 0.0


def test_threshold_supports_unbounded_margins_and_all_negative_point():
    labels = np.array([0, 1, 0, 1])
    inverted_margins = np.array([3.0, -2.0, 3.0, -2.0])
    threshold = select_threshold(labels, inverted_margins)
    assert threshold > inverted_margins.max()
    assert binary_metrics(labels, inverted_margins, threshold)["balanced_accuracy"] == 0.5
