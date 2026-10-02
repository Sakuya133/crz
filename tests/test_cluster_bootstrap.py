from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cxr_steganalysis.evaluation.cluster_bootstrap import (
    bootstrap_prediction_files,
    patient_cluster_paired_bootstrap,
    resample_patient_clusters,
    save_bootstrap_results,
)
from cxr_steganalysis.evaluation.cross_view import RAW_PREDICTION_COLUMNS
from cxr_steganalysis.provenance import sha256_file


def _prediction_frame(source_view: str, scores: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    threshold = 0.5 if source_view == "PA" else 0.45
    for pair_index, (cover_score, stego_score) in enumerate(scores):
        patient_id = str(pair_index // 2 + 1)
        pair_id = f"pair-{pair_index}"
        for class_name, label, score in (
            ("cover", 0, cover_score),
            ("stego", 1, stego_score),
        ):
            rows.append({
                "pair_id": pair_id,
                "image_id": f"image-{pair_index}.png",
                "patient_id": patient_id,
                "split": "test",
                "view_position": "PA",
                "source_view": source_view,
                "target_view": "PA",
                "class_name": class_name,
                "label": label,
                "logit": float(np.log(score / (1.0 - score))),
                "score": score,
                "predicted_label": int(score >= threshold),
                "threshold": threshold,
                "patch_index": 0,
                "crop_top": pair_index,
                "crop_left": pair_index + 1,
                "checkpoint_path": f"/{source_view.lower()}/best.pt",
                "checkpoint_sha256": ("a" if source_view == "PA" else "b") * 64,
                "config_sha256": "c" * 64,
                "manifest_sha256": "d" * 64,
                "seed": 1337,
                "payload_bpp": 0.2,
                "algorithm": "lsb_matching",
            })
    return pd.DataFrame(rows)[RAW_PREDICTION_COLUMNS]


@pytest.fixture
def paired_predictions():
    matched = _prediction_frame(
        "PA", [(0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.35, 0.65)]
    )
    mismatched = _prediction_frame(
        "AP", [(0.25, 0.75), (0.35, 0.60), (0.55, 0.65), (0.60, 0.70)]
    )
    return matched, mismatched


def test_bootstrap_is_deterministic_and_reports_paired_deltas(paired_predictions):
    matched, mismatched = paired_predictions
    first = patient_cluster_paired_bootstrap(
        matched, mismatched, target_view="PA", replicates=200, seed=1337
    )
    second = patient_cluster_paired_bootstrap(
        matched, mismatched, target_view="PA", replicates=200, seed=1337
    )
    assert first == second
    assert first["delta_definition"] == "mismatched - matched"
    assert first["patients"] == 2
    assert first["pairs"] == 4
    assert first["samples"] == 8
    assert first["replicates_valid"] == 200
    assert first["replicates_invalid"] == 0
    expected_delta = (
        first["estimates"]["roc_auc"]["mismatched"]["point_estimate"]
        - first["estimates"]["roc_auc"]["matched"]["point_estimate"]
    )
    assert first["estimates"]["roc_auc"]["delta"]["point_estimate"] == pytest.approx(
        expected_delta
    )


def test_resampling_retains_every_row_in_selected_patient_clusters(paired_predictions):
    matched, _mismatched = paired_predictions
    positions, selected = resample_patient_clusters(
        matched, np.random.default_rng(7)
    )
    resampled = matched.iloc[positions]
    for patient_id in matched["patient_id"].unique():
        expected = int((selected == patient_id).sum()) * int(
            (matched["patient_id"] == patient_id).sum()
        )
        assert int((resampled["patient_id"] == patient_id).sum()) == expected


@pytest.mark.parametrize("failure", ["key", "label"])
def test_paired_bootstrap_rejects_unaligned_keys_or_labels(
    paired_predictions, failure
):
    matched, mismatched = paired_predictions
    broken = mismatched.copy()
    if failure == "key":
        broken.loc[0, "pair_id"] = "wrong-pair"
    else:
        broken.loc[0, "label"] = 1
    with pytest.raises(ValueError):
        patient_cluster_paired_bootstrap(
            matched, broken, target_view="PA", replicates=10, seed=1337
        )


def test_paired_bootstrap_rejects_duplicate_or_missing_patient(paired_predictions):
    matched, mismatched = paired_predictions
    duplicate = pd.concat([matched, matched.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="composite key"):
        patient_cluster_paired_bootstrap(
            duplicate, mismatched, target_view="PA", replicates=10
        )
    missing = mismatched.copy()
    missing.loc[0, "patient_id"] = None
    with pytest.raises(ValueError, match="patient_id"):
        patient_cluster_paired_bootstrap(
            matched, missing, target_view="PA", replicates=10
        )


def test_bootstrap_file_provenance_and_outputs(tmp_path, paired_predictions):
    matched, mismatched = paired_predictions
    matched_path = tmp_path / "matched_predictions.csv"
    mismatched_path = tmp_path / "mismatched_predictions.csv"
    # Lab files contain both targets; the other target must be filtered out.
    other = lambda f: f.assign(target_view="AP", view_position="AP")
    pd.concat([matched, other(matched)]).to_csv(matched_path, index=False)
    pd.concat([mismatched, other(mismatched)]).to_csv(mismatched_path, index=False)
    results = bootstrap_prediction_files(
        matched_path,
        mismatched_path,
        target_view="PA",
        replicates=25,
        seed=1337,
    )
    assert results["matched_prediction_sha256"] == sha256_file(matched_path)
    assert results["mismatched_prediction_sha256"] == sha256_file(mismatched_path)
    output = tmp_path / "statistics" / "target_pa"
    save_bootstrap_results(results, output)
    assert output.with_suffix(".json").is_file()
    table = pd.read_csv(output.with_suffix(".csv"))
    assert len(table) == 15  # Five metrics, each baseline/candidate/delta.
    assert "false_positive_rate" in set(table.metric)
    assert set(table["comparison"]) == {"matched", "mismatched", "delta"}


def test_method_mode_explicit_and_protocol_guard(paired_predictions):
    baseline, candidate = paired_predictions
    baseline = baseline.assign(model_name="highpass", model_version="v1", score_type="sigmoid_probability", protocol_id="A", manifest_identity="e"*64, crop_seed=1337, patch_size=256)
    candidate = candidate.assign(model_name="srnet", model_version="v2", score_type="sigmoid_probability", protocol_id="A", manifest_identity="e"*64, crop_seed=1337, patch_size=256)
    with pytest.raises(ValueError, match="different model"):
        patient_cluster_paired_bootstrap(baseline, candidate, target_view="PA", replicates=10)
    candidate.source_view = "PA"
    result = patient_cluster_paired_bootstrap(baseline, candidate, target_view="PA", replicates=10, comparison_mode="method")
    assert result["delta_definition"] == "candidate - baseline"
    with pytest.raises(ValueError, match="protocol mismatch"):
        patient_cluster_paired_bootstrap(baseline, candidate.assign(crop_seed=42), target_view="PA", replicates=10, comparison_mode="method")
    with pytest.raises(ValueError, match="crop_top"):
        patient_cluster_paired_bootstrap(baseline, candidate.assign(crop_top=candidate.crop_top+1), target_view="PA", replicates=10, comparison_mode="method")


def test_joint_mitigation_bootstrap_keeps_patients_across_views(paired_predictions):
    from cxr_steganalysis.evaluation.mixed_bootstrap import mixed_patient_bootstrap
    from cxr_steganalysis.training.metrics import binary_metrics
    left, right = paired_predictions
    frames=[]
    for frame in (left,right):
        frame=frame.assign(source_view="ALL", model_name="highpass", model_version="v1",
                           protocol_id="B",manifest_identity="e"*64,crop_seed=1337,patch_size=256,normalization="unit_range")
        ap=frame.assign(target_view="AP",view_position="AP",pair_id="ap-"+frame.pair_id,image_id="ap-"+frame.image_id)
        frames.append(pd.concat([ap,frame],ignore_index=True))
    result=mixed_patient_bootstrap(*frames,replicates=50,seed=7)
    assert result==mixed_patient_bootstrap(*frames,replicates=50,seed=7)
    assert [r["target_view"] for r in result]==["macro","worst_view"]
    assert all(r["patients"]==2 and r["pairs"]==8 and r["samples"]==16 for r in result)
    points=[binary_metrics(f.label.to_numpy(),f.score.to_numpy(),f.threshold.iloc[0])["balanced_accuracy"] for f in frames]
    assert result[0]["estimates"]["balanced_accuracy"]["delta"]["point_estimate"]==pytest.approx(points[1]-points[0])
    broken=frames[1].copy()
    broken.loc[broken.target_view=="AP","threshold"]+=.01
    with pytest.raises(ValueError,match="threshold"):
        mixed_patient_bootstrap(frames[0],broken,replicates=10)
