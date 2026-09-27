from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cxr_steganalysis.config import load_config
from cxr_steganalysis.data.metadata import (
    discover_local_pngs,
    filter_metadata_to_local_images,
)
from cxr_steganalysis.data.split import (
    assert_patient_disjoint,
    build_patient_assignments,
    create_patient_split,
    sample_split_by_view,
    scope_official_image_lists,
    validate_patient_assignments,
    validate_pilot_split,
)


def _save_image(path: Path, mode: str = "L") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = 50 if mode == "L" else (50, 50, 50, 255)
    Image.new(mode, (24, 24), color).save(path, format="PNG")


def _normalized_metadata(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "image_id", "patient_id", "view_position", "finding_labels",
    ])


def test_full_metadata_and_raw_subset_return_only_local_mode_l_candidates(tmp_path):
    raw = tmp_path / "raw"
    _save_image(raw / "00000001_000.png", "L")
    _save_image(raw / "00000002_000.png", "RGBA")
    metadata = _normalized_metadata([
        {"image_id": "00000001_000.png", "patient_id": "1", "view_position": "AP", "finding_labels": ""},
        {"image_id": "00000002_000.png", "patient_id": "2", "view_position": "PA", "finding_labels": ""},
        {"image_id": "00000003_000.png", "patient_id": "3", "view_position": "AP", "finding_labels": ""},
    ])

    candidates, report = filter_metadata_to_local_images(
        metadata, raw, allowed_views=["AP", "PA"], required_image_mode="L"
    )

    assert candidates["image_id"].tolist() == ["00000001_000.png"]
    assert report["num_png_files_found"] == 2
    assert report["num_metadata_matches"] == 2
    assert report["num_metadata_without_file"] == 1
    assert report["num_files_without_metadata"] == 0
    assert report["image_mode_counts"] == {"L": 1, "RGBA": 1}
    assert report["num_excluded_by_mode"] == 1
    assert report["excluded_mode_view_counts"] == {"RGBA": {"PA": 1}}
    assert report["candidate_view_counts"] == {"AP": 1}


def test_duplicate_png_basename_is_a_hard_failure(tmp_path):
    raw = tmp_path / "raw"
    _save_image(raw / "a" / "same.png")
    _save_image(raw / "b" / "same.png")
    with pytest.raises(ValueError, match="Duplicate PNG basename"):
        discover_local_pngs(raw)


def test_too_small_image_has_an_explicit_exclusion_reason(tmp_path):
    raw = tmp_path / "raw"
    _save_image(raw / "00000001_000.png", "L")
    metadata = _normalized_metadata([{
        "image_id": "00000001_000.png",
        "patient_id": "1",
        "view_position": "AP",
        "finding_labels": "",
    }])
    candidates, report = filter_metadata_to_local_images(
        metadata,
        raw,
        minimum_image_size=32,
    )
    assert candidates.empty
    assert report["num_excluded_by_size"] == 1
    assert report["excluded_images"] == [{
        "image_id": "00000001_000.png",
        "view_position": "AP",
        "reason": "smaller_than_patch:24x24",
    }]


def _mixed_patient_pool(tmp_path: Path) -> tuple[pd.DataFrame, set[str], set[str], set[str]]:
    rows = []
    official_train = set()
    official_test = set()
    test_patients = {"10", "11"}
    for patient in range(12):
        patient_id = str(patient)
        for view in ("AP", "PA"):
            for image_number in range(2):
                image_id = f"{patient:08d}_{view.lower()}_{image_number:03d}.png"
                path = tmp_path / "raw" / image_id
                _save_image(path)
                rows.append({
                    "image_id": image_id,
                    "patient_id": patient_id,
                    "view_position": view,
                    "finding_labels": "No Finding",
                    "cover_path": str(path.resolve()),
                    "image_mode": "L",
                })
                (official_test if patient_id in test_patients else official_train).add(image_id)
    return pd.DataFrame(rows), official_train, official_test, test_patients


def test_official_scoping_assignment_and_sampling_are_exact_and_deterministic(tmp_path):
    pool, official_train, official_test, test_patients = _mixed_patient_pool(tmp_path)
    full_train = official_train | {"absent_train.png"}
    full_test = official_test | {"absent_test.png"}
    scoped_train, scoped_test, scope_report = scope_official_image_lists(
        pool["image_id"], full_train, full_test
    )
    assert scoped_train == official_train
    assert scoped_test == official_test
    assert scope_report["train_entries_excluded"] == 1
    assert scope_report["test_entries_excluded"] == 1

    assigned = create_patient_split(
        pool.sample(frac=1.0, random_state=99),
        seed=1337,
        official_train_ids=scoped_train,
        official_test_ids=scoped_test,
    )
    assignments = build_patient_assignments(assigned, test_patients)
    validate_patient_assignments(assignments, test_patients)
    assert len(assignments) == 12
    assert set(assignments.loc[
        assignments["patient_id"].isin(test_patients), "split"
    ]) == {"test"}
    assert assignments["view_group"].eq("MIXED").all()

    quotas = {"train": 2, "validation": 1, "test": 1}
    first = sample_split_by_view(assigned, quotas, seed=1337)
    second = sample_split_by_view(
        assigned.iloc[::-1].reset_index(drop=True), quotas, seed=1337
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby(["split", "view_position"]).size().to_dict() == {
        ("test", "AP"): 1,
        ("test", "PA"): 1,
        ("train", "AP"): 2,
        ("train", "PA"): 2,
        ("validation", "AP"): 1,
        ("validation", "PA"): 1,
    }
    assert first.groupby("patient_id")["split"].nunique().max() == 1
    validate_pilot_split(
        first,
        quotas,
        official_test_patient_ids=test_patients,
        check_files=True,
    )


def test_non_l_cover_cannot_enter_validated_pilot_split(tmp_path):
    pool, official_train, official_test, test_patients = _mixed_patient_pool(tmp_path)
    assigned = create_patient_split(
        pool,
        seed=1337,
        official_train_ids=official_train,
        official_test_ids=official_test,
    )
    quotas = {"train": 2, "validation": 1, "test": 1}
    sampled = sample_split_by_view(assigned, quotas, seed=1337)
    rgba_path = tmp_path / "raw" / "replacement_rgba.png"
    _save_image(rgba_path, "RGBA")
    sampled.loc[0, "cover_path"] = str(rgba_path)
    sampled.loc[0, "image_mode"] = "RGBA"
    with pytest.raises(ValueError, match="non-required recorded image modes"):
        validate_pilot_split(
            sampled,
            quotas,
            official_test_patient_ids=test_patients,
            check_files=True,
        )


def test_insufficient_split_view_quota_is_a_hard_failure(tmp_path):
    pool, official_train, official_test, _ = _mixed_patient_pool(tmp_path)
    assigned = create_patient_split(
        pool,
        seed=1337,
        official_train_ids=official_train,
        official_test_ids=official_test,
    )
    with pytest.raises(ValueError, match="Insufficient quota"):
        sample_split_by_view(
            assigned,
            {"train": 100, "validation": 1, "test": 1},
            seed=1337,
        )


def test_patient_overlap_remains_a_hard_failure():
    frame = pd.DataFrame([
        {"image_id": "a.png", "patient_id": "1", "split": "train"},
        {"image_id": "b.png", "patient_id": "1", "split": "test"},
    ])
    with pytest.raises(AssertionError, match="overlap"):
        assert_patient_disjoint(frame)


def test_legacy_config_does_not_enable_available_filtering(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("seed: 17\n", encoding="utf-8")
    config = load_config(config_path)
    assert config["pilot"]["available_only"] is False
    metadata = _normalized_metadata([
        {"image_id": f"{patient:08d}_000.png", "patient_id": str(patient),
         "view_position": "AP" if patient % 2 else "PA", "finding_labels": ""}
        for patient in range(10)
    ])
    split = create_patient_split(metadata, seed=17)
    assert len(split) == len(metadata)
    assert set(split["split"]) == {"train", "validation", "test"}
