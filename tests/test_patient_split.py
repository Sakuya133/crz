from __future__ import annotations

import pandas as pd
import pytest

from cxr_steganalysis.data.split import (
    assert_patient_disjoint,
    create_patient_split,
    validate_official_patient_partition,
)


def make_metadata(num_patients: int = 20) -> pd.DataFrame:
    rows = []
    for patient in range(num_patients):
        for image in range(2 if patient % 3 == 0 else 1):
            rows.append({
                "image_id": f"{patient:08d}_{image:03d}.png",
                "patient_id": f"{patient:08d}",
                "view_position": "PA" if patient % 2 == 0 else "AP",
                "finding_labels": "No Finding",
            })
    return pd.DataFrame(rows)


def test_patient_split_is_disjoint_and_deterministic():
    metadata = make_metadata()
    first = create_patient_split(metadata, seed=1337)
    second = create_patient_split(metadata, seed=1337)
    assert first["split"].tolist() == second["split"].tolist()
    assert_patient_disjoint(first)
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first.groupby("patient_id")["split"].nunique().max() == 1


def test_official_test_is_preserved_by_patient():
    metadata = make_metadata(12)
    test_patient = "00000000"
    official_test = {metadata.iloc[0]["image_id"]}
    official_train = set(metadata.loc[metadata["patient_id"] != test_patient, "image_id"])
    split = create_patient_split(
        metadata,
        seed=5,
        official_train_ids=official_train,
        official_test_ids=official_test,
    )
    assert set(split.loc[split["patient_id"] == test_patient, "split"]) == {"test"}
    assert_patient_disjoint(split)


def test_overlap_assertion_is_hard_failure():
    frame = make_metadata(3)
    frame["split"] = ["train", "validation", "test", "test"]
    with pytest.raises(AssertionError, match="overlap"):
        assert_patient_disjoint(frame)


def test_official_patient_overlap_is_checked_before_local_filtering():
    metadata = pd.DataFrame([
        {"image_id": "a.png", "patient_id": "1", "view_position": "AP"},
        {"image_id": "b.png", "patient_id": "1", "view_position": "PA"},
    ])
    with pytest.raises(AssertionError, match="full metadata"):
        validate_official_patient_partition(metadata, {"a.png"}, {"b.png"})


def test_no_available_official_test_never_falls_back_to_random():
    frame=make_metadata(12)
    with pytest.raises(ValueError,match="refusing random-test fallback"):
        create_patient_split(frame,1337,official_train_ids=set(frame.image_id),official_test_ids=set())
