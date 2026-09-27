import pandas as pd
import pytest
from cxr_steganalysis.data.full_dataset import assign_all_patients


def test_full_assignment_preserves_mixed_view_patient_and_exposure():
    metadata = pd.DataFrame([dict(image_id=f"{p}_{i}", patient_id=str(p), view_position=v) for p in range(10) for i,v in enumerate(("AP", "PA"))])
    train = set(metadata[metadata.patient_id.astype(int) < 8].image_id)
    test = set(metadata[metadata.patient_id.astype(int) >= 8].image_id)
    locked = pd.DataFrame([dict(patient_id="0", split="validation"), dict(patient_id="8", split="test")])
    a = assign_all_patients(metadata, locked, train, test, 1337, {"8"})
    b = assign_all_patients(metadata.sample(frac=1), locked, train, test, 1337, {"8"})
    pd.testing.assert_frame_equal(a, b)
    assert a.set_index("patient_id").loc["0", "split"] == "validation"
    assert a.set_index("patient_id").loc["8", "test_cohort"] == "pilot_exposed"
    assert a.set_index("patient_id").loc["9", "test_cohort"] == "confirmatory_unseen_patient"
    with pytest.raises(ValueError, match="conflicts"):
        assign_all_patients(metadata, locked.assign(split="train"), train, test, 1337, {"8"})


def test_full_official_missing_membership_rejected():
    metadata = pd.DataFrame([dict(image_id=str(p), patient_id=str(p)) for p in range(4)])
    with pytest.raises(ValueError, match="cover the full metadata"):
        assign_all_patients(metadata, pd.DataFrame(columns=["patient_id", "split"]), {"0"}, {"1"}, 1337, set())
