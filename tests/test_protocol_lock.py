from copy import deepcopy
import pandas as pd
import pytest
from cxr_steganalysis.config import load_config
from cxr_steganalysis.protocol_lock import freeze_protocol, validate_test_access, settings_identity


def test_confirmatory_access_requires_explicit_matching_freeze(tmp_path):
    config = load_config("configs/full_all_eligible.yaml")
    frame = pd.DataFrame(dict(pair_id=["a"], patient_id=["1"], split=["test"],
                              test_cohort=["confirmatory_unseen_patient"], cover_path=["old"]))
    with pytest.raises(ValueError, match="explicit --test-cohort"):
        validate_test_access(config, frame, None)
    with pytest.raises(ValueError, match="requires --protocol-lock"):
        validate_test_access(config, frame, "confirmatory_unseen_patient")
    lock = tmp_path / "lock.json"
    freeze_protocol([config], frame, lock)
    assert len(validate_test_access(config, frame, "confirmatory_unseen_patient", lock)) == 64
    with pytest.raises(FileExistsError):
        freeze_protocol([config], frame, lock)
    changed = deepcopy(config)
    changed["learning_rate"] *= 10
    with pytest.raises(ValueError, match="Scientific settings"):
        validate_test_access(changed, frame, "confirmatory_unseen_patient", lock)
    with pytest.raises(ValueError, match="enrollment"):
        validate_test_access(config, frame.assign(patient_id="2"), "confirmatory_unseen_patient", lock)
    relocated = deepcopy(config)
    relocated["paths"]["raw_dir"] = "/kaggle/input/images"
    relocated["training"]["device"] = "cpu"
    assert settings_identity(config) == settings_identity(relocated)
    assert validate_test_access(relocated, frame.assign(cover_path="new"), "confirmatory_unseen_patient", lock)
    assert validate_test_access(config, frame, "pilot_exposed") is None


def test_pilot_remains_backward_compatible():
    config = load_config("configs/pilot_existing.yaml")
    assert validate_test_access(config, pd.DataFrame({"pair_id": ["a"]}), None) is None
