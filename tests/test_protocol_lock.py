from copy import deepcopy
import json
import pandas as pd
import pytest
from cxr_steganalysis.config import load_config
from cxr_steganalysis.experiment import select_size_matched
from cxr_steganalysis.protocol_lock import freeze_plan, validate_test_access, settings_identity, verify_plan_membership


def _frame():
    return pd.DataFrame(dict(pair_id=["a", "b"], patient_id=["1", "2"], split=["test", "train"], view_position=["AP", "PA"],
                             test_cohort=["confirmatory_unseen_patient", "not_test"], cover_path=["old", "old2"]))


def _run(config, view):
    run = deepcopy(config)
    run["training"]["train_view"] = view
    run["paths"]["output_dir"] = f"outputs/{view}"
    return run


def test_confirmatory_access_requires_explicit_matching_whole_plan_lock(tmp_path):
    config = load_config("configs/full_all_eligible.yaml")
    frame = _frame()
    manifest = tmp_path / "cover_stego.csv"
    frame.to_csv(manifest, index=False)
    config["paths"]["pair_manifest"] = str(manifest)
    with pytest.raises(ValueError, match="explicit --test-cohort"):
        validate_test_access(config, frame, None)
    with pytest.raises(ValueError, match="requires --protocol-lock"):
        validate_test_access(config, frame, "confirmatory_unseen_patient")
    lock = tmp_path / "lock.json"
    runs = [(view, _run(config, view)) for view in ("AP", "PA")]
    entry = dict(role="primary", config=config, manifest_path=manifest, runs=runs)
    with pytest.raises(FileNotFoundError, match="Whole-plan lock missing"):
        verify_plan_membership(lock, [r for _, r in runs])
    document = freeze_plan([entry], lock)
    assert {c["name"] for c in document["comparisons"] if c["primary"]} == {f"mismatch_{m}_{t}" for m in ("highpass", "srnet") for t in ("AP", "PA")}
    ap = runs[0][1]
    assert len(validate_test_access(ap, frame, "confirmatory_unseen_patient", lock)) == 64
    assert verify_plan_membership(lock, [r for _, r in runs])
    with pytest.raises(FileExistsError):
        freeze_plan([entry], lock)
    changed = deepcopy(ap)
    changed["learning_rate"] *= 10
    with pytest.raises(ValueError, match="Scientific settings"):
        validate_test_access(changed, frame, "confirmatory_unseen_patient", lock)
    undeclared_seed = deepcopy(ap)
    undeclared_seed["seeds"]["training"] = 7
    with pytest.raises(ValueError, match="not part of the frozen plan"):
        verify_plan_membership(lock, [undeclared_seed])
    with pytest.raises(ValueError, match="enrollment"):
        validate_test_access(ap, frame.assign(patient_id="2"), "confirmatory_unseen_patient", lock)
    relocated = deepcopy(ap)
    relocated["paths"]["raw_dir"] = "/kaggle/input/images"
    relocated["training"]["device"] = "cpu"
    assert settings_identity(ap) == settings_identity(relocated)
    assert validate_test_access(relocated, frame.assign(cover_path="new"), "confirmatory_unseen_patient", lock)
    assert validate_test_access(ap, frame, "pilot_exposed") is None


def test_plan_lock_rejects_ap_pa_budget_difference(tmp_path):
    config = load_config("configs/full_all_eligible.yaml")
    manifest = tmp_path / "cover_stego.csv"
    _frame().to_csv(manifest, index=False)
    ap, pa = _run(config, "AP"), _run(config, "PA")
    pa["epochs"] += 1
    with pytest.raises(ValueError, match="differ beyond source view"):
        freeze_plan([dict(role="primary", config=config, manifest_path=manifest, runs=[("AP", ap), ("PA", pa)])], tmp_path / "lock.json")


def test_size_matched_subset_is_deterministic_and_keeps_test():
    rows = []
    for split, n_ap, n_pa in (("train", 3, 7), ("validation", 4, 2), ("test", 5, 9)):
        for view, n in (("AP", n_ap), ("PA", n_pa)):
            rows += [dict(pair_id=f"{split}{view}{i}", patient_id=f"{view}{i % 2}{split}", split=split, view_position=view) for i in range(n)]
    frame = pd.DataFrame(rows)
    control = select_size_matched(frame, 20261002)
    counts = control.groupby(["split", "view_position"]).size().to_dict()
    assert counts == {("train", "AP"): 3, ("train", "PA"): 3, ("validation", "AP"): 2, ("validation", "PA"): 2, ("test", "AP"): 5, ("test", "PA"): 9}
    assert set(control.pair_id) <= set(frame.pair_id)
    # Selection depends on the sampling seed only, never on input row order.
    assert set(select_size_matched(frame.sample(frac=1, random_state=3), 20261002).pair_id) == set(control.pair_id)
    assert set(select_size_matched(frame, 1).pair_id) != set(control.pair_id)


def test_pilot_remains_backward_compatible():
    config = load_config("configs/pilot_existing.yaml")
    assert validate_test_access(config, pd.DataFrame({"pair_id": ["a"]}), None) is None
