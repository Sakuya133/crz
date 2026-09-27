"""Classifier split invariants with small synthetic feature matrices (not SRM results)."""
import json
import numpy as np
import pandas as pd
import pytest
from cxr_steganalysis.models import srm_svm


def test_scaler_never_fits_validation_or_test(tmp_path,monkeypatch):
    pytest.importorskip("sklearn")
    # Keep the production 34,671-column shape; train/validation distributions
    # deliberately differ so a train+validation fit would fail this assertion.
    rng=np.random.default_rng(4)
    features=rng.normal(size=(16,34671)).astype(np.float32)
    features[8:12] += 10
    features[12:] += 100
    rows=pd.DataFrame(dict(feature_row=range(16),split=["train"]*8+["validation"]*4+["test"]*4,
                           view_position=["AP"]*16,label=[0,1]*8))
    manifest=tmp_path/"manifest.csv";manifest.write_text("synthetic fixture,not real SRM cache\n")
    monkeypatch.setattr(srm_svm,"_load_feature_cache",lambda *_:(rows,features,dict(binding={},features_sha256="synthetic")))
    path=srm_svm.train_srm_svm(manifest,tmp_path/"cache",tmp_path/"model",source_view="AP",seed=1337,c_grid=[.01])
    parameters=np.load(path)
    np.testing.assert_allclose(parameters["mean"],features[:8].astype(np.float64).mean(axis=0),rtol=0,atol=1e-12)
    before=parameters["coef"].copy()
    # Test labels/features must have no effect on fitting or validation threshold.
    features[12:] *= -13
    rows.loc[rows.split=="test","label"] = 1-rows.loc[rows.split=="test","label"]
    other=srm_svm.train_srm_svm(manifest,tmp_path/"cache",tmp_path/"other",source_view="AP",seed=1337,c_grid=[.01])
    np.testing.assert_array_equal(np.load(other)["coef"],before)
    assert np.load(path)["threshold"]==np.load(other)["threshold"]
    assert "not probability" in json.loads(path.with_suffix(".json").read_text())["score_type"].replace("_"," ")


def test_queue_lock_prevents_second_runner(tmp_path,monkeypatch):
    import fcntl
    from cxr_steganalysis import queue
    from cxr_steganalysis.config import load_config
    monkeypatch.setattr(queue,"LOCK_DIRECTORY",tmp_path/"locks")
    queue.LOCK_DIRECTORY.mkdir()
    with (queue.LOCK_DIRECTORY/".one_queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.execute(dict(dataset_profile="synthetic_smoke",stages=[]),{tmp_path/"c.yaml":load_config("configs/smoke.yaml")},tmp_path/"run",validate_inputs=False)
