#!/usr/bin/env python3
"""Tiny CPU-only standalone workflow, clearly SYNTHETIC (never reads NIH)."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from _bootstrap import bootstrap
bootstrap()
import numpy as np
import pandas as pd
from PIL import Image
from cxr_steganalysis.config import load_config
from cxr_steganalysis.lab import atomic_json, write_config, require_empty
from cxr_steganalysis.data.manifest import generate_stego_records, build_pair_manifest
from cxr_steganalysis.data.portable import export_pilot, import_pilot, check_data
from cxr_steganalysis.queue import make_plan, execute


def fixture(root):
    root = Path(root).resolve()
    require_empty(root)
    raw = root / "raw"; raw.mkdir()
    original = root / "original"; original.mkdir()
    rows, metadata, assignment = [], [], []
    rng = np.random.default_rng(15)
    train, test = [], []
    for patient in range(12):
        split = "train" if patient < 8 else "validation" if patient < 10 else "test"
        assignment.append(dict(patient_id=str(patient),split=split))
        # Each patient has BOTH views, exercising the global leakage invariant.
        for index,view in enumerate(("AP","PA")):
            image = f"{patient:08d}_{index:03d}.png"
            path = raw / image
            Image.fromarray(rng.integers(0,256,(40,40),dtype=np.uint8)).save(path)
            rows.append(dict(image_id=image,patient_id=str(patient),view_position=view,finding_labels="synthetic",split=split,cover_path=str(path)))
            metadata.append({"Image Index":image,"Patient ID":str(patient),"View Position":view,"Finding Labels":"synthetic"})
            (test if split == "test" else train).append(image)
    split = pd.DataFrame(rows)
    generated = generate_stego_records(split,root / "original_stego",.2,1337)
    pairs = build_pair_manifest(split,generated)
    pairs.to_csv(original / "cover_stego.csv",index=False)
    pd.DataFrame(assignment).to_csv(original / "patient_assignments.csv",index=False)
    pd.DataFrame(metadata).to_csv(root / "metadata.csv",index=False)
    (root / "train.txt").write_text("\n".join(train)+"\n")
    (root / "test.txt").write_text("\n".join(test)+"\n")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["paths"].update(metadata=str(root/"metadata.csv"),official_train_list=str(root/"train.txt"),official_test_list=str(root/"test.txt"),raw_dir=str(raw),stego_dir=str(root/"relocated_stego"))
    for key,name in (("pair_manifest","cover_stego.csv"),("split_manifest","splits.csv"),("generation_manifest","stego_generation.csv"),("patient_assignments","patient_assignments.csv")):
        config["paths"][key] = str(root/"imported"/name)
    config["lab"]["c_grid"] = [.01]
    export_pilot(original,root/"bundle")
    imported = import_pilot(root/"bundle",config,generate=True)
    checked = check_data(config,verify_hashes=True)
    assert imported["semantic_identity_before"] == imported["semantic_identity_after"]
    write_config(root/"config.yaml",config)
    return config,checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--skip-srm",action="store_true",help="Core CNN smoke only, explicitly not full comparator verification")
    a = p.parse_args()
    root = a.output.resolve(); require_empty(root)
    started = time.monotonic()
    config,checked = fixture(root/"fixture")
    models = ["highpass","srnet"] + ([] if a.skip_srm else ["srm_svm"])
    plan,configs = make_plan(config,root/"A",models,[1337])
    code = execute(plan,configs,root/"A",budget_hours=.25)
    if code: raise RuntimeError(f"Synthetic A queue failed with exit {code}; see logs")
    # Resume a completed queue must verify artifacts and do no training work.
    code = execute(plan,configs,root/"A",resume=True,budget_hours=.25)
    if code: raise RuntimeError("Completed queue resume failed")
    mixed = deepcopy(config)
    mixed["protocol_id"] = "SYNTHETIC-ONLY-lab-smoke-B"
    mixed["training"].update(train_view="ALL",mixed_pairs_per_view=4,selection_metric="macro_view_auc",groupdro_step_size=.01)
    plan_b,configs_b = make_plan(mixed,root/"B",["highpass"],[1337])
    code = execute(plan_b,configs_b,root/"B",budget_hours=.25)
    if code: raise RuntimeError(f"Synthetic B queue failed with exit {code}; see logs")
    receipt = dict(status="verified_on_synthetic_only",dataset="24 synthetic pairs / 12 synthetic patients, both views per patient",device="cpu",
                   models=models,protocols=["A","B: mixed ERM/GroupDRO"],bootstrap_replicates=50,
                   verified=["private manifest export/import", "full-image regeneration byte identity", "official/global patient split", "CLI train/evaluate", "completed queue resume", "source-validation threshold", "paired bootstrap", "automatic reports"],
                   seconds=time.monotonic()-started,data=checked,NIH_training="not_run")
    atomic_json(root/"verification.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
