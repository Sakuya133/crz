from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd
import pytest
from cxr_steganalysis.config import load_config
from cxr_steganalysis.data.portable import import_pilot, check_data, png_index
from cxr_steganalysis.lab import load_lab_config
from cxr_steganalysis.queue import make_plan, execute
from cxr_steganalysis.provenance import sha256_file


@pytest.fixture(autouse=True)
def isolated_test_queue_lock(tmp_path, monkeypatch):
    # State-machine unit tests must not contend with a real lab/smoke queue.
    import cxr_steganalysis.queue as queue
    monkeypatch.setattr(queue, "LOCK_DIRECTORY", tmp_path / "locks")


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts"
    if str(path) not in sys.path: sys.path.insert(0,str(path))
    spec = importlib.util.spec_from_file_location("lab_test_"+name,path/(name+".py"))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def portable_fixture(tmp_path):
    root = tmp_path / "fixture"
    config,checked = script("smoke").fixture(root)
    return root,config,checked


def test_import_relocation_byte_identity_and_no_overwrite(portable_fixture):
    root,config,checked = portable_fixture
    assert checked["pairs"] == 24 and checked["patients"] == 12
    receipt = json.loads((root/"imported/relocation.json").read_text())
    assert receipt["semantic_identity_before"] == receipt["semantic_identity_after"]
    assert receipt["source_manifest_sha256"] != receipt["relocated_manifest_sha256"]
    with pytest.raises(FileExistsError): import_pilot(root/"bundle",config,generate=True)
    before = sha256_file(root/"imported/cover_stego.csv")
    import_pilot(root/"bundle",config,generate=True,resume=True)
    assert sha256_file(root/"imported/cover_stego.csv") == before


def test_corruption_and_duplicate_basename_rejected(portable_fixture):
    root,config,_ = portable_fixture
    image = next((root/"relocated_stego").glob("*.png"))
    image.write_bytes(image.read_bytes()+b"modified")
    with pytest.raises(ValueError,match="SHA-256 mismatch"): check_data(config,True)
    raw = next((root/"raw").glob("*.png"))
    (root/"raw/duplicate").mkdir()
    (root/"raw/duplicate"/raw.name).write_bytes(raw.read_bytes())
    with pytest.raises(ValueError,match="Duplicate basename"): png_index(root/"raw")


def test_official_global_leakage_and_training_seed_independence(portable_fixture):
    root,config,_ = portable_fixture
    changed = deepcopy(config);changed["seed"] = 42;changed["seeds"]["training"] = 42
    a,b = check_data(config),check_data(changed)
    assert a["manifest_identity"] == b["manifest_identity"]
    assert a["seeds"]["embedding"] == b["seeds"]["embedding"] == 1337
    assignment = pd.read_csv(root/"imported/patient_assignments.csv",dtype={"patient_id":str})
    assignment.loc[assignment.split == "test","split"] = "train"
    assignment.to_csv(root/"imported/patient_assignments.csv",index=False)
    with pytest.raises(ValueError,match="locked patient assignment"): check_data(config)


def test_metadata_wrong_view_and_missing_official_list_fail(portable_fixture):
    root,config,_ = portable_fixture
    metadata = pd.read_csv(root/"metadata.csv")
    metadata.loc[0,"View Position"] = "PA"
    metadata.to_csv(root/"metadata.csv",index=False)
    with pytest.raises(ValueError,match="view mismatch"): check_data(config)
    metadata.loc[0,"View Position"] = "AP";metadata.to_csv(root/"metadata.csv",index=False)
    config["paths"]["official_test_list"] = str(root/"missing.txt")
    with pytest.raises(FileNotFoundError): check_data(config)


def test_pilot_hash_guard_refuses_new_split(portable_fixture):
    root,config,_ = portable_fixture
    config["dataset_profile"] = "pilot_existing"
    with pytest.raises(ValueError,match="pinned audited pilot"): import_pilot(root/"bundle",config,resume=True)


def test_path_only_local_overlay_and_dryrun_no_writes(tmp_path):
    local = tmp_path/"local.yaml";local.write_text("paths:\n  raw_dir: /readonly/lab/images\n")
    config = load_lab_config("configs/pilot_existing.yaml",local)
    plan,configs = make_plan(config,tmp_path/"queue",["highpass","srnet","srm_svm"],[1337,2026,42])
    assert not (tmp_path/"queue").exists()
    assert sum(s["kind"] == "cnn" for s in plan["stages"]) == 12
    assert sum(s["kind"] == "svm" for s in plan["stages"]) == 2
    assert all(c["seeds"]["embedding"] == c["seeds"]["crop"] == c["seeds"]["split"] == 1337 for c in configs.values())
    local.write_text("seed: 42\n")
    with pytest.raises(ValueError,match="paths-only"): load_lab_config("configs/pilot_existing.yaml",local)


def minimal_queue(tmp_path,code="from pathlib import Path; Path('MARKER').write_text('done')"):
    root = tmp_path/"queue"; marker=root/"done.json"
    command=[sys.executable,"-c",code.replace("MARKER",str(marker))]
    config=load_config("configs/smoke.yaml")
    plan=dict(dataset_profile="synthetic_smoke",stages=[dict(id="probe",kind="probe",markers=[str(marker)],command=command)])
    return root,marker,plan,{root/"config.yaml":config}


def test_queue_resume_overwrite_and_artifact_tampering(tmp_path):
    root,marker,plan,configs=minimal_queue(tmp_path)
    assert execute(plan,configs,root,validate_inputs=False)==0
    before=sha256_file(marker)
    with pytest.raises(FileExistsError):execute(plan,configs,root,validate_inputs=False)
    assert execute(plan,configs,root,resume=True,validate_inputs=False)==0
    assert sha256_file(marker)==before
    marker.write_text("changed")
    with pytest.raises(ValueError,match="modified"):execute(plan,configs,root,resume=True,validate_inputs=False)


def test_queue_failure_exit_status_and_budget_not_completed(tmp_path):
    root,_,plan,configs=minimal_queue(tmp_path,"import sys; sys.exit(7)")
    assert execute(plan,configs,root,validate_inputs=False)==1
    state=json.loads((root/"queue.json").read_text())
    assert state["status"]=="failed" and state["stages"][0]["returncode"]==7
    root,_,plan,configs=minimal_queue(tmp_path/"next")
    assert execute(plan,configs,root,budget_hours=1e-12,validate_inputs=False)==124
    assert json.loads((root/"queue.json").read_text())["stages"][0]["status"]=="pending"
    assert execute(plan,configs,root,resume=True,validate_inputs=False)==0


def test_queue_restarts_cnn_killed_before_first_checkpoint(tmp_path):
    root,marker,plan,configs=minimal_queue(tmp_path)
    run,crashed=root/"run",tmp_path/"crashed"
    plan["stages"][0].update(kind="cnn",run_dir=str(run))
    # First attempt dies during epoch 0 after writing provenance only.
    code=("import pathlib,sys; r,c,m=map(pathlib.Path,sys.argv[1:])\n"
          "if not c.exists(): c.touch(); r.mkdir(); (r/'provenance.json').write_text('{}'); sys.exit(1)\n"
          "assert not r.exists(); m.write_text('{}')")
    plan["stages"][0]["command"]=[sys.executable,"-c",code,str(run),str(crashed),str(marker)]
    assert execute(plan,configs,root,validate_inputs=False)==1
    assert execute(plan,configs,root,resume=True,validate_inputs=False)==0
    assert (Path(json.loads((root/"queue.json").read_text())["stages"][0]["preserved_incomplete_fit"])/"provenance.json").is_file()


def test_mitigation_plan_has_macro_selection_and_equal_data(tmp_path):
    config=load_config("configs/pilot_mixed.yaml")
    plan,configs=make_plan(config,tmp_path/"B",seeds=[1337])
    runconfigs=[configs[Path(r["config"])] for r in plan["runs"]]
    assert {c["training"]["objective"] for c in runconfigs}=={"erm","groupdro"}
    assert all(c["training"]["selection_metric"]=="macro_view_auc" for c in runconfigs)
    for c in runconfigs:c["training"].pop("objective");c["paths"].pop("output_dir")
    assert runconfigs[0]==runconfigs[1]


def test_full_preflight_missing_data_cannot_prepare(tmp_path):
    folder=tmp_path/"preflight";folder.mkdir()
    (folder/"summary.json").write_text(json.dumps(dict(missing_images=10,exclusions={})))
    with pytest.raises(ValueError,match="incomplete"):script("full_data").ready(folder)


def test_doctor_accepts_existing_runtime_without_project_venv(monkeypatch, capsys):
    from types import SimpleNamespace
    doctor = script("doctor")
    monkeypatch.setattr(sys, "argv", ["doctor.py"])
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setenv("PYTHONPATH", "/lab/existing/site-packages")
    monkeypatch.setattr(doctor.importlib.metadata, "version", lambda name: "available")
    monkeypatch.setattr(doctor.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="No broken requirements found.", stderr=""))
    assert doctor.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready" and report["errors"] == []
    assert report["isolated_venv"] is False
    assert report["warnings"]  # Record external paths without requiring a new env.


def synthetic_full(portable_fixture):
    from PIL import Image
    import numpy as np
    from cxr_steganalysis.data.full_dataset import full_preflight
    from cxr_steganalysis.config import resolve_config_path
    root,config,_=portable_fixture
    metadata=pd.read_csv(root/"metadata.csv",dtype=str)
    added=[]
    with (root/"test.txt").open("a") as official:
        for patient in (12,13):
            for index,view in enumerate(("AP","PA")):
                name=f"{patient:08d}_{index:03d}.png"
                Image.fromarray(np.full((40,40),120,dtype=np.uint8)).save(root/"raw"/name)
                official.write(name+"\n")
                added.append({"Image Index":name,"Patient ID":str(patient),"View Position":view,"Finding Labels":"synthetic"})
    pd.concat([metadata,pd.DataFrame(added)],ignore_index=True).to_csv(root/"metadata.csv",index=False)
    config.update(dataset_profile="full_all_eligible",protocol_id="SYNTHETIC-ONLY-full-enrollment",full_expected_metadata_images=28)
    inventory=root/"full"
    config["paths"].update(pilot_pair_manifest=str(root/"imported/cover_stego.csv"),pilot_patient_assignments=str(root/"imported/patient_assignments.csv"),
                           split_manifest=str(inventory/"splits.csv"),patient_assignments=str(inventory/"patient_assignments.csv"),
                           pair_manifest=str(inventory/"bpp02/cover_stego.csv"),stego_dir=str(inventory/"bpp02/stego"))
    result=full_preflight(config,inventory)
    assert result["missing_images"]==0 and result["patients_locked"]==12
    script("full_data").ready(inventory)
    script("prepare_full").prepare(config,inventory,inventory/"bpp02")
    return root,config,inventory


def test_synthetic_full_extension_preserves_exposed_patients(portable_fixture):
    from cxr_steganalysis.config import resolve_config_path
    root,config,inventory=synthetic_full(portable_fixture)
    assert check_data(config,True)["pairs"]==28
    frame=pd.read_csv(resolve_config_path(config,"pair_manifest"),dtype={"patient_id":str})
    assert set(frame.loc[frame.test_cohort=="pilot_exposed","patient_id"])=={"10","11"}
    assert set(frame.loc[frame.test_cohort=="confirmatory_unseen_patient","patient_id"])=={"12","13"}
    assert set(frame.loc[frame.patient_id=="8","split"])=={"validation"}
    # A hidden "full" subsample is rejected even if ordinary disjoint checks pass.
    frame.iloc[1:].to_csv(resolve_config_path(config,"pair_manifest"),index=False)
    with pytest.raises(ValueError,match="every eligible image"):check_data(config)


def test_size_control_reuses_files_decodes_once_and_rejects_tampering(portable_fixture):
    from cxr_steganalysis.config import resolve_config_path
    from cxr_steganalysis.data.full_dataset import build_size_control
    root,config,inventory=synthetic_full(portable_fixture)
    full=pd.read_csv(resolve_config_path(config,"pair_manifest"),dtype={"patient_id":str})
    # Make PA train larger than AP train (as in NIH) by dropping AP train images from the full manifest copy.
    source=inventory/"bpp02_unbalanced/cover_stego.csv"; source.parent.mkdir()
    drop=full[(full.split=="train")&(full.view_position=="AP")].pair_id.head(3)
    full[~full.pair_id.isin(drop)].to_csv(source,index=False)
    (source.parent/"complete.json").write_text(json.dumps(dict(manifest_sha256=sha256_file(source))))
    control=deepcopy(config)
    control.update(dataset_profile="full_size_matched",protocol_id="SYNTHETIC-ONLY-size-matched")
    control["seeds"]["sampling"]=20261002
    control["paths"].update(size_control_source_manifest=str(source),pair_manifest=str(inventory/"size_matched/cover_stego.csv"))
    receipt=build_size_control(control)
    assert receipt["pairs_per_view"]=={"train":5,"validation":2} and receipt["sampling_seed"]==20261002
    frame=pd.read_csv(inventory/"size_matched/cover_stego.csv",dtype={"patient_id":str})
    assert set(frame.stego_path)<=set(full.stego_path)  # same files, no copies
    assert (frame.split=="test").sum()==(full.split=="test").sum()
    assert build_size_control(control)==receipt  # idempotent, never redrawn
    # The full-enrollment check applies to the source; this unbalanced copy is not complete.
    with pytest.raises(ValueError,match="every eligible image"):check_data(control)
    control["paths"]["size_control_source_manifest"]=str(resolve_config_path(config,"pair_manifest"))
    other=inventory/"size_matched_full"; control["paths"]["pair_manifest"]=str(other/"cover_stego.csv")
    build_size_control(control)
    first=check_data(control)
    assert first["files_fully_decoded"]==2*first["pairs"]
    assert check_data(control)["files_fully_decoded"]==0  # header-only re-check
    tampered=pd.read_csv(other/"cover_stego.csv",dtype={"patient_id":str})
    tampered.iloc[1:].to_csv(other/"cover_stego.csv",index=False)
    with pytest.raises(ValueError,match="deterministic subset"):check_data(control)
