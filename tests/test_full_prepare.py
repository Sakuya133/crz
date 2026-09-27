import importlib.util
import json
from pathlib import Path
import pandas as pd
from cxr_steganalysis.config import load_config
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.data.manifest import validate_pair_manifest


def test_streaming_full_prepare_resume_preserves_pixels_and_binding(tmp_path):
    from test_training_smoke import _manifest
    frame = _manifest(tmp_path)
    split = frame[["image_id","patient_id","view_position","finding_labels","split","cover_path"]].copy()
    split["width"]=32
    split["height"]=32
    preflight=tmp_path/"preflight"
    preflight.mkdir()
    split.to_csv(preflight/"splits.csv",index=False)
    (preflight/"summary.json").write_text(json.dumps(dict(split_identity=sha256_file(preflight/"splits.csv"))))
    spec=importlib.util.spec_from_file_location("prepare_full",Path("scripts/prepare_full.py"))
    module=importlib.util.module_from_spec(spec)
    # scripts use a simple local bootstrap import, just like CLI entry points.
    import sys
    sys.path.insert(0,str(Path("scripts").resolve()))
    spec.loader.exec_module(module)
    c=load_config("configs/full_all_eligible.yaml")
    c["payload_bpp"]=.5
    c["seeds"]["embedding"]=22
    output=tmp_path/"prepared"
    module.prepare(c,preflight,output,max_pairs=2)
    assert not (output/"complete.json").exists()
    before={p.name:sha256_file(p) for p in (output/"stego").glob("*.png")}
    module.prepare(c,preflight,output)
    after=pd.read_csv(output/"cover_stego.csv",dtype={"patient_id":str,"seed":str})
    validate_pair_manifest(after,verify_hashes=True)
    assert after.stego_sha256.tolist()==frame.stego_sha256.tolist()
    for name,digest in before.items():
        assert sha256_file(output/"stego"/name)==digest
    module.prepare(c,preflight,output)
    import pytest
    c["payload_bpp"]=.4
    with pytest.raises(ValueError,match="binding changed"):
        module.prepare(c,preflight,output)
