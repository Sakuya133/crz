import json
from cxr_steganalysis.config import load_config
from cxr_steganalysis.experiment import save_run_provenance


def test_resume_retains_first_snapshot_and_records_current_source(tmp_path):
    config = load_config("configs/pilot_existing.yaml")
    source = tmp_path / "src" / "fixture.py"
    source.parent.mkdir()
    source.write_text("VERSION = 1\n")
    config["_project_root"] = str(tmp_path)
    output = tmp_path / "run"
    save_run_provenance(output, config, {"fixture": True})
    original = (output / "provenance.json").read_bytes()
    original_archive = (output / "source_snapshot.tar.gz").read_bytes()
    source.write_text("VERSION = 2\n")
    config["epochs"] += 1
    save_run_provenance(output, config, {"fixture": True})
    assert (output / "provenance.json").read_bytes() == original
    assert (output / "source_snapshot.tar.gz").read_bytes() == original_archive
    resume = list(output.glob("resume_provenance/*/provenance.json"))
    assert len(resume) == 1
    assert json.loads(resume[0].read_text())["source_identity"] != json.loads(original)["source_identity"]
