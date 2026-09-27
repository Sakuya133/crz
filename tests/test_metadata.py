from __future__ import annotations

import pandas as pd
import pytest

from cxr_steganalysis.data.metadata import infer_patient_id, inspect_metadata, load_metadata


def test_metadata_fallback_and_inspection(tmp_path):
    path = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame({
        "Image Index": ["00000001_000.png", "00000001_001.png", "odd-name.png"],
        "Finding Labels": ["No Finding", "Mass|Effusion", "No Finding"],
        "View Position": ["PA", "AP", "PA"],
    }).to_csv(path, index=False)

    frame = load_metadata(path)
    assert frame["patient_id"].iloc[0] == "00000001"
    assert frame["patient_id"].iloc[0] == frame["patient_id"].iloc[1]
    assert frame["patient_id"].iloc[2].startswith("inferred-")
    assert infer_patient_id("odd-name.png") == frame["patient_id"].iloc[2]

    report = inspect_metadata(frame)
    assert report["num_images"] == 3
    assert report["num_unique_patients"] == 2
    assert report["num_ap"] == 1
    assert report["num_pa"] == 2
    assert report["num_no_finding"] == 2
    assert report["pathology_distribution"]["Effusion"] == 1


def test_metadata_requires_image_index(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Patient ID": ["1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Image Index"):
        load_metadata(path)


def test_missing_value_report_precedes_fallback(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({
        "Image Index": ["00000001_000.png"],
        "Patient ID": [""],
        "View Position": [""],
    }).to_csv(path, index=False)
    frame = load_metadata(path)
    report = inspect_metadata(frame)
    assert frame.iloc[0]["patient_id"] == "00000001"
    assert frame.iloc[0]["view_position"] == "UNKNOWN"
    assert report["missing_values"]["patient_id"] == 1
    assert report["missing_values"]["view_position"] == 1


def test_explicit_and_inferred_numeric_ids_cannot_split_same_patient(tmp_path):
    path=tmp_path/"partial.csv"
    pd.DataFrame({"Image Index":["00000001_000.png","00000001_001.png"],"Patient ID":["1",""],"View Position":["AP","PA"]}).to_csv(path,index=False)
    frame=load_metadata(path)
    assert frame.patient_id.tolist()==["1","1"]
