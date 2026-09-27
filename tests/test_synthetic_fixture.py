from __future__ import annotations

from PIL import Image

from cxr_steganalysis.data.metadata import inspect_metadata, load_metadata, read_image_list
from cxr_steganalysis.data.synthetic import create_synthetic_nih_fixture


def test_synthetic_fixture_has_nih_shape_and_no_overlap(tmp_path):
    paths = create_synthetic_nih_fixture(
        tmp_path, num_patients=10, images_per_patient=2, image_size=24
    )
    frame = load_metadata(paths["metadata"])
    report = inspect_metadata(
        frame,
        raw_dir=paths["raw_dir"],
        official_train=read_image_list(paths["official_train_list"]),
        official_test=read_image_list(paths["official_test_list"]),
    )
    assert report["num_images"] == 20
    assert report["num_unique_patients"] == 10
    assert report["official_lists"]["patient_overlap_count"] == 0
    assert report["num_missing_image_files"] == 0
    assert frame.groupby("patient_id").size().max() == 2
    assert set(frame["view_position"]) == {"AP", "PA"}
    assert Image.open(paths["raw_dir"] / frame.iloc[0]["image_id"]).mode == "L"
