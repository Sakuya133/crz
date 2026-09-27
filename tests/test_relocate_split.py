from __future__ import annotations

import pandas as pd
from PIL import Image

from cxr_steganalysis.data.relocate import relocate_split


def test_relocate_split_preserves_identity_columns(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        name = f"{index:08d}_000.png"
        old_path = old_root / name
        Image.new("L", (32, 32), index).save(old_path)
        Image.new("L", (32, 32), index).save(new_root / name)
        rows.append({
            "image_id": name,
            "patient_id": str(index),
            "view_position": "AP" if index % 2 else "PA",
            "split": split,
            "cover_path": str(old_path),
            "image_mode": "L",
        })
    original = pd.DataFrame(rows)
    path = tmp_path / "split.csv"
    original.to_csv(path, index=False)
    relocated = relocate_split(path, new_root)
    identity = ["image_id", "patient_id", "view_position", "split"]
    pd.testing.assert_frame_equal(original[identity], relocated[identity])
    assert all(str(new_root) in value for value in relocated["cover_path"])
