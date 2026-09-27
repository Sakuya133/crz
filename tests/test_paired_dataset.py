from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from cxr_steganalysis.data.dataset import PairedPatchDataset
from cxr_steganalysis.data.manifest import build_pair_manifest, generate_stego_records


def test_cover_stego_use_identical_crop_coordinates(tmp_path):
    cover = np.arange(40 * 40, dtype=np.uint16).reshape(40, 40).astype(np.uint8)
    cover_path = tmp_path / "00000001_000.png"
    Image.fromarray(cover, mode="L").save(cover_path)
    split = pd.DataFrame([{
        "image_id": cover_path.name,
        "patient_id": "1",
        "view_position": "PA",
        "finding_labels": "No Finding",
        "split": "train",
        "cover_path": str(cover_path),
    }])
    generation = generate_stego_records(split, tmp_path / "stego", 0.8, 7)
    manifest = build_pair_manifest(split, generation)
    dataset = PairedPatchDataset(manifest, patch_size=16, patches_per_image=2, seed=99)
    sample = dataset[1]
    top = sample["metadata"]["crop_top"]
    left = sample["metadata"]["crop_left"]
    stego = np.asarray(Image.open(manifest.iloc[0]["stego_path"]))
    expected_cover = cover[top:top + 16, left:left + 16].astype(np.float32) / 255.0
    expected_stego = stego[top:top + 16, left:left + 16].astype(np.float32) / 255.0
    assert np.array_equal(sample["cover"].numpy()[0], expected_cover)
    assert np.array_equal(sample["stego"].numpy()[0], expected_stego)
    assert dataset[1]["metadata"] == sample["metadata"]
