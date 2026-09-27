from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cxr_steganalysis.data.manifest import (
    build_pair_manifest,
    generate_stego_records,
    validate_pair_manifest,
)


def test_build_and_validate_manifest(tmp_path):
    rows = []
    for index, split in enumerate(["train", "validation", "test"]):
        path = tmp_path / f"{index:08d}_000.png"
        Image.fromarray(np.full((20, 20), 40 + index, dtype=np.uint8), mode="L").save(path)
        rows.append({
            "image_id": path.name,
            "patient_id": str(index),
            "view_position": "PA" if index % 2 == 0 else "AP",
            "finding_labels": "No Finding",
            "split": split,
            "cover_path": str(path),
        })
    split_frame = pd.DataFrame(rows)
    generation = generate_stego_records(split_frame, tmp_path / "stego", 0.2, 1337)
    manifest = build_pair_manifest(split_frame, generation)
    validate_pair_manifest(manifest)
    assert list(manifest.columns) == [
        "image_id", "patient_id", "view_position", "finding_labels", "split",
        "pair_id", "cover_path", "stego_path", "algorithm", "payload_bpp",
        "seed", "cover_sha256", "stego_sha256", "embedded_bits", "changed_pixels",
    ]
    assert manifest["pair_id"].is_unique
    assert manifest["cover_path"].is_unique
    assert manifest["stego_path"].is_unique

    validate_pair_manifest(manifest, verify_hashes=True)
    stego_path = manifest.loc[0, "stego_path"]
    stego = np.asarray(Image.open(stego_path), dtype=np.uint8).copy()
    stego[0, 0] ^= 1
    Image.fromarray(stego, mode="L").save(stego_path)
    with pytest.raises(ValueError, match="Stego SHA-256 mismatch"):
        validate_pair_manifest(manifest, verify_hashes=True)
