"""Portable remapping of a fixed split to a new cover-image root."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cxr_steganalysis.data.metadata import discover_local_pngs
from cxr_steganalysis.data.split import assert_patient_disjoint


def relocate_split(input_path: str | Path, raw_dir: str | Path) -> pd.DataFrame:
    """Change paths only; never change selected images, patients, views, or splits."""
    frame = pd.read_csv(input_path, dtype={"patient_id": str})
    required = {"image_id", "patient_id", "view_position", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Existing split is missing columns: {missing}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Existing split contains duplicate image_id values")
    assert_patient_disjoint(frame)
    local = discover_local_pngs(raw_dir).set_index("image_id")
    if not local.index.is_unique:
        raise ValueError("New raw root contains duplicate image IDs")
    missing_files = sorted(set(frame["image_id"]) - set(local.index))
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} selected images are absent from the new raw root: "
            f"{missing_files[:10]}"
        )
    relocated = frame.copy()
    selected = local.loc[relocated["image_id"]].reset_index()
    if "image_mode" in relocated.columns:
        recorded = relocated["image_mode"].astype(str).reset_index(drop=True)
        actual = selected["image_mode"].astype(str).reset_index(drop=True)
        if not recorded.equals(actual):
            raise ValueError("Image mode at the new root differs from the recorded split")
    relocated["cover_path"] = selected["cover_path"].to_numpy()
    relocated["image_mode"] = selected["image_mode"].to_numpy()
    relocated["image_width"] = selected["image_width"].to_numpy()
    relocated["image_height"] = selected["image_height"].to_numpy()
    assert_patient_disjoint(relocated)
    return relocated
