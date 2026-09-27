"""Paired cover-stego patch dataset with deterministic shared crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from cxr_steganalysis.data.manifest import validate_pair_manifest
from cxr_steganalysis.reproducibility import stable_seed


def _load_l_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Expected grayscale mode L, got {image.mode}: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def deterministic_crop_coordinates(
    height: int,
    width: int,
    patch_size: int,
    seed: int,
    pair_id: str,
    patch_index: int,
) -> tuple[int, int]:
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Image {width}x{height} is smaller than requested patch {patch_size}x{patch_size}; "
            "resize is intentionally disabled"
        )
    crop_seed = stable_seed(seed, pair_id, patch_index, "crop")
    rng = np.random.default_rng(crop_seed)
    top = int(rng.integers(0, height - patch_size + 1))
    left = int(rng.integers(0, width - patch_size + 1))
    return top, left


class PairedPatchDataset(Dataset[dict[str, Any]]):
    """Return paired cover/stego patches cropped at identical coordinates."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str | None = None,
        view_position: str | None = None,
        patch_size: int = 256,
        patches_per_image: int = 1,
        seed: int = 1337,
        normalize: str = "unit_range",
        validate_files: bool = True,
    ) -> None:
        frame = pd.read_csv(manifest, dtype={"patient_id": str, "seed": str}) \
            if isinstance(manifest, (str, Path)) else manifest.copy()
        if split is not None:
            frame = frame[frame["split"] == split]
        if view_position is not None and view_position.upper() != "ALL":
            frame = frame[frame["view_position"] == view_position.upper()]
        frame = frame.reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"No pairs after filtering split={split!r}, view_position={view_position!r}"
            )
        validate_pair_manifest(frame, check_files=validate_files)
        if patch_size <= 0 or patches_per_image <= 0:
            raise ValueError("patch_size and patches_per_image must be positive")
        if normalize not in {"unit_range", "none"}:
            raise ValueError("normalize must be 'unit_range' or 'none'")
        self.frame = frame
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.seed = int(seed)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.frame) * self.patches_per_image

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pair_index, patch_index = divmod(index, self.patches_per_image)
        row = self.frame.iloc[pair_index]
        cover = _load_l_image(row["cover_path"])
        stego = _load_l_image(row["stego_path"])
        if cover.shape != stego.shape:
            raise ValueError(f"Pair {row['pair_id']} has mismatched array shapes")
        top, left = deterministic_crop_coordinates(
            cover.shape[0], cover.shape[1], self.patch_size,
            self.seed, str(row["pair_id"]), patch_index,
        )
        selection = np.s_[top:top + self.patch_size, left:left + self.patch_size]
        cover_patch = cover[selection].astype(np.float32)
        stego_patch = stego[selection].astype(np.float32)
        if self.normalize == "unit_range":
            cover_patch /= 255.0
            stego_patch /= 255.0
        return {
            "cover": torch.from_numpy(cover_patch).unsqueeze(0),
            "stego": torch.from_numpy(stego_patch).unsqueeze(0),
            "metadata": {
                "pair_id": str(row["pair_id"]),
                "image_id": str(row["image_id"]),
                "patient_id": str(row["patient_id"]),
                "view_position": str(row["view_position"]),
                "split": str(row["split"]),
                "patch_index": patch_index,
                "crop_top": top,
                "crop_left": left,
            },
        }


def flatten_paired_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten B paired samples to a balanced 2B binary classification batch."""
    cover = batch["cover"]
    stego = batch["stego"]
    images = torch.cat([cover, stego], dim=0)
    labels = torch.cat([
        torch.zeros(cover.shape[0], dtype=torch.float32),
        torch.ones(stego.shape[0], dtype=torch.float32),
    ])
    return images, labels
