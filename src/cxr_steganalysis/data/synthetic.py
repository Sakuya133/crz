"""Synthetic NIH-like fixtures for pipeline testing; contains no patient data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _synthetic_xray(size: int, seed: int, view: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    body = 105 + 80 * np.exp(-((xx / 0.78) ** 2 + (yy / 1.05) ** 2) * 1.8)
    left_lung = -42 * np.exp(-(((xx + 0.30) / 0.25) ** 2 + ((yy + 0.05) / 0.62) ** 2))
    right_lung = -42 * np.exp(-(((xx - 0.30) / 0.25) ** 2 + ((yy + 0.05) / 0.62) ** 2))
    spine = 22 * np.exp(-((xx / 0.055) ** 2 + ((yy + 0.02) / 0.85) ** 2))
    view_offset = 6.0 if view == "AP" else -3.0
    noise = rng.normal(0, 5.0 if view == "AP" else 4.0, size=(size, size))
    return np.clip(body + left_lung + right_lung + spine + view_offset + noise, 0, 255).astype(
        np.uint8
    )


def create_synthetic_nih_fixture(
    root: str | Path,
    num_patients: int = 15,
    images_per_patient: int = 2,
    image_size: int = 288,
    seed: int = 1337,
) -> dict[str, Path]:
    """Create small NIH-shaped metadata, official lists, and grayscale PNG files."""
    if num_patients < 10:
        raise ValueError("num_patients must be at least 10 for non-empty three-way splits")
    if images_per_patient < 1 or image_size < 16:
        raise ValueError("images_per_patient must be positive and image_size at least 16")
    root = Path(root)
    metadata_dir = root / "metadata"
    raw_dir = root / "raw"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    train_names: list[str] = []
    test_names: list[str] = []
    test_start = int(round(num_patients * 0.8))
    labels = ["No Finding", "Effusion", "Mass|Atelectasis"]
    for patient in range(num_patients):
        patient_id = f"{patient:08d}"
        view = "PA" if patient % 2 == 0 else "AP"
        for image_number in range(images_per_patient):
            image_id = f"{patient_id}_{image_number:03d}.png"
            array = _synthetic_xray(
                image_size, seed + patient * 101 + image_number, view
            )
            Image.fromarray(array, mode="L").save(raw_dir / image_id, format="PNG")
            rows.append({
                "Image Index": image_id,
                "Finding Labels": labels[(patient + image_number) % len(labels)],
                "Follow-up #": image_number,
                "Patient ID": patient_id,
                "Patient Age": f"{30 + patient % 50:03d}Y",
                "Patient Gender": "M" if patient % 2 else "F",
                "View Position": view,
            })
            (test_names if patient >= test_start else train_names).append(image_id)

    metadata_path = metadata_dir / "Data_Entry_2017.csv"
    train_path = metadata_dir / "train_val_list.txt"
    test_path = metadata_dir / "test_list.txt"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    train_path.write_text("\n".join(train_names) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(test_names) + "\n", encoding="utf-8")
    return {
        "root": root,
        "metadata": metadata_path,
        "raw_dir": raw_dir,
        "official_train_list": train_path,
        "official_test_list": test_path,
    }
