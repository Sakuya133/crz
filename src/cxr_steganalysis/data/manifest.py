"""Cover-stego generation records and pair manifest validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from cxr_steganalysis.data.split import assert_patient_disjoint
from cxr_steganalysis.stego.lsb_matching import embed_lsb_matching_file, sha256_file


PAIR_COLUMNS = [
    "image_id",
    "patient_id",
    "view_position",
    "finding_labels",
    "split",
    "pair_id",
    "cover_path",
    "stego_path",
    "algorithm",
    "payload_bpp",
    "seed",
    "cover_sha256",
    "stego_sha256",
    "embedded_bits",
    "changed_pixels",
]


def make_pair_id(image_id: str, algorithm: str, payload_bpp: float) -> str:
    value = f"{image_id}\x1f{algorithm}\x1f{float(payload_bpp):.12g}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def generate_stego_records(
    split_frame: pd.DataFrame,
    output_dir: str | Path,
    payload_bpp: float,
    global_seed: int,
    allow_conversion: bool = False,
) -> pd.DataFrame:
    required = {"image_id", "cover_path"}
    missing = sorted(required - set(split_frame.columns))
    if missing:
        raise ValueError(f"Split manifest missing required column(s): {', '.join(missing)}")
    output_dir = Path(output_dir)
    records: list[dict[str, Any]] = []
    for row in split_frame.itertuples(index=False):
        image_id = str(row.image_id)
        cover_path = Path(row.cover_path)
        if not cover_path.is_file():
            raise FileNotFoundError(f"Cover image not found for {image_id}: {cover_path}")
        # image_id can contain subdirectories in custom datasets; retain only a safe filename.
        stego_path = output_dir / Path(image_id).name
        result = embed_lsb_matching_file(
            cover_path,
            payload_bpp=payload_bpp,
            global_seed=global_seed,
            image_id=image_id,
            output_path=stego_path,
            allow_conversion=allow_conversion,
        )
        record = {
            "image_id": image_id,
            "stego_path": str(stego_path.resolve()),
            **result.metadata.to_dict(),
        }
        records.append(record)
    return pd.DataFrame(records)


def build_pair_manifest(
    split_frame: pd.DataFrame,
    generation_frame: pd.DataFrame,
) -> pd.DataFrame:
    required_split = {
        "image_id", "patient_id", "view_position", "finding_labels", "split", "cover_path"
    }
    required_generation = {
        "image_id", "stego_path", "algorithm", "payload_bpp", "seed",
        "cover_sha256", "stego_sha256", "embedded_bits", "changed_pixels",
    }
    missing_split = sorted(required_split - set(split_frame.columns))
    missing_generation = sorted(required_generation - set(generation_frame.columns))
    if missing_split:
        raise ValueError(f"Split manifest missing column(s): {', '.join(missing_split)}")
    if missing_generation:
        raise ValueError(
            f"Generation manifest missing column(s): {', '.join(missing_generation)}"
        )
    if generation_frame["image_id"].duplicated().any():
        raise ValueError("Generation manifest contains duplicate image_id values")
    merged = split_frame.merge(
        generation_frame[list(required_generation)],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if merged["stego_path"].isna().any():
        missing_ids = merged.loc[merged["stego_path"].isna(), "image_id"].tolist()
        raise ValueError(f"No stego generation record for: {missing_ids[:10]}")
    merged["pair_id"] = merged.apply(
        lambda row: make_pair_id(row["image_id"], row["algorithm"], row["payload_bpp"]),
        axis=1,
    )
    manifest = merged[PAIR_COLUMNS].copy()
    validate_pair_manifest(manifest, check_files=True)
    return manifest


def validate_pair_manifest(
    frame: pd.DataFrame,
    check_files: bool = True,
    *,
    verify_hashes: bool = False,
) -> None:
    missing = sorted(set(PAIR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Pair manifest missing required column(s): {', '.join(missing)}")
    if frame["pair_id"].duplicated().any():
        raise ValueError("Pair manifest contains duplicate pair_id values")
    unknown_splits = sorted(set(frame["split"].astype(str)) - {"train", "validation", "test"})
    if unknown_splits:
        raise ValueError(f"Pair manifest contains unknown split name(s): {unknown_splits}")
    cover_paths = frame["cover_path"].map(lambda value: str(Path(value).expanduser().resolve()))
    stego_paths = frame["stego_path"].map(lambda value: str(Path(value).expanduser().resolve()))
    if cover_paths.duplicated().any():
        raise ValueError("Pair manifest contains duplicate cover paths")
    if stego_paths.duplicated().any():
        raise ValueError("Pair manifest contains duplicate stego paths")
    if set(cover_paths) & set(stego_paths):
        raise ValueError("A path cannot be used as both cover and stego")
    assert_patient_disjoint(frame)
    if frame.groupby("pair_id")["split"].nunique().max() > 1:
        raise AssertionError("A cover-stego pair spans multiple splits")

    if check_files:
        for row in frame.itertuples(index=False):
            cover_path = Path(row.cover_path)
            stego_path = Path(row.stego_path)
            if not cover_path.is_file() or not stego_path.is_file():
                raise FileNotFoundError(
                    f"Missing cover/stego file for pair {row.pair_id}: "
                    f"{cover_path}, {stego_path}"
                )
            with Image.open(cover_path) as cover, Image.open(stego_path) as stego:
                if cover.mode != "L" or stego.mode != "L":
                    raise ValueError(
                        f"Pair {row.pair_id} must use grayscale mode L, got "
                        f"{cover.mode}/{stego.mode}"
                    )
                if cover.size != stego.size:
                    raise ValueError(
                        f"Pair {row.pair_id} has mismatched sizes: {cover.size}/{stego.size}"
                    )
            if verify_hashes:
                if sha256_file(cover_path) != str(row.cover_sha256):
                    raise ValueError(
                        f"Cover SHA-256 mismatch for pair {row.pair_id}: {cover_path}"
                    )
                if sha256_file(stego_path) != str(row.stego_sha256):
                    raise ValueError(
                        f"Stego SHA-256 mismatch for pair {row.pair_id}: {stego_path}"
                    )


def save_manifest(frame: pd.DataFrame, output: str | Path) -> None:
    validate_pair_manifest(frame, check_files=True)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
