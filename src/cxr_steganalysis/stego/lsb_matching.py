"""Deterministic LSB Matching for 8-bit grayscale images."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cxr_steganalysis.reproducibility import stable_seed


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LSBMatchingMetadata:
    algorithm: str
    payload_bpp: float
    embedded_bits: int
    changed_pixels: int
    seed: int
    cover_sha256: str
    stego_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LSBMatchingResult:
    image: np.ndarray
    selected_indices: np.ndarray
    message_bits: np.ndarray
    metadata: LSBMatchingMetadata


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("LSB Matching input must be a NumPy array")
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 pixels, got {image.dtype}")


def embed_lsb_matching(
    image: np.ndarray,
    payload_bpp: float,
    seed: int,
) -> LSBMatchingResult:
    """Embed pseudorandom bits at unique pixel positions using LSB Matching."""
    _validate_image(image)
    if not 0.0 <= float(payload_bpp) <= 1.0:
        raise ValueError(f"payload_bpp must be in [0, 1], got {payload_bpp}")

    cover = image.copy()
    stego = image.copy()
    num_pixels = image.size
    # A payload alpha bpp embeds floor(alpha * H * W) message bits. The
    # changed-pixel fraction is typically about alpha/2 and is recorded separately.
    message_length = int(np.floor(float(payload_bpp) * num_pixels))
    rng = np.random.default_rng(seed)
    if message_length == 0:
        selected = np.empty(0, dtype=np.int64)
        bits = np.empty(0, dtype=np.uint8)
    else:
        selected = rng.choice(num_pixels, size=message_length, replace=False)
        bits = rng.integers(0, 2, size=message_length, dtype=np.uint8)
        flat = stego.reshape(-1)
        values = flat[selected]
        mismatch = (values & 1) != bits
        mismatch_indices = selected[mismatch]
        mismatch_values = flat[mismatch_indices]
        directions = rng.choice(np.array([-1, 1], dtype=np.int16), size=len(mismatch_indices))
        directions[mismatch_values == 0] = 1
        directions[mismatch_values == 255] = -1
        flat[mismatch_indices] = (
            mismatch_values.astype(np.int16) + directions
        ).astype(np.uint8)

    metadata = LSBMatchingMetadata(
        algorithm="lsb_matching",
        payload_bpp=float(payload_bpp),
        embedded_bits=message_length,
        changed_pixels=int(np.count_nonzero(cover != stego)),
        seed=int(seed),
        cover_sha256=sha256_bytes(cover.tobytes()),
        stego_sha256=sha256_bytes(stego.tobytes()),
    )
    return LSBMatchingResult(stego, selected, bits, metadata)


def load_grayscale_png(path: str | Path, allow_conversion: bool = False) -> np.ndarray:
    image_path = Path(path)
    if image_path.suffix.lower() != ".png":
        raise ValueError(f"Only PNG input is supported, got: {image_path}")
    with Image.open(image_path) as image:
        if image.mode != "L":
            if not allow_conversion:
                raise ValueError(
                    f"Expected grayscale mode L, got {image.mode} for {image_path}. "
                    "Set allow_conversion explicitly to convert."
                )
            image = image.convert("L")
        array = np.asarray(image, dtype=np.uint8).copy()
    _validate_image(array)
    return array


def embed_lsb_matching_file(
    input_path: str | Path,
    payload_bpp: float,
    global_seed: int,
    image_id: str,
    output_path: str | Path | None = None,
    allow_conversion: bool = False,
) -> LSBMatchingResult:
    """Load a PNG, derive its per-image seed, embed, and optionally save it."""
    input_path = Path(input_path)
    array = load_grayscale_png(input_path, allow_conversion=allow_conversion)
    derived_seed = stable_seed(global_seed, image_id, "lsb_matching", float(payload_bpp))
    result = embed_lsb_matching(array, payload_bpp, derived_seed)
    if output_path is None:
        return result

    output_path = Path(output_path)
    if output_path.resolve() == input_path.resolve() or output_path.exists():
        raise FileExistsError(f"Refusing to overwrite raw/existing PNG: {output_path}")
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Stego output must be PNG, got: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image, mode="L").save(output_path, format="PNG")
    file_metadata = LSBMatchingMetadata(
        algorithm=result.metadata.algorithm,
        payload_bpp=result.metadata.payload_bpp,
        embedded_bits=result.metadata.embedded_bits,
        changed_pixels=result.metadata.changed_pixels,
        seed=result.metadata.seed,
        cover_sha256=sha256_file(input_path),
        stego_sha256=sha256_file(output_path),
    )
    return LSBMatchingResult(
        result.image, result.selected_indices, result.message_bits, file_metadata
    )
