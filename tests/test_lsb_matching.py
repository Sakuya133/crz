from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cxr_steganalysis.stego.lsb_matching import (
    embed_lsb_matching,
    embed_lsb_matching_file,
    load_grayscale_png,
)


def test_same_seed_is_identical_and_input_not_mutated():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    original = image.copy()
    first = embed_lsb_matching(image, 0.4, 123)
    second = embed_lsb_matching(image, 0.4, 123)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.selected_indices, second.selected_indices)
    assert np.array_equal(image, original)
    assert first.image.dtype == np.uint8
    assert first.image.shape == image.shape


def test_different_seeds_generally_differ():
    image = np.full((32, 32), 127, dtype=np.uint8)
    assert not np.array_equal(
        embed_lsb_matching(image, 0.5, 1).image,
        embed_lsb_matching(image, 0.5, 2).image,
    )


def test_boundaries_magnitude_and_unselected_pixels():
    image = np.tile(np.array([0, 255, 10, 11], dtype=np.uint8), (16, 16))
    result = embed_lsb_matching(image, 0.75, 77)
    difference = result.image.astype(np.int16) - image.astype(np.int16)
    assert np.abs(difference).max() <= 1
    assert np.array_equal(
        result.image.reshape(-1)[result.selected_indices] & 1,
        result.message_bits,
    )
    assert result.image[image == 0].min() >= 0
    assert result.image[image == 255].max() <= 255
    selected_mask = np.zeros(image.size, dtype=bool)
    selected_mask[result.selected_indices] = True
    assert np.array_equal(
        result.image.reshape(-1)[~selected_mask], image.reshape(-1)[~selected_mask]
    )


def test_payload_zero_and_invalid_payloads():
    image = np.full((9, 11), 42, dtype=np.uint8)
    result = embed_lsb_matching(image, 0, 1)
    assert np.array_equal(result.image, image)
    assert result.metadata.embedded_bits == 0
    assert result.metadata.changed_pixels == 0
    for payload in (-0.01, 1.01):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            embed_lsb_matching(image, payload, 1)


def test_message_length_matches_floor_bpp_definition():
    image = np.zeros((7, 9), dtype=np.uint8)
    result = embed_lsb_matching(image, 0.2, 10)
    assert len(result.selected_indices) == int(np.floor(0.2 * 7 * 9))
    assert len(np.unique(result.selected_indices)) == len(result.selected_indices)


def test_non_grayscale_rejected_by_default_and_explicit_conversion(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    with pytest.raises(ValueError, match="mode L"):
        load_grayscale_png(path)
    converted = load_grayscale_png(path, allow_conversion=True)
    assert converted.shape == (8, 8)
    output = tmp_path / "stego.png"
    embed_lsb_matching_file(path, 0.2, 1337, "rgb.png", output, allow_conversion=True)
    assert Image.open(output).mode == "L"
