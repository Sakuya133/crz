from __future__ import annotations

import numpy as np
import pytest

from cxr_steganalysis.models.srm_svm import SRM_DIMENSION, srm_vector, _bounded_ordered_map


def test_srm_rejects_normalized_or_non_grayscale_input():
    with pytest.raises(ValueError, match="2D uint8"):
        srm_vector(np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="2D uint8"):
        srm_vector(np.zeros((32, 32, 3), dtype=np.uint8))


def test_full_srm_dimension_when_optional_dependency_is_available():
    pytest.importorskip("sealwatch")
    image = np.random.default_rng(7).integers(0, 256, (32, 32), dtype=np.uint8)
    first, schema = srm_vector(image)
    second, second_schema = srm_vector(image)
    assert len(first) == SRM_DIMENSION
    assert np.array_equal(first, second)
    assert schema == second_schema


def test_srm_submission_is_bounded_and_ordered():
    from concurrent.futures import ThreadPoolExecutor
    consumed = []
    def jobs():
        for index in range(100):
            consumed.append(index)
            yield index
    with ThreadPoolExecutor(max_workers=2) as pool:
        stream = _bounded_ordered_map(pool, lambda i: i*i, jobs(), 4)
        assert next(stream) == 0
        assert len(consumed) == 4  # not all 100 tasks submitted/eagerly retained
        assert [0] + list(stream) == [i*i for i in range(100)]
