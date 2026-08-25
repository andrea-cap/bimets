from __future__ import annotations

import numpy as np
import pytest

from bimets import BimetsMask, BimetsSeries, Frequency, YearPeriod, timeseries


def test_series_arithmetic_aligns_on_the_temporal_intersection() -> None:
    first = timeseries([1, 2, 3, 4, 5, 6], start=(2000, 1), freq="Q")
    second = timeseries([10, 11, 12, 13], start=(2000, 3), freq="Q")

    result = first + second

    assert isinstance(result, BimetsSeries)
    assert result.start == YearPeriod(2000, 3)
    assert result.end == YearPeriod(2001, 2)
    assert result.freq == Frequency.QUARTERLY
    assert result.metadata == {}
    np.testing.assert_array_equal(result.values, [13, 15, 17, 19])
    np.testing.assert_array_equal((second - first).values, [7, 7, 7, 7])
    np.testing.assert_array_equal((first * second).values, [30, 44, 60, 78])


def test_scalar_arithmetic_and_unary_operators() -> None:
    source = timeseries([-2, 3, 4], start=(2020, 1), freq="Y")

    np.testing.assert_array_equal((source + 2).values, [0, 5, 6])
    np.testing.assert_array_equal((2 + source).values, [0, 5, 6])
    np.testing.assert_array_equal((source - 2).values, [-4, 1, 2])
    np.testing.assert_array_equal((10 - source).values, [12, 7, 6])
    np.testing.assert_array_equal((source * 2).values, [-4, 6, 8])
    np.testing.assert_allclose((12 / source).values, [-6, 4, 3])
    np.testing.assert_array_equal((source // 2).values, [-1, 1, 2])
    np.testing.assert_array_equal((10 // source).values, [-5, 3, 2])
    np.testing.assert_array_equal((source % 2).values, [0, 1, 0])
    np.testing.assert_array_equal((10 % source).values, [0, 1, 2])
    np.testing.assert_array_equal((source**2).values, [4, 9, 16])
    np.testing.assert_array_equal((2**source).values, [0.25, 8, 16])
    np.testing.assert_array_equal((-source).values, [2, -3, -4])
    np.testing.assert_array_equal(abs(source).values, [2, 3, 4])
    assert +source is source


def test_arithmetic_propagates_missing_values_and_non_finite_results() -> None:
    source = timeseries([1, np.nan, 0])

    np.testing.assert_allclose((source + 1).values, [2, np.nan, 1], equal_nan=True)
    np.testing.assert_allclose((1 / source).values, [1, np.nan, np.inf], equal_nan=True)


def test_arithmetic_requires_compatible_operands() -> None:
    quarterly = timeseries([1, 2], start=(2020, 1), freq="Q")
    monthly = timeseries([1, 2], start=(2020, 1), freq="M")
    distant = timeseries([1], start=(2030, 1), freq="Q")

    with pytest.raises(ValueError, match="same frequency"):
        _ = quarterly + monthly
    with pytest.raises(ValueError, match="do not intersect"):
        _ = quarterly + distant
    with pytest.raises(TypeError, match="numeric scalar"):
        _ = quarterly + "1"
    with pytest.raises(TypeError, match="numeric scalar"):
        _ = quarterly + True


def test_comparisons_return_temporally_indexed_tri_state_masks() -> None:
    source = timeseries([1, np.nan, 3, 4], start=(2020, 1), freq="Q")

    greater = source > 2
    equal = source == 3
    different = source != 3

    assert isinstance(greater, BimetsMask)
    assert greater.start == source.start
    assert greater.end == source.end
    assert list(greater) == [False, None, True, True]
    assert list(equal) == [False, None, True, False]
    assert list(different) == [True, None, False, True]
    assert list(source <= 3) == [True, None, True, False]
    assert list(source > 2) == [False, None, True, True]


def test_series_comparisons_align_like_arithmetic() -> None:
    first = timeseries([1, 2, 3], start=(2020, 1), freq="Q")
    second = timeseries([2, 4, 3], start=(2020, 2), freq="Q")

    result = first < second

    assert result.start == YearPeriod(2020, 2)
    assert list(result) == [False, True]


def test_mask_logical_operations_use_three_valued_logic() -> None:
    first = BimetsMask([True, False, None], start=(2020, 1), freq="Q")
    second = BimetsMask([None, None, True], start=(2020, 1), freq="Q")

    assert list(first & second) == [None, False, None]
    assert list(first | second) == [True, None, True]
    assert list(first ^ second) == [None, None, None]
    assert list(~first) == [False, True, None]
    assert list(first & True) == [True, False, None]
    assert list(False | first) == [True, False, None]
    assert first.any() is True
    assert first.all() is False
    assert (
        BimetsMask([None, False], start=(2020, 1), freq="Q").any(skip_missing=False)
        is None
    )
    assert (
        BimetsMask([None, True], start=(2020, 1), freq="Q").all(skip_missing=False)
        is None
    )


def test_masks_align_slice_and_validate_usage() -> None:
    first = BimetsMask([True, False, True], start=(2020, 1), freq="Q")
    second = BimetsMask([True, True], start=(2020, 2), freq="Q")

    aligned = first & second
    assert aligned.start == YearPeriod(2020, 2)
    assert list(aligned) == [False, True]
    assert first[0] is True
    assert list(first[1:]) == [False, True]
    assert first.values.flags.writeable is False
    with pytest.raises(ValueError, match="ambiguous"):
        bool(first)
    with pytest.raises(ValueError, match="step"):
        _ = first[::2]
    with pytest.raises(ValueError, match="must not be empty"):
        _ = first[1:1]
    with pytest.raises(TypeError, match="boolean or missing"):
        BimetsMask([1], start=(2020, 1), freq="Q")
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        BimetsMask([], start=(2020, 1), freq="Q")
    with pytest.raises(ValueError, match="1-9999"):
        BimetsMask([True], start=(10000, 1), freq="Q")
    with pytest.raises(TypeError, match="BimetsMask or boolean"):
        _ = first & 1


def test_mask_rejects_incompatible_ranges_and_frequencies() -> None:
    quarterly = BimetsMask([True], start=(2020, 1), freq="Q")
    monthly = BimetsMask([True], start=(2020, 1), freq="M")
    distant = BimetsMask([True], start=(2030, 1), freq="Q")

    with pytest.raises(ValueError, match="same frequency"):
        _ = quarterly | monthly
    with pytest.raises(ValueError, match="do not intersect"):
        _ = quarterly | distant
