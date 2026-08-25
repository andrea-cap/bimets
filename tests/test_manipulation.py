from __future__ import annotations

import numpy as np
import pytest

from _paper_models import PAPER_DOI
from bimets import (
    YearPeriod,
    cumprod,
    cumsum,
    indexnum,
    movavg,
    movtot,
    tabulate,
    timeseries,
    tsdeltap,
    tsextend,
    tsjoin,
    tslag,
    tsmerge,
    tsproject,
)


@pytest.mark.source(PAPER_DOI)
def test_time_series_manipulation_chain_from_paper() -> None:
    """Reproduce the composed time-series example in paper section 2.3."""
    first = timeseries(range(1, 101), start=(2000, 1), freq="M")
    second = timeseries(-np.arange(1, 101), start=(2005, 1), freq="M")

    extended = tsextend(first, up_to=(2020, 4), mode="quadratic")
    merged = tsmerge(extended, second, method="sum")
    projected = tsproject(merged, (2004, 2), (2006, 4))
    transformed = movavg(tsdeltap(tslag(projected, 2), 2), 5)
    table = tabulate(
        transformed,
        first,
        headers=("moving_average", "original"),
        start=(2004, 8),
        end=(2004, 12),
    )

    np.testing.assert_allclose(
        table["moving_average"].to_numpy(),
        [np.nan, np.nan, 3.849002, 3.776275, 3.706247],
        atol=5e-7,
        equal_nan=True,
    )
    np.testing.assert_allclose(table["original"].to_numpy(), [56, 57, 58, 59, 60])


@pytest.mark.source("bimets-R")
def test_cumulative_operations_match_help_examples() -> None:
    monthly = timeseries(range(1, 31), start=(2000, 1), freq="M")
    np.testing.assert_array_equal(cumsum(monthly).values, np.cumsum(range(1, 31)))

    selected = monthly.project((2000, 4), (2001, 7))
    yearly = cumsum(selected, mode="yearly")
    np.testing.assert_array_equal(
        yearly.values,
        [4, 9, 15, 22, 30, 39, 49, 60, 72, 13, 27, 42, 58, 75, 93, 112],
    )
    product = cumprod(timeseries(range(1, 11), freq="M"))
    np.testing.assert_array_equal(
        product.values, [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    )


@pytest.mark.source("bimets-R")
def test_cumulative_operations_accept_r_style_calculation_ranges() -> None:
    source = timeseries(range(1, 9), start=(2020, 1), freq="Q")

    summed = cumsum(source, start=(2020, 3), end=(2021, 2))
    product = source.cumulative_product(start=(2020, 3), end=(2021, 2))

    assert summed.start == YearPeriod(2020, 3)
    np.testing.assert_array_equal(summed.values, [3, 7, 12, 18])
    np.testing.assert_array_equal(product.values, [3, 12, 60, 360])


@pytest.mark.source("native")
def test_cumulative_ranges_allow_one_implicit_source_boundary() -> None:
    source = timeseries([1, 2, 3, 4], start=(2020, 1), freq="Q")
    np.testing.assert_array_equal(source.cumulative_sum(start=(2020, 3)).values, [3, 7])
    np.testing.assert_array_equal(cumprod(source, end=(2020, 2)).values, [1, 2])


def test_cumulative_skip_missing_matches_documented_semantics() -> None:
    source = timeseries([1, np.nan, 2, np.nan, 3])
    np.testing.assert_array_equal(
        cumsum(source, skip_missing=True).values, [1, 1, 3, 3, 6]
    )
    np.testing.assert_array_equal(
        cumprod(source, skip_missing=True).values, [1, 1, 2, 2, 6]
    )


@pytest.mark.source("bimets-R")
def test_moving_operations_match_help_examples() -> None:
    source = timeseries(
        [1, 2, 3, 4, np.nan, 1, 2, 3, 4, 5],
        start=(2000, 1),
        freq="M",
    )
    average = movavg(source, 4, direction="center")
    total = movtot(source, 3)

    assert average.start == YearPeriod(2000, 3)
    np.testing.assert_allclose(
        average.values, [2.5, np.nan, np.nan, np.nan, np.nan, 2.5, 3.5], equal_nan=True
    )
    assert total.start == YearPeriod(2000, 3)
    np.testing.assert_allclose(
        total.values, [6, 9, np.nan, np.nan, np.nan, 6, 9, 12], equal_nan=True
    )
    np.testing.assert_allclose(
        movavg(source, 4, skip_missing=True).values,
        [2.5, 3, 8 / 3, 7 / 3, 2, 2.5, 3.5],
    )


@pytest.mark.source("bimets-R")
def test_moving_average_preserves_an_all_missing_window() -> None:
    source = timeseries(
        [1, 2, 3, 4, np.nan, np.nan, np.nan, 5, 6, 7],
        start=(2000, 1),
        freq="M",
    )

    result = movavg(source, 3, skip_missing=True)

    np.testing.assert_allclose(
        result.values, [2, 3, 3.5, 4, np.nan, 5, 5.5, 6], equal_nan=True
    )


@pytest.mark.source("bimets-R")
def test_index_number_matches_help_examples() -> None:
    yearly = timeseries(range(1, 21), start=(2000, 1))
    np.testing.assert_allclose(
        indexnum(yearly, 2005).values, np.arange(1, 21) * (100 / 6)
    )

    quarterly = timeseries(range(1, 21), start=(2000, 1), freq="Q")
    rebased = indexnum(quarterly, 2000)
    np.testing.assert_array_equal(rebased.values[:4], [40, 80, 120, 160])


@pytest.mark.source("bimets-R")
def test_join_matches_help_example_and_validates_gaps() -> None:
    first = timeseries(range(1, 11), start=(1985, 1))
    second = timeseries(range(1, 11), start=(2000, 1))

    with pytest.raises(ValueError, match="gap"):
        tsjoin(first, second)
    joined = tsjoin(first, second, allow_gap=True)
    assert joined.start == YearPeriod(1985, 1)
    np.testing.assert_allclose(
        joined.values,
        [*range(1, 11), *([np.nan] * 5), *range(1, 11)],
        equal_nan=True,
    )


def test_merge_uses_argument_priority_and_aggregation() -> None:
    first = timeseries([3, np.nan, 5], start=(2000, 1))
    second = timeseries([10, 20, 30], start=(2001, 1))

    np.testing.assert_allclose(
        tsmerge(first, second).values, [3, 10, 5, 30], equal_nan=True
    )
    np.testing.assert_allclose(
        tsmerge(first, second, method="sum").values, [3, 10, 25, 30], equal_nan=True
    )
    np.testing.assert_allclose(
        tsmerge(first, second, method="sum", skip_missing=False).values,
        [np.nan, np.nan, 25, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tsmerge(first, second, method="average").values, [3, 10, 12.5, 30]
    )
    np.testing.assert_allclose(
        tsmerge(first, second, method="max").values, [3, 10, 20, 30]
    )
    np.testing.assert_allclose(
        tsmerge(first, second, method="min").values, [3, 10, 5, 30]
    )


@pytest.mark.source("bimets-R")
def test_extend_growth4_matches_help_example() -> None:
    source = timeseries(range(1, 11), start=(2000, 1))
    extended = tsextend(source, back_to=(1990, 1), up_to=(2020, 1), mode="growth4")
    backward_rate = (10 / 26) ** 0.25
    forward_rate = (34 / 18) ** 0.25

    assert extended.start == YearPeriod(1990, 1)
    assert extended.end == YearPeriod(2020, 1)
    np.testing.assert_allclose(
        extended.values[:10], backward_rate ** np.arange(10, 0, -1)
    )
    np.testing.assert_array_equal(extended.values[10:20], np.arange(1, 11))
    np.testing.assert_allclose(
        extended.values[20:], 10 * forward_rate ** np.arange(1, 12)
    )


@pytest.mark.source("bimets-R")
def test_extend_preserves_sparse_source_boundaries() -> None:
    source = timeseries(
        [np.nan, 2, np.nan, 2, 2, 2, np.nan, 2, np.nan, np.nan],
        start=(2000, 1),
    )
    result = tsextend(source, back_to=(1990, 1), up_to=(2020, 1), mode="mean4")

    np.testing.assert_allclose(
        result.values,
        [*([np.nan] * 10), *source.values, *([np.nan] * 11)],
        equal_nan=True,
    )


@pytest.mark.source("bimets-R")
def test_extend_all_missing_and_zero_modes_match_r() -> None:
    missing = timeseries([np.nan], start=(2000, 1), freq="Q")
    projected = tsextend(
        missing, back_to=(1998, 2), up_to=(2003, 1), mode="myrate", factor=1.5
    )
    zeros = tsextend(
        timeseries([np.nan] * 10, start=(2000, 1)),
        back_to=(1990, 1),
        up_to=(2020, 1),
        mode="zero",
    )

    assert np.isnan(projected.values).all()
    np.testing.assert_array_equal(zeros.values, np.zeros(31))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("missing", [np.nan, 1, 2, 3, np.nan]),
        ("zero", [0, 1, 2, 3, 0]),
        ("constant", [1, 1, 2, 3, 3]),
        ("linear", [0, 1, 2, 3, 4]),
        ("quadratic", [0, 1, 2, 3, 4]),
    ],
)
def test_extend_modes(mode: str, expected: list[float]) -> None:
    actual = tsextend(
        timeseries([1, 2, 3], start=(2000, 1)),
        back_to=(1999, 1),
        up_to=(2003, 1),
        mode=mode,
    )
    np.testing.assert_allclose(actual.values, expected, equal_nan=True)


def test_extend_configurable_and_short_history_modes() -> None:
    source = timeseries([1, 2, 3], start=(2000, 1))
    constant = tsextend(
        source,
        back_to=(1999, 1),
        up_to=(2003, 1),
        mode="myconst",
        factor=7,
    )
    rate = tsextend(
        source,
        back_to=(1999, 1),
        up_to=(2003, 1),
        mode="myrate",
        factor=2,
    )
    mean = tsextend(
        timeseries([1, 2, 3, 4], start=(2000, 1)),
        back_to=(1999, 1),
        up_to=(2004, 1),
        mode="mean4",
    )
    short_growth = tsextend(
        source,
        back_to=(1999, 1),
        up_to=(2003, 1),
        mode="growth4",
    )

    np.testing.assert_array_equal(constant.values, [7, 1, 2, 3, 7])
    np.testing.assert_array_equal(rate.values, [2, 1, 2, 3, 6])
    np.testing.assert_array_equal(mean.values, [2.5, 1, 2, 3, 4, 2.5])
    assert np.isnan(short_growth.values[[0, -1]]).all()


def test_series_methods_delegate_to_manipulation_operations() -> None:
    source = timeseries([1, 2, 3, 4], start=(2000, 1))

    np.testing.assert_array_equal(source.cumulative_sum().values, cumsum(source).values)
    np.testing.assert_array_equal(
        source.cumulative_product().values, cumprod(source).values
    )
    np.testing.assert_array_equal(
        source.moving_average(2).values, movavg(source, 2).values
    )
    np.testing.assert_array_equal(source.moving_sum(2).values, movtot(source, 2).values)
    np.testing.assert_array_equal(
        source.index_number(2000).values, indexnum(source, 2000).values
    )
    np.testing.assert_array_equal(
        source.extend(up_to=(2004, 1), mode="constant").values,
        tsextend(source, up_to=(2004, 1), mode="constant").values,
    )


def test_manipulation_validation() -> None:
    source = timeseries([1, 2, 3], freq="Q")
    with pytest.raises(ValueError, match="window"):
        movavg(source, 4)
    with pytest.raises(ValueError, match="direction"):
        movtot(source, 2, direction="sideways")
    with pytest.raises(ValueError, match="base year"):
        indexnum(source, 1999)
    with pytest.raises(ValueError, match="base_year"):
        indexnum(source, True)
    with pytest.raises(ValueError, match="missing values"):
        indexnum(timeseries([1, np.nan], start=(2000, 1), freq=2), 2000)
    with pytest.raises(ValueError, match="average is zero"):
        indexnum(timeseries([-1, 1], start=(2000, 1), freq=2), 2000)
    with pytest.raises(ValueError, match="factor"):
        tsextend(source, back_to=(1999, 1), mode="myrate")
    with pytest.raises(ValueError, match="unknown extension"):
        tsextend(source, mode="cubic")
    all_missing = tsextend(timeseries([np.nan]), up_to=(2001, 1), mode="growth")
    assert np.isnan(all_missing.values).all()
    with pytest.raises(ValueError, match="same frequency"):
        tsmerge(source, timeseries([1], freq="M"))
    with pytest.raises(ValueError, match="at least one"):
        tsmerge()
    with pytest.raises(ValueError, match="unknown merge"):
        tsmerge(source, method="median")
    with pytest.raises(ValueError, match="same frequency"):
        tsjoin(source, timeseries([1], freq="M"))
    with pytest.raises(ValueError, match="outside"):
        tsjoin(source, source, join_period=(2002, 1))
    with pytest.raises(ValueError, match="mode must"):
        cumsum(source, mode="week")


def test_monthly_cumulative_mode_supports_multiple_frequencies() -> None:
    quarterly = cumsum(
        timeseries([1, 2, 3, 4], start=(2000, 1), freq="Q"), mode="monthly"
    )
    twice_monthly = cumsum(
        timeseries([1, 2, 3, 4], start=(2000, 1), freq=24), mode="monthly"
    )

    np.testing.assert_array_equal(quarterly.values, [1, 2, 3, 4])
    np.testing.assert_array_equal(twice_monthly.values, [1, 3, 3, 7])
