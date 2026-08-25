from __future__ import annotations

import numpy as np
import pytest

from bimets import (
    Frequency,
    YearPeriod,
    annual,
    convert_frequency,
    daily,
    monthly,
    quarterly,
    semiannual,
    timeseries,
)


@pytest.mark.source("bimets-R")
def test_monthly_to_yearly_matches_help_example() -> None:
    source = timeseries(range(37), start=(2000, 1), freq="M")
    values = source.values.copy()
    values[9] = np.nan
    source = timeseries(values, start=source.start, freq=source.freq)

    result = annual(source, method="sum")

    assert result.start == YearPeriod(2000, 1)
    np.testing.assert_allclose(result.values, [np.nan, 210, 354], equal_nan=True)


@pytest.mark.source("bimets-R")
def test_daily_to_yearly_non_missing_average_matches_help_example() -> None:
    source = timeseries(range(367), start=(2000, 1), freq="D")
    values = source.values.copy()
    values[9] = np.nan
    result = annual(timeseries(values, start=(2000, 1), freq="D"), method="nave")

    assert result.start == YearPeriod(2000, 1)
    np.testing.assert_allclose(result.values, [(sum(range(366)) - 9) / 365])


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("stock", [np.nan, 6]),
        ("nstock", [2, 6]),
        ("sum", [np.nan, 15]),
        ("nsum", [3, 15]),
        ("average", [np.nan, 5]),
        ("naverage", [1.5, 5]),
    ],
)
def test_all_aggregation_methods(method: str, expected: list[float]) -> None:
    source = timeseries([1, 2, np.nan, 4, 5, 6], freq="M")
    result = quarterly(source, method=method)
    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_stock_accepts_partial_initial_period_but_flows_require_complete_period() -> (
    None
):
    source = timeseries([2, 3], start=(2000, 2), freq="M")

    stock = quarterly(source, method="stock")
    assert stock.start == YearPeriod(2000, 1)
    np.testing.assert_array_equal(stock.values, [3])
    with pytest.raises(ValueError, match="complete target period"):
        quarterly(source, method="sum")


@pytest.mark.source("bimets-R")
def test_quarterly_to_semiannual_matches_help_example() -> None:
    source = timeseries(range(14, -1, -1), start=(2000, 1), freq="Q")
    result = semiannual(source, method="nave")

    assert result.start == YearPeriod(2000, 1)
    np.testing.assert_array_equal(result.values, [13.5, 11.5, 9.5, 7.5, 5.5, 3.5, 1.5])


def test_repeat_disaggregation_covers_complete_source_periods() -> None:
    yearly = timeseries([1, 2], start=(2000, 1), freq="Y")
    result = quarterly(yearly)

    assert result.start == YearPeriod(2000, 1)
    np.testing.assert_array_equal(result.values, [1, 1, 1, 1, 2, 2, 2, 2])
    assert convert_frequency(result, "Q") is result


@pytest.mark.parametrize(
    ("method", "start_period"),
    [("interp_begin", 1), ("interp_center", 3), ("interp_end", 4)],
)
def test_yearly_to_quarterly_interpolation(method: str, start_period: int) -> None:
    source = timeseries([1, 2, 3], start=(2000, 1), freq="Y")
    result = quarterly(source, method=method)

    assert result.start == YearPeriod(2000, start_period)
    np.testing.assert_array_equal(
        result.values, [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    )


@pytest.mark.source("bimets-R")
def test_interpolation_preserves_valid_anchor_before_missing_value() -> None:
    source = timeseries([1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10], start=(2000, 1), freq="Y")
    result = quarterly(source, method="interp_center")

    assert result[12] == 4
    assert np.isnan(result.values[13:20]).all()
    assert result[20] == 6


@pytest.mark.source("bimets-R")
def test_annual_to_semiannual_interpolation_matches_help_example() -> None:
    source = timeseries(range(1, 11), start=(2000, 1), freq="Y")
    result = semiannual(source, method="interp_end")

    assert result.start == YearPeriod(2000, 2)
    np.testing.assert_array_equal(result.values, np.arange(1, 10.5, 0.5))


@pytest.mark.source("bimets-R")
def test_monthly_to_daily_repeat_matches_help_example() -> None:
    source = timeseries([1, 2, 3, 4], start=(2000, 1), freq="M")
    result = daily(source)

    assert result.start == YearPeriod(2000, 1)
    assert result.end == YearPeriod(2000, 121)
    np.testing.assert_array_equal(
        result.values,
        [*([1] * 31), *([2] * 29), *([3] * 31), *([4] * 30)],
    )


def test_daily_interpolation_uses_calendar_anchors() -> None:
    source = timeseries([1, 2], start=(2000, 1), freq="Q")
    result = daily(source, method="interp_center")

    assert result.start == YearPeriod(2000, 46)
    assert result.end == YearPeriod(2000, 136)
    assert len(result) == 91
    assert result[0] == 1
    assert result[-1] == 2
    np.testing.assert_allclose(result.values, np.linspace(1, 2, 91))


@pytest.mark.parametrize(
    ("freq", "method", "start", "end", "length"),
    [
        ("Y", "interp_begin", (2000, 1), (2001, 1), 367),
        ("Y", "interp_center", (2000, 183), (2001, 182), 366),
        ("Y", "interp_end", (2000, 366), (2001, 365), 366),
        ("S", "interp_begin", (2000, 1), (2000, 183), 183),
        ("S", "interp_center", (2000, 92), (2000, 275), 184),
        ("S", "interp_end", (2000, 182), (2000, 366), 185),
        ("Q", "interp_begin", (2000, 1), (2000, 92), 92),
        ("Q", "interp_end", (2000, 91), (2000, 182), 92),
        ("M", "interp_begin", (2000, 1), (2000, 32), 32),
        ("M", "interp_center", (2000, 15), (2000, 46), 32),
        ("M", "interp_end", (2000, 31), (2000, 60), 30),
    ],
)
def test_daily_interpolation_alignment_matches_documented_behavior(
    freq: str,
    method: str,
    start: tuple[int, int],
    end: tuple[int, int],
    length: int,
) -> None:
    result = daily(timeseries([1, 2], start=(2000, 1), freq=freq), method=method)

    assert result.start == YearPeriod(*start)
    assert result.end == YearPeriod(*end)
    assert len(result) == length
    assert result[0] == 1
    assert result[-1] == 2


def test_monthly_stock_from_daily_uses_calendar_month_ends() -> None:
    source = timeseries(range(1, 367), start=(2000, 1), freq="D")
    result = monthly(source, method="stock")

    np.testing.assert_array_equal(
        result.values, [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    )


def test_frequency_conversion_validation() -> None:
    monthly_series = timeseries([1, 2, 3], freq="M")
    yearly = timeseries([1], freq="Y")

    with pytest.raises(ValueError, match="aggregation method"):
        annual(monthly_series)
    with pytest.raises(ValueError, match="unknown aggregation"):
        annual(monthly_series, method="median")
    with pytest.raises(ValueError, match="unknown disaggregation"):
        monthly(yearly, method="spline")
    with pytest.raises(ValueError, match="at least two"):
        monthly(yearly, method="interp_begin")
    with pytest.raises(ValueError, match="supports yearly"):
        convert_frequency(timeseries([1], freq=24), "Y", method="sum")


def test_series_to_frequency_method() -> None:
    source = timeseries(range(1, 13), freq="M")
    np.testing.assert_array_equal(
        source.to_frequency("Q", method="sum").values,
        quarterly(source, method="sum").values,
    )
    assert source.to_frequency(Frequency.MONTHLY) is source


@pytest.mark.source("native")
@pytest.mark.parametrize(
    ("target", "method"),
    [("Q", "sum"), ("D", "repeat"), ("D", "interp_begin")],
)
def test_frequency_conversion_preserves_metadata(target: str, method: str) -> None:
    source = timeseries(
        range(1, 13),
        start=(2000, 1),
        freq="M",
        title="GDP",
        units="index",
        source="Example database",
    )

    result = convert_frequency(source, target, method=method)

    assert result.metadata == source.metadata
