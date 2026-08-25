from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pytest

from bimets import BimetsSeries, Frequency, YearPeriod, timeseries


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, Frequency.YEARLY),
        ("Y", Frequency.YEARLY),
        ("a", Frequency.YEARLY),
        ("S", Frequency.SEMIANNUAL),
        ("Q", Frequency.QUARTERLY),
        ("M", Frequency.MONTHLY),
        ("W", Frequency.WEEKLY),
        ("D", Frequency.DAILY),
        (24, Frequency.PERIODS_24),
    ],
)
def test_frequency_parsing(value: int | str, expected: Frequency) -> None:
    assert Frequency.parse(value) is expected


@pytest.mark.parametrize("value", [True, 5, "unknown", 12.0])
def test_frequency_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError, match="frequency"):
        Frequency.parse(value)  # type: ignore[arg-type]


def test_year_period_normalizes_and_shifts() -> None:
    assert YearPeriod.normalize(2000, 15, 12) == YearPeriod(2001, 3)
    assert YearPeriod(2000, 4).shift(1, 4) == YearPeriod(2001, 1)
    assert YearPeriod(2000, 1).shift(-1, 4) == YearPeriod(1999, 4)
    assert YearPeriod(2000, 3).ordinal(4) < YearPeriod(2000, 4).ordinal(4)


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda: YearPeriod(True, 1), TypeError),
        (lambda: YearPeriod(2000, False), TypeError),
        (lambda: YearPeriod(2000, 0), ValueError),
        (lambda: YearPeriod.normalize(2000, False, 4), TypeError),
        (lambda: YearPeriod.normalize(2000, 0, 4), ValueError),
        (lambda: YearPeriod(2000, 1).shift(True, 4), TypeError),
        (lambda: YearPeriod(2000, 1).ordinal(False), ValueError),
        (lambda: YearPeriod(2000, 1).ordinal(0), ValueError),
    ],
)
def test_year_period_validation(
    operation: Callable[[], object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        operation()


def test_series_properties_indexing_and_slicing() -> None:
    series = timeseries([1, 2, 3, 4], start=(2020, 3), freq="Q")

    assert len(series) == 4
    assert list(series) == [1.0, 2.0, 3.0, 4.0]
    assert series.start == YearPeriod(2020, 3)
    assert series.end == YearPeriod(2021, 2)
    assert series.period_at(0) == YearPeriod(2020, 3)
    assert series.period_at(-1) == YearPeriod(2021, 2)
    assert series.at_period(2021, 1) == 3
    assert series[1] == 2
    assert series[1:3].start == YearPeriod(2020, 4)
    np.testing.assert_array_equal(series[1:3].values, [2, 3])
    assert "length=4" in repr(series)


@pytest.mark.source("bimets-R")
def test_annual_print_matches_r_demo() -> None:
    series = timeseries(
        [100, 105, 111, 118, 126],
        start=(2020, 1),
        freq="A",
        source="sintetico",
        title="Indice annuale",
        units="indice",
    )

    assert (
        str(series)
        == """Time Series:
Start = 2020
End = 2024
Frequency = 1
[1] 100 105 111 118 126
attr(,"Source")
[1] sintetico
attr(,"Title")
[1] Indice annuale
attr(,"Units")
[1] indice"""
    )


@pytest.mark.source("bimets-R")
def test_quarterly_print_matches_r_demo_layout() -> None:
    series = timeseries(range(100, 108), start=(2023, 1), freq="Q")

    assert (
        str(series)
        == """     Qtr1 Qtr2 Qtr3 Qtr4
2023  100  101  102  103
2024  104  105  106  107"""
    )


@pytest.mark.source("bimets-R")
def test_monthly_print_matches_r_demo_layout() -> None:
    series = timeseries(range(100, 124), start=(2024, 1), freq="M")

    assert (
        str(series)
        == """     Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
2024 100 101 102 103 104 105 106 107 108 109 110 111
2025 112 113 114 115 116 117 118 119 120 121 122 123"""
    )


@pytest.mark.source("native")
def test_non_matrix_print_handles_ranges_wrapping_and_special_values() -> None:
    series = BimetsSeries(
        [*range(1, 31), np.nan, np.inf, -np.inf],
        start=(2023, 2),
        freq="S",
        metadata={"scale_factor": 2, "owner": "internal"},
    )

    rendered = str(series)

    assert "Start = c(2023, 2)" in rendered
    assert "End = c(2039, 2)" in rendered
    assert "Frequency = 2" in rendered
    assert "\n[" in rendered[rendered.index("[1]") + 3 :]
    assert "NA Inf -Inf" in rendered
    assert 'attr(,"ScaleFac")\n[1] 2' in rendered
    assert 'attr(,"owner")\n[1] internal' in rendered


@pytest.mark.source("native")
def test_partial_quarterly_print_leaves_unobserved_cycles_blank() -> None:
    series = timeseries([10, 20, 30], start=(2020, 3), freq="Q")

    assert (
        str(series)
        == """     Qtr1 Qtr2 Qtr3 Qtr4
2020             10   20
2021   30"""
    )


@pytest.mark.source("native")
def test_repr_is_a_compact_technical_view_with_data_and_metadata() -> None:
    series = timeseries(
        range(100, 112),
        start=(2023, 1),
        freq="Q",
        title="GDP",
    )

    rendered = repr(series)

    assert rendered.startswith("BimetsSeries(values=[100, 101, 102, 103, ...")
    assert "length=12" in rendered
    assert "start=(2023, 1)" in rendered
    assert "end=(2025, 4)" in rendered
    assert "freq=4" in rendered
    assert "metadata={'title': 'GDP'}" in rendered


@pytest.mark.source("bimets-R")
def test_r_style_year_period_indexing_matches_idxover_help() -> None:
    series = timeseries(range(26), start=(2000, 1), freq="M")

    assert series[[2000, 5]] == 4
    assert series[[2002, 2]] == 25

    start = [2001, 2]
    end = [2001, 4]
    assert series[[start]] == 13

    selected = series[[start, end]]
    assert isinstance(selected, BimetsSeries)
    assert selected.start == YearPeriod(2001, 2)
    assert selected.end == YearPeriod(2001, 4)
    np.testing.assert_array_equal(selected.values, [13, 14, 15])


@pytest.mark.source("bimets-R")
def test_r_style_range_indexing_is_inclusive_and_clips_to_overlap() -> None:
    series = timeseries(range(8), start=(2000, 1), freq="Q")

    selected = series[[[1999, 4], [2000, 2]]]
    assert isinstance(selected, BimetsSeries)
    assert selected.start == YearPeriod(2000, 1)
    assert selected.end == YearPeriod(2000, 2)
    np.testing.assert_array_equal(selected.values, [0, 1])

    with pytest.raises(ValueError, match="do not overlap"):
        _ = series[[[1998, 1], [1999, 1]]]


@pytest.mark.source("bimets-R")
def test_r_style_date_indexing_matches_idxover_help() -> None:
    monthly = timeseries(range(26), start=(2000, 1), freq="M")

    assert monthly["2001-01"] == 12
    assert monthly[date(2001, 1, 20)] == 12
    np.testing.assert_array_equal(
        monthly["2000-09/2001-01"].values,  # type: ignore[union-attr]
        [8, 9, 10, 11, 12],
    )
    np.testing.assert_array_equal(
        monthly["2000-09/"].values,  # type: ignore[union-attr]
        np.arange(8, 26),
    )
    np.testing.assert_array_equal(
        monthly["/2001-01"].values,  # type: ignore[union-attr]
        np.arange(13),
    )
    np.testing.assert_array_equal(
        monthly["2001"].values,  # type: ignore[union-attr]
        np.arange(12, 24),
    )

    quarterly = timeseries(range(26), start=(2000, 1), freq="Q")
    np.testing.assert_array_equal(
        quarterly["2001"].values,  # type: ignore[union-attr]
        [4, 5, 6, 7],
    )
    assert quarterly["2001-02"] == 4

    yearly = timeseries(range(1, 26), start=(2000, 1))
    assert yearly["2002-12-31"] == 3
    np.testing.assert_array_equal(
        yearly["2000/2004"].values,  # type: ignore[union-attr]
        [1, 2, 3, 4, 5],
    )

    daily = timeseries(range(1, 27), start=(2000, 1), freq="D")
    assert daily["2000-01-12"] == 12


@pytest.mark.source("bimets-R")
def test_immutable_year_period_updates_match_idxover_help() -> None:
    source = timeseries(range(26), start=(2000, 1), freq="M")

    missing = source.with_values([2000, 5], np.nan)
    assert np.isnan(missing[[2000, 5]])

    extended = source.with_values([2002, 2], [-1, -2, -3, -4, -5])
    assert extended.start == YearPeriod(2000, 1)
    assert extended.end == YearPeriod(2002, 6)
    np.testing.assert_array_equal(extended.values[-5:], [-1, -2, -3, -4, -5])

    start = [2001, 2]
    end = [2001, 4]
    broadcast = source.with_values([start, end], [0])
    np.testing.assert_array_equal(broadcast[[start, end]].values, [0, 0, 0])  # type: ignore[union-attr]
    replaced = source.with_values([start, end], [-2, -4, -6])
    np.testing.assert_array_equal(replaced[[start, end]].values, [-2, -4, -6])  # type: ignore[union-attr]

    np.testing.assert_array_equal(source.values, np.arange(26))


@pytest.mark.source("bimets-R")
def test_immutable_date_updates_match_idxover_help() -> None:
    source = timeseries(range(26), start=(2000, 1), freq="M")

    updated = source.with_values("2000-08", 9.9)
    updated = updated.with_values("2000-09/2001-01", 11.11)

    assert updated["2000-08"] == pytest.approx(9.9)
    np.testing.assert_allclose(
        updated["2000-09/2001-01"].values,  # type: ignore[union-attr]
        np.full(5, 11.11),
    )
    np.testing.assert_array_equal(source.values, np.arange(26))


def test_immutable_updates_support_python_positions_and_validation() -> None:
    source = timeseries([1, 2, 3, 4], start=(2020, 1), freq="Q")

    assert source.with_values(-1, 9).values.tolist() == [1, 2, 3, 9]
    assert source.with_values(slice(1, 3), [8, 7]).values.tolist() == [1, 8, 7, 4]
    assert source.with_values([2019, 4], 0).start == YearPeriod(2019, 4)

    with pytest.raises(ValueError, match="use extend=True"):
        source.with_values([2021, 1], 5, extend=False)
    with pytest.raises(ValueError, match="requires one or 2"):
        source.with_values([[2020, 1], [2020, 2]], [1, 2, 3])
    with pytest.raises(ValueError, match="must not be empty"):
        source.with_values([2020, 1], [])
    with pytest.raises(ValueError, match="one-dimensional"):
        source.with_values([2020, 1], [[1, 2]])
    with pytest.raises(ValueError, match="exactly one"):
        _ = source["2020/2021/2022"]
    with pytest.raises(ValueError, match="must use"):
        _ = source["January 2020"]


@pytest.mark.parametrize(
    ("key", "error", "message"),
    [
        ([2000, 0], ValueError, "must be positive"),
        ([2000, True], TypeError, "must be integers"),
        ([2000, "Q1"], TypeError, "must be integers"),
        ([2000], TypeError, "two-element sequence"),
        ([[2000]], ValueError, "exactly year and period"),
        ([[2000, 1], [2000, 2], [2000, 3]], ValueError, "inclusive start/end"),
    ],
)
def test_r_style_index_validation(
    key: list[object], error: type[Exception], message: str
) -> None:
    series = timeseries([1, 2, 3, 4], freq="Q")

    with pytest.raises(error, match=message):
        _ = series[key]  # type: ignore[index]


def test_r_style_index_rejects_python_tuple_subscript() -> None:
    series = timeseries([1, 2, 3, 4], freq="Q")

    with pytest.raises(TypeError, match="BIMETS year-period key"):
        _ = series[2000, 1]  # type: ignore[call-overload]


def test_values_and_metadata_are_read_only() -> None:
    series = BimetsSeries([1, 2], metadata={"units": "index"})

    with pytest.raises(ValueError, match="read-only"):
        series.values[0] = 9
    with pytest.raises(TypeError):
        series.metadata["units"] = "percent"  # type: ignore[index]


def test_constructor_copies_a_supplied_numpy_array() -> None:
    source = np.array([1.0, 2.0])
    series = BimetsSeries(source)

    assert source.flags.writeable
    source[0] = 9
    np.testing.assert_array_equal(series.values, [1, 2])


@pytest.mark.parametrize(
    ("values", "start", "freq", "message"),
    [
        ([], (2000, 1), 1, "must not be empty"),
        ([[1, 2]], (2000, 1), 1, "one-dimensional"),
        ([1], (0, 1), 1, "1-9999"),
        ([1, 2], (9999, 1), 1, "1-9999"),
    ],
)
def test_series_validation(
    values: object,
    start: tuple[int, int],
    freq: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BimetsSeries(values, start=start, freq=freq)  # type: ignore[arg-type]


@pytest.mark.source("bimets-R")
def test_series_accepts_year_one_like_r_ts() -> None:
    series = BimetsSeries([np.nan], start=(1, 1), freq=1)

    assert series.start == YearPeriod(1, 1)


def test_series_reports_invalid_access_and_slice() -> None:
    series = timeseries([1, 2, 3])

    with pytest.raises(IndexError):
        series.period_at(3)
    with pytest.raises(IndexError):
        series.at_period(1999, 1)
    with pytest.raises(ValueError, match="step"):
        _ = series[::2]
    with pytest.raises(ValueError, match="must not be empty"):
        _ = series[1:1]


def test_operation_validation_and_edge_cases() -> None:
    series = timeseries([1, 2, 3, 4], freq="Q")

    with pytest.raises(ValueError, match="positive integer"):
        series.delta(lag=0)
    with pytest.raises(ValueError, match="consume"):
        series.delta(lag=4)
    with pytest.raises(ValueError, match="divisible"):
        series.delta_percent(lag=3, annualize=True)
    with pytest.raises(ValueError, match="precedes"):
        series.project((2001, 1), (2000, 1))
    with pytest.raises(ValueError, match="do not overlap"):
        series.project((2010, 1), (2011, 1))
    extended = series.project((2010, 1), (2010, 2), extend=True)
    assert np.isnan(extended.values).all()
    assert timeseries([np.nan, np.nan]).trim() is None


def test_constructor_metadata_validation() -> None:
    with pytest.raises(TypeError, match="scale_factor"):
        timeseries([1], scale_factor=True)
    with pytest.raises(ValueError, match="scale_factor"):
        timeseries([1], scale_factor=-1)
