from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bimets import Frequency, from_pandas, timeseries, to_pandas


def test_year_period_round_trip_supports_non_calendar_frequency() -> None:
    source = timeseries(
        [1, 2, np.nan, 4],
        start=(2020, 23),
        freq=24,
        units="index",
    )

    pandas_series = to_pandas(source)
    assert isinstance(pandas_series.index, pd.MultiIndex)
    assert list(pandas_series.index) == [(2020, 23), (2020, 24), (2021, 1), (2021, 2)]
    assert pandas_series.attrs["bimets_frequency"] == 24

    restored = from_pandas(pandas_series)
    assert restored.start == source.start
    assert restored.freq is Frequency.PERIODS_24
    assert restored.metadata == {"units": "index"}
    np.testing.assert_allclose(restored.values, source.values, equal_nan=True)


@pytest.mark.parametrize(
    ("freq", "start", "expected_label"),
    [
        ("Y", (2020, 1), "2020"),
        ("Q", (2020, 3), "2020Q3"),
        ("M", (2020, 3), "2020-03"),
    ],
)
def test_period_index_round_trip(
    freq: str,
    start: tuple[int, int],
    expected_label: str,
) -> None:
    source = timeseries([1, 2, 3], start=start, freq=freq)

    pandas_series = to_pandas(source, index="period")
    assert isinstance(pandas_series.index, pd.PeriodIndex)
    assert str(pandas_series.index[0]) == expected_label

    restored = from_pandas(pandas_series)
    assert restored.start == source.start
    assert restored.freq == source.freq
    np.testing.assert_array_equal(restored.values, source.values)


def test_datetime_index_honours_date_in_period() -> None:
    monthly = timeseries([1, 2], start=(2020, 1), freq="M")

    first = to_pandas(monthly, index="datetime", date_in_period="first")
    last = to_pandas(monthly, index="datetime", date_in_period="last")

    assert list(first.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert list(last.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]

    restored = from_pandas(last)
    assert restored.start == monthly.start
    assert restored.freq == monthly.freq
    np.testing.assert_array_equal(restored.values, monthly.values)

    with pytest.raises(ValueError, match="date_in_period"):
        to_pandas(monthly, index="datetime", date_in_period="middle")  # type: ignore[arg-type]


@pytest.mark.parametrize(("freq", "start"), [("Y", (2020, 1)), ("Q", (2020, 2))])
def test_datetime_round_trip_for_yearly_and_quarterly(
    freq: str, start: tuple[int, int]
) -> None:
    source = timeseries([1, 2, 3], start=start, freq=freq)
    restored = from_pandas(to_pandas(source, index="datetime"))

    assert restored.start == source.start
    assert restored.freq == source.freq
    np.testing.assert_array_equal(restored.values, source.values)


def test_pandas_adapter_validation() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        from_pandas([1, 2, 3])
    with pytest.raises(ValueError, match="must not be empty"):
        from_pandas(pd.Series([], dtype=float, index=pd.PeriodIndex([], freq="M")))

    duplicate = pd.Series(
        [1, 2],
        index=pd.MultiIndex.from_tuples(
            [(2020, 1), (2020, 1)], names=["year", "period"]
        ),
    )
    with pytest.raises(ValueError, match="duplicates"):
        from_pandas(duplicate, freq=12)

    decreasing = pd.Series(
        [1, 2],
        index=pd.MultiIndex.from_tuples(
            [(2020, 2), (2020, 1)], names=["year", "period"]
        ),
    )
    with pytest.raises(ValueError, match="increasing"):
        from_pandas(decreasing, freq=12)

    wrong_names = pd.Series(
        [1],
        index=pd.MultiIndex.from_tuples([(2020, 1)], names=["y", "p"]),
    )
    with pytest.raises(ValueError, match="named year and period"):
        from_pandas(wrong_names, freq=12)

    irregular = pd.Series(
        [1, 2],
        index=pd.MultiIndex.from_tuples(
            [(2020, 1), (2020, 3)], names=["year", "period"]
        ),
    )
    with pytest.raises(ValueError, match="not a regular"):
        from_pandas(irregular, freq=12)

    irregular_periods = pd.Series(
        [1, 2], index=pd.PeriodIndex(["2020-01", "2020-03"], freq="M")
    )
    with pytest.raises(ValueError, match="not a regular"):
        from_pandas(irregular_periods)

    no_frequency = pd.Series(
        [1],
        index=pd.MultiIndex.from_tuples([(2020, 1)], names=["year", "period"]),
    )
    with pytest.raises(ValueError, match="frequency is required"):
        from_pandas(no_frequency)


def test_unsupported_pandas_representations_are_rejected() -> None:
    semiannual = timeseries([1, 2], freq=2)
    with pytest.raises(ValueError, match="only annual"):
        to_pandas(semiannual, index="period")
    with pytest.raises(ValueError, match="unsupported pandas index"):
        to_pandas(timeseries([1]), index="invalid")  # type: ignore[arg-type]

    dated = pd.Series([1], index=pd.DatetimeIndex(["2020-01-01"]))
    with pytest.raises(ValueError, match="frequency is required"):
        from_pandas(dated)

    weekly = pd.Series([1], index=pd.period_range("2020-01-01", periods=1, freq="W"))
    with pytest.raises(ValueError, match="unsupported pandas PeriodIndex"):
        from_pandas(weekly)

    monthly = pd.Series([1], index=pd.period_range("2020-01", periods=1, freq="M"))
    with pytest.raises(ValueError, match="conflicts"):
        from_pandas(monthly, freq="Q")
