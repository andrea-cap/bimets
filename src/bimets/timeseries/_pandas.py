"""Pandas adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod
from bimets.timeseries._series import BimetsSeries, MetadataValue

PandasIndex = Literal["year-period", "period", "datetime"]
DateInPeriod = Literal["first", "last"]


def to_pandas(
    series: BimetsSeries,
    *,
    index: PandasIndex = "year-period",
    date_in_period: DateInPeriod = "last",
) -> Any:
    """Convert a BIMETS series to a pandas Series.

    Parameters
    ----------
    series : BimetsSeries
        Series to convert.
    index : {"year-period", "period", "datetime"}, default="year-period"
        Pandas index representation. ``year-period`` is lossless for every
        BIMETS frequency; the other representations support yearly, quarterly,
        and monthly series.
    date_in_period : {"first", "last"}, default="last"
        Timestamp boundary used by a ``datetime`` index.

    Returns
    -------
    pandas.Series
        Copy of the observations with BIMETS metadata stored in ``attrs``.

    Raises
    ------
    ValueError
        If the index representation is invalid or unsupported for the series
        frequency.

    Examples
    --------
    >>> from bimets import timeseries, to_pandas
    >>> series = timeseries([10, 20], start=(2020, 1), freq="Q")
    >>> converted = to_pandas(series)
    >>> converted.index.names
    FrozenList(['year', 'period'])
    >>> converted.tolist()
    [10.0, 20.0]
    """
    if index == "year-period":
        periods = [series.period_at(position) for position in range(len(series))]
        pandas_index = pd.MultiIndex.from_arrays(
            [
                [period.year for period in periods],
                [period.period for period in periods],
            ],
            names=["year", "period"],
        )
    else:
        pandas_index = _to_period_index(series, pd)
        if index == "datetime":
            if date_in_period not in {"first", "last"}:
                raise ValueError("date_in_period must be first or last")
            how = "start" if date_in_period == "first" else "end"
            pandas_index = pandas_index.to_timestamp(how=how).normalize()
        elif index != "period":
            raise ValueError(f"unsupported pandas index representation: {index}")

    output = pd.Series(series.values.copy(), index=pandas_index)
    output.attrs.update(series.metadata)
    output.attrs["bimets_frequency"] = int(series.freq)
    return output


def from_pandas(
    pandas_series: Any,
    *,
    freq: int | str | Frequency | None = None,
) -> BimetsSeries:
    """Convert a regular pandas Series to :class:`BimetsSeries`.

    Parameters
    ----------
    pandas_series : pandas.Series
        Non-empty numeric series with an increasing, unique ``PeriodIndex``,
        ``DatetimeIndex``, or ``MultiIndex`` named ``year`` and ``period``.
    freq : int, str, or Frequency, optional
        Explicit BIMETS frequency. It is required when it cannot be inferred
        from the index or the ``bimets_frequency`` attribute.

    Returns
    -------
    BimetsSeries
        Regular BIMETS representation with copied values and metadata.

    Raises
    ------
    TypeError
        If the input or its index type is unsupported.
    ValueError
        If the index is empty, duplicated, unordered, irregular, or conflicts
        with the supplied frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from bimets import from_pandas
    >>> source = pd.Series([1.0, 2.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
    >>> converted = from_pandas(source)
    >>> converted.start, converted.freq
    (YearPeriod(year=2020, period=1), <Frequency.QUARTERLY: 4>)
    """
    if not isinstance(pandas_series, pd.Series):
        raise TypeError("from_pandas expects a pandas Series")
    if pandas_series.empty:
        raise ValueError("pandas Series must not be empty")
    if not pandas_series.index.is_unique:
        raise ValueError("pandas index must not contain duplicates")
    if not pandas_series.index.is_monotonic_increasing:
        raise ValueError("pandas index must be increasing")

    parsed_frequency: Frequency
    start: YearPeriod
    if isinstance(pandas_series.index, pd.MultiIndex):
        if list(pandas_series.index.names) != ["year", "period"]:
            raise ValueError("MultiIndex levels must be named year and period")
        parsed_frequency = _resolve_frequency(pandas_series, freq)
        first = pandas_series.index[0]
        start = YearPeriod(int(first[0]), int(first[1]))
    elif isinstance(pandas_series.index, pd.PeriodIndex):
        inferred_frequency = _frequency_from_period_index(pandas_series.index)
        parsed_frequency = inferred_frequency if freq is None else Frequency.parse(freq)
        if parsed_frequency != inferred_frequency:
            raise ValueError("provided frequency conflicts with PeriodIndex")
        first_period = pandas_series.index[0]
        if parsed_frequency is Frequency.YEARLY:
            start = YearPeriod(int(first_period.year), 1)
        elif parsed_frequency is Frequency.QUARTERLY:
            start = YearPeriod(int(first_period.year), int(first_period.quarter))
        else:
            start = YearPeriod(int(first_period.year), int(first_period.month))
    elif isinstance(pandas_series.index, pd.DatetimeIndex):
        parsed_frequency = _resolve_frequency(pandas_series, freq)
        period_frequency = _pandas_period_frequency(parsed_frequency)
        period_index = pandas_series.index.to_period(period_frequency)
        first_period = period_index[0]
        if parsed_frequency is Frequency.YEARLY:
            start = YearPeriod(int(first_period.year), 1)
        elif parsed_frequency is Frequency.QUARTERLY:
            start = YearPeriod(int(first_period.year), int(first_period.quarter))
        else:
            start = YearPeriod(int(first_period.year), int(first_period.month))
    else:
        raise TypeError(
            "pandas index must be a PeriodIndex, DatetimeIndex, "
            "or year-period MultiIndex"
        )

    metadata = cast(
        Mapping[str, MetadataValue],
        {
            key: value
            for key, value in pandas_series.attrs.items()
            if key != "bimets_frequency"
        },
    )
    result = BimetsSeries(
        pandas_series.to_numpy(dtype=np.float64),
        start=start,
        freq=parsed_frequency,
        metadata=metadata,
    )
    expected = [result.period_at(position) for position in range(len(result))]
    if isinstance(pandas_series.index, pd.MultiIndex):
        actual = [
            YearPeriod(int(year), int(period)) for year, period in pandas_series.index
        ]
    elif isinstance(pandas_series.index, pd.PeriodIndex):
        actual = [
            _year_period_from_pandas_period(item, parsed_frequency)
            for item in pandas_series.index
        ]
    elif isinstance(pandas_series.index, pd.DatetimeIndex):
        period_frequency = _pandas_period_frequency(parsed_frequency)
        actual = [
            _year_period_from_pandas_period(item, parsed_frequency)
            for item in pandas_series.index.to_period(period_frequency)
        ]
    else:
        raise AssertionError("unreachable pandas index type")
    if actual != expected:
        raise ValueError("pandas index is not a regular BIMETS sequence")
    return result


def _resolve_frequency(
    pandas_series: Any, freq: int | str | Frequency | None
) -> Frequency:
    """Resolve an explicit or inferred time-series frequency."""
    if freq is not None:
        return Frequency.parse(freq)
    stored = pandas_series.attrs.get("bimets_frequency")
    if stored is None:
        raise ValueError("frequency is required for a year-period MultiIndex")
    return Frequency.parse(int(stored))


def _frequency_from_period_index(index: Any) -> Frequency:
    """Infer the BIMETS frequency from a pandas period index."""
    frequency_name = str(index.freqstr).upper()
    if frequency_name.startswith(("Y", "A")):
        return Frequency.YEARLY
    if frequency_name.startswith("Q"):
        return Frequency.QUARTERLY
    if frequency_name == "M":
        return Frequency.MONTHLY
    raise ValueError(f"unsupported pandas PeriodIndex frequency: {index.freqstr}")


def _to_period_index(series: BimetsSeries, pd: Any) -> Any:
    """Convert a BIMETS index to a pandas period index."""
    if series.freq is Frequency.YEARLY:
        start = pd.Period(str(series.start.year), freq="Y")
    elif series.freq is Frequency.QUARTERLY:
        start = pd.Period(
            f"{series.start.year}Q{series.start.period}",
            freq="Q",
        )
    elif series.freq is Frequency.MONTHLY:
        start = pd.Period(
            f"{series.start.year}-{series.start.period:02d}",
            freq="M",
        )
    else:
        raise ValueError(
            "PeriodIndex conversion supports only annual, quarterly, and monthly series"
        )
    return pd.period_range(start=start, periods=len(series), freq=start.freq)


def _pandas_period_frequency(freq: Frequency) -> str:
    """Return the pandas frequency code for a BIMETS frequency."""
    if freq is Frequency.YEARLY:
        return "Y"
    if freq is Frequency.QUARTERLY:
        return "Q"
    if freq is Frequency.MONTHLY:
        return "M"
    raise ValueError(
        "DatetimeIndex conversion supports only annual, quarterly, and monthly series"
    )


def _year_period_from_pandas_period(period: Any, freq: Frequency) -> YearPeriod:
    """Convert a pandas period to a BIMETS year-period index."""
    if freq is Frequency.YEARLY:
        return YearPeriod(int(period.year), 1)
    if freq is Frequency.QUARTERLY:
        return YearPeriod(int(period.year), int(period.quarter))
    return YearPeriod(int(period.year), int(period.month))
