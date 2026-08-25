"""Inspection and tabular presentation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from bimets.timeseries._calendar import get_dates, get_year_periods
from bimets.timeseries._display import frequency_table
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod
from bimets.timeseries._series import BimetsSeries, MetadataValue


@dataclass(frozen=True, slots=True)
class SeriesInfo:
    """Range, frequency, and standard metadata of a time series.

    Attributes
    ----------
    start, end : YearPeriod
        Inclusive series bounds.
    freq : Frequency
        Number of observations per year.
    source, title, units : str, int, float, bool, or None
        Standard metadata values.
    scale_factor : str, int, float, bool, or None
        Display scale metadata, defaulting to zero.
    """

    start: YearPeriod
    end: YearPeriod
    freq: Frequency
    source: MetadataValue = None
    title: MetadataValue = None
    units: MetadataValue = None
    scale_factor: MetadataValue = 0


def series_info(series: BimetsSeries) -> SeriesInfo:
    """Return structured information about a time series.

    Parameters
    ----------
    series : BimetsSeries
        Series to inspect.

    Returns
    -------
    SeriesInfo
        Bounds, frequency, and standard metadata.

    Examples
    --------
    >>> from bimets import series_info, timeseries
    >>> series = timeseries([1, 2], start=(2020, 1), freq="Q", title="GDP")
    >>> info = series_info(series)
    >>> info.start, info.title
    (YearPeriod(year=2020, period=1), 'GDP')
    """
    return SeriesInfo(
        start=series.start,
        end=series.end,
        freq=series.freq,
        source=series.metadata.get("source"),
        title=series.metadata.get("title"),
        units=series.metadata.get("units"),
        scale_factor=series.metadata.get("scale_factor", 0),
    )


def get_range(
    *series: BimetsSeries, kind: Literal["inner", "outer"] = "inner"
) -> tuple[YearPeriod, YearPeriod] | None:
    """Return the intersection or union of series ranges.

    Parameters
    ----------
    *series : BimetsSeries
        One or more series with a common frequency.
    kind : {"inner", "outer"}, default="inner"
        Select intersection or union.

    Returns
    -------
    tuple of YearPeriod or None
        Inclusive range, or ``None`` when the intersection is empty.

    Raises
    ------
    ValueError
        If no series is supplied, frequencies differ, or ``kind`` is invalid.
    """
    if not series:
        raise ValueError("at least one series is required")
    if kind not in {"inner", "outer"}:
        raise ValueError("kind must be inner or outer")
    freq = series[0].freq
    if any(item.freq != freq for item in series[1:]):
        raise ValueError("all series must have the same frequency")
    starts = [item.start for item in series]
    ends = [item.end for item in series]

    def key(value: YearPeriod) -> int:
        """Return the sortable ordinal for a period."""
        return value.ordinal(freq)

    if kind == "inner":
        start, end = max(starts, key=key), min(ends, key=key)
        return None if key(start) > key(end) else (start, end)
    return min(starts, key=key), max(ends, key=key)


def tabulate(
    *series: BimetsSeries,
    headers: Sequence[str] | None = None,
    start: YearPeriod | tuple[int, int] | None = None,
    end: YearPeriod | tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Return aligned series data in a pandas table.

    Parameters
    ----------
    *series : BimetsSeries
        One or more series with a common frequency.
    headers : sequence of str, optional
        Column names. Metadata titles or generated names are used by default.
    start, end : YearPeriod or tuple of int, optional
        Inclusive output bounds, defaulting to the outer range.

    Returns
    -------
    pandas.DataFrame
        Aligned values with a frequency-aware ``Date``/``Prd.`` MultiIndex.
    """
    if not series:
        raise ValueError("at least one series is required")
    freq = series[0].freq
    if any(item.freq != freq for item in series[1:]):
        raise ValueError("all series must have the same frequency")
    if headers is not None and len(headers) != len(series):
        raise ValueError("headers and series counts must match")
    names = (
        list(headers)
        if headers is not None
        else [
            str(item.metadata.get("title") or f"series_{index + 1}")
            for index, item in enumerate(series)
        ]
    )
    common = get_range(*series, kind="outer")
    assert common is not None
    selected_start = (
        common[0]
        if start is None
        else YearPeriod.normalize(*start, freq)
        if isinstance(start, tuple)
        else start
    )
    selected_end = (
        common[1]
        if end is None
        else YearPeriod.normalize(*end, freq)
        if isinstance(end, tuple)
        else end
    )
    if selected_end.ordinal(freq) < selected_start.ordinal(freq):
        raise ValueError("end precedes start")
    values = np.column_stack(
        [
            item.project(selected_start, selected_end, extend=True).values
            for item in series
        ]
    )
    return frequency_table(
        values,
        start=selected_start,
        freq=freq,
        columns=names,
    )


TsInfoValue = YearPeriod | Frequency | MetadataValue


def tsinfo(*series: BimetsSeries, mode: str) -> TsInfoValue | tuple[TsInfoValue, ...]:
    """Retrieve one information field from one or more series.

    Parameters
    ----------
    *series : BimetsSeries
        One or more series to inspect.
    mode : str
        Python modes are ``start``, ``end``, ``frequency``/``freq``,
        ``source``, ``title``, ``units``, and ``scale_factor``/``factor``.
        BIMETS R modes ``STARTY``, ``STARTP``, ``ENDY``, ``ENDP``, ``START2``,
        ``END2``, ``START``, ``END``, ``FREQ``, ``SOURCE``, ``TITLE``,
        ``UNITS``, and ``FACTOR`` are also accepted.

    Returns
    -------
    YearPeriod, Frequency, scalar, None, or tuple
        Requested value for one series, or an ordered tuple for multiple
        series. Uppercase R modes ``START`` and ``END`` return fractional
        year values; lowercase Python modes return ``YearPeriod`` objects.

    Raises
    ------
    ValueError
        If no series is supplied or ``mode`` is unknown.
    """
    if not series:
        raise ValueError("at least one series is required")
    values = tuple(_tsinfo_value(item, mode) for item in series)
    return values[0] if len(values) == 1 else values


def _tsinfo_value(series: BimetsSeries, mode: str) -> TsInfoValue:
    """Return one TSINFO field for one series."""
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    if mode == "START":
        return series.start.year + series.start.period / int(series.freq)
    if mode == "END":
        return series.end.year + series.end.period / int(series.freq)
    normalized = mode.lower()
    if normalized == "start":
        return series.start
    if normalized == "end":
        return series.end
    if normalized in {"frequency", "freq"}:
        return series.freq
    if normalized == "starty":
        return series.start.year
    if normalized == "startp":
        return series.start.period
    if normalized == "endy":
        return series.end.year
    if normalized == "endp":
        return series.end.period
    if normalized == "start2":
        return series.start
    if normalized == "end2":
        return series.end
    if normalized in {"source", "title", "units"}:
        return series.metadata.get(normalized)
    if normalized in {"scale_factor", "factor"}:
        return series.metadata.get("scale_factor", 0)
    raise ValueError(f"unknown information mode: {mode}")


def magnitude(series: BimetsSeries) -> float:
    """Return the Euclidean magnitude, ignoring missing observations.

    Parameters
    ----------
    series : BimetsSeries
        Numeric series.

    Returns
    -------
    float
        ``sqrt(sum(x**2))`` over non-missing observations.

    Examples
    --------
    >>> from bimets import magnitude, timeseries
    >>> magnitude(timeseries([3, 4]))
    5.0
    """
    return float(np.sqrt(np.nansum(np.square(series.values))))


def verify_magnitude(
    series: Sequence[BimetsSeries], *, threshold: float = 1e-6
) -> list[int]:
    """Find series whose magnitude exceeds a threshold.

    Parameters
    ----------
    series : sequence of BimetsSeries
        Series to inspect.
    threshold : float, default=1e-6
        Finite, non-negative strict lower bound.

    Returns
    -------
    list of int
        Zero-based indexes of values exceeding ``threshold``.

    Raises
    ------
    ValueError
        If ``threshold`` is negative or non-finite.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative number")
    return [index for index, item in enumerate(series) if magnitude(item) > threshold]


__all__ = [
    "SeriesInfo",
    "get_dates",
    "get_range",
    "get_year_periods",
    "magnitude",
    "series_info",
    "tabulate",
    "tsinfo",
    "verify_magnitude",
]
