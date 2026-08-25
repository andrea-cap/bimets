"""Aggregation and disaggregation of BIMETS time series."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import date

import numpy as np

from bimets.timeseries._calendar import date_to_year_period, year_period_to_date
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod, _from_ordinal
from bimets.timeseries._series import BimetsSeries

Aggregation = str
Interpolation = str


def convert_frequency(
    series: BimetsSeries,
    freq: int | str | Frequency,
    *,
    method: str | None = None,
) -> BimetsSeries:
    """Aggregate or disaggregate a series to a supported target frequency.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    freq : int, str, or Frequency
        Target frequency.
    method : str, optional
        Aggregation method (``stock``, ``nstock``, ``sum``, ``nsum``,
        ``average``/``ave``, or ``naverage``/``nave``) or disaggregation method
        (``repeat``, ``interp_begin``, ``interp_center``, or ``interp_end``).
        Disaggregation defaults to ``repeat``.

    Returns
    -------
    BimetsSeries
        Converted series. Metadata is preserved.

    Raises
    ------
    ValueError
        If either frequency is unsupported for conversion, aggregation has no
        method, or the selected method is invalid.

    Notes
    -----
    Frequency conversion is supported between yearly, semiannual, quarterly,
    monthly, and daily series. Other BIMETS frequencies remain valid series
    frequencies but cannot currently be converted.

    Examples
    --------
    >>> from bimets import convert_frequency, timeseries
    >>> quarterly = timeseries([1, 2, 3, 4], start=(2020, 1), freq="Q")
    >>> annual = convert_frequency(quarterly, "Y", method="sum")
    >>> annual.values.tolist()
    [10.0]
    """
    target = Frequency.parse(freq)
    source = series.freq
    if target is source:
        return series
    supported = {
        Frequency.YEARLY,
        Frequency.SEMIANNUAL,
        Frequency.QUARTERLY,
        Frequency.MONTHLY,
        Frequency.DAILY,
    }
    if source not in supported or target not in supported:
        raise ValueError(
            "frequency conversion supports yearly, semiannual, quarterly, "
            "monthly, and daily series"
        )
    if int(target) < int(source):
        return _aggregate(series, target, method)
    return _disaggregate(series, target, method)


def annual(series: BimetsSeries, *, method: str | None = None) -> BimetsSeries:
    """Convert a series to yearly frequency.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    method : str, optional
        Aggregation or disaggregation method.

    Returns
    -------
    BimetsSeries
        Yearly series.

    See Also
    --------
    convert_frequency : General frequency conversion.
    BimetsSeries.to_frequency : Equivalent method form.
    """
    return convert_frequency(series, Frequency.YEARLY, method=method)


def semiannual(series: BimetsSeries, *, method: str | None = None) -> BimetsSeries:
    """Convert a series to semiannual frequency.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    method : str, optional
        Aggregation or disaggregation method.

    Returns
    -------
    BimetsSeries
        Semiannual series.
    """
    return convert_frequency(series, Frequency.SEMIANNUAL, method=method)


def quarterly(series: BimetsSeries, *, method: str | None = None) -> BimetsSeries:
    """Convert a series to quarterly frequency.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    method : str, optional
        Aggregation or disaggregation method.

    Returns
    -------
    BimetsSeries
        Quarterly series.
    """
    return convert_frequency(series, Frequency.QUARTERLY, method=method)


def monthly(series: BimetsSeries, *, method: str | None = None) -> BimetsSeries:
    """Convert a series to monthly frequency.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    method : str, optional
        Aggregation or disaggregation method.

    Returns
    -------
    BimetsSeries
        Monthly series.
    """
    return convert_frequency(series, Frequency.MONTHLY, method=method)


def daily(series: BimetsSeries, *, method: str | None = None) -> BimetsSeries:
    """Convert a series to daily frequency.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    method : str, optional
        Disaggregation method.

    Returns
    -------
    BimetsSeries
        Daily series.
    """
    return convert_frequency(series, Frequency.DAILY, method=method)


def _aggregate(
    series: BimetsSeries, target: Frequency, method: str | None
) -> BimetsSeries:
    """Aggregate for internal processing."""
    if method is None:
        raise ValueError("an aggregation method is required")
    normalized = method.lower()
    aliases = {"ave": "average", "nave": "naverage"}
    normalized = aliases.get(normalized, normalized)
    reducers: dict[str, Callable[[np.ndarray], float]] = {
        "stock": lambda values: float(values[-1]),
        "nstock": _last_non_missing,
        "sum": lambda values: float(np.sum(values)),
        "nsum": _non_missing_sum,
        "average": lambda values: float(np.mean(values)),
        "naverage": _non_missing_average,
    }
    if normalized not in reducers:
        raise ValueError(f"unknown aggregation method: {method}")

    grouped: defaultdict[YearPeriod, list[tuple[int, float]]] = defaultdict(list)
    for position, value in enumerate(series.values):
        source_period = series.period_at(position)
        target_period = _target_period(source_period, series.freq, target)
        if target_period is not None:
            grouped[target_period].append((source_period.period, float(value)))

    output_periods: list[YearPeriod] = []
    output_values: list[float] = []
    for period in sorted(grouped, key=lambda item: item.ordinal(target)):
        observations = grouped[period]
        source_periods = [item[0] for item in observations]
        if normalized in {"stock", "nstock"}:
            if not _group_reaches_end(period, source_periods, series.freq, target):
                continue
        elif not _group_is_complete(period, source_periods, series.freq, target):
            continue
        values = np.asarray([item[1] for item in observations], dtype=np.float64)
        output_periods.append(period)
        output_values.append(reducers[normalized](values))
    if not output_values:
        raise ValueError("input series does not span a complete target period")
    return BimetsSeries(
        output_values,
        start=output_periods[0],
        freq=target,
        metadata=series.metadata,
    )


def _disaggregate(
    series: BimetsSeries, target: Frequency, method: str | None
) -> BimetsSeries:
    """Disaggregate for internal processing."""
    normalized = "repeat" if method is None else method.lower()
    if normalized in {"null", "duplicate"}:
        normalized = "repeat"
    if normalized not in {"repeat", "interp_begin", "interp_center", "interp_end"}:
        raise ValueError(f"unknown disaggregation method: {method}")
    if normalized == "repeat":
        return _repeat(series, target)
    if len(series) < 2:
        raise ValueError("interpolation requires at least two observations")
    return _interpolate(series, target, normalized)


def _repeat(series: BimetsSeries, target: Frequency) -> BimetsSeries:
    """Repeat for internal processing."""
    if target is not Frequency.DAILY:
        ratio = int(target) // int(series.freq)
        first_period = (series.start.period - 1) * ratio + 1
        return BimetsSeries(
            np.repeat(series.values, ratio),
            start=(series.start.year, first_period),
            freq=target,
            metadata=series.metadata,
        )

    start_ordinal = _daily_bound(series.start, series.freq, first=True)
    end_ordinal = _daily_bound(series.end, series.freq, first=False)
    output = np.empty(end_ordinal - start_ordinal + 1, dtype=np.float64)
    for offset in range(len(output)):
        ordinal = start_ordinal + offset
        year, day_zero = divmod(ordinal, 366)
        day = day_zero + 1
        actual = _daily_date(year, day)
        source_period = (
            YearPeriod(year, int(series.freq))
            if actual is None
            else date_to_year_period(actual, series.freq)
        )
        source_position = source_period.ordinal(series.freq) - series.start.ordinal(
            series.freq
        )
        output[offset] = series.values[source_position]
    return BimetsSeries(
        output,
        start=_from_daily_ordinal(start_ordinal),
        freq=Frequency.DAILY,
        metadata=series.metadata,
    )


def _interpolate(series: BimetsSeries, target: Frequency, method: str) -> BimetsSeries:
    """Interpolate for internal processing."""
    if target is Frequency.DAILY:
        anchors = [_daily_interpolation_start(series.start, series.freq, method)]
        for index in range(len(series) - 1):
            anchors.append(
                anchors[-1]
                + _daily_interpolation_width(
                    series.period_at(index), series.freq, method
                )
            )
    else:
        ratio = int(target) // int(series.freq)
        offsets = {
            "interp_begin": 0,
            "interp_center": ratio // 2,
            "interp_end": ratio - 1,
        }
        anchors = [
            (period.year * int(target) + (period.period - 1) * ratio + offsets[method])
            for period in (series.period_at(index) for index in range(len(series)))
        ]
    values: list[float] = []
    for index in range(len(series) - 1):
        width = anchors[index + 1] - anchors[index]
        start_value = series.values[index]
        end_value = series.values[index + 1]
        values.append(float(start_value))
        values.extend(
            float(start_value + step * (end_value - start_value) / width)
            for step in range(1, width)
        )
    values.append(float(series.values[-1]))
    start = (
        _from_daily_ordinal(anchors[0])
        if target is Frequency.DAILY
        else _from_ordinal(anchors[0], target)
    )
    return BimetsSeries(
        values,
        start=start,
        freq=target,
        metadata=series.metadata,
    )


def _target_period(
    period: YearPeriod, source: Frequency, target: Frequency
) -> YearPeriod | None:
    """Map a source observation to its target-frequency period."""
    if source is Frequency.DAILY:
        converted = year_period_to_date(period, source, date_in_period="last")
        return None if converted is None else date_to_year_period(converted, target)
    ratio = int(source) // int(target)
    return YearPeriod(period.year, (period.period - 1) // ratio + 1)


def _group_reaches_end(
    target_period: YearPeriod,
    source_periods: list[int],
    source: Frequency,
    target: Frequency,
) -> bool:
    """Return whether group reaches end."""
    if source is Frequency.DAILY:
        end = year_period_to_date(target_period, target, date_in_period="last")
        assert end is not None
        return end.timetuple().tm_yday in source_periods
    ratio = int(source) // int(target)
    return target_period.period * ratio in source_periods


def _group_is_complete(
    target_period: YearPeriod,
    source_periods: list[int],
    source: Frequency,
    target: Frequency,
) -> bool:
    """Return whether group complete."""
    if source is Frequency.DAILY:
        first = year_period_to_date(target_period, target, date_in_period="first")
        last = year_period_to_date(target_period, target, date_in_period="last")
        assert first is not None and last is not None
        expected = last.timetuple().tm_yday - first.timetuple().tm_yday + 1
        return len(source_periods) == expected
    return len(source_periods) == int(source) // int(target)


def _last_non_missing(values: np.ndarray) -> float:
    """Return the last finite value, or missing if none exists."""
    valid = values[~np.isnan(values)]
    return np.nan if valid.size == 0 else float(valid[-1])


def _non_missing_sum(values: np.ndarray) -> float:
    """Sum finite values while preserving all-missing groups."""
    return np.nan if np.isnan(values).all() else float(np.nansum(values))


def _non_missing_average(values: np.ndarray) -> float:
    """Average finite values while preserving all-missing groups."""
    return np.nan if np.isnan(values).all() else float(np.nanmean(values))


def _daily_bound(period: YearPeriod, freq: Frequency, *, first: bool) -> int:
    """Convert a period boundary to a daily ordinal."""
    converted = year_period_to_date(
        period, freq, date_in_period="first" if first else "last"
    )
    assert converted is not None
    day = converted.timetuple().tm_yday
    if not first and converted.month == 12 and converted.day == 31:
        day = 366
    return period.year * 366 + day - 1


def _daily_interpolation_start(period: YearPeriod, freq: Frequency, method: str) -> int:
    """Return the daily interpolation anchor for a source period."""
    first = _daily_bound(period, freq, first=True)
    last_date = year_period_to_date(period, freq, date_in_period="last")
    assert last_date is not None
    last = period.year * 366 + last_date.timetuple().tm_yday - 1
    if method == "interp_begin":
        return first
    if method == "interp_end":
        return last
    leap = _is_leap(period.year)
    if freq is Frequency.YEARLY:
        day = 183 if leap else 182
    elif freq is Frequency.SEMIANNUAL:
        day = (92 if leap else 91) if period.period == 1 else (275 if leap else 274)
    elif freq is Frequency.QUARTERLY:
        starts = (
            46,
            136 if leap else 135,
            197 if leap else 196,
            320 if leap else 319,
        )
        day = starts[period.period - 1]
    elif freq is Frequency.MONTHLY:
        day = first - period.year * 366 + 15
    else:
        return (first + last + 1) // 2
    return period.year * 366 + day - 1


def _daily_interpolation_width(period: YearPeriod, freq: Frequency, method: str) -> int:
    """Return the number of days covered by an interpolation span."""
    if freq is Frequency.QUARTERLY:
        leap = _is_leap(period.year)
        if method == "interp_begin":
            return ((91 if leap else 90), 91, 92, (92 if leap else 93))[
                period.period - 1
            ]
        if method == "interp_center":
            return ((90 if leap else 89), 92, 92, (92 if leap else 93))[
                period.period - 1
            ]
        if period.period == 4:
            next_leap = _is_leap(period.year + 1)
            return 90 if leap else 92 if next_leap else 91
        return (91, 92, 92)[period.period - 1]
    current = _daily_interpolation_start(period, freq, method)
    following = period.shift(1, freq)
    return _daily_interpolation_start(following, freq, method) - current


def _daily_date(year: int, day: int) -> date | None:
    """Convert a year and day number to a valid date when possible."""
    candidate = date(year, 1, 1).toordinal() + day - 1
    converted = date.fromordinal(candidate)
    return converted if converted.year == year else None


def _is_leap(year: int) -> bool:
    """Return whether a year is a leap year."""
    return date(year, 12, 31).timetuple().tm_yday == 366


def _from_daily_ordinal(ordinal: int) -> YearPeriod:
    """Convert a daily ordinal to a year-period index."""
    year, day = divmod(ordinal, 366)
    return YearPeriod(year, day + 1)
