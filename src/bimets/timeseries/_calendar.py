"""Calendar conversion for BIMETS year-period indexes."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod
from bimets.timeseries._series import BimetsSeries

DateInPeriod = Literal["first", "last"]


def date_to_year_period(value: date, freq: int | str | Frequency) -> YearPeriod:
    """Convert a calendar date to a BIMETS year-period.

    Parameters
    ----------
    value : datetime.date
        Calendar date to convert.
    freq : int, str, or Frequency
        Target BIMETS frequency.

    Returns
    -------
    YearPeriod
        Year and one-based period containing the date.

    Examples
    --------
    >>> from datetime import date
    >>> from bimets import date_to_year_period
    >>> date_to_year_period(date(2020, 5, 20), "Q")
    YearPeriod(year=2020, period=2)
    """
    if not isinstance(value, date):
        raise TypeError("value must be a date")
    parsed = Frequency.parse(freq)
    day = value.day
    month = value.month
    if parsed is Frequency.DAILY:
        period = value.timetuple().tm_yday
    elif parsed is Frequency.WEEKLY:
        period = min(53, (value.timetuple().tm_yday - 1) // 7 + 1)
    elif int(parsed) <= 12:
        period = min(int(parsed), (month * int(parsed) - 1) // 12 + 1)
    else:
        periods_per_month = int(parsed) // 12
        if periods_per_month == 2:
            within_month = 1 if day <= 15 else 2
        else:
            within_month = 1 if day <= 10 else 2 if day <= 20 else 3
        period = (month - 1) * periods_per_month + within_month
    return YearPeriod(value.year, period)


def year_period_to_date(
    value: YearPeriod | tuple[int, int],
    freq: int | str | Frequency,
    *,
    date_in_period: DateInPeriod = "last",
) -> date | None:
    """Convert a BIMETS year-period to its first or last calendar date.

    Parameters
    ----------
    value : YearPeriod or tuple of int
        Year and one-based period to convert.
    freq : int, str, or Frequency
        Frequency used to interpret the period.
    date_in_period : {"first", "last"}, default="last"
        Select the first or last date in the period.

    Returns
    -------
    datetime.date or None
        Requested calendar boundary. Daily period 366 has no representation
        in a non-leap year and returns ``None``.

    Raises
    ------
    ValueError
        If the period exceeds the frequency or ``date_in_period`` is invalid.

    Daily period 366 has no calendar representation in a non-leap year and is
    therefore returned as ``None``.

    Examples
    --------
    >>> from bimets import year_period_to_date
    >>> year_period_to_date((2020, 2), "Q")
    datetime.date(2020, 6, 30)
    """
    parsed = Frequency.parse(freq)
    if date_in_period not in {"first", "last"}:
        raise ValueError("date_in_period must be first or last")
    period = value if isinstance(value, YearPeriod) else YearPeriod(*value)
    if period.period > int(parsed):
        raise ValueError("period exceeds frequency")
    year = period.year

    if parsed is Frequency.DAILY:
        candidate = date(year, 1, 1) + timedelta(days=period.period - 1)
        return candidate if candidate.year == year else None
    if parsed is Frequency.WEEKLY:
        day_number = (period.period - 1) * 7 + 1
        if date_in_period == "last":
            day_number += 6
        last_day = _days_in_year(year)
        return date(year, 1, 1) + timedelta(days=min(day_number, last_day) - 1)
    if int(parsed) <= 12:
        start_month = (period.period - 1) * 12 // int(parsed) + 1
        end_month = period.period * 12 // int(parsed)
        return (
            date(year, start_month, 1)
            if date_in_period == "first"
            else _last_day_of_month(year, end_month)
        )

    periods_per_month = int(parsed) // 12
    month, part = divmod(period.period - 1, periods_per_month)
    month += 1
    first_days = (1, 16) if periods_per_month == 2 else (1, 11, 21)
    last_days = (15, 31) if periods_per_month == 2 else (10, 20, 31)
    if date_in_period == "first":
        return date(year, month, first_days[part])
    return date(year, month, min(last_days[part], _last_day_of_month(year, month).day))


@overload
def get_dates(
    series: BimetsSeries,
    index: int,
    *,
    date_in_period: DateInPeriod = "last",
    format: None = None,
) -> date | None: ...


@overload
def get_dates(
    series: BimetsSeries,
    index: int,
    *,
    date_in_period: DateInPeriod = "last",
    format: str,
) -> str | None: ...


@overload
def get_dates(
    series: BimetsSeries,
    index: Sequence[int] | None = None,
    *,
    date_in_period: DateInPeriod = "last",
    format: None = None,
) -> list[date | None]: ...


@overload
def get_dates(
    series: BimetsSeries,
    index: Sequence[int] | None = None,
    *,
    date_in_period: DateInPeriod = "last",
    format: str,
) -> list[str | None]: ...


def get_dates(
    series: BimetsSeries,
    index: int | Sequence[int] | None = None,
    *,
    date_in_period: DateInPeriod = "last",
    format: str | None = None,
) -> Any:
    """Return calendar dates for zero-based observation indexes.

    Parameters
    ----------
    series : BimetsSeries
        Series whose indexes are converted.
    index : int, sequence of int, or None, optional
        Zero-based indexes. A scalar produces a scalar result; ``None``
        selects every observation.
    date_in_period : {"first", "last"}, default="last"
        Select period starts or ends.
    format : str, optional
        ``datetime.date.strftime`` format. ``%q`` is additionally supported
        as the calendar quarter number.

    Returns
    -------
    datetime.date, str, None, or list
        Calendar values matching the shape requested by ``index``.

    Examples
    --------
    >>> from bimets import get_dates, timeseries
    >>> series = timeseries([1, 2], start=(2020, 1), freq="Q")
    >>> get_dates(series)
    [datetime.date(2020, 3, 31), datetime.date(2020, 6, 30)]
    >>> get_dates(series, 1, format="%Y-Q%q")
    '2020-Q2'
    """
    scalar = isinstance(index, int)
    selected: Sequence[int]
    if isinstance(index, int):
        selected = [index]
    elif index is None:
        selected = range(len(series))
    else:
        selected = index
    output: list[date | str | None] = []
    for position in selected:
        period = series.period_at(position)
        converted = year_period_to_date(
            period, series.freq, date_in_period=date_in_period
        )
        output.append(_format_date(converted, format, series.freq))
    return output[0] if scalar else output


@overload
def get_year_periods(series: BimetsSeries) -> list[YearPeriod]: ...


@overload
def get_year_periods(
    series: BimetsSeries,
    *,
    years: str,
    periods: str,
    join: Literal[False] = False,
) -> dict[str, NDArray[np.int64]]: ...


@overload
def get_year_periods(
    series: BimetsSeries,
    *,
    years: str | None = None,
    periods: str | None = None,
    join: Literal[True],
) -> NDArray[np.int64]: ...


def get_year_periods(
    series: BimetsSeries,
    *,
    years: str | None = None,
    periods: str | None = None,
    join: bool = False,
) -> list[YearPeriod] | dict[str, NDArray[np.int64]] | NDArray[np.int64]:
    """Return the year-period index of every observation.

    Parameters
    ----------
    series : BimetsSeries
        Series to inspect.
    years, periods : str, optional
        Names for separate integer arrays, corresponding to BIMETS R ``YEARS``
        and ``PERIODS``. Supply both to request a mapping instead of typed
        ``YearPeriod`` values.
    join : bool, default=False
        Return an ``(n, 2)`` integer array of year-period pairs.

    Returns
    -------
    list of YearPeriod, dict, or numpy.ndarray
        Typed indexes by default, named arrays when ``years`` and ``periods``
        are supplied, or a joined matrix when ``join`` is true.

    Raises
    ------
    TypeError
        If names are not strings or ``join`` is not boolean.
    ValueError
        If only one output name is supplied or names are empty or equal.
    """
    if not isinstance(join, bool):
        raise TypeError("join must be boolean")
    if (years is None) != (periods is None):
        raise ValueError("years and periods must be supplied together")
    if years is not None and (
        not isinstance(years, str) or not isinstance(periods, str)
    ):
        raise TypeError("years and periods must be strings")
    if years is not None and (not years or not periods or years == periods):
        raise ValueError("years and periods must be non-empty distinct names")

    indexes = [series.period_at(position) for position in range(len(series))]
    joined = np.asarray(
        [(period.year, period.period) for period in indexes], dtype=np.int64
    )
    if join:
        return joined
    if years is not None and periods is not None:
        return {years: joined[:, 0], periods: joined[:, 1]}
    return indexes


def _format_date(
    value: date | None, format_string: str | None, freq: Frequency
) -> date | str | None:
    """Format a date value for external representation."""
    if value is None or format_string is None:
        return value
    if "%q" in format_string:
        quarter = (value.month - 1) // 3 + 1
        format_string = format_string.replace("%q", str(quarter))
    if "%j" in format_string and freq is Frequency.DAILY:
        return value.strftime(format_string)
    return value.strftime(format_string)


def _last_day_of_month(year: int, month: int) -> date:
    """Return the final calendar date of a month."""
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _days_in_year(year: int) -> int:
    """Return the number of days in a calendar year."""
    return (date(year + 1, 1, 1) - date(year, 1, 1)).days
