"""Shared frequency-aware tabular representation of time series."""

from __future__ import annotations

from calendar import month_abbr
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from bimets.timeseries._calendar import year_period_to_date
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod


class FrequencyTable(pd.DataFrame):  # type: ignore[misc]
    """DataFrame whose text display uses BIMETS-style numeric formatting."""

    @property
    def _constructor(self) -> type[FrequencyTable]:
        """Preserve frequency-aware display across pandas operations."""
        return FrequencyTable

    def __str__(self) -> str:
        """Render the table with compact numbers and blank missing values."""
        return self.to_string(  # type: ignore[no-any-return]
            na_rep="",
            float_format=lambda value: _format_value(float(value)),
        )

    def __repr__(self) -> str:
        """Use the user-facing table in interactive Python sessions."""
        return str(self)


def period_label(period: YearPeriod, freq: Frequency) -> str:
    """Return the BIMETS-style display label for a year-period index."""
    if freq in {Frequency.YEARLY, Frequency.SEMIANNUAL}:
        return str(period.year)
    if freq is Frequency.QUARTERLY:
        return f"{period.year} Q{period.period}"
    if freq is Frequency.MONTHLY:
        return f"{month_abbr[period.period]} {period.year}"
    boundary = year_period_to_date(period, freq)
    return str(period) if boundary is None else boundary.isoformat()


def frequency_index(
    start: YearPeriod,
    length: int,
    freq: Frequency,
) -> pd.MultiIndex:
    """Build the frequency-aware ``Date``/``Prd.`` display index."""
    periods = [start.shift(position, freq) for position in range(length)]
    return pd.MultiIndex.from_arrays(
        [
            [period_label(period, freq) for period in periods],
            [period.period for period in periods],
        ],
        names=["Date", "Prd."],
    )


def frequency_table(
    values: ArrayLike,
    *,
    start: YearPeriod,
    freq: Frequency,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return values in the shared frequency-aware display table."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    return FrequencyTable(
        array,
        index=frequency_index(start, len(array), freq),
        columns=list(columns),
    )


def format_series(
    values: ArrayLike,
    *,
    start: YearPeriod,
    end: YearPeriod,
    freq: Frequency,
    metadata: Mapping[str, Any],
) -> str:
    """Render a series using the frequency-dependent layout of R ``ts``."""
    array = np.asarray(values, dtype=np.float64)
    if freq is Frequency.QUARTERLY:
        return _wide_series(array, start, end, freq, ("Qtr1", "Qtr2", "Qtr3", "Qtr4"))
    if freq is Frequency.MONTHLY:
        return _wide_series(array, start, end, freq, tuple(month_abbr[1:]))
    return _linear_series(array, start, end, freq, metadata)


def technical_series_repr(
    values: ArrayLike,
    *,
    start: YearPeriod,
    end: YearPeriod,
    freq: Frequency,
    metadata: Mapping[str, Any],
) -> str:
    """Return a compact Python-oriented representation of a series."""
    array = np.asarray(values, dtype=np.float64)
    preview = [_format_value(value) for value in array]
    if len(preview) > 8:
        preview = [*preview[:4], "...", *preview[-3:]]
    values_text = f"[{', '.join(preview)}]"
    return (
        f"BimetsSeries(values={values_text}, length={len(array)}, "
        f"start=({start.year}, {start.period}), end=({end.year}, {end.period}), "
        f"freq={int(freq)}, metadata={dict(metadata)!r})"
    )


def _wide_series(
    values: np.ndarray,
    start: YearPeriod,
    end: YearPeriod,
    freq: Frequency,
    columns: Sequence[str],
) -> str:
    """Arrange quarterly or monthly observations by year and cycle."""
    years = list(range(start.year, end.year + 1))
    rows = np.full((len(years), len(columns)), np.nan, dtype=np.float64)
    for position, value in enumerate(values):
        period = start.shift(position, freq)
        rows[period.year - start.year, period.period - 1] = value
    formatted = [
        ["" if np.isnan(value) else _format_value(value) for value in row]
        for row in rows
    ]
    widths = [
        max(len(name), *(len(row[index]) for row in formatted))
        for index, name in enumerate(columns)
    ]
    year_width = max(len(str(year)) for year in years)
    lines = [
        " " * (year_width + 1)
        + " ".join(
            name.rjust(width) for name, width in zip(columns, widths, strict=True)
        )
    ]
    lines.extend(
        str(year).rjust(year_width)
        + " "
        + " ".join(value.rjust(width) for value, width in zip(row, widths, strict=True))
        for year, row in zip(years, formatted, strict=True)
    )
    return "\n".join(line.rstrip() for line in lines)


def _linear_series(
    values: np.ndarray,
    start: YearPeriod,
    end: YearPeriod,
    freq: Frequency,
    metadata: Mapping[str, Any],
) -> str:
    """Render the vector-style ``print.ts`` layout used by other frequencies."""
    start_text = _r_period(start, freq)
    end_text = _r_period(end, freq)
    lines = [
        "Time Series:",
        f"Start = {start_text}",
        f"End = {end_text}",
        f"Frequency = {int(freq)}",
        *_vector_lines(values),
    ]
    labels = {
        "source": "Source",
        "title": "Title",
        "units": "Units",
        "scale_factor": "ScaleFac",
    }
    for name, value in metadata.items():
        lines.extend((f'attr(,"{labels.get(name, name)}")', f"[1] {value}"))
    return "\n".join(lines)


def _vector_lines(values: np.ndarray, width: int = 80) -> list[str]:
    """Format a numeric vector with R-style one-based line prefixes."""
    lines: list[str] = []
    current = ""
    first_position = 1
    for position, value in enumerate(values, start=1):
        token = _format_value(value)
        prefix = f"[{first_position}] " if not current else ""
        candidate = f"{current} {token}" if current else f"{prefix}{token}"
        if current and len(candidate) > width:
            lines.append(current)
            first_position = position
            current = f"[{first_position}] {token}"
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _r_period(period: YearPeriod, freq: Frequency) -> str:
    """Format a start or end period like R's ``print.ts`` header."""
    if freq is Frequency.YEARLY:
        return str(period.year)
    return f"c({period.year}, {period.period})"


def _format_value(value: float) -> str:
    """Format one numeric value using compact R-like conventions."""
    if np.isnan(value):
        return "NA"
    if np.isposinf(value):
        return "Inf"
    if np.isneginf(value):
        return "-Inf"
    return format(value, ".7g")
