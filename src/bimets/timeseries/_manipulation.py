"""Time-series manipulation operations derived from BIMETS semantics."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod, _from_ordinal, normalize_year_period
from bimets.timeseries._series import BimetsSeries, _require_positive_integer


def cumulative_sum(
    series: BimetsSeries,
    *,
    mode: str | None = None,
    skip_missing: bool = False,
    start: YearPeriod | tuple[int, int] | None = None,
    end: YearPeriod | tuple[int, int] | None = None,
) -> BimetsSeries:
    """Return cumulative sums, matching ``CUMSUM`` for time-series inputs."""
    selected = _cumulative_range(series, start, end)
    groups = _cumulative_groups(selected, mode)
    values = _cumulative(selected.values, np.add, groups, skip_missing)
    return BimetsSeries(values, start=selected.start, freq=selected.freq)


def cumulative_product(
    series: BimetsSeries,
    *,
    skip_missing: bool = False,
    start: YearPeriod | tuple[int, int] | None = None,
    end: YearPeriod | tuple[int, int] | None = None,
) -> BimetsSeries:
    """Return cumulative products, matching ``CUMPROD``."""
    selected = _cumulative_range(series, start, end)
    values = _cumulative(selected.values, np.multiply, None, skip_missing)
    return BimetsSeries(values, start=selected.start, freq=selected.freq)


def _cumulative_range(
    series: BimetsSeries,
    start: YearPeriod | tuple[int, int] | None,
    end: YearPeriod | tuple[int, int] | None,
) -> BimetsSeries:
    """Project an optional cumulative-operation range."""
    if start is None and end is None:
        return series
    return series.project(start or series.start, end or series.end)


def moving_average(
    series: BimetsSeries,
    window: int,
    *,
    direction: str = "back",
    skip_missing: bool = False,
) -> BimetsSeries:
    """Return averages over complete moving windows."""
    return _moving(series, window, direction, skip_missing, average=True)


def moving_sum(
    series: BimetsSeries,
    window: int,
    *,
    direction: str = "back",
    skip_missing: bool = False,
) -> BimetsSeries:
    """Return totals over complete moving windows."""
    return _moving(series, window, direction, skip_missing, average=False)


def index_number(series: BimetsSeries, base_year: int) -> BimetsSeries:
    """Rebase a series to 100 at the average of ``base_year``."""
    if isinstance(base_year, bool) or not isinstance(base_year, int) or base_year < 0:
        raise ValueError("base_year must be a non-negative integer")
    positions = np.array(
        [series.period_at(index).year == base_year for index in range(len(series))]
    )
    if not positions.any():
        raise ValueError("series is not defined in the base year")
    base_values = series.values[positions]
    if np.isnan(base_values).any():
        raise ValueError("base year contains missing values")
    base = float(np.mean(base_values))
    if base == 0:
        raise ValueError("base year average is zero")
    return BimetsSeries(
        series.values * (100.0 / base),
        start=series.start,
        freq=series.freq,
    )


def merge(
    *series: BimetsSeries,
    method: str | None = None,
    skip_missing: bool = True,
) -> BimetsSeries:
    """Merge series over their union range, matching ``TSMERGE``."""
    if not series:
        raise ValueError("at least one series is required")
    freq = series[0].freq
    if any(item.freq != freq for item in series[1:]):
        raise ValueError("all series must have the same frequency")
    operation = None if method is None else method.lower()
    if operation not in {None, "average", "ave", "sum", "max", "min"}:
        raise ValueError(f"unknown merge method: {method}")

    start_ordinal = min(item.start.ordinal(freq) for item in series)
    end_ordinal = max(item.end.ordinal(freq) for item in series)
    matrix = np.full((len(series), end_ordinal - start_ordinal + 1), np.nan)
    covered = np.zeros(matrix.shape, dtype=bool)
    for row, item in enumerate(series):
        offset = item.start.ordinal(freq) - start_ordinal
        matrix[row, offset : offset + len(item)] = item.values
        covered[row, offset : offset + len(item)] = True

    output = np.full(matrix.shape[1], np.nan)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        if operation is None:
            for row, value in enumerate(values):
                if covered[row, column] and (not skip_missing or not np.isnan(value)):
                    output[column] = value
                    break
                if not skip_missing and not covered[row, column]:
                    break
            continue
        if not skip_missing and (
            not covered[:, column].all() or np.isnan(values).any()
        ):
            continue
        active = values[~np.isnan(values)]
        if active.size == 0:
            continue
        if operation in {"average", "ave"}:
            output[column] = np.mean(active)
        elif operation == "sum":
            output[column] = np.sum(active)
        elif operation == "max":
            output[column] = np.max(active)
        else:
            output[column] = np.min(active)
    return BimetsSeries(
        output,
        start=_from_ordinal(start_ordinal, freq),
        freq=freq,
    )


def join(
    first: BimetsSeries,
    second: BimetsSeries,
    *,
    join_period: YearPeriod | tuple[int, int] | None = None,
    allow_gap: bool = False,
) -> BimetsSeries:
    """Use ``first`` before a join period and ``second`` from that period on."""
    if first.freq != second.freq:
        raise ValueError("series must have the same frequency")
    freq = first.freq
    switch = (
        second.start
        if join_period is None
        else normalize_year_period(join_period, freq)
    )
    switch_ordinal = switch.ordinal(freq)
    if not second.start.ordinal(freq) <= switch_ordinal <= second.end.ordinal(freq):
        raise ValueError("join period is outside the second series")
    if second.start.ordinal(freq) - first.end.ordinal(freq) > 1 and not allow_gap:
        raise ValueError("there is a gap between the series")

    start_ordinal = min(first.start.ordinal(freq), switch_ordinal)
    end_ordinal = second.end.ordinal(freq)
    output = np.full(end_ordinal - start_ordinal + 1, np.nan)
    first_end = min(first.end.ordinal(freq), switch_ordinal - 1)
    if first.start.ordinal(freq) <= first_end:
        count = first_end - first.start.ordinal(freq) + 1
        offset = first.start.ordinal(freq) - start_ordinal
        output[offset : offset + count] = first.values[:count]
    second_offset = switch_ordinal - second.start.ordinal(freq)
    target_offset = switch_ordinal - start_ordinal
    output[target_offset:] = second.values[second_offset:]
    return BimetsSeries(
        output,
        start=_from_ordinal(start_ordinal, freq),
        freq=freq,
    )


def extend(
    series: BimetsSeries,
    *,
    back_to: YearPeriod | tuple[int, int] | None = None,
    up_to: YearPeriod | tuple[int, int] | None = None,
    mode: str = "growth",
    factor: float | None = None,
) -> BimetsSeries:
    """Extend a series using the modes supported by BIMETS ``TSEXTEND``."""
    extension_mode = mode.lower()
    valid_modes = {
        "missing",
        "zero",
        "constant",
        "mean4",
        "growth",
        "growth4",
        "linear",
        "quadratic",
        "myconst",
        "myrate",
    }
    if extension_mode not in valid_modes:
        raise ValueError(f"unknown extension mode: {mode}")
    if extension_mode in {"myconst", "myrate"} and factor is None:
        raise ValueError(f"factor is required for {extension_mode} mode")

    freq = series.freq
    requested_start = (
        series.start if back_to is None else normalize_year_period(back_to, freq)
    )
    requested_end = series.end if up_to is None else normalize_year_period(up_to, freq)
    output_start = min(requested_start, series.start, key=lambda x: x.ordinal(freq))
    output_end = max(requested_end, series.end, key=lambda x: x.ordinal(freq))
    output = series.project(output_start, output_end, extend=True).values.copy()
    offset = series.start.ordinal(freq) - output_start.ordinal(freq)
    finite = np.flatnonzero(~np.isnan(series.values))
    if extension_mode == "zero":
        output[np.isnan(output)] = 0.0
        return BimetsSeries(output, start=output_start, freq=freq)
    if finite.size == 0:
        return BimetsSeries(output, start=output_start, freq=freq)

    if requested_start.ordinal(freq) < series.start.ordinal(freq):
        boundary = offset + int(finite[0])
        _fill_extension(
            output,
            series.values,
            range(boundary - 1, -1, -1),
            boundary,
            extension_mode,
            factor,
            backwards=True,
        )
    if requested_end.ordinal(freq) > series.end.ordinal(freq):
        boundary = offset + int(finite[-1])
        _fill_extension(
            output,
            series.values,
            range(boundary + 1, len(output)),
            boundary,
            extension_mode,
            factor,
            backwards=False,
        )
    return BimetsSeries(output, start=output_start, freq=freq)


def _cumulative(
    source: NDArray[np.float64],
    operation: Callable[[float, float], np.float64],
    groups: list[object] | None,
    skip_missing: bool,
) -> NDArray[np.float64]:
    """Apply a cumulative transformation, optionally within groups."""
    output = source.copy()
    for index in range(1, len(output)):
        if groups is not None and groups[index] != groups[index - 1]:
            continue
        previous, current = output[index - 1], output[index]
        if skip_missing:
            if not np.isnan(previous):
                output[index] = (
                    previous if np.isnan(current) else operation(current, previous)
                )
        else:
            output[index] = operation(current, previous)
    return output


def _cumulative_groups(series: BimetsSeries, mode: str | None) -> list[object] | None:
    """Build grouping keys for cumulative transformations."""
    if mode is None:
        return None
    normalized = mode.lower()
    periods = [series.period_at(index) for index in range(len(series))]
    if normalized in {"year", "yearly"}:
        return [period.year for period in periods]
    if normalized not in {"month", "monthly"}:
        raise ValueError("mode must be yearly, monthly, or None")
    return [(period.year, _period_month(period, series.freq)) for period in periods]


def _period_month(period: YearPeriod, freq: Frequency) -> int:
    """Map a period index to its representative month."""
    periods_per_year = int(freq)
    if periods_per_year <= 36 and (
        12 % periods_per_year == 0 or periods_per_year % 12 == 0
    ):
        return min(12, (period.period * 12 - 1) // periods_per_year + 1)
    days_in_year = (date(period.year + 1, 1, 1) - date(period.year, 1, 1)).days
    day_number = min(
        days_in_year,
        (period.period * days_in_year + periods_per_year - 1) // periods_per_year,
    )
    return (date(period.year, 1, 1) + timedelta(days=day_number - 1)).month


def _moving(
    series: BimetsSeries,
    window: int,
    direction: str,
    skip_missing: bool,
    *,
    average: bool,
) -> BimetsSeries:
    """Apply a trailing moving-window transformation."""
    _require_positive_integer(window, "window")
    if window > len(series):
        raise ValueError("series has fewer observations than the window")
    normalized_direction = direction.lower()
    shifts = {"ahead": 0, "center": window // 2, "back": window - 1}
    if normalized_direction not in shifts:
        raise ValueError("direction must be ahead, center, or back")
    windows = np.lib.stride_tricks.sliding_window_view(series.values, window)
    if skip_missing:
        values = np.nansum(windows, axis=1)
        counts = np.count_nonzero(~np.isnan(windows), axis=1)
        if average:
            values = np.divide(
                values,
                counts,
                out=np.full(values.shape, np.nan),
                where=counts != 0,
            )
        else:
            values[counts == 0] = np.nan
    else:
        values = np.mean(windows, axis=1) if average else np.sum(windows, axis=1)
    return BimetsSeries(
        values,
        start=series.start.shift(shifts[normalized_direction], series.freq),
        freq=series.freq,
    )


def _fill_extension(
    output: NDArray[np.float64],
    source: NDArray[np.float64],
    positions: range,
    boundary: int,
    mode: str,
    factor: float | None,
    *,
    backwards: bool,
) -> None:
    """Fill values introduced while extending a series range."""
    if mode == "missing":
        output[list(positions)] = np.nan
        return
    if mode in {"zero", "myconst"}:
        output[list(positions)] = 0.0 if mode == "zero" else float(factor)  # type: ignore[arg-type]
        return
    finite = np.flatnonzero(~np.isnan(source))
    if backwards:
        first = int(finite[0])
        edge = source[first : first + 8]
    else:
        last = int(finite[-1])
        edge = source[max(0, last - 7) : last + 1]
    if mode == "constant":
        output[list(positions)] = edge[0] if backwards else edge[-1]
        return
    if mode == "mean4":
        selected = edge[:4] if backwards else edge[-4:]
        value = float(np.mean(selected)) if selected.size == 4 else np.nan
        output[list(positions)] = value
        return

    if mode == "myrate":
        multiplier = float(factor)  # type: ignore[arg-type]
    elif mode == "growth":
        selected = edge[:2] if backwards else edge[-2:]
        multiplier = (
            (
                float(selected[0] / selected[1])
                if backwards
                else float(selected[-1] / selected[-2])
            )
            if selected.size == 2
            else np.nan
        )
    elif mode == "growth4":
        if edge.size < 8:
            multiplier = np.nan
        elif backwards:
            multiplier = float((np.sum(edge[:4]) / np.sum(edge[4:8])) ** 0.25)
        else:
            multiplier = float((np.sum(edge[-4:]) / np.sum(edge[-8:-4])) ** 0.25)
    elif mode == "linear":
        selected = edge[:2] if backwards else edge[-2:]
        difference = (
            (
                float(selected[0] - selected[1])
                if backwards
                else float(selected[-1] - selected[-2])
            )
            if selected.size == 2
            else np.nan
        )
        previous = boundary
        for position in positions:
            output[position] = output[previous] + difference
            previous = position
        return
    elif mode == "quadratic":
        selected = edge[:3] if backwards else edge[-3:]
        if selected.size < 3:
            second = slope = np.nan
        elif backwards:
            a, b, c = selected
            second = float((a - 2 * b + c) / 2)
            slope = float((3 * a - 4 * b + c) / 2)
        else:
            a, b, c = selected[-1], selected[-2], selected[-3]
            second = float((a - 2 * b + c) / 2)
            slope = float((3 * a - 4 * b + c) / 2)
        for distance, position in enumerate(positions, start=1):
            output[position] = output[boundary] + (slope + second * distance) * distance
        return
    else:
        raise AssertionError("unreachable extension mode")

    previous = boundary
    for position in positions:
        output[position] = output[previous] * multiplier
        previous = position
