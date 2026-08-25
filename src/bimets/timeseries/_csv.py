"""CSV exchange for collections of BIMETS time series."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from bimets.timeseries._calendar import date_to_year_period, get_dates
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod
from bimets.timeseries._inspection import get_range
from bimets.timeseries._series import BimetsSeries

SeriesCollection = Mapping[str, BimetsSeries] | Sequence[BimetsSeries] | BimetsSeries


def bimets_to_csv(
    series: SeriesCollection,
    path: str | Path,
    *,
    merged: bool = False,
    delimiter: str = ",",
    decimal_separator: str = ".",
    date_format: str = "%Y/%m/%d",
    missing: str = "NA",
    overwrite: bool = False,
    append: bool = False,
    separator_metadata: bool = True,
    freq_header_prefix: str = "FREQ_",
    time_range: tuple[int, int, int, int] | None = None,
    name_metadata_key: str | None = None,
    title_lines: str | Sequence[str] | None = None,
    plain_table: bool = False,
) -> Path:
    """Export one or more series to a BIMETS-compatible CSV table.

    Parameters
    ----------
    series : BimetsSeries, sequence, or mapping
        Series to export. Mapping keys become column names.
    path : str or pathlib.Path
        Destination file.
    merged : bool, default=False
        Write one shared date column. This requires a common frequency. The
        default paired layout writes one date/value pair per series.
    delimiter : str, default=","
        CSV field delimiter.
    decimal_separator : str, default="."
        Decimal character used for numeric values.
    date_format : str, default="%Y/%m/%d"
        ``strftime`` format used for dates. ``%q`` is available as a BIMETS
        quarter placeholder.
    missing : str, default="NA"
        Text representation of missing observations.
    overwrite : bool, default=False
        Permit replacement of an existing destination.
    append : bool, default=False
        Append a complete table to an existing file. It is mutually exclusive
        with ``overwrite``.
    separator_metadata : bool, default=True
        Write an initial ``sep=`` declaration.
    freq_header_prefix : str, default="FREQ_"
        Prefix written before each numeric frequency, corresponding to
        ``freqHeaderPrefix`` in BIMETS R.
    time_range : tuple of four int, optional
        Inclusive ``(start_year, start_period, end_year, end_period)`` exported
        range, corresponding to ``TSRANGE``.
    name_metadata_key : str, optional
        Metadata field used to name unnamed sequence inputs, corresponding to
        ``attributeOfNames``. Mapping keys continue to take precedence.
    title_lines : str or sequence of str, optional
        Preamble lines written before the CSV header.
    plain_table : bool, default=False
        Write a merged semicolon-delimited table headed by ``DATE`` without a
        ``sep=`` declaration. This matches BIMETS R ``plainTable``.

    Returns
    -------
    pathlib.Path
        Destination path.

    Raises
    ------
    FileExistsError
        If the destination exists and ``overwrite`` is false.
    ValueError
        If options are inconsistent, no series is supplied, the export range
        is invalid, or merged series have mixed frequencies.

    See Also
    --------
    csv_to_bimets : Import a compatible table.
    """
    destination = Path(path)
    _validate_frequency_prefix(freq_header_prefix)
    if overwrite and append:
        raise ValueError("overwrite and append are mutually exclusive")
    if destination.exists() and not overwrite and not append:
        raise FileExistsError(f"CSV file already exists: {destination}")
    if name_metadata_key is not None and (
        not isinstance(name_metadata_key, str) or not name_metadata_key
    ):
        raise ValueError("name_metadata_key must be a non-empty string")
    names, values = _normalize_collection(series, name_metadata_key)
    if not values:
        raise ValueError("at least one series is required")
    effective_merged = merged or plain_table
    values = _project_collection(values, time_range, merged=effective_merged)
    effective_delimiter = ";" if plain_table else delimiter
    effective_decimal_separator = "." if plain_table else decimal_separator
    with destination.open(
        "a" if append else "w", newline="", encoding="utf-8"
    ) as stream:
        if separator_metadata and not plain_table:
            stream.write(f"sep={effective_delimiter}\n")
        writer = csv.writer(stream, delimiter=effective_delimiter)
        if title_lines is not None:
            lines = [title_lines] if isinstance(title_lines, str) else list(title_lines)
            for line in lines:
                writer.writerow([line, ""])
        if effective_merged:
            _write_merged(
                writer,
                names,
                values,
                date_format,
                missing,
                effective_decimal_separator,
                freq_header_prefix,
                plain_table,
            )
        else:
            _write_paired(
                writer,
                names,
                values,
                date_format,
                missing,
                effective_decimal_separator,
                freq_header_prefix,
            )
    return destination


def csv_to_bimets(
    path: str | Path,
    *,
    merged: bool = False,
    delimiter: str = ",",
    decimal_separator: str = ".",
    date_format: str = "%Y/%m/%d",
    missing: str = "NA",
    skip_lines: int = 0,
    freq_header_prefix: str | None = "FREQ_",
) -> dict[str, BimetsSeries]:
    """Import series from a BIMETS-compatible CSV table.

    Parameters
    ----------
    path : str or pathlib.Path
        Source file.
    merged : bool, default=False
        Read the shared-date layout instead of paired date/value columns.
    delimiter : str, default=","
        Fallback delimiter when the file has no ``sep=`` declaration.
    decimal_separator : str, default="."
        Decimal character used by numeric values.
    date_format : str, default="%Y/%m/%d"
        Format used to parse calendar dates. ``%q`` is available as a BIMETS
        quarter placeholder.
    missing : str, default="NA"
        Text representation interpreted as a missing observation.
    skip_lines : int, default=0
        Additional lines skipped before the table header, after an optional
        ``sep=`` declaration. This corresponds to ``skipLines`` in BIMETS R.
    freq_header_prefix : str or None, default="FREQ_"
        Prefix identifying an explicit frequency in a header. If the prefix is
        absent, or this argument is ``None``, infer frequency from the dates as
        BIMETS R does.

    Returns
    -------
    dict of str to BimetsSeries
        Imported series in file order.

    Raises
    ------
    ValueError
        If the file is empty or its headers, frequencies, dates, or values are
        invalid.
    TypeError
        If ``skip_lines`` is not an integer.

    See Also
    --------
    bimets_to_csv : Export compatible tables.
    """
    source = Path(path)
    if isinstance(skip_lines, bool) or not isinstance(skip_lines, int):
        raise TypeError("skip_lines must be an integer")
    if skip_lines < 0:
        raise ValueError("skip_lines must be non-negative")
    if freq_header_prefix is not None:
        _validate_frequency_prefix(freq_header_prefix)
    with source.open(newline="", encoding="utf-8") as stream:
        first_line = stream.readline().rstrip("\r\n")
        if first_line.startswith("sep=") and len(first_line) > 4:
            effective_delimiter = first_line[4:]
        else:
            effective_delimiter = delimiter
            stream.seek(0)
        for _ in range(skip_lines):
            if stream.readline() == "":
                raise ValueError("CSV file has no header after skipped lines")
        reader = csv.reader(stream, delimiter=effective_delimiter)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("CSV file is empty") from error
        rows = list(reader)
    return (
        _read_merged(
            header,
            rows,
            date_format,
            missing,
            decimal_separator,
            freq_header_prefix,
        )
        if merged
        else _read_paired(
            header,
            rows,
            date_format,
            missing,
            decimal_separator,
            freq_header_prefix,
        )
    )


def _normalize_collection(
    collection: SeriesCollection,
    name_metadata_key: str | None = None,
) -> tuple[list[str], list[BimetsSeries]]:
    """Normalize collection for internal processing."""
    if isinstance(collection, BimetsSeries):
        return [_series_name(collection, 1, name_metadata_key)], [collection]
    if isinstance(collection, Mapping):
        return list(collection), list(collection.values())
    values = list(collection)
    return [
        _series_name(item, index + 1, name_metadata_key)
        for index, item in enumerate(values)
    ], values


def _series_name(series: BimetsSeries, index: int, metadata_key: str | None) -> str:
    """Resolve the exported name of an unnamed series."""
    selected = series.metadata.get(metadata_key) if metadata_key is not None else None
    if selected is not None:
        return str(selected)
    return str(series.metadata.get("title") or f"series_{index}")


def _project_collection(
    series: list[BimetsSeries],
    time_range: tuple[int, int, int, int] | None,
    *,
    merged: bool,
) -> list[BimetsSeries]:
    """Apply an optional BIMETS export range to every series."""
    if time_range is None:
        return series
    if (
        not isinstance(time_range, tuple)
        or len(time_range) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in time_range
        )
    ):
        raise ValueError("time_range must contain four integers")
    requested_start = (time_range[0], time_range[1])
    requested_end = (time_range[2], time_range[3])
    if not merged:
        return [item.project(requested_start, requested_end) for item in series]

    common = get_range(*series, kind="outer")
    assert common is not None
    freq = series[0].freq
    normalized_start = YearPeriod.normalize(*requested_start, freq)
    normalized_end = YearPeriod.normalize(*requested_end, freq)
    selected_start = max(
        normalized_start, common[0], key=lambda period: period.ordinal(freq)
    )
    selected_end = min(
        normalized_end, common[1], key=lambda period: period.ordinal(freq)
    )
    if selected_end.ordinal(freq) < selected_start.ordinal(freq):
        raise ValueError("time_range does not overlap the merged series range")
    return [item.project(selected_start, selected_end, extend=True) for item in series]


def _write_paired(
    writer: Any,
    names: list[str],
    series: list[BimetsSeries],
    date_format: str,
    missing: str,
    decimal_separator: str,
    freq_header_prefix: str,
) -> None:
    """Write paired for internal processing."""
    header: list[str] = []
    columns: list[list[str]] = []
    for name, item in zip(names, series, strict=True):
        header.extend([name, f"{freq_header_prefix}{int(item.freq)}"])
        dates = get_dates(item, format=date_format)
        assert isinstance(dates, list)
        periods = [item.period_at(position) for position in range(len(item))]
        columns.append(
            [
                value
                for pair in zip(
                    [
                        _format_date(value, period)
                        for value, period in zip(dates, periods, strict=True)
                    ],
                    [
                        _format_value(value, missing, decimal_separator)
                        for value in item.values
                    ],
                    strict=True,
                )
                for value in pair
            ]
        )
    writer.writerow(header)
    width = max(len(column) for column in columns)
    for row in range(0, width, 2):
        output: list[str] = []
        for column in columns:
            output.extend(column[row : row + 2] if row < len(column) else ["", ""])
        writer.writerow(output)


def _write_merged(
    writer: Any,
    names: list[str],
    series: list[BimetsSeries],
    date_format: str,
    missing: str,
    decimal_separator: str,
    freq_header_prefix: str,
    plain_table: bool,
) -> None:
    """Write merged for internal processing."""
    freq = series[0].freq
    if any(item.freq != freq for item in series[1:]):
        raise ValueError("merged CSV export requires a common frequency")
    common = get_range(*series, kind="outer")
    assert common is not None
    template = series[0].project(*common, extend=True)
    dates = get_dates(template, format=date_format)
    assert isinstance(dates, list)
    periods = [template.period_at(position) for position in range(len(template))]
    first_header = "DATE" if plain_table else f"{freq_header_prefix}{int(freq)}"
    writer.writerow([first_header, *names])
    projected = [item.project(*common, extend=True).values for item in series]
    for position, current_date in enumerate(dates):
        writer.writerow(
            [
                _format_date(current_date, periods[position]),
                *[
                    _format_value(values[position], missing, decimal_separator)
                    for values in projected
                ],
            ]
        )


def _read_paired(
    header: list[str],
    rows: list[list[str]],
    date_format: str,
    missing: str,
    decimal_separator: str,
    freq_header_prefix: str | None,
) -> dict[str, BimetsSeries]:
    """Read paired for internal processing."""
    if len(header) == 0 or len(header) % 2 != 0:
        raise ValueError("paired CSV must contain date/value column pairs")
    output: dict[str, BimetsSeries] = {}
    for column in range(0, len(header), 2):
        name = header[column]
        pairs = [
            (row[column], row[column + 1])
            for row in rows
            if len(row) > column + 1 and row[column] != ""
        ]
        if not pairs:
            raise ValueError(f"series {name!r} contains no observations")
        freq = _parse_frequency_header(header[column + 1], freq_header_prefix)
        parsed_dates = [_parse_date(value, date_format) for value, _ in pairs]
        if freq is None:
            freq = _infer_frequency(parsed_dates)
        output[name] = _series_from_dated_values(
            name,
            parsed_dates,
            [_parse_value(value, missing, decimal_separator) for _, value in pairs],
            freq,
        )
    return output


def _read_merged(
    header: list[str],
    rows: list[list[str]],
    date_format: str,
    missing: str,
    decimal_separator: str,
    freq_header_prefix: str | None,
) -> dict[str, BimetsSeries]:
    """Read merged for internal processing."""
    if len(header) < 2:
        raise ValueError("merged CSV must contain a date and at least one value column")
    populated = [row for row in rows if row and row[0] != ""]
    if not populated:
        raise ValueError("merged CSV contains no observations")
    freq = _parse_frequency_header(header[0], freq_header_prefix)
    parsed_dates = [_parse_date(row[0], date_format) for row in populated]
    if freq is None:
        freq = _infer_frequency(parsed_dates)
    output: dict[str, BimetsSeries] = {}
    for column, name in enumerate(header[1:], start=1):
        values = [
            _parse_value(
                row[column] if len(row) > column else missing,
                missing,
                decimal_separator,
            )
            for row in populated
        ]
        output[name] = _series_from_dated_values(name, parsed_dates, values, freq)
    return output


def _parse_frequency_header(value: str, prefix: str | None) -> Frequency | None:
    """Parse an explicit frequency or request date-based inference."""
    compact = "".join(value.split())
    if prefix is None or not compact.startswith(prefix):
        return None
    raw_frequency = compact[len(prefix) :]
    if not raw_frequency:
        raise ValueError(f"invalid frequency header: {value}")
    try:
        return Frequency.parse(int(raw_frequency))
    except ValueError as error:
        raise ValueError(f"invalid frequency header: {value}") from error


def _validate_frequency_prefix(prefix: str) -> None:
    """Validate a CSV frequency-header prefix."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("freq_header_prefix must be a non-empty string")


def _format_value(value: float, missing: str, decimal_separator: str) -> str:
    """Format value for internal processing."""
    if np.isnan(value):
        return missing
    formatted = format(float(value), ".17g")
    return formatted.replace(".", decimal_separator)


def _parse_value(value: str, missing: str, decimal_separator: str) -> float:
    """Parse value for internal processing."""
    if value == missing or value == "":
        return np.nan
    return float(value.replace(decimal_separator, "."))


def _format_date(value: str | None, period: YearPeriod) -> str:
    """Format a date value for external representation."""
    if value is not None:
        return str(value)
    return f"{period.year}-P{period.period}"


def _parse_date(value: str, date_format: str) -> date | YearPeriod:
    """Parse a CSV calendar date or a non-calendar BIMETS period label."""
    if "-P" in value:
        year, period = value.split("-P", 1)
        return YearPeriod(int(year), int(period))
    if "%q" in date_format:
        return _parse_quarter_date(value, date_format)
    try:
        return datetime.strptime(value, date_format).date()
    except ValueError as error:
        raise ValueError(f"invalid CSV date: {value!r}") from error


def _parse_quarter_date(value: str, date_format: str) -> date:
    """Parse the BIMETS ``%q`` quarter placeholder into a calendar date."""
    if "%Y" not in date_format:
        raise ValueError("quarterly date_format must contain %Y")
    pattern = re.escape(date_format)
    pattern = pattern.replace("%Y", r"(?P<year>\d{4})")
    pattern = pattern.replace("%q", r"(?P<quarter>[1-4])")
    match = re.fullmatch(pattern, value)
    if match is None:
        raise ValueError(f"invalid CSV date: {value!r}")
    return date(int(match["year"]), 3 * int(match["quarter"]), 1)


def _infer_frequency(values: list[date | YearPeriod]) -> Frequency:
    """Infer BIMETS frequency from the minimum positive calendar spacing."""
    if any(isinstance(value, YearPeriod) for value in values):
        raise ValueError("cannot infer frequency from BIMETS period labels")
    dates = [value for value in values if isinstance(value, date)]
    if len(dates) == 1:
        return Frequency.DAILY
    positive_spacings = [
        (right - left).days for left, right in pairwise(dates) if right > left
    ]
    if not positive_spacings:
        raise ValueError("cannot infer frequency from non-increasing dates")
    spacing = min(positive_spacings)
    thresholds = (
        (7, Frequency.DAILY),
        (8, Frequency.WEEKLY),
        (10, Frequency.PERIODS_36),
        (28, Frequency.PERIODS_24),
        (89, Frequency.MONTHLY),
        (120, Frequency.QUARTERLY),
        (181, Frequency.THREE_PER_YEAR),
        (365, Frequency.SEMIANNUAL),
    )
    return next(
        (freq for limit, freq in thresholds if spacing < limit), Frequency.YEARLY
    )


def _series_from_dated_values(
    name: str,
    dates: list[date | YearPeriod],
    values: list[float],
    freq: Frequency,
) -> BimetsSeries:
    """Validate a dated column and fill uncovered periods with missing values."""
    periods = [
        value if isinstance(value, YearPeriod) else date_to_year_period(value, freq)
        for value in dates
    ]
    for period in periods:
        if period.period > int(freq):
            raise ValueError(f"date period {period} exceeds frequency {int(freq)}")
    ordinals = [period.ordinal(freq) for period in periods]
    if any(right <= left for left, right in pairwise(ordinals)):
        raise ValueError(
            f"dates for series {name!r} must identify unique increasing periods"
        )
    regular = np.full(ordinals[-1] - ordinals[0] + 1, np.nan, dtype=np.float64)
    for ordinal, value in zip(ordinals, values, strict=True):
        regular[ordinal - ordinals[0]] = value
    return BimetsSeries(
        regular,
        start=periods[0],
        freq=freq,
        metadata={"title": name},
    )
