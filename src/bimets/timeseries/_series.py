"""Canonical BIMETS time-series type."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from datetime import date
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bimets.timeseries._alignment import align_values
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod, _from_ordinal, normalize_year_period
from bimets.timeseries._mask import BimetsMask

type MetadataValue = str | int | float | bool | None
type RPeriodKey = list[int] | tuple[int, int]
type RIndexKey = list[Any]


def _restore_series(
    values: NDArray[np.float64],
    start: YearPeriod,
    freq: Frequency,
    metadata: Mapping[str, MetadataValue],
) -> BimetsSeries:
    """Reconstruct an immutable series during process deserialization."""
    return BimetsSeries(values, start=start, freq=freq, metadata=metadata)


class BimetsSeries:
    """A regular, univariate BIMETS time series.

    Values are stored in a read-only contiguous ``float64`` NumPy array. Time is
    represented by a start year-period plus a fixed number of periods per year.

    Parameters
    ----------
    values : array-like
        Non-empty, one-dimensional numeric observations. Missing observations
        can be represented by ``numpy.nan``.
    start : YearPeriod or tuple of int, default=(2000, 1)
        Year and one-based period of the first observation.
    freq : int, str, or Frequency, default=Frequency.YEARLY
        Number of periods per year or a supported frequency alias.
    metadata : mapping, optional
        Descriptive values associated with the series. A read-only copy is
        stored.

    Raises
    ------
    TypeError
        If the frequency, start index, or observations have invalid types.
    ValueError
        If observations are empty or multidimensional, the frequency is not
        supported, or the resulting range lies outside 1--9999.

    Notes
    -----
    Instances are immutable. Operations create new series and never modify
    the source observations. BIMETS R-style year-period access is supported in
    immutable form: ``series[[year, period]]`` returns one observation and
    ``series[[start, end]]`` returns an inclusive range when ``start`` and
    ``end`` are two-element sequences. ISO date strings select observations or
    ranges, and :meth:`with_values` provides the corresponding immutable update.

    Examples
    --------
    >>> from bimets import BimetsSeries
    >>> series = BimetsSeries(
    ...     [100, 105, 110],
    ...     start=(2020, 1),
    ...     freq="Q",
    ...     metadata={"title": "GDP"},
    ... )
    >>> series.start, series.end
    (YearPeriod(year=2020, period=1), YearPeriod(year=2020, period=3))
    >>> series.values.tolist()
    [100.0, 105.0, 110.0]
    >>> series[[2020, 2]]
    105.0
    >>> series[[[2020, 1], [2020, 2]]].values.tolist()
    [100.0, 105.0]
    """

    __slots__ = ("_freq", "_metadata", "_start", "_values")

    def __init__(
        self,
        values: ArrayLike,
        *,
        start: YearPeriod | tuple[int, int] = (2000, 1),
        freq: int | str | Frequency = Frequency.YEARLY,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> None:
        parsed_frequency = Frequency.parse(freq)
        parsed_start = normalize_year_period(start, parsed_frequency)
        # Always own the storage: making a borrowed ndarray read-only would be a
        # surprising side effect for the caller.
        array = np.array(values, dtype=np.float64, copy=True, order="C")
        if array.ndim != 1:
            raise ValueError("BimetsSeries values must be one-dimensional")
        if array.size == 0:
            raise ValueError("BimetsSeries values must not be empty")

        end = parsed_start.shift(int(array.size) - 1, parsed_frequency)
        if parsed_start.year < 1 or end.year > 9999:
            raise ValueError("BimetsSeries must lie in the year range 1-9999")

        array.setflags(write=False)
        self._values = array
        self._start = parsed_start
        self._freq = parsed_frequency
        self._metadata = MappingProxyType(dict(metadata or {}))

    @property
    def values(self) -> NDArray[np.float64]:
        """Read-only observations."""
        return self._values

    @property
    def start(self) -> YearPeriod:
        """First observation's year-period."""
        return self._start

    @property
    def end(self) -> YearPeriod:
        """Last observation's year-period."""
        return self.start.shift(len(self) - 1, self.freq)

    @property
    def freq(self) -> Frequency:
        """Number of periods per year."""
        return self._freq

    @property
    def metadata(self) -> Mapping[str, MetadataValue]:
        """Read-only descriptive metadata."""
        return self._metadata

    def __len__(self) -> int:
        return int(self.values.size)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize constructor state without the internal mapping proxy."""
        return (
            _restore_series,
            (self.values, self.start, self.freq, dict(self.metadata)),
        )

    def __iter__(self) -> Iterator[float]:
        return (float(value) for value in self.values)

    def __add__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.add)

    def __radd__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.add, reverse=True)

    def __sub__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.subtract)

    def __rsub__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.subtract, reverse=True)

    def __mul__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.multiply)

    def __rmul__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.multiply, reverse=True)

    def __truediv__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.divide)

    def __rtruediv__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.divide, reverse=True)

    def __floordiv__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.floor_divide)

    def __rfloordiv__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.floor_divide, reverse=True)

    def __mod__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.remainder)

    def __rmod__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.remainder, reverse=True)

    def __pow__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.power)

    def __rpow__(self, other: object) -> BimetsSeries:
        return self._arithmetic(other, np.power, reverse=True)

    def __neg__(self) -> BimetsSeries:
        return BimetsSeries(-self.values, start=self.start, freq=self.freq)

    def __pos__(self) -> BimetsSeries:
        return self

    def __abs__(self) -> BimetsSeries:
        return BimetsSeries(np.abs(self.values), start=self.start, freq=self.freq)

    def __lt__(self, other: object) -> BimetsMask:
        return self._compare(other, np.less)

    def __le__(self, other: object) -> BimetsMask:
        return self._compare(other, np.less_equal)

    def __gt__(self, other: object) -> BimetsMask:
        return self._compare(other, np.greater)

    def __ge__(self, other: object) -> BimetsMask:
        return self._compare(other, np.greater_equal)

    def __eq__(self, other: object) -> BimetsMask:  # type: ignore[override]
        return self._compare(other, np.equal)

    def __ne__(self, other: object) -> BimetsMask:  # type: ignore[override]
        return self._compare(other, np.not_equal)

    @overload
    def __getitem__(self, key: int) -> float: ...

    @overload
    def __getitem__(self, key: slice) -> BimetsSeries: ...

    @overload
    def __getitem__(self, key: list[int]) -> float: ...

    @overload
    def __getitem__(self, key: list[RPeriodKey]) -> float | BimetsSeries: ...

    @overload
    def __getitem__(self, key: str | date) -> float | BimetsSeries: ...

    def __getitem__(
        self, key: int | slice | RIndexKey | str | date
    ) -> float | BimetsSeries:
        if isinstance(key, int):
            return float(self.values[key])
        if isinstance(key, list):
            return self._r_index(key)
        if isinstance(key, (str, date)):
            selection_start, selection_end, single = self._date_selection(key)
            if single:
                return self.at_period(selection_start.year, selection_start.period)
            return self.project(selection_start, selection_end)
        if not isinstance(key, slice):
            raise TypeError(
                "BimetsSeries index must be an integer, slice, date selector, "
                "or BIMETS year-period key"
            )
        slice_start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("BimetsSeries slicing does not support a step")
        if stop <= slice_start:
            raise ValueError("BimetsSeries slices must not be empty")
        return BimetsSeries(
            self.values[slice_start:stop],
            start=self.start.shift(slice_start, self.freq),
            freq=self.freq,
            metadata=self.metadata,
        )

    def _r_index(self, key: RIndexKey) -> float | BimetsSeries:
        """Evaluate a BIMETS R-style year-period read key."""
        if len(key) == 2 and not any(isinstance(item, (list, tuple)) for item in key):
            year, period = _year_period_pair(key, "year-period index")
            return self.at_period(year, period)
        if len(key) == 1:
            year, period = _year_period_pair(key[0], "start index")
            return self.at_period(year, period)
        if len(key) == 2:
            start = _year_period_pair(key[0], "start index")
            end = _year_period_pair(key[1], "end index")
            return self.project(start, end)
        raise ValueError(
            "BIMETS year-period index must contain one period pair or "
            "an inclusive start/end pair"
        )

    def _date_selection(self, key: str | date) -> tuple[YearPeriod, YearPeriod, bool]:
        """Resolve a BIMETS date selector to inclusive period bounds."""
        if isinstance(key, date):
            period = _period_from_date(key, self.freq)
            return period, period, True

        selector = key.strip()
        if not selector:
            raise ValueError("date selector must not be empty")
        if "/" in selector:
            if selector.count("/") != 1:
                raise ValueError("date range must contain exactly one '/' separator")
            left, right = selector.split("/")
            start = _date_boundary(left, self.freq, first=True) if left else self.start
            end = _date_boundary(right, self.freq, first=False) if right else self.end
            if end.ordinal(self.freq) < start.ordinal(self.freq):
                raise ValueError("date range end precedes start")
            return start, end, False
        if re.fullmatch(r"\d{4}", selector):
            year = int(selector)
            return YearPeriod(year, 1), YearPeriod(year, int(self.freq)), False
        period = _period_from_date(_parse_iso_date(selector), self.freq)
        return period, period, True

    def with_values(
        self,
        key: int | slice | RIndexKey | str | date,
        values: ArrayLike,
        *,
        extend: bool = True,
    ) -> BimetsSeries:
        """Return a copy with observations replaced at an index selection.

        Parameters
        ----------
        key : int, slice, date selector, or BIMETS year-period key
            The same selectors accepted by ``series[key]``. A single
            year-period key followed by multiple values assigns them
            sequentially, matching ``ts[[year, period]] <- values`` in BIMETS
            R. Date strings accept ``YYYY-MM-DD``, ``YYYY-MM``, ``YYYY``, and
            closed or open ``start/end`` ranges.
        values : array-like
            A scalar is broadcast over a range. A sequence must either match a
            fixed selection exactly or, for a single year-period key, defines
            the number of consecutive observations to replace.
        extend : bool, default=True
            Extend the series with missing observations if the replacement
            exceeds its range. Set to false to reject out-of-range updates.

        Returns
        -------
        BimetsSeries
            Updated series. The source series and its read-only values remain
            unchanged.

        Raises
        ------
        IndexError
            If a positional index is outside the series.
        ValueError
            If the selector is empty or reversed, replacement length is
            incompatible, or extension is disabled for an out-of-range update.

        Examples
        --------
        >>> from bimets import timeseries
        >>> source = timeseries([1, 2, 3], start=(2020, 1), freq="Q")
        >>> updated = source.with_values([[2020, 2], [2020, 3]], 9)
        >>> updated.values.tolist()
        [1.0, 9.0, 9.0]
        >>> source.values.tolist()
        [1.0, 2.0, 3.0]
        >>> source.with_values([2020, 4], [4, 5]).values.tolist()
        [1.0, 2.0, 3.0, 4.0, 5.0]
        """
        replacement = np.asarray(values, dtype=np.float64)
        if replacement.ndim > 1:
            raise ValueError("replacement values must be scalar or one-dimensional")
        if replacement.ndim == 1 and replacement.size == 0:
            raise ValueError("replacement values must not be empty")

        start, end, sequential = self._replacement_selection(key)
        if sequential and replacement.ndim == 1:
            end = start.shift(int(replacement.size) - 1, self.freq)

        first = start.ordinal(self.freq)
        last = end.ordinal(self.freq)
        if last < first:
            raise ValueError("replacement end precedes start")
        length = last - first + 1
        if replacement.ndim == 0 or replacement.size == 1:
            assigned = np.full(length, float(replacement.reshape(-1)[0]))
        elif replacement.size == length:
            assigned = replacement
        else:
            raise ValueError(
                f"replacement requires one or {length} values, got {replacement.size}"
            )

        own_first = self.start.ordinal(self.freq)
        own_last = self.end.ordinal(self.freq)
        if not extend and (first < own_first or last > own_last):
            raise ValueError("replacement lies outside series; use extend=True")
        output_start = min(first, own_first)
        output_end = max(last, own_last)
        output = np.full(output_end - output_start + 1, np.nan)
        source_offset = own_first - output_start
        output[source_offset : source_offset + len(self)] = self.values
        replacement_offset = first - output_start
        output[replacement_offset : replacement_offset + length] = assigned
        return BimetsSeries(
            output,
            start=_from_ordinal(output_start, self.freq),
            freq=self.freq,
            metadata=self.metadata,
        )

    def _replacement_selection(
        self, key: int | slice | RIndexKey | str | date
    ) -> tuple[YearPeriod, YearPeriod, bool]:
        """Resolve an immutable replacement selector and its sequential mode."""
        if isinstance(key, int):
            period = self.period_at(key)
            return period, period, False
        if isinstance(key, slice):
            slice_start, stop, step = key.indices(len(self))
            if step != 1:
                raise ValueError("BimetsSeries slicing does not support a step")
            if stop <= slice_start:
                raise ValueError("BimetsSeries slices must not be empty")
            return self.period_at(slice_start), self.period_at(stop - 1), False
        if isinstance(key, (str, date)):
            return self._date_selection(key)
        if not isinstance(key, list):
            raise TypeError(
                "replacement index must be an integer, slice, date selector, "
                "or BIMETS year-period key"
            )
        if len(key) == 2 and not any(isinstance(item, (list, tuple)) for item in key):
            pair = _year_period_pair(key, "year-period index")
            period = normalize_year_period(pair, self.freq)
            return period, period, True
        if len(key) == 1:
            pair = _year_period_pair(key[0], "start index")
            period = normalize_year_period(pair, self.freq)
            return period, period, True
        if len(key) == 2:
            period_start = normalize_year_period(
                _year_period_pair(key[0], "start index"), self.freq
            )
            period_end = normalize_year_period(
                _year_period_pair(key[1], "end index"), self.freq
            )
            return period_start, period_end, False
        raise ValueError(
            "BIMETS year-period index must contain one period pair or "
            "an inclusive start/end pair"
        )

    def period_at(self, position: int) -> YearPeriod:
        """Return the year-period at a zero-based position.

        Parameters
        ----------
        position : int
            Observation position. Negative indexing follows Python rules.

        Returns
        -------
        YearPeriod
            Index corresponding to ``position``.

        Raises
        ------
        IndexError
            If the position is outside the series.
        """
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("BimetsSeries position out of range")
        return self.start.shift(position, self.freq)

    def at_period(self, year: int, period: int) -> float:
        """Return the observation for a year-period.

        Parameters
        ----------
        year : int
            Calendar year.
        period : int
            One-based period within ``year``.

        Returns
        -------
        float
            Observation at the requested index.

        Raises
        ------
        IndexError
            If the normalized year-period is outside the series.
        """
        requested = YearPeriod.normalize(year, period, self.freq)
        position = requested.ordinal(self.freq) - self.start.ordinal(self.freq)
        if position < 0 or position >= len(self):
            raise IndexError("BimetsSeries year-period out of range")
        return float(self.values[position])

    def at_date(self, value: date) -> float:
        """Return the observation containing a calendar date.

        Parameters
        ----------
        value : datetime.date
            Date to locate in the series calendar.

        Returns
        -------
        float
            Observation whose period contains ``value``.
        """
        from bimets.timeseries._calendar import date_to_year_period

        period = date_to_year_period(value, self.freq)
        return self.at_period(period.year, period.period)

    def between_dates(self, start: date, end: date) -> BimetsSeries:
        """Return the inclusive range containing two calendar dates.

        Parameters
        ----------
        start, end : datetime.date
            Bounds converted to the corresponding year-period indexes.

        Returns
        -------
        BimetsSeries
            Projection containing both boundary periods.
        """
        from bimets.timeseries._calendar import date_to_year_period

        first = date_to_year_period(start, self.freq)
        last = date_to_year_period(end, self.freq)
        return self.project(first, last)

    def lag(self, periods: int = 1) -> BimetsSeries:
        """Shift observations forward in time.

        Parameters
        ----------
        periods : int, default=1
            Number of periods by which to move the index.

        Returns
        -------
        BimetsSeries
            Series with unchanged values and a later start index.

        Examples
        --------
        >>> from bimets import timeseries
        >>> series = timeseries([10, 20], start=(2020, 1), freq="Q")
        >>> lagged = series.lag(1)
        >>> lagged.start, lagged.values.tolist()
        (YearPeriod(year=2020, period=2), [10.0, 20.0])
        """
        return BimetsSeries(
            self.values,
            start=self.start.shift(periods, self.freq),
            freq=self.freq,
            metadata=self.metadata,
        )

    def lead(self, periods: int = 1) -> BimetsSeries:
        """Shift observations backward in time.

        Parameters
        ----------
        periods : int, default=1
            Number of periods by which to move the index backward.

        Returns
        -------
        BimetsSeries
            Series with unchanged values and an earlier start index.
        """
        return self.lag(-periods)

    def delta(self, lag: int = 1, order: int = 1) -> BimetsSeries:
        """Return lagged differences.

        Parameters
        ----------
        lag : int, default=1
            Distance between observations in each difference.
        order : int, default=1
            Number of times to apply the difference.

        Returns
        -------
        BimetsSeries
            Differenced observations, starting ``lag * order`` periods after
            the source series.

        Raises
        ------
        ValueError
            If ``lag`` or ``order`` is not positive, or if differencing would
            consume the complete series.

        Examples
        --------
        >>> from bimets import timeseries
        >>> series = timeseries([10, 13, 18], start=(2020, 1), freq="Q")
        >>> series.delta().values.tolist()
        [3.0, 5.0]
        """
        _require_positive_integer(lag, "lag")
        _require_positive_integer(order, "order")
        if len(self) <= lag * order:
            raise ValueError("lag and order consume the complete series")

        result = self.values
        for _ in range(order):
            result = result[lag:] - result[:-lag]
        return BimetsSeries(
            result,
            start=self.start.shift(lag * order, self.freq),
            freq=self.freq,
        )

    def delta_log(self, lag: int = 1) -> BimetsSeries:
        """Return lagged differences of natural logarithms.

        Parameters
        ----------
        lag : int, default=1
            Distance between logged observations.

        Returns
        -------
        BimetsSeries
            Log-difference series.

        Notes
        -----
        Non-positive inputs follow NumPy logarithm semantics and may produce
        infinite or missing observations.
        """
        _require_positive_integer(lag, "lag")
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(self.values)
        return BimetsSeries(
            logged,
            start=self.start,
            freq=self.freq,
        ).delta(lag=lag)

    def delta_percent(self, lag: int = 1, *, annualize: bool = False) -> BimetsSeries:
        """Return percentage changes over a selected lag.

        Parameters
        ----------
        lag : int, default=1
            Distance between current and previous observations.
        annualize : bool, default=False
            Raise the growth factor to ``frequency / lag`` before converting
            it to a percentage.

        Returns
        -------
        BimetsSeries
            Percentage changes beginning ``lag`` periods after the source.

        Raises
        ------
        ValueError
            If ``lag`` is invalid, consumes the series, or does not divide the
            frequency when ``annualize`` is true.

        Examples
        --------
        >>> from bimets import timeseries
        >>> series = timeseries([100, 110, 121], start=(2020, 1), freq="Q")
        >>> series.delta_percent().values.round(8).tolist()
        [10.0, 10.0]
        """
        _require_positive_integer(lag, "lag")
        if len(self) <= lag:
            raise ValueError("lag consumes the complete series")
        if annualize and int(self.freq) % lag != 0:
            raise ValueError("annualization requires frequency to be divisible by lag")

        current = self.values[lag:]
        previous = self.values[:-lag]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = current / previous
            if annualize:
                ratio = np.power(ratio, int(self.freq) / lag)
            result = 100.0 * (ratio - 1.0)
        return BimetsSeries(
            result,
            start=self.start.shift(lag, self.freq),
            freq=self.freq,
        )

    def project(
        self,
        start: YearPeriod | tuple[int, int],
        end: YearPeriod | tuple[int, int],
        *,
        extend: bool = False,
    ) -> BimetsSeries:
        """Project the series into an inclusive year-period range.

        Parameters
        ----------
        start, end : YearPeriod or tuple of int
            Inclusive requested bounds.
        extend : bool, default=False
            If true, preserve bounds outside the source and fill uncovered
            observations with ``numpy.nan``. Otherwise clip to the overlap.

        Returns
        -------
        BimetsSeries
            Projected series.

        Raises
        ------
        ValueError
            If the bounds are reversed or do not overlap when ``extend`` is
            false.

        Examples
        --------
        >>> from bimets import timeseries
        >>> series = timeseries([1, 2, 3, 4], start=(2020, 1), freq="Q")
        >>> series.project((2020, 2), (2020, 3)).values.tolist()
        [2.0, 3.0]
        """
        requested_start = normalize_year_period(start, self.freq)
        requested_end = normalize_year_period(end, self.freq)
        start_ordinal = requested_start.ordinal(self.freq)
        end_ordinal = requested_end.ordinal(self.freq)
        if end_ordinal < start_ordinal:
            raise ValueError("projection end precedes projection start")

        own_start = self.start.ordinal(self.freq)
        own_end = self.end.ordinal(self.freq)
        overlap_start = max(start_ordinal, own_start)
        overlap_end = min(end_ordinal, own_end)
        if overlap_start > overlap_end and not extend:
            raise ValueError("projection range and series do not overlap")

        if not extend:
            selected_start = overlap_start
            selected_end = overlap_end
            first = selected_start - own_start
            last = selected_end - own_start + 1
            return BimetsSeries(
                self.values[first:last],
                start=_from_ordinal(selected_start, self.freq),
                freq=self.freq,
                metadata=self.metadata,
            )

        output = np.full(end_ordinal - start_ordinal + 1, np.nan)
        if overlap_start <= overlap_end:
            source_start = overlap_start - own_start
            source_end = overlap_end - own_start + 1
            target_start = overlap_start - start_ordinal
            target_end = overlap_end - start_ordinal + 1
            output[target_start:target_end] = self.values[source_start:source_end]
        return BimetsSeries(
            output,
            start=requested_start,
            freq=self.freq,
            metadata=self.metadata,
        )

    def trim(
        self,
        value: float | None = None,
        *,
        leading: bool = True,
        trailing: bool = True,
    ) -> BimetsSeries | None:
        """Remove leading or trailing missing or target values.

        Parameters
        ----------
        value : float, optional
            Value to trim. ``None`` and ``numpy.nan`` select missing values.
        leading : bool, default=True
            Remove matching observations from the beginning.
        trailing : bool, default=True
            Remove matching observations from the end.

        Returns
        -------
        BimetsSeries or None
            Trimmed series, or ``None`` if every observation is removed.

        Examples
        --------
        >>> import numpy as np
        >>> from bimets import timeseries
        >>> series = timeseries([np.nan, 1, 2, np.nan], freq="Q")
        >>> series.trim().values.tolist()
        [1.0, 2.0]
        """
        if value is None or np.isnan(value):
            keep = ~np.isnan(self.values)
        else:
            keep = (self.values != value) | np.isnan(self.values)
        positions = np.flatnonzero(keep)
        if positions.size == 0:
            return None

        first = int(positions[0]) if leading else 0
        last = int(positions[-1]) + 1 if trailing else len(self)
        return BimetsSeries(
            self.values[first:last],
            start=self.start.shift(first, self.freq),
            freq=self.freq,
        )

    def cumulative_sum(
        self,
        *,
        mode: str | None = None,
        skip_missing: bool = False,
        start: YearPeriod | tuple[int, int] | None = None,
        end: YearPeriod | tuple[int, int] | None = None,
    ) -> BimetsSeries:
        """Return cumulative sums, optionally resetting by calendar group.

        Parameters
        ----------
        mode : {None, "yearly", "monthly"}, optional
            Group at whose boundary the cumulative sum restarts.
        skip_missing : bool, default=False
            Ignore missing observations instead of propagating them.
        start, end : YearPeriod or tuple of int, optional
            Inclusive calculation bounds. Omitted bounds use the source range.

        Returns
        -------
        BimetsSeries
            Cumulative sums on the original range.
        """
        from bimets.timeseries._manipulation import cumulative_sum

        return cumulative_sum(
            self,
            mode=mode,
            skip_missing=skip_missing,
            start=start,
            end=end,
        )

    def cumulative_product(
        self,
        *,
        skip_missing: bool = False,
        start: YearPeriod | tuple[int, int] | None = None,
        end: YearPeriod | tuple[int, int] | None = None,
    ) -> BimetsSeries:
        """Return cumulative products.

        Parameters
        ----------
        skip_missing : bool, default=False
            Ignore missing observations instead of propagating them.
        start, end : YearPeriod or tuple of int, optional
            Inclusive calculation bounds. Omitted bounds use the source range.

        Returns
        -------
        BimetsSeries
            Cumulative products on the original range.
        """
        from bimets.timeseries._manipulation import cumulative_product

        return cumulative_product(self, skip_missing=skip_missing, start=start, end=end)

    def moving_average(
        self, window: int, *, direction: str = "back", skip_missing: bool = False
    ) -> BimetsSeries:
        """Return averages over complete moving windows.

        Parameters
        ----------
        window : int
            Positive number of observations in each window.
        direction : {"back", "center", "ahead"}, default="back"
            Position of the output observation relative to its window.
        skip_missing : bool, default=False
            Ignore missing observations when computing each average.

        Returns
        -------
        BimetsSeries
            Moving averages over the reduced complete-window range.
        """
        from bimets.timeseries._manipulation import moving_average

        return moving_average(
            self, window, direction=direction, skip_missing=skip_missing
        )

    def moving_sum(
        self, window: int, *, direction: str = "back", skip_missing: bool = False
    ) -> BimetsSeries:
        """Return sums over complete moving windows.

        Parameters
        ----------
        window : int
            Positive number of observations in each window.
        direction : {"back", "center", "ahead"}, default="back"
            Position of the output observation relative to its window.
        skip_missing : bool, default=False
            Ignore missing observations when computing each sum.

        Returns
        -------
        BimetsSeries
            Moving sums over the reduced complete-window range.

        Examples
        --------
        >>> from bimets import timeseries
        >>> series = timeseries([1, 2, 3, 4], freq="Q")
        >>> series.moving_sum(2).values.tolist()
        [3.0, 5.0, 7.0]
        """
        from bimets.timeseries._manipulation import moving_sum

        return moving_sum(self, window, direction=direction, skip_missing=skip_missing)

    def index_number(self, base_year: int) -> BimetsSeries:
        """Rebase the series so that its base-year average is 100.

        Parameters
        ----------
        base_year : int
            Year whose complete available average defines the base.

        Returns
        -------
        BimetsSeries
            Rebased series on the original range.
        """
        from bimets.timeseries._manipulation import index_number

        return index_number(self, base_year)

    def extend(
        self,
        *,
        back_to: YearPeriod | tuple[int, int] | None = None,
        up_to: YearPeriod | tuple[int, int] | None = None,
        mode: str = "growth",
        factor: float | None = None,
    ) -> BimetsSeries:
        """Extend the definition range using a BIMETS extrapolation mode.

        Parameters
        ----------
        back_to, up_to : YearPeriod or tuple of int, optional
            New inclusive lower and upper bounds. At least one is required.
        mode : str, default="growth"
            Extension rule: ``missing``, ``zero``, ``constant``, ``mean4``,
            ``growth``, ``growth4``, ``linear``, ``quadratic``, ``myconst``,
            or ``myrate``.
        factor : float, optional
            User value required by ``myconst`` and ``myrate``.

        Returns
        -------
        BimetsSeries
            Extended series.

        Raises
        ------
        ValueError
            If bounds or mode are invalid, or the selected mode lacks enough
            valid observations.
        """
        from bimets.timeseries._manipulation import extend

        return extend(self, back_to=back_to, up_to=up_to, mode=mode, factor=factor)

    def to_frequency(
        self,
        freq: int | str | Frequency,
        *,
        method: str | None = None,
    ) -> BimetsSeries:
        """Aggregate or disaggregate to another frequency.

        Parameters
        ----------
        freq : int, str, or Frequency
            Target frequency.
        method : str, optional
            Aggregation or interpolation method. Aggregation requires an
            explicit method; disaggregation defaults to repetition.

        Returns
        -------
        BimetsSeries
            Converted series.

        See Also
        --------
        convert_frequency : Equivalent functional API.
        """
        from bimets.timeseries._frequency_conversion import convert_frequency

        return convert_frequency(self, freq, method=method)

    def dates(
        self,
        *,
        date_in_period: Literal["first", "last"] = "last",
        format: str | None = None,
    ) -> list[date | str | None]:
        """Return calendar dates for all observations.

        Parameters
        ----------
        date_in_period : {"first", "last"}, default="last"
            Select the first or last calendar day in each period.
        format : str, optional
            ``datetime.date.strftime`` format. If omitted, return date objects.

        Returns
        -------
        list of datetime.date, str, or None
            One calendar value for every observation. Invalid placeholder days
            in daily series are represented by ``None``.
        """
        from bimets.timeseries._calendar import get_dates

        if format is None:
            return cast(
                list[date | str | None],
                get_dates(self, date_in_period=date_in_period),
            )
        return cast(
            list[date | str | None],
            get_dates(self, date_in_period=date_in_period, format=format),
        )

    def _arithmetic(
        self,
        other: object,
        operation: np.ufunc,
        *,
        reverse: bool = False,
    ) -> BimetsSeries:
        """Apply an arithmetic operation over aligned observations."""
        left_values: object
        right_values: object
        if isinstance(other, BimetsSeries):
            left_values, right_values, start = align_values(self, other)
        else:
            scalar = _numeric_scalar(other)
            left_values, right_values, start = self.values, scalar, self.start
        with np.errstate(all="ignore"):
            result = (
                operation(right_values, left_values)
                if reverse
                else operation(left_values, right_values)
            )
        return BimetsSeries(result, start=start, freq=self.freq)

    def _compare(
        self,
        other: object,
        operation: np.ufunc,
    ) -> BimetsMask:
        """Compare an operand with the series over aligned observations."""
        left_values: object
        right_values: object
        if isinstance(other, BimetsSeries):
            left_values, right_values, start = align_values(self, other)
        else:
            scalar = _numeric_scalar(other)
            left_values, right_values, start = self.values, scalar, self.start
        missing = np.isnan(np.asarray(left_values, dtype=np.float64)) | np.isnan(
            np.asarray(right_values, dtype=np.float64)
        )
        with np.errstate(all="ignore"):
            compared = np.asarray(operation(left_values, right_values), dtype=object)
        compared[missing] = None
        return BimetsMask(compared, start=start, freq=self.freq)

    def __repr__(self) -> str:
        from bimets.timeseries._display import technical_series_repr

        return technical_series_repr(
            self.values,
            start=self.start,
            end=self.end,
            freq=self.freq,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        """Return the user-facing, frequency-dependent series display."""
        from bimets.timeseries._display import format_series

        return format_series(
            self.values,
            start=self.start,
            end=self.end,
            freq=self.freq,
            metadata=self.metadata,
        )


def timeseries(
    values: ArrayLike,
    *,
    start: YearPeriod | tuple[int, int] = (2000, 1),
    freq: int | str | Frequency = Frequency.YEARLY,
    source: str | None = None,
    title: str | None = None,
    units: str | None = None,
    scale_factor: int = 0,
) -> BimetsSeries:
    """Construct a regular BIMETS time series.

    Parameters
    ----------
    values : array-like
        Non-empty, one-dimensional numeric observations.
    start : YearPeriod or tuple of int, default=(2000, 1)
        Year and one-based period of the first observation.
    freq : int, str, or Frequency, default=Frequency.YEARLY
        Number of periods per year or a supported alias.
    source, title, units : str, optional
        Standard descriptive metadata.
    scale_factor : int, default=0
        Non-negative display scale stored as metadata. Zero is omitted.

    Returns
    -------
    BimetsSeries
        Immutable series containing the observations and supplied metadata.

    Raises
    ------
    TypeError
        If ``scale_factor`` is not an integer or another argument has an
        invalid type.
    ValueError
        If the values, range, frequency, or scale factor are invalid.

    Examples
    --------
    >>> from bimets import timeseries
    >>> gdp = timeseries(
    ...     [100, 102, 105],
    ...     start=(2020, 1),
    ...     freq="Q",
    ...     title="GDP",
    ...     units="index",
    ... )
    >>> gdp
    BimetsSeries(values=[100, 102, 105], length=3, start=(2020, 1), end=(2020, 3), freq=4, metadata={'title': 'GDP', 'units': 'index'})
    >>> dict(gdp.metadata)
    {'title': 'GDP', 'units': 'index'}
    """
    metadata: dict[str, MetadataValue] = {}
    if source is not None:
        metadata["source"] = source
    if title is not None:
        metadata["title"] = title
    if units is not None:
        metadata["units"] = units
    if scale_factor != 0:
        if isinstance(scale_factor, bool) or not isinstance(scale_factor, int):
            raise TypeError("scale_factor must be an integer")
        if scale_factor < 0:
            raise ValueError("scale_factor must not be negative")
        metadata["scale_factor"] = scale_factor
    return BimetsSeries(
        values,
        start=start,
        freq=freq,
        metadata=metadata,
    )


def is_bimets(value: object) -> bool:
    """Return whether a value is a :class:`BimetsSeries`.

    Parameters
    ----------
    value : object
        Value to inspect.

    Returns
    -------
    bool
        ``True`` only for ``BimetsSeries`` instances.

    Examples
    --------
    >>> from bimets import is_bimets, timeseries
    >>> is_bimets(timeseries([1, 2]))
    True
    >>> is_bimets([1, 2])
    False
    """
    return isinstance(value, BimetsSeries)


def _is_integer(value: object) -> bool:
    """Return whether a value is an integer scalar."""
    return not isinstance(value, bool) and isinstance(value, Integral)


def _year_period_pair(value: object, label: str) -> tuple[int, int]:
    """Validate and unpack a two-item year-period key."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"BIMETS {label} must be a two-element sequence")
    if len(value) != 2:
        raise ValueError(f"BIMETS {label} must contain exactly year and period")
    year, period = value
    if not _is_integer(year) or not _is_integer(period):
        raise TypeError(f"BIMETS {label} year and period must be integers")
    if year <= 0 or period <= 0:
        raise ValueError(f"BIMETS {label} year and period must be positive")
    return int(year), int(period)


def _require_positive_integer(value: int, name: str) -> None:
    """Validate positive integer for internal processing."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _parse_iso_date(value: str) -> date:
    """Parse iso date for internal processing."""
    if re.fullmatch(r"\d{4}-\d{2}", value):
        value = f"{value}-01"
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "date selector must use YYYY, YYYY-MM, YYYY-MM-DD, or start/end"
        ) from error


def _period_from_date(value: date, freq: Frequency) -> YearPeriod:
    """Map a calendar date to a period at the requested frequency."""
    from bimets.timeseries._calendar import date_to_year_period

    return date_to_year_period(value, freq)


def _date_boundary(value: str, freq: Frequency, *, first: bool) -> YearPeriod:
    """Convert an ISO date into a containing period boundary."""
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        return YearPeriod(int(value), 1 if first else int(freq))
    return _period_from_date(_parse_iso_date(value), freq)


def _numeric_scalar(value: object) -> float:
    """Convert a scalar operand to a floating-point value."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("operation requires a BimetsSeries or numeric scalar")
    return float(value)
