"""Named collections of BIMETS time series."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from bimets.timeseries._csv import bimets_to_csv, csv_to_bimets
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod
from bimets.timeseries._inspection import get_range, tabulate
from bimets.timeseries._pandas import from_pandas, to_pandas
from bimets.timeseries._series import BimetsSeries, MetadataValue


def _restore_dataset(
    series: Mapping[str, BimetsSeries], metadata: Mapping[str, MetadataValue]
) -> BimetsDataset:
    """Reconstruct an immutable dataset during process deserialization."""
    return BimetsDataset(series, metadata=metadata)


class BimetsDataset(Mapping[str, BimetsSeries]):
    """An immutable named collection of time series.

    Parameters
    ----------
    series : mapping of str to BimetsSeries
        Non-empty collection. Names must be non-empty strings and retain their
        insertion order.
    metadata : mapping, optional
        Dataset-level metadata. A read-only copy is stored.

    Raises
    ------
    TypeError
        If a value is not a ``BimetsSeries``.
    ValueError
        If the collection is empty or contains an invalid name.

    Examples
    --------
    >>> from bimets import BimetsDataset, timeseries
    >>> gdp = timeseries([100, 102], start=(2020, 1), freq="Q")
    >>> cpi = timeseries([98, 99], start=(2020, 1), freq="Q")
    >>> data = BimetsDataset({"gdp": gdp, "cpi": cpi})
    >>> data.names
    ('gdp', 'cpi')
    >>> data.homogeneous_frequency
    <Frequency.QUARTERLY: 4>
    """

    __slots__ = ("_metadata", "_series")

    def __init__(
        self,
        series: Mapping[str, BimetsSeries],
        *,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> None:
        if not series:
            raise ValueError("BimetsDataset must contain at least one series")
        copied: dict[str, BimetsSeries] = {}
        for name, value in series.items():
            if not isinstance(name, str) or name == "":
                raise ValueError("dataset names must be non-empty strings")
            if not isinstance(value, BimetsSeries):
                raise TypeError("dataset values must be BimetsSeries objects")
            copied[name] = value
        self._series = MappingProxyType(copied)
        self._metadata = MappingProxyType(dict(metadata or {}))

    def __getitem__(self, name: str) -> BimetsSeries:
        return self._series[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._series)

    def __len__(self) -> int:
        return len(self._series)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize constructor state without internal mapping proxies."""
        return _restore_dataset, (dict(self), dict(self.metadata))

    @property
    def names(self) -> tuple[str, ...]:
        """Series names in insertion order."""
        return tuple(self._series)

    @property
    def metadata(self) -> Mapping[str, MetadataValue]:
        """Read-only dataset-level metadata."""
        return self._metadata

    @property
    def homogeneous_frequency(self) -> Frequency | None:
        """Common frequency, or ``None`` for a heterogeneous dataset."""
        frequencies = {item.freq for item in self.values()}
        return next(iter(frequencies)) if len(frequencies) == 1 else None

    def select(self, names: Sequence[str]) -> BimetsDataset:
        """Return selected variables in the requested order.

        Parameters
        ----------
        names : sequence of str
            Non-empty variable selection.

        Returns
        -------
        BimetsDataset
            New dataset containing only ``names``.
        """
        if not names:
            raise ValueError("at least one name must be selected")
        return BimetsDataset(
            {name: self[name] for name in names}, metadata=self.metadata
        )

    def drop(self, names: str | Sequence[str]) -> BimetsDataset:
        """Return a dataset without selected variables.

        Parameters
        ----------
        names : str or sequence of str
            Variables to remove.

        Returns
        -------
        BimetsDataset
            New dataset containing the remaining variables.

        Raises
        ------
        KeyError
            If a requested name is unknown.
        ValueError
            If removing the names would produce an empty dataset.
        """
        removed = {names} if isinstance(names, str) else set(names)
        unknown = removed.difference(self)
        if unknown:
            raise KeyError(f"unknown dataset names: {sorted(unknown)}")
        remaining = {name: value for name, value in self.items() if name not in removed}
        if not remaining:
            raise ValueError("drop would produce an empty dataset")
        return BimetsDataset(remaining, metadata=self.metadata)

    def rename(self, names: Mapping[str, str]) -> BimetsDataset:
        """Return a dataset with selected variables renamed.

        Parameters
        ----------
        names : mapping of str to str
            Existing names mapped to their replacements.

        Returns
        -------
        BimetsDataset
            Renamed dataset, preserving variable order.

        Raises
        ------
        KeyError
            If a source name is unknown.
        ValueError
            If the replacements create duplicate names.
        """
        unknown = set(names).difference(self)
        if unknown:
            raise KeyError(f"unknown dataset names: {sorted(unknown)}")
        renamed = {names.get(name, name): value for name, value in self.items()}
        if len(renamed) != len(self):
            raise ValueError("renaming would create duplicate names")
        return BimetsDataset(renamed, metadata=self.metadata)

    def with_series(
        self, name: str, series: BimetsSeries, *, replace: bool = False
    ) -> BimetsDataset:
        """Return a dataset with a variable added or replaced.

        Parameters
        ----------
        name : str
            Variable name.
        series : BimetsSeries
            Value to store.
        replace : bool, default=False
            Permit replacement when ``name`` already exists.

        Returns
        -------
        BimetsDataset
            Updated immutable dataset.
        """
        if name in self and not replace:
            raise KeyError(f"dataset already contains {name!r}")
        updated = dict(self.items())
        updated[name] = series
        return BimetsDataset(updated, metadata=self.metadata)

    def combine(
        self, other: Mapping[str, BimetsSeries], *, replace: bool = False
    ) -> BimetsDataset:
        """Combine this dataset with another named collection.

        Parameters
        ----------
        other : mapping of str to BimetsSeries
            Variables appended to the dataset.
        replace : bool, default=False
            Permit values from ``other`` to replace duplicate names.

        Returns
        -------
        BimetsDataset
            Combined dataset.
        """
        overlap = set(self).intersection(other)
        if overlap and not replace:
            raise KeyError(f"duplicate dataset names: {sorted(overlap)}")
        combined = dict(self.items())
        combined.update(other)
        return BimetsDataset(combined, metadata=self.metadata)

    def assign_range(
        self,
        values: Mapping[str, float | Sequence[float]],
        *,
        start: YearPeriod | tuple[int, int],
        end: YearPeriod | tuple[int, int] | None = None,
        extend: bool = False,
    ) -> BimetsDataset:
        """Return a dataset with values replaced over an inclusive range.

        Parameters
        ----------
        values : mapping
            Existing variable names mapped to a numeric scalar or a sequence.
            Scalars are broadcast over the selected range; sequences must have
            exactly one value per selected period.
        start, end : YearPeriod or tuple of int
            Inclusive assignment bounds. ``end`` defaults to ``start``.
        extend : bool, default=False
            Extend affected series with missing observations when the selected
            range exceeds their current bounds.

        Returns
        -------
        BimetsDataset
            Updated immutable dataset. Unaffected series are shared with the
            original dataset.

        Raises
        ------
        KeyError
            If an assigned variable is not present.
        ValueError
            If the range is reversed, lies outside a series without
            ``extend=True``, or a sequence has an incompatible length.

        Examples
        --------
        >>> from bimets import BimetsDataset, timeseries
        >>> data = BimetsDataset({
        ...     "policy": timeseries([1, 1, 1], start=(2020, 1), freq="Q")
        ... })
        >>> scenario = data.assign_range(
        ...     {"policy": [2, 3]}, start=(2020, 2), end=(2020, 3)
        ... )
        >>> scenario["policy"].values.tolist()
        [1.0, 2.0, 3.0]
        >>> data["policy"].values.tolist()
        [1.0, 1.0, 1.0]
        """
        if not values:
            raise ValueError("at least one variable must be assigned")
        unknown = set(values).difference(self)
        if unknown:
            raise KeyError(f"unknown dataset names: {sorted(unknown)}")

        updated = dict(self.items())
        for name, replacement in values.items():
            series = self[name]
            first = _normalize_bound(start, series.freq)
            last = _normalize_bound(end if end is not None else start, series.freq)
            first_ordinal = first.ordinal(series.freq)
            last_ordinal = last.ordinal(series.freq)
            if last_ordinal < first_ordinal:
                raise ValueError("assignment end precedes start")

            source_start = series.start.ordinal(series.freq)
            source_end = series.end.ordinal(series.freq)
            if not extend and (
                first_ordinal < source_start or last_ordinal > source_end
            ):
                raise ValueError(
                    f"assignment range lies outside series {name!r}; use extend=True"
                )
            assigned = (
                series.project(
                    first if first_ordinal < source_start else series.start,
                    last if last_ordinal > source_end else series.end,
                    extend=True,
                )
                if extend
                else series
            )
            length = last_ordinal - first_ordinal + 1
            replacement_array = np.asarray(replacement, dtype=float)
            if replacement_array.ndim == 0:
                replacement_array = np.full(length, float(replacement_array))
            elif replacement_array.ndim != 1:
                raise ValueError("assigned values must be a scalar or one-dimensional")
            elif len(replacement_array) != length:
                raise ValueError(
                    f"assignment for {name!r} requires {length} values, "
                    f"got {len(replacement_array)}"
                )
            output = assigned.values.copy()
            offset = first_ordinal - assigned.start.ordinal(series.freq)
            output[offset : offset + length] = replacement_array
            updated[name] = BimetsSeries(
                output,
                start=assigned.start,
                freq=assigned.freq,
                metadata=series.metadata,
            )
        return BimetsDataset(updated, metadata=self.metadata)

    def range(
        self,
        *,
        kind: Literal["inner", "outer"] = "inner",
        names: Sequence[str] | None = None,
    ) -> tuple[YearPeriod, YearPeriod] | None:
        """Return the intersection or union of variable ranges.

        Parameters
        ----------
        kind : {"inner", "outer"}, default="inner"
            Select temporal intersection or union.
        names : sequence of str, optional
            Restrict the calculation to selected variables.

        Returns
        -------
        tuple of YearPeriod or None
            Inclusive bounds, or ``None`` for an empty intersection.

        Raises
        ------
        ValueError
            If selected series do not share a frequency.
        """
        selected = self if names is None else self.select(names)
        if selected.homogeneous_frequency is None:
            raise ValueError("range requires series with a common frequency")
        return get_range(*selected.values(), kind=kind)

    def align(
        self,
        *,
        kind: Literal["inner", "outer"] = "inner",
        names: Sequence[str] | None = None,
    ) -> BimetsDataset:
        """Project selected variables to a common range.

        Parameters
        ----------
        kind : {"inner", "outer"}, default="inner"
            Use the intersection or union. Outer alignment fills uncovered
            observations with ``numpy.nan``.
        names : sequence of str, optional
            Variables to align; unselected variables are omitted.

        Returns
        -------
        BimetsDataset
            Dataset whose variables share identical bounds.

        Raises
        ------
        ValueError
            If frequencies differ or an inner intersection is empty.
        """
        selected = self if names is None else self.select(names)
        common = selected.range(kind=kind)
        if common is None:
            raise ValueError("selected series ranges do not intersect")
        return BimetsDataset(
            {
                name: value.project(*common, extend=kind == "outer")
                for name, value in selected.items()
            },
            metadata=self.metadata,
        )

    def map(self, function: Callable[[BimetsSeries], BimetsSeries]) -> BimetsDataset:
        """Apply a transformation independently to every variable.

        Parameters
        ----------
        function : callable
            Function accepting and returning a ``BimetsSeries``.

        Returns
        -------
        BimetsDataset
            Dataset of transformed variables with the original names.

        Examples
        --------
        >>> from bimets import BimetsDataset, timeseries
        >>> data = BimetsDataset({
        ...     "gdp": timeseries([100, 102], freq="Q"),
        ...     "cpi": timeseries([98, 99], freq="Q"),
        ... })
        >>> changes = data.map(lambda series: series.delta_percent())
        >>> changes.names
        ('gdp', 'cpi')
        """
        transformed: dict[str, BimetsSeries] = {}
        for name, value in self.items():
            result = function(value)
            if not isinstance(result, BimetsSeries):
                raise TypeError("dataset map function must return BimetsSeries")
            transformed[name] = result
        return BimetsDataset(transformed, metadata=self.metadata)

    def to_pandas(
        self,
        *,
        index: Literal["year-period", "period", "datetime"] = "year-period",
        date_in_period: Literal["first", "last"] = "last",
    ) -> dict[str, Any]:
        """Convert every variable to a pandas Series.

        Parameters
        ----------
        index : {"year-period", "period", "datetime"}, default="year-period"
            Pandas index representation.
        date_in_period : {"first", "last"}, default="last"
            Timestamp boundary for datetime indexes.

        Returns
        -------
        dict of str to pandas.Series
            Converted variables in dataset order.
        """
        return {
            name: to_pandas(
                value,
                index=index,
                date_in_period=date_in_period,
            )
            for name, value in self.items()
        }

    @classmethod
    def from_pandas(
        cls,
        series: Mapping[str, Any],
        *,
        freq: int | str | Frequency | None = None,
    ) -> BimetsDataset:
        """Construct a dataset from named pandas Series.

        Parameters
        ----------
        series : mapping of str to pandas.Series
            Named regular pandas series.
        freq : int, str, or Frequency, optional
            Explicit frequency used when it cannot be inferred.

        Returns
        -------
        BimetsDataset
            Converted immutable collection.
        """
        return cls(
            {name: from_pandas(value, freq=freq) for name, value in series.items()}
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        freq: int | str | Frequency | None = None,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> BimetsDataset:
        """Construct a homogeneous dataset from a pandas DataFrame.

        Parameters
        ----------
        frame : pandas.DataFrame
            Non-empty table whose columns are variable names and whose index is
            a regular ``PeriodIndex``, ``DatetimeIndex``, or year-period
            ``MultiIndex`` accepted by :func:`bimets.from_pandas`.
        freq : int, str, or Frequency, optional
            Explicit frequency. It defaults to ``frame.attrs`` metadata when
            produced by :meth:`to_frame`, or to index inference when possible.
        metadata : mapping, optional
            Dataset metadata. It defaults to metadata stored by :meth:`to_frame`.

        Returns
        -------
        BimetsDataset
            Immutable dataset containing one series per column.

        Raises
        ------
        TypeError
            If ``frame`` is not a pandas DataFrame.
        ValueError
            If the table is empty or has invalid column names.

        Examples
        --------
        >>> import pandas as pd
        >>> from bimets import BimetsDataset
        >>> frame = pd.DataFrame(
        ...     {"gdp": [100, 102], "rate": [2.0, 2.5]},
        ...     index=pd.period_range("2020Q1", periods=2, freq="Q"),
        ... )
        >>> data = BimetsDataset.from_frame(frame)
        >>> data.names, data["gdp"].values.tolist()
        (('gdp', 'rate'), [100.0, 102.0])
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("from_frame expects a pandas DataFrame")
        if frame.empty or len(frame.columns) == 0:
            raise ValueError("pandas DataFrame must not be empty")
        if not frame.columns.is_unique:
            raise ValueError("DataFrame columns must not contain duplicates")
        if any(not isinstance(name, str) or name == "" for name in frame.columns):
            raise ValueError("DataFrame columns must be non-empty strings")

        stored_frequency = frame.attrs.get("bimets_frequency")
        effective_frequency = freq if freq is not None else stored_frequency
        stored_metadata = frame.attrs.get("bimets_metadata")
        effective_metadata = metadata if metadata is not None else stored_metadata
        if effective_metadata is not None and not isinstance(
            effective_metadata, Mapping
        ):
            raise TypeError("bimets_metadata must be a mapping")
        return cls(
            {
                name: from_pandas(frame[name], freq=effective_frequency)
                for name in frame.columns
            },
            metadata=effective_metadata,
        )

    def to_frame(
        self,
        *,
        start: YearPeriod | tuple[int, int] | None = None,
        end: YearPeriod | tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Align a homogeneous dataset into a pandas DataFrame.

        Parameters
        ----------
        start, end : YearPeriod or tuple of int, optional
            Inclusive table bounds. Defaults to the outer dataset range.

        Returns
        -------
        pandas.DataFrame
            Variables as columns on a year-period ``MultiIndex``.

        Raises
        ------
        ValueError
            If the dataset contains mixed frequencies or bounds are invalid.
        """
        freq = self.homogeneous_frequency
        if freq is None:
            raise ValueError("to_frame requires series with a common frequency")
        frame = pd.DataFrame(
            tabulate(*self.values(), headers=self.names, start=start, end=end)
        )
        common = get_range(*self.values(), kind="outer")
        assert common is not None
        selected_start = (
            common[0]
            if start is None
            else YearPeriod.normalize(*start, freq)
            if isinstance(start, tuple)
            else start
        )
        periods = [
            selected_start.shift(position, freq) for position in range(len(frame))
        ]
        frame.index = pd.MultiIndex.from_arrays(
            [
                [period.year for period in periods],
                [period.period for period in periods],
            ],
            names=["year", "period"],
        )
        frame.attrs["bimets_frequency"] = int(freq)
        frame.attrs["bimets_metadata"] = dict(self.metadata)
        return frame

    def to_csv(self, path: str | Path, **options: Any) -> Path:
        """Export this dataset to a BIMETS-compatible CSV file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.
        **options
            Additional options forwarded to :func:`bimets_to_csv`.

        Returns
        -------
        pathlib.Path
            Destination path.
        """
        return bimets_to_csv(self, path, **options)

    @classmethod
    def from_csv(cls, path: str | Path, **options: Any) -> BimetsDataset:
        """Import a dataset from a BIMETS-compatible CSV file.

        Parameters
        ----------
        path : str or pathlib.Path
            Source file.
        **options
            Additional options forwarded to :func:`csv_to_bimets`.

        Returns
        -------
        BimetsDataset
            Imported named collection.
        """
        return cls(csv_to_bimets(path, **options))

    def __repr__(self) -> str:
        freq = self.homogeneous_frequency
        frequency_text = "mixed" if freq is None else str(int(freq))
        return f"BimetsDataset(names={self.names!r}, freq={frequency_text})"


def _normalize_bound(
    value: YearPeriod | tuple[int, int], freq: Frequency
) -> YearPeriod:
    """Normalize bound for internal processing."""
    return (
        YearPeriod.normalize(*value, freq)
        if isinstance(value, tuple)
        else YearPeriod.normalize(value.year, value.period, freq)
    )
