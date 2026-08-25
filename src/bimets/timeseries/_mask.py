"""Tri-state boolean series used by time-series comparisons."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bimets.timeseries._alignment import align_values
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod, normalize_year_period

MaskValue = bool | None


class BimetsMask:
    """An immutable regular boolean series with missing-value support.

    Parameters
    ----------
    values : array-like
        Non-empty, one-dimensional values containing only booleans, ``None``,
        or ``numpy.nan``. Missing values are normalized to ``None``.
    start : YearPeriod or tuple of int
        Year and one-based period of the first value.
    freq : int, str, or Frequency
        Number of periods per year or a supported alias.

    Notes
    -----
    Logical operators implement three-valued Kleene logic. A mask cannot be
    converted directly to ``bool``; use :meth:`any` or :meth:`all`.

    Examples
    --------
    >>> from bimets import BimetsMask
    >>> mask = BimetsMask([True, None, False], start=(2020, 1), freq="Q")
    >>> list(mask)
    [True, None, False]
    >>> list(mask & True)
    [True, None, False]
    """

    __slots__ = ("_freq", "_start", "_values")

    def __init__(
        self,
        values: ArrayLike | Sequence[MaskValue],
        *,
        start: YearPeriod | tuple[int, int],
        freq: int | str | Frequency,
    ) -> None:
        parsed_frequency = Frequency.parse(freq)
        parsed_start = normalize_year_period(start, parsed_frequency)
        source = np.asarray(cast(Any, values), dtype=object)
        if source.ndim != 1 or source.size == 0:
            raise ValueError(
                "BimetsMask values must be a non-empty one-dimensional array"
            )
        end = parsed_start.shift(int(source.size) - 1, parsed_frequency)
        if parsed_start.year < 1 or end.year > 9999:
            raise ValueError("BimetsMask must lie in the year range 1-9999")
        normalized = np.empty(source.size, dtype=object)
        for index, value in enumerate(source):
            if value is None or (
                isinstance(value, (float, np.floating)) and np.isnan(value)
            ):
                normalized[index] = None
            elif isinstance(value, (bool, np.bool_)):
                normalized[index] = bool(value)
            else:
                raise TypeError("BimetsMask values must be boolean or missing")
        normalized.setflags(write=False)
        self._values = normalized
        self._start = parsed_start
        self._freq = parsed_frequency

    @property
    def values(self) -> NDArray[np.object_]:
        """Read-only values containing ``True``, ``False``, or ``None``."""
        return self._values

    @property
    def start(self) -> YearPeriod:
        """First mask value's year-period."""
        return self._start

    @property
    def end(self) -> YearPeriod:
        """Last mask value's year-period."""
        return self.start.shift(len(self) - 1, self.freq)

    @property
    def freq(self) -> Frequency:
        """Number of mask values per year."""
        return self._freq

    def __len__(self) -> int:
        return int(self.values.size)

    def __iter__(self) -> Iterator[MaskValue]:
        return (None if value is None else bool(value) for value in self.values)

    @overload
    def __getitem__(self, key: int) -> MaskValue: ...

    @overload
    def __getitem__(self, key: slice) -> BimetsMask: ...

    def __getitem__(self, key: int | slice) -> MaskValue | BimetsMask:
        if isinstance(key, int):
            value = self.values[key]
            return None if value is None else bool(value)
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("BimetsMask slicing does not support a step")
        if stop <= start:
            raise ValueError("BimetsMask slices must not be empty")
        return BimetsMask(
            self.values[start:stop],
            start=self.start.shift(start, self.freq),
            freq=self.freq,
        )

    def __and__(self, other: object) -> BimetsMask:
        return self._logical(other, "and")

    def __rand__(self, other: object) -> BimetsMask:
        return self._logical(other, "and")

    def __or__(self, other: object) -> BimetsMask:
        return self._logical(other, "or")

    def __ror__(self, other: object) -> BimetsMask:
        return self._logical(other, "or")

    def __xor__(self, other: object) -> BimetsMask:
        return self._logical(other, "xor")

    def __rxor__(self, other: object) -> BimetsMask:
        return self._logical(other, "xor")

    def __invert__(self) -> BimetsMask:
        return BimetsMask(
            [None if value is None else not value for value in self.values],
            start=self.start,
            freq=self.freq,
        )

    def any(self, *, skip_missing: bool = True) -> bool | None:
        """Return whether any observation is true.

        Parameters
        ----------
        skip_missing : bool, default=True
            Ignore missing observations. If false and no value is true, return
            ``None`` when at least one value is missing.

        Returns
        -------
        bool or None
            Three-valued reduction of the mask.
        """
        values = list(self)
        if any(value is True for value in values):
            return True
        if not skip_missing and any(value is None for value in values):
            return None
        return False

    def all(self, *, skip_missing: bool = True) -> bool | None:
        """Return whether all observations are true.

        Parameters
        ----------
        skip_missing : bool, default=True
            Ignore missing observations. If false and no value is false,
            return ``None`` when at least one value is missing.

        Returns
        -------
        bool or None
            Three-valued reduction of the mask.
        """
        values = list(self)
        if any(value is False for value in values):
            return False
        if not skip_missing and any(value is None for value in values):
            return None
        return True

    def __bool__(self) -> bool:
        raise ValueError(
            "the truth value of a BimetsMask is ambiguous; use any() or all()"
        )

    def _logical(self, other: object, operation: str) -> BimetsMask:
        """Apply a three-valued logical operation to aligned masks."""
        if isinstance(other, BimetsMask):
            left, right, start = align_values(self, other)
        elif isinstance(other, (bool, np.bool_)):
            left, right, start = (
                self.values,
                np.full(len(self), bool(other)),
                self.start,
            )
        else:
            raise TypeError("logical operations require a BimetsMask or boolean")
        values = [
            _kleene(left_value, right_value, operation)
            for left_value, right_value in zip(left, right, strict=True)
        ]
        return BimetsMask(values, start=start, freq=self.freq)

    def __repr__(self) -> str:
        return (
            f"BimetsMask(length={len(self)}, start={self.start!r}, "
            f"end={self.end!r}, freq={int(self.freq)})"
        )


def _kleene(left: object, right: object, operation: str) -> MaskValue:
    """Apply a scalar three-valued Boolean operation."""
    if operation == "and":
        if left is False or right is False:
            return False
        return None if left is None or right is None else True
    if operation == "or":
        if left is True or right is True:
            return True
        return None if left is None or right is None else False
    if left is None or right is None:
        return None
    return bool(left) ^ bool(right)
