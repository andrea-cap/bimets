"""Shared temporal alignment helpers."""

from __future__ import annotations

from typing import Any, Protocol

from numpy.typing import NDArray

from bimets.timeseries._frequency import Frequency
from bimets.timeseries._index import YearPeriod, _from_ordinal


class IndexedValues(Protocol):
    @property
    def start(self) -> YearPeriod:
        """Return the first indexed period."""
        ...

    @property
    def end(self) -> YearPeriod:
        """Return the last indexed period."""
        ...

    @property
    def freq(self) -> Frequency:
        """Return the observation frequency."""
        ...

    @property
    def values(self) -> NDArray[Any]:
        """Return the indexed values."""
        ...


def align_values(
    left: IndexedValues, right: IndexedValues
) -> tuple[NDArray[Any], NDArray[Any], YearPeriod]:
    """Align two regular indexed arrays over their temporal intersection."""
    if left.freq != right.freq:
        raise ValueError("series must have the same frequency")
    freq = left.freq
    left_start = left.start.ordinal(freq)
    right_start = right.start.ordinal(freq)
    start = max(left_start, right_start)
    end = min(left.end.ordinal(freq), right.end.ordinal(freq))
    if end < start:
        raise ValueError("series ranges do not intersect")
    size = end - start + 1
    left_offset = start - left_start
    right_offset = start - right_start
    return (
        left.values[left_offset : left_offset + size],
        right.values[right_offset : right_offset + size],
        _from_ordinal(start, freq),
    )
