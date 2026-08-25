"""Year-period indexing independent of calendar libraries."""

from dataclasses import dataclass

from bimets.timeseries._frequency import Frequency


@dataclass(frozen=True, order=True, slots=True)
class YearPeriod:
    """A one-based period within a year.

    Parameters
    ----------
    year : int
        Calendar year.
    period : int
        Positive one-based period. Values above a particular frequency are
        allowed until normalized in a frequency-aware operation.

    Examples
    --------
    >>> from bimets import YearPeriod
    >>> YearPeriod.normalize(2020, 5, "Q")
    YearPeriod(year=2021, period=1)
    """

    year: int
    period: int

    def __post_init__(self) -> None:
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise TypeError("year must be an integer")
        if isinstance(self.period, bool) or not isinstance(self.period, int):
            raise TypeError("period must be an integer")
        if self.period < 1:
            raise ValueError("period must be positive")

    @classmethod
    def normalize(
        cls,
        year: int,
        period: int,
        freq: int | str | Frequency,
    ) -> "YearPeriod":
        """Normalize an overflowing one-based period.

        Parameters
        ----------
        year, period : int
            Unnormalized year and positive one-based period.
        freq : int, str, or Frequency
            Periods per year.

        Returns
        -------
        YearPeriod
            Equivalent index whose period falls within the selected year.
        """
        periods_per_year = _periods_per_year(freq)
        if isinstance(period, bool) or not isinstance(period, int):
            raise TypeError("period must be an integer")
        if period < 1:
            raise ValueError("period must be positive")
        year_offset, zero_based_period = divmod(period - 1, periods_per_year)
        return cls(year + year_offset, zero_based_period + 1)

    def ordinal(self, freq: int | str | Frequency) -> int:
        """Return a sortable integer for this year-period.

        Parameters
        ----------
        freq : int, str, or Frequency
            Periods per year used to interpret the index.

        Returns
        -------
        int
            Zero-based absolute period count relative to year zero.
        """
        periods_per_year = _periods_per_year(freq)
        normalized = self.normalize(self.year, self.period, periods_per_year)
        return normalized.year * periods_per_year + normalized.period - 1

    def shift(
        self,
        periods: int,
        freq: int | str | Frequency,
    ) -> "YearPeriod":
        """Shift this index by an integer number of periods.

        Parameters
        ----------
        periods : int
            Signed number of periods to add.
        freq : int, str, or Frequency
            Periods per year used for the shift.

        Returns
        -------
        YearPeriod
            Shifted, normalized index.
        """
        if isinstance(periods, bool) or not isinstance(periods, int):
            raise TypeError("periods must be an integer")
        periods_per_year = _periods_per_year(freq)
        shifted = self.ordinal(periods_per_year) + periods
        year, zero_based_period = divmod(shifted, periods_per_year)
        return YearPeriod(year, zero_based_period + 1)


def normalize_year_period(
    year_period: YearPeriod | tuple[int, int],
    freq: int | str | Frequency,
) -> YearPeriod:
    """Normalize a year-period for a selected frequency.

    Parameters
    ----------
    year_period : YearPeriod or tuple of int
        Year and positive one-based period.
    freq : int, str, or Frequency
        Periods per year.

    Returns
    -------
    YearPeriod
        Normalized index.

    Examples
    --------
    >>> from bimets import normalize_year_period
    >>> normalize_year_period((2020, 13), "M")
    YearPeriod(year=2021, period=1)
    """
    if isinstance(year_period, YearPeriod):
        year, period = year_period.year, year_period.period
    else:
        year, period = year_period
    return YearPeriod.normalize(year, period, freq)


def num_periods(
    start: YearPeriod | tuple[int, int],
    end: YearPeriod | tuple[int, int],
    freq: int | str | Frequency,
) -> int:
    """Return the signed distance between two year-period indexes.

    Parameters
    ----------
    start, end : YearPeriod or tuple of int
        Boundary indexes. The calculation is ``end - start`` and is therefore
        not an inclusive observation count.
    freq : int, str, or Frequency
        Periods per year.

    Returns
    -------
    int
        Signed number of periods separating the indexes.

    Examples
    --------
    >>> from bimets import num_periods
    >>> num_periods((2020, 1), (2021, 1), "Q")
    4
    """
    normalized_start = normalize_year_period(start, freq)
    normalized_end = normalize_year_period(end, freq)
    return normalized_end.ordinal(freq) - normalized_start.ordinal(freq)


def _from_ordinal(ordinal: int, freq: Frequency) -> YearPeriod:
    """Convert an absolute ordinal to a normalized year-period index."""
    year, zero_based_period = divmod(ordinal, int(freq))
    return YearPeriod(year, zero_based_period + 1)


def _periods_per_year(freq: int | str | Frequency) -> int:
    """Return the number of periods represented by a frequency."""
    if isinstance(freq, bool):
        raise ValueError("frequency must be a positive integer")
    if isinstance(freq, int):
        if freq <= 0:
            raise ValueError("frequency must be a positive integer")
        return freq
    return int(Frequency.parse(freq))
