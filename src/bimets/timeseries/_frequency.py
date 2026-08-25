"""Supported BIMETS frequencies."""

from __future__ import annotations

from enum import IntEnum


class Frequency(IntEnum):
    """Supported numbers of observations in a BIMETS year.

    The integer value of each member is the fixed number of periods allocated
    to a year. Daily series use 366 slots so that the same year-period indexing
    works for leap and non-leap years.

    Examples
    --------
    >>> from bimets import Frequency
    >>> int(Frequency.QUARTERLY)
    4
    >>> Frequency.parse("M")
    <Frequency.MONTHLY: 12>
    """

    YEARLY = 1
    SEMIANNUAL = 2
    THREE_PER_YEAR = 3
    QUARTERLY = 4
    MONTHLY = 12
    PERIODS_24 = 24
    PERIODS_36 = 36
    WEEKLY = 53
    DAILY = 366

    @classmethod
    def parse(cls, value: int | str | Frequency) -> Frequency:
        """Return the frequency represented by a value or alias.

        Parameters
        ----------
        value : int, str, or Frequency
            Existing member, supported integer value, or one of ``A``, ``Y``,
            ``S``, ``Q``, ``M``, ``W``, and ``D`` (case-insensitive).

        Returns
        -------
        Frequency
            Normalized enumeration member.

        Raises
        ------
        ValueError
            If the value does not identify a supported frequency.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError("frequency must not be boolean")
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError as error:
                raise ValueError(f"unsupported BIMETS frequency: {value}") from error
        if isinstance(value, str):
            aliases = {
                "A": cls.YEARLY,
                "Y": cls.YEARLY,
                "S": cls.SEMIANNUAL,
                "Q": cls.QUARTERLY,
                "M": cls.MONTHLY,
                "W": cls.WEEKLY,
                "D": cls.DAILY,
            }
            normalized = value.strip().upper()
            if normalized in aliases:
                return aliases[normalized]
        raise ValueError(f"unsupported BIMETS frequency: {value!r}")
