"""Functional API for BIMETS time-series operations."""

from bimets.timeseries._index import YearPeriod
from bimets.timeseries._manipulation import (
    cumulative_product,
    cumulative_sum,
    extend,
    index_number,
    join,
    merge,
    moving_average,
    moving_sum,
)
from bimets.timeseries._series import BimetsSeries


def tslag(series: BimetsSeries, periods: int = 1) -> BimetsSeries:
    """Shift a series forward in time.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    periods : int, default=1
        Signed number of periods by which to move the index.

    Returns
    -------
    BimetsSeries
        Series with unchanged observations and a shifted range.

    See Also
    --------
    BimetsSeries.lag : Equivalent method form.

    Examples
    --------
    >>> from bimets import timeseries, tslag
    >>> series = timeseries([1, 2], start=(2020, 1), freq="Q")
    >>> tslag(series).start
    YearPeriod(year=2020, period=2)
    """
    return series.lag(periods)


def tslead(series: BimetsSeries, periods: int = 1) -> BimetsSeries:
    """Shift a series backward in time.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    periods : int, default=1
        Signed number of periods by which to move the index backward.

    Returns
    -------
    BimetsSeries
        Series with unchanged observations and a shifted range.

    See Also
    --------
    BimetsSeries.lead : Equivalent method form.
    """
    return series.lead(periods)


def tsdelta(series: BimetsSeries, lag: int = 1, order: int = 1) -> BimetsSeries:
    """Return lagged differences.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    lag : int, default=1
        Distance between observations in each difference.
    order : int, default=1
        Number of successive differences.

    Returns
    -------
    BimetsSeries
        Differenced series.

    See Also
    --------
    BimetsSeries.delta : Equivalent method form.
    """
    return series.delta(lag=lag, order=order)


def tsdeltalog(series: BimetsSeries, lag: int = 1) -> BimetsSeries:
    """Return lagged differences of natural logarithms.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    lag : int, default=1
        Distance between logged observations.

    Returns
    -------
    BimetsSeries
        Log-difference series.

    See Also
    --------
    BimetsSeries.delta_log : Equivalent method form.
    """
    return series.delta_log(lag=lag)


def tsdeltap(
    series: BimetsSeries,
    lag: int = 1,
    *,
    annualize: bool = False,
) -> BimetsSeries:
    """Return percentage changes over a selected lag.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    lag : int, default=1
        Distance between current and previous observations.
    annualize : bool, default=False
        Convert the selected growth factor to an annual rate.

    Returns
    -------
    BimetsSeries
        Percentage-change series.

    See Also
    --------
    BimetsSeries.delta_percent : Equivalent method form.
    """
    return series.delta_percent(lag=lag, annualize=annualize)


def tsproject(
    series: BimetsSeries,
    start: YearPeriod | tuple[int, int],
    end: YearPeriod | tuple[int, int],
    *,
    extend: bool = False,
) -> BimetsSeries:
    """Project a series into an inclusive year-period range.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    start, end : YearPeriod or tuple of int
        Requested bounds.
    extend : bool, default=False
        Fill uncovered requested periods with missing values instead of
        clipping to the source range.

    Returns
    -------
    BimetsSeries
        Projected series.

    See Also
    --------
    BimetsSeries.project : Equivalent method form.
    """
    return series.project(start, end, extend=extend)


def tstrim(
    series: BimetsSeries,
    value: float | None = None,
    *,
    leading: bool = True,
    trailing: bool = True,
) -> BimetsSeries | None:
    """Trim leading or trailing missing or target values.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    value : float, optional
        Value to trim; ``None`` selects missing observations.
    leading, trailing : bool, default=True
        Enable trimming at each boundary.

    Returns
    -------
    BimetsSeries or None
        Trimmed series, or ``None`` if every observation is removed.

    See Also
    --------
    BimetsSeries.trim : Equivalent method form.
    """
    return series.trim(value=value, leading=leading, trailing=trailing)


def cumsum(
    series: BimetsSeries,
    *,
    mode: str | None = None,
    skip_missing: bool = False,
    start: YearPeriod | tuple[int, int] | None = None,
    end: YearPeriod | tuple[int, int] | None = None,
) -> BimetsSeries:
    """Return cumulative sums.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    mode : {None, "yearly", "monthly"}, optional
        Optional reset group.
    skip_missing : bool, default=False
        Ignore missing observations instead of propagating them.
    start, end : YearPeriod or tuple of int, optional
        Inclusive calculation bounds, corresponding to BIMETS R ``TSRANGE``.
        An omitted bound defaults to the source boundary.

    Returns
    -------
    BimetsSeries
        Cumulative sums.

    See Also
    --------
    BimetsSeries.cumulative_sum : Equivalent method form.
    """
    return cumulative_sum(
        series,
        mode=mode,
        skip_missing=skip_missing,
        start=start,
        end=end,
    )


def cumprod(
    series: BimetsSeries,
    *,
    skip_missing: bool = False,
    start: YearPeriod | tuple[int, int] | None = None,
    end: YearPeriod | tuple[int, int] | None = None,
) -> BimetsSeries:
    """Return cumulative products.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    skip_missing : bool, default=False
        Ignore missing observations instead of propagating them.
    start, end : YearPeriod or tuple of int, optional
        Inclusive calculation bounds, corresponding to BIMETS R ``TSRANGE``.
        An omitted bound defaults to the source boundary.

    Returns
    -------
    BimetsSeries
        Cumulative products.

    See Also
    --------
    BimetsSeries.cumulative_product : Equivalent method form.
    """
    return cumulative_product(series, skip_missing=skip_missing, start=start, end=end)


def movavg(
    series: BimetsSeries,
    window: int,
    *,
    direction: str = "back",
    skip_missing: bool = False,
) -> BimetsSeries:
    """Return averages over complete moving windows.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    window : int
        Positive number of observations per window.
    direction : {"back", "center", "ahead"}, default="back"
        Window alignment.
    skip_missing : bool, default=False
        Ignore missing observations.

    Returns
    -------
    BimetsSeries
        Moving-average series.

    See Also
    --------
    BimetsSeries.moving_average : Equivalent method form.
    """
    return moving_average(
        series, window, direction=direction, skip_missing=skip_missing
    )


def movtot(
    series: BimetsSeries,
    window: int,
    *,
    direction: str = "back",
    skip_missing: bool = False,
) -> BimetsSeries:
    """Return totals over complete moving windows.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    window : int
        Positive number of observations per window.
    direction : {"back", "center", "ahead"}, default="back"
        Window alignment.
    skip_missing : bool, default=False
        Ignore missing observations.

    Returns
    -------
    BimetsSeries
        Moving-total series.

    See Also
    --------
    BimetsSeries.moving_sum : Equivalent method form.
    """
    return moving_sum(series, window, direction=direction, skip_missing=skip_missing)


def indexnum(series: BimetsSeries, base_year: int) -> BimetsSeries:
    """Rebase a series to 100 in a base year.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    base_year : int
        Year whose average defines the base.

    Returns
    -------
    BimetsSeries
        Rebased series.

    See Also
    --------
    BimetsSeries.index_number : Equivalent method form.
    """
    return index_number(series, base_year)


def tsmerge(
    *series: BimetsSeries,
    method: str | None = None,
    skip_missing: bool = True,
) -> BimetsSeries:
    """Merge series over their union range.

    Parameters
    ----------
    *series : BimetsSeries
        Two or more series with a common frequency.
    method : {None, "sum", "average", "max", "min"}, optional
        Aggregation applied where ranges overlap. ``None`` selects the first
        non-missing value according to argument order.
    skip_missing : bool, default=True
        Ignore missing or uncovered inputs. If false, propagate them.

    Returns
    -------
    BimetsSeries
        Merged series covering the union range.

    Examples
    --------
    >>> from bimets import timeseries, tsmerge
    >>> first = timeseries([1, 2], start=(2020, 1), freq="Q")
    >>> second = timeseries([10, 20], start=(2020, 2), freq="Q")
    >>> tsmerge(first, second, method="sum").values.tolist()
    [1.0, 12.0, 20.0]
    """
    return merge(*series, method=method, skip_missing=skip_missing)


def tsjoin(
    first: BimetsSeries,
    second: BimetsSeries,
    *,
    join_period: YearPeriod | tuple[int, int] | None = None,
    allow_gap: bool = False,
) -> BimetsSeries:
    """Join two series at a selected period.

    Parameters
    ----------
    first, second : BimetsSeries
        Series with a common frequency. Values before the join come from
        ``first`` and values from the join onward come from ``second``.
    join_period : YearPeriod or tuple of int, optional
        First period taken from ``second``. Defaults to its start.
    allow_gap : bool, default=False
        Fill a gap between inputs with missing observations.

    Returns
    -------
    BimetsSeries
        Joined series.
    """
    return join(first, second, join_period=join_period, allow_gap=allow_gap)


def tsextend(
    series: BimetsSeries,
    *,
    back_to: YearPeriod | tuple[int, int] | None = None,
    up_to: YearPeriod | tuple[int, int] | None = None,
    mode: str = "growth",
    factor: float | None = None,
) -> BimetsSeries:
    """Extend a series using a BIMETS extrapolation mode.

    Parameters
    ----------
    series : BimetsSeries
        Input series.
    back_to, up_to : YearPeriod or tuple of int, optional
        New inclusive bounds.
    mode : str, default="growth"
        Extension rule; see :meth:`BimetsSeries.extend`.
    factor : float, optional
        User value for ``myconst`` or ``myrate``.

    Returns
    -------
    BimetsSeries
        Extended series.

    See Also
    --------
    BimetsSeries.extend : Equivalent method form.
    """
    return extend(series, back_to=back_to, up_to=up_to, mode=mode, factor=factor)
