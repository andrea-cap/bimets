#!/usr/bin/env python3
"""BIMETS time-series operations demonstrated alongside the R script."""

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from bimets import *


# Helpers to reproduce R base functions.
def section(title: str) -> None:
    """Print a section heading like the R demo."""
    separator = "=" * 78
    print(f"\n{separator}\n{title}\n{separator}")


def show_object(label: str, value: Any) -> Any:
    """Print a labeled value using its natural representation."""
    print(f"\n--- {label} ---")
    print(value)
    return value


def show_table(
    *series: BimetsSeries,
    headers: tuple[str, ...],
    start: tuple[int, int] | None = None,
    end: tuple[int, int] | None = None,
    digits: int | None = None,
) -> None:
    """Print TABIT output with the display-only rounding used by R."""
    table = TABIT(*series, headers=headers, start=start, end=end)
    print(table if digits is None else table.round(digits))


def main() -> None:
    """Run the complete time-series demonstration."""
    # ---- Series creation and metadata ----------------------------------------

    section("SERIES CREATION AND METADATA")

    annual_series = TSERIES(
        [100, 105, 111, 118, 126],
        start=(2020, 1),
        freq="A",
        title="Annual index",
        units="index",
        source="synthetic",
        scale_factor=0,
    )

    semiannual_series = TSERIES(
        range(1, 9),
        start=(2023, 1),
        freq="S",
    )

    quarterly_series = TSERIES(
        range(100, 124),
        start=(2023, 1),
        freq="Q",
        title="Quarterly index",
        units="index",
    )

    monthly_series = TSERIES(
        range(100, 124),
        start=(2024, 1),
        freq="M",
        title="Monthly index",
        units="index",
    )

    weekly_series = TSERIES(
        range(1, 11),
        start=(2025, 1),
        freq="W",
    )

    # Python converts a date explicitly to the BIMETS year-period index.
    daily_series = TSERIES(
        range(1, 21),
        start=date2yp(date.fromisoformat("2025-01-01"), "D"),
        freq="D",
    )

    show_object("Annual series", annual_series)
    show_object("Quarterly series", quarterly_series)
    show_object("Monthly series", monthly_series)

    # TIMESERIES is an alias of TSERIES in the Python port too.
    alias_series = TIMESERIES(
        [10, 20, 30],
        start=(2022, 1),
        freq="A",
    )
    show_object("Series created with TIMESERIES()", alias_series)

    # Keep the other series alive to exercise every supported frequency.
    assert is_bimets(semiannual_series)
    assert is_bimets(weekly_series)

    # ---- Inspection and compliance ------------------------------------------

    section("INSPECTION AND COMPLIANCE")

    show_object("Python class", type(monthly_series))
    show_object("BIMETS-compliant?", is_bimets(monthly_series))
    show_object("Range and frequency with TSLOOK()", TSLOOK(monthly_series))

    show_object(
        "Frequencies with TSINFO(..., mode='FREQ')",
        [
            TSINFO(item, mode="FREQ")
            for item in (annual_series, quarterly_series, monthly_series)
        ],
    )

    # START is the Python counterpart of START2.
    show_object(
        "Initial period with TSINFO(..., mode='START')",
        [
            TSINFO(item, mode="START")
            for item in (annual_series, quarterly_series, monthly_series)
        ],
    )

    show_object(
        "Annual-series title",
        TSINFO(annual_series, mode="TITLE"),
    )
    show_object("Monthly-series dates", GETDATE(monthly_series))
    show_object(
        "Monthly-series year-period values",
        GETYEARPERIOD(monthly_series),
    )

    print("\nTABIT with multiple series:")
    show_table(
        quarterly_series,
        TSLAG(quarterly_series, 1),
        start=(2023, 1),
        end=(2024, 2),
        headers=("original", "lag_1"),
    )

    # ---- Indexing and modification ------------------------------------------

    section("INDEXING AND MODIFICATION")

    # By position: Python uses zero-based indexes and stop-exclusive slices.
    show_object("First three observations", monthly_series[0:3])

    modified_series = monthly_series.with_values(0, 999)
    show_object("Immutable update by position", modified_series[0:3])

    # By year-period: the outer list reproduces R double-bracket indexing.
    show_object("Observation at 2024-03", monthly_series[[2024, 3]])
    show_object(
        "Range 2024-03 / 2024-06",
        monthly_series[[[2024, 3], [2024, 6]]],
    )

    modified_series = monthly_series.with_values([2024, 3], np.nan)
    modified_series = modified_series.with_values(
        [[2024, 4], [2024, 6]],
        [200, 201, 202],
    )
    show_object("Immutable updates by year-period", modified_series)

    # An update beyond the current end extends the new series.
    index_extended_series = monthly_series.with_values(
        [2026, 1],
        [200, 201, 202],
    )
    show_object(
        "Extension through an immutable update",
        TSLOOK(index_extended_series),
    )

    # By date.
    show_object(
        "Daily observation on January 5",
        daily_series["2025-01-05"],
    )
    show_object(
        "Daily range",
        daily_series["2025-01-05/2025-01-10"],
    )

    modified_daily_series = daily_series.with_values("2025-01-05", np.nan)
    modified_daily_series = modified_daily_series.with_values(
        "2025-01-07/2025-01-09", 50
    )
    show_object("Immutable update by date", modified_daily_series)

    # ---- Lags, leads, and differences ---------------------------------------

    section("LAGS, LEADS, AND DIFFERENCES")

    lag_1 = TSLAG(monthly_series, 1)
    lead_1 = TSLEAD(monthly_series, 1)
    delta_1 = TSDELTA(monthly_series, lag=1)
    second_order_delta = TSDELTA(monthly_series, lag=1, order=2)
    percentage_delta = TSDELTAP(monthly_series, lag=1)
    annual_percentage_delta = TSDELTAP(monthly_series, lag=12)
    log_delta = TSDELTALOG(monthly_series, lag=1)

    show_table(
        monthly_series,
        lag_1,
        lead_1,
        delta_1,
        percentage_delta,
        log_delta,
        start=(2024, 1),
        end=(2024, 8),
        headers=(
            "original",
            "lag_1",
            "lead_1",
            "delta",
            "delta_pct",
            "log_delta",
        ),
        digits=4,
    )

    show_object("Second-order difference", second_order_delta)
    show_object("Twelve-month percentage change", annual_percentage_delta)

    # ---- Moving and cumulative operations ----------------------------------

    section("MOVING AND CUMULATIVE OPERATIONS")

    series_with_na = TSERIES(
        [1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10],
        start=(2024, 1),
        freq="M",
    )

    trailing_moving_average = MOVAVG(
        series_with_na,
        3,
        direction="BACK",
        skip_missing=True,
    )
    centered_moving_average = MOVAVG(
        series_with_na,
        3,
        direction="CENTER",
        skip_missing=True,
    )
    moving_sum = MOVSUM(
        series_with_na,
        3,
        direction="BACK",
        skip_missing=True,
    )

    # MOVTOT is the same callable as MOVSUM.
    moving_sum_alias = MOVTOT(
        series_with_na,
        3,
        skip_missing=True,
    )
    cumulative_sum = CUMSUM(series_with_na, skip_missing=True)

    factor_series = TSERIES(
        [1.02, 1.01, 0.99, 1.03, 1.02],
        start=(2024, 1),
        freq="Q",
    )
    cumulative_product = CUMPROD(factor_series)

    show_table(
        series_with_na,
        trailing_moving_average,
        centered_moving_average,
        moving_sum,
        cumulative_sum,
        headers=(
            "original",
            "trailing_average",
            "centered_average",
            "moving_sum",
            "cumulative_sum",
        ),
        digits=3,
    )

    show_object(
        "MOVSUM and MOVTOT return the same values",
        np.allclose(
            moving_sum.values,
            moving_sum_alias.values,
            equal_nan=True,
        ),
    )

    show_table(
        factor_series,
        cumulative_product,
        headers=("factor", "cumulative_product"),
        digits=5,
    )

    # ---- Projection, extension, trim, join, and merge -----------------------

    section("PROJECTION, EXTENSION, TRIM, JOIN, AND MERGE")

    projection = TSPROJECT(monthly_series, (2024, 4), (2024, 9))
    show_object("TSPROJECT: April-September 2024", projection)

    missing_extension = TSEXTEND(
        annual_series,
        back_to=(2018, 1),
        up_to=(2027, 1),
        mode="MISSING",
    )
    constant_extension = TSEXTEND(
        annual_series,
        up_to=(2027, 1),
        mode="CONSTANT",
    )
    rate_extension = TSEXTEND(
        annual_series,
        up_to=(2027, 1),
        mode="MYRATE",
        factor=5,
    )

    show_table(
        missing_extension,
        constant_extension,
        rate_extension,
        headers=("missing", "constant", "increase_5"),
    )

    series_to_trim = TSERIES(
        [np.nan, np.nan, 10, 20, 30, np.nan],
        start=(2020, 1),
        freq="A",
    )
    trimmed_series = TSTRIM(series_to_trim)
    assert trimmed_series is not None
    show_table(
        series_to_trim,
        trimmed_series,
        headers=("with_missing_ends", "trimmed"),
    )

    historical_series = TSERIES(
        [10, 11, 12, 13, 14, 15],
        start=(2020, 1),
        freq="A",
    )
    new_series = TSERIES(
        [100, 101, 102, 103],
        start=(2023, 1),
        freq="A",
    )
    joined_series = TSJOIN(historical_series, new_series)
    show_table(
        historical_series,
        new_series,
        joined_series,
        headers=("historical_series", "new_series", "join"),
    )

    first_series = TSERIES(
        [1, 2, np.nan, np.nan, 5, 6],
        start=(2020, 1),
        freq="A",
    )
    second_series = TSERIES(
        [30, 40, 50, 60],
        start=(2022, 1),
        freq="A",
    )

    priority_merge = TSMERGE(first_series, second_series)
    # MV=FALSE in the R example corresponds to retaining the univariate result;
    # the Python API expresses the effective missing-value rule explicitly.
    sum_merge = TSMERGE(first_series, second_series, method="SUM", skip_missing=True)
    average_merge = TSMERGE(
        first_series, second_series, method="AVE", skip_missing=True
    )
    maximum_merge = TSMERGE(
        first_series, second_series, method="MAX", skip_missing=True
    )
    minimum_merge = TSMERGE(
        first_series, second_series, method="MIN", skip_missing=True
    )

    show_table(
        first_series,
        second_series,
        priority_merge,
        sum_merge,
        average_merge,
        maximum_merge,
        minimum_merge,
        headers=(
            "first_series",
            "second_series",
            "priority",
            "sum",
            "average",
            "max",
            "min",
        ),
    )

    # ---- Rebasing ------------------------------------------------------------

    section("REBASING")

    index_base_2024 = INDEXNUM(monthly_series, 2024)
    show_table(
        monthly_series,
        index_base_2024,
        headers=("original", "base_2024_equals_100"),
        digits=3,
    )

    # ---- Aggregation and disaggregation ------------------------------------

    section("AGGREGATION AND DISAGGREGATION")

    quarterly_average = QUARTERLY(monthly_series, method="AVE")
    quarterly_sum = QUARTERLY(monthly_series, method="SUM")
    quarterly_stock = QUARTERLY(monthly_series, method="STOCK")
    show_table(
        quarterly_average,
        quarterly_sum,
        quarterly_stock,
        headers=("average", "sum", "stock"),
    )

    annual_average = YEARLY(monthly_series, method="AVE")
    annual_sum = YEARLY(monthly_series, method="SUM")
    annual_stock = YEARLY(monthly_series, method="STOCK")
    show_table(
        annual_average,
        annual_sum,
        annual_stock,
        headers=("average", "sum", "stock"),
    )

    semiannual_average = SEMIANNUAL(monthly_series, method="AVE")
    show_object("Monthly -> semiannual using average", semiannual_average)

    simple_annual_series = TSERIES(
        [100, 110, 125],
        start=(2022, 1),
        freq="A",
    )
    annual_to_quarterly = QUARTERLY(
        simple_annual_series,
        method="INTERP_CENTER",
    )
    annual_to_monthly = MONTHLY(
        simple_annual_series,
        method="INTERP_CENTER",
    )
    monthly_to_daily = DAILY(
        TSERIES(
            [100, 103, 107],
            start=(2025, 1),
            freq="M",
        ),
        method="INTERP_CENTER",
    )

    # Keep the quarterly conversion among the checks shared with the R example.
    assert is_bimets(annual_to_quarterly)
    show_object(
        "First observations of the interpolated monthly series",
        annual_to_monthly[0:8],
    )
    show_object(
        "First observations of the interpolated daily series",
        monthly_to_daily[0:10],
    )

    long_daily_series = TSERIES(
        range(1, 91),
        start=date2yp(date.fromisoformat("2025-01-01"), "D"),
        freq="D",
    )
    daily_to_monthly_sum = MONTHLY(long_daily_series, method="SUM")
    daily_to_monthly_average = MONTHLY(long_daily_series, method="AVE")
    daily_to_monthly_stock = MONTHLY(long_daily_series, method="STOCK")
    show_table(
        daily_to_monthly_sum,
        daily_to_monthly_average,
        daily_to_monthly_stock,
        headers=("sum", "average", "stock"),
    )

    # ---- Conversion between classes -----------------------------------------

    section("BIMETS / PANDAS CONVERSION")

    # pandas.Series provides interoperability in place of the R ts/xts types.
    pandas_series = to_pandas(monthly_series, index="datetime")
    show_object("Class after to_pandas()", type(pandas_series))
    show_object("First pandas rows", pandas_series.head())

    back_to_bimets_series = from_pandas(pandas_series)
    show_object(
        "BIMETS-compliant again?",
        is_bimets(back_to_bimets_series),
    )

    period_index_series = to_pandas(back_to_bimets_series, index="period")
    show_object("Class with pandas PeriodIndex", type(period_index_series.index))

    # ---- CSV import/export ---------------------------------------------------

    section("CSV IMPORT / EXPORT")

    output_directory = Path.cwd()
    csv_file = output_directory / "bimets_series.csv"

    series_list = {
        "monthly": monthly_series,
        "quarterly": quarterly_series,
    }

    BIMETS2CSV(
        series_list,
        csv_file,
        delimiter=";",
        decimal_separator=",",
        date_format="%Y%m%d",
        overwrite=True,
    )
    imported_series = CSV2BIMETS(
        csv_file,
        delimiter=";",
        decimal_separator=",",
        date_format="%Y%m%d",
    )

    print(f"\nCSV created at:\n{csv_file}")
    show_object("Names of imported series", list(imported_series))

    for name, series in imported_series.items():
        print(f"\nImported series: {name}")
        print(TSLOOK(series))

    # ---- Final checks --------------------------------------------------------

    section("FINAL CHECKS")

    assert is_bimets(annual_series)
    assert is_bimets(monthly_series)
    assert is_bimets(quarterly_average)
    assert is_bimets(annual_to_monthly)
    assert csv_file.exists()

    print(f"\nExecution completed.\nOutput available at:\n{csv_file}")


if __name__ == "__main__":
    main()
