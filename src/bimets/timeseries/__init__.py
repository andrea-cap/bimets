"""BIMETS time-series representation and operations."""

from bimets.timeseries._calendar import (
    date_to_year_period,
    get_dates,
    get_year_periods,
    year_period_to_date,
)
from bimets.timeseries._csv import (
    bimets_to_csv,
    csv_to_bimets,
)
from bimets.timeseries._dataset import BimetsDataset
from bimets.timeseries._frequency import Frequency
from bimets.timeseries._frequency_conversion import (
    annual,
    convert_frequency,
    daily,
    monthly,
    quarterly,
    semiannual,
)
from bimets.timeseries._index import YearPeriod, normalize_year_period, num_periods
from bimets.timeseries._inspection import (
    SeriesInfo,
    get_range,
    magnitude,
    series_info,
    tabulate,
    tsinfo,
    verify_magnitude,
)
from bimets.timeseries._mask import BimetsMask
from bimets.timeseries._operations import (
    cumprod,
    cumsum,
    indexnum,
    movavg,
    movtot,
    tsdelta,
    tsdeltalog,
    tsdeltap,
    tsextend,
    tsjoin,
    tslag,
    tslead,
    tsmerge,
    tsproject,
    tstrim,
)
from bimets.timeseries._pandas import from_pandas, to_pandas
from bimets.timeseries._series import BimetsSeries, is_bimets, timeseries

# BIMETS R compatibility names. These are aliases, not separate implementations;
# the canonical Python call signatures and lowercase names remain unchanged.
ANNUAL = annual
BIMETS2CSV = bimets_to_csv
CSV2BIMETS = csv_to_bimets
CUMPROD = cumprod
CUMSUM = cumsum
CUMULO = cumsum
DAILY = daily
date2yp = date_to_year_period
DELTA = tsdelta
DELTAP = tsdeltap
EXTEND = tsextend
GETDATE = get_dates
GETRANGE = get_range
GETYEARPERIOD = get_year_periods
INDEXNUM = indexnum
MAVE = movavg
MONTHLY = monthly
MOVAVG = movavg
MOVSUM = movtot
MOVTOT = movtot
MSUM = movtot
MTOT = movtot
NUMPERIOD = num_periods
normalizeYP = normalize_year_period
QUARTERLY = quarterly
SEMIANNUAL = semiannual
TABIT = tabulate
TIMESERIES = timeseries
TSDELTA = tsdelta
TSDELTALOG = tsdeltalog
TSDELTAP = tsdeltap
TSDATES = get_year_periods
TSERIES = timeseries
TSEXTEND = tsextend
TSINFO = tsinfo
TSJOIN = tsjoin
TSLAG = tslag
TSLEAD = tslead
TSLOOK = series_info
TSMERGE = tsmerge
TSPROJECT = tsproject
TSTRIM = tstrim
VERIFY_MAGNITUDE = verify_magnitude
YEARLY = annual

__all__ = [
    "ANNUAL",
    "BIMETS2CSV",
    "CSV2BIMETS",
    "CUMPROD",
    "CUMSUM",
    "CUMULO",
    "DAILY",
    "DELTA",
    "DELTAP",
    "EXTEND",
    "GETDATE",
    "GETRANGE",
    "GETYEARPERIOD",
    "INDEXNUM",
    "MAVE",
    "MONTHLY",
    "MOVAVG",
    "MOVSUM",
    "MOVTOT",
    "MSUM",
    "MTOT",
    "NUMPERIOD",
    "QUARTERLY",
    "SEMIANNUAL",
    "TABIT",
    "TIMESERIES",
    "TSDATES",
    "TSDELTA",
    "TSDELTALOG",
    "TSDELTAP",
    "TSERIES",
    "TSEXTEND",
    "TSINFO",
    "TSJOIN",
    "TSLAG",
    "TSLEAD",
    "TSLOOK",
    "TSMERGE",
    "TSPROJECT",
    "TSTRIM",
    "VERIFY_MAGNITUDE",
    "YEARLY",
    "BimetsDataset",
    "BimetsMask",
    "BimetsSeries",
    "Frequency",
    "SeriesInfo",
    "YearPeriod",
    "annual",
    "bimets_to_csv",
    "convert_frequency",
    "csv_to_bimets",
    "cumprod",
    "cumsum",
    "daily",
    "date2yp",
    "date_to_year_period",
    "from_pandas",
    "get_dates",
    "get_range",
    "get_year_periods",
    "indexnum",
    "is_bimets",
    "magnitude",
    "monthly",
    "movavg",
    "movtot",
    "normalizeYP",
    "normalize_year_period",
    "num_periods",
    "quarterly",
    "semiannual",
    "series_info",
    "tabulate",
    "timeseries",
    "to_pandas",
    "tsdelta",
    "tsdeltalog",
    "tsdeltap",
    "tsextend",
    "tsinfo",
    "tsjoin",
    "tslag",
    "tslead",
    "tsmerge",
    "tsproject",
    "tstrim",
    "verify_magnitude",
    "year_period_to_date",
]
