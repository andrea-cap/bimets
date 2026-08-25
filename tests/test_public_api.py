from __future__ import annotations

from inspect import signature

import pytest

import bimets
from bimets import BimetsSeries, cumprod, cumsum, movavg, movtot, tsmerge


def test_public_exports_are_unique_and_available() -> None:
    assert len(bimets.__all__) == len(set(bimets.__all__))
    assert all(hasattr(bimets, name) for name in bimets.__all__)


@pytest.mark.source("bimets-R")
def test_compatibility_aliases_reference_the_canonical_api() -> None:
    aliases = {
        "ANNUAL": "annual",
        "BIMETS2CSV": "bimets_to_csv",
        "CSV2BIMETS": "csv_to_bimets",
        "CUMPROD": "cumprod",
        "CUMSUM": "cumsum",
        "CUMULO": "cumsum",
        "DAILY": "daily",
        "DELTA": "tsdelta",
        "DELTAP": "tsdeltap",
        "ESTIMATE": "estimate",
        "EXTEND": "tsextend",
        "GETDATE": "get_dates",
        "GETRANGE": "get_range",
        "GETYEARPERIOD": "get_year_periods",
        "INDEXNUM": "indexnum",
        "LOAD_MODEL": "load_model",
        "LOAD_MODEL_DATA": "bind_model_data",
        "MAVE": "movavg",
        "MONTHLY": "monthly",
        "MOVAVG": "movavg",
        "MOVSUM": "movtot",
        "MOVTOT": "movtot",
        "MSUM": "movtot",
        "MTOT": "movtot",
        "MULTMATRIX": "multiplier_matrix",
        "NUMPERIOD": "num_periods",
        "OPTIMIZE": "optimize_model",
        "QUARTERLY": "quarterly",
        "RENORM": "renormalize",
        "SEMIANNUAL": "semiannual",
        "SIMULATE": "simulate",
        "STOCHSIMULATE": "stochastic_simulate",
        "TABIT": "tabulate",
        "TIMESERIES": "timeseries",
        "TSDATES": "get_year_periods",
        "TSDELTA": "tsdelta",
        "TSDELTALOG": "tsdeltalog",
        "TSDELTAP": "tsdeltap",
        "TSERIES": "timeseries",
        "TSEXTEND": "tsextend",
        "TSINFO": "tsinfo",
        "TSJOIN": "tsjoin",
        "TSLAG": "tslag",
        "TSLEAD": "tslead",
        "TSLOOK": "series_info",
        "TSMERGE": "tsmerge",
        "TSPROJECT": "tsproject",
        "TSTRIM": "tstrim",
        "VERIFY_MAGNITUDE": "verify_magnitude",
        "YEARLY": "annual",
        "date2yp": "date_to_year_period",
        "normalizeYP": "normalize_year_period",
    }

    for alias, canonical in aliases.items():
        assert getattr(bimets, alias) is getattr(bimets, canonical)

    removed_non_r_aliases = {
        "bimets2csv",
        "csv2bimets",
        "getdate",
        "getrange",
        "getyearperiod",
        "multmatrix",
        "optimize",
        "renorm",
        "stochsimulate",
        "tabit",
        "tslook",
        "yearly",
    }
    assert removed_non_r_aliases.isdisjoint(bimets.__all__)
    assert all(not hasattr(bimets, name) for name in removed_non_r_aliases)


def test_missing_value_keyword_is_consistent_across_public_operations() -> None:
    functions = (cumsum, cumprod, movavg, movtot)
    methods = (
        BimetsSeries.cumulative_sum,
        BimetsSeries.cumulative_product,
        BimetsSeries.moving_average,
        BimetsSeries.moving_sum,
    )

    for operation in (*functions, *methods):
        parameters = signature(operation).parameters
        assert "skip_missing" in parameters
        assert "ignore_na" not in parameters

    merge_parameters = signature(tsmerge).parameters
    assert "method" in merge_parameters
    assert "skip_missing" in merge_parameters
    assert "function" not in merge_parameters
    assert "missing_values" not in merge_parameters
