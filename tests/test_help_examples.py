from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from bimets import (
    BimetsSeries,
    YearPeriod,
    normalize_year_period,
    num_periods,
    timeseries,
    tsdelta,
    tsdeltalog,
    tsdeltap,
    tslag,
    tslead,
    tsproject,
    tstrim,
)

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True, slots=True)
class ExpectedSeries:
    freq: int
    start: YearPeriod
    values: np.ndarray[tuple[int], np.dtype[np.float64]]


@lru_cache(maxsize=1)
def _series_fixtures() -> dict[str, ExpectedSeries]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with (FIXTURES / "help_series.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            grouped.setdefault(row["case"], []).append(row)

    fixtures: dict[str, ExpectedSeries] = {}
    for name, rows in grouped.items():
        values = np.array(
            [np.nan if row["value"] == "NA" else float(row["value"]) for row in rows],
            dtype=np.float64,
        )
        fixtures[name] = ExpectedSeries(
            freq=int(rows[0]["frequency"]),
            start=YearPeriod(
                int(rows[0]["start_year"]),
                int(rows[0]["start_period"]),
            ),
            values=values,
        )
    return fixtures


@lru_cache(maxsize=1)
def _scalar_fixtures() -> dict[str, tuple[int, int | None]]:
    fixtures: dict[str, tuple[int, int | None]] = {}
    with (FIXTURES / "help_scalars.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            second = None if row["value_2"] == "NA" else int(row["value_2"])
            fixtures[row["case"]] = (int(row["value_1"]), second)
    return fixtures


def _assert_matches_help(case: str, actual: BimetsSeries) -> None:
    expected = _series_fixtures()[case]
    assert int(actual.freq) == expected.freq
    assert actual.start == expected.start
    np.testing.assert_allclose(
        actual.values,
        expected.values,
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
    )


def test_timeseries_matches_documented_constructor() -> None:
    actual = timeseries(
        [5, *range(1, 11), np.nan, 8],
        start=(2020, 1),
        freq=1,
        source="mySource",
        title="myTitle",
        units="myUnits",
        scale_factor=2,
    )

    _assert_matches_help("timeseries", actual)
    assert actual.metadata == {
        "source": "mySource",
        "title": "myTitle",
        "units": "myUnits",
        "scale_factor": 2,
    }


def test_lag_and_lead_match_help_examples() -> None:
    values = np.arange(10, 0, -1, dtype=np.float64)
    values[4] = np.nan
    source = timeseries(values, start=(2000, 1), freq="A")

    _assert_matches_help("tslag_5", tslag(source, 5))
    _assert_matches_help("tslead_5", tslead(source, 5))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_delta_matches_help_example(order: int) -> None:
    source = timeseries(range(1, 11), start=(2000, 1), freq="A")
    _assert_matches_help(f"tsdelta_order_{order}", tsdelta(source, 1, order))


def test_log_delta_matches_help_example() -> None:
    source = timeseries(range(1, 11), start=(2000, 1), freq="A")
    _assert_matches_help("tsdeltalog", tsdeltalog(source))


def test_percentage_delta_matches_help_examples() -> None:
    quarterly = timeseries(range(10, -1, -1), start=(2000, 1), freq="Q")
    _assert_matches_help("tsdeltap", tsdeltap(quarterly))

    daily = timeseries(
        1 - np.arange(10, dtype=np.float64) * 0.001,
        start=(2000, 1),
        freq=366,
    )
    _assert_matches_help(
        "tsdeltap_annualized",
        tsdeltap(daily, annualize=True),
    )


def test_projection_matches_help_examples() -> None:
    source = timeseries(range(1, 11), start=(2000, 1), freq=1)
    _assert_matches_help("tsproject", tsproject(source, (2002, 1), (2005, 1)))
    _assert_matches_help(
        "tsproject_extended",
        tsproject(source, (1998, 1), (2002, 1), extend=True),
    )


def test_trim_matches_help_examples() -> None:
    source = timeseries([np.nan, *range(1, 11), np.nan], start=(2000, 1))

    trimmed = tstrim(source)
    keep_leading = tstrim(source, leading=False)
    keep_trailing = tstrim(source, trailing=False)
    assert trimmed is not None
    assert keep_leading is not None
    assert keep_trailing is not None
    _assert_matches_help("tstrim", trimmed)
    _assert_matches_help("tstrim_keep_leading", keep_leading)
    _assert_matches_help("tstrim_keep_trailing", keep_trailing)

    zero_source = timeseries(
        [0, 0, np.nan, *range(1, 11), np.nan, 0],
        start=(2000, 1),
    )
    zero_trimmed = tstrim(zero_source, 0)
    assert zero_trimmed is not None
    _assert_matches_help("tstrim_zero", zero_trimmed)


def test_scalar_help_examples() -> None:
    fixtures = _scalar_fixtures()
    normalized = normalize_year_period((2, 13), 4)
    assert (normalized.year, normalized.period) == fixtures["normalize_year_period"]
    assert num_periods((2, 3), (3, 4), 5) == fixtures["num_periods"][0]
