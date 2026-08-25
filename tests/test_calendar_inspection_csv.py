from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bimets import (
    TABIT,
    BimetsSeries,
    Frequency,
    SeriesInfo,
    YearPeriod,
    bimets_to_csv,
    csv_to_bimets,
    date_to_year_period,
    get_dates,
    get_range,
    get_year_periods,
    is_bimets,
    magnitude,
    series_info,
    tabulate,
    timeseries,
    tsinfo,
    verify_magnitude,
    year_period_to_date,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("freq", "period", "first", "last"),
    [
        ("Y", 1, date(2020, 1, 1), date(2020, 12, 31)),
        ("S", 2, date(2020, 7, 1), date(2020, 12, 31)),
        (3, 2, date(2020, 5, 1), date(2020, 8, 31)),
        ("Q", 3, date(2020, 7, 1), date(2020, 9, 30)),
        ("M", 2, date(2020, 2, 1), date(2020, 2, 29)),
        (24, 2, date(2020, 1, 16), date(2020, 1, 31)),
        (36, 3, date(2020, 1, 21), date(2020, 1, 31)),
        ("W", 2, date(2020, 1, 8), date(2020, 1, 14)),
        ("D", 60, date(2020, 2, 29), date(2020, 2, 29)),
    ],
)
def test_year_period_calendar_conversion(
    freq: int | str, period: int, first: date, last: date
) -> None:
    value = YearPeriod(2020, period)
    assert year_period_to_date(value, freq, date_in_period="first") == first
    assert year_period_to_date(value, freq) == last
    assert date_to_year_period(first, freq) == value


def test_daily_padding_period_has_no_date_in_non_leap_year() -> None:
    assert year_period_to_date((2021, 366), "D") is None
    assert year_period_to_date((2020, 366), "D") == date(2020, 12, 31)


def test_get_dates_supports_positions_formats_and_method_form() -> None:
    source = timeseries([1, 2, 3], start=(2020, 1), freq="Q")

    assert get_dates(source, index=1) == date(2020, 6, 30)
    assert get_dates(source, [0, 2], date_in_period="first") == [
        date(2020, 1, 1),
        date(2020, 7, 1),
    ]
    assert get_dates(source, 1, format="%Y Q%q") == "2020 Q2"
    assert source.dates(format="%Y-%m-%d") == [
        "2020-03-31",
        "2020-06-30",
        "2020-09-30",
    ]
    assert get_year_periods(source) == [
        YearPeriod(2020, 1),
        YearPeriod(2020, 2),
        YearPeriod(2020, 3),
    ]
    assert source.at_date(date(2020, 5, 1)) == 2
    np.testing.assert_array_equal(
        source.between_dates(date(2020, 2, 1), date(2020, 8, 1)).values,
        [1, 2, 3],
    )


@pytest.mark.source("bimets-R")
def test_get_year_periods_supports_named_and_joined_r_outputs() -> None:
    source = timeseries([1, 2, 3], start=(2020, 3), freq="Q")

    named = get_year_periods(source, years="YEAR", periods="PRD")
    np.testing.assert_array_equal(named["YEAR"], [2020, 2020, 2021])
    np.testing.assert_array_equal(named["PRD"], [3, 4, 1])
    np.testing.assert_array_equal(
        get_year_periods(source, join=True),
        [[2020, 3], [2020, 4], [2021, 1]],
    )


@pytest.mark.source("native")
def test_get_year_periods_validates_r_output_options() -> None:
    source = timeseries([1])
    with pytest.raises(ValueError, match="supplied together"):
        get_year_periods(source, years="YEAR")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="distinct"):
        get_year_periods(source, years="YEAR", periods="YEAR")
    with pytest.raises(TypeError, match="join"):
        get_year_periods(source, join=1)  # type: ignore[call-overload]


def test_calendar_validation() -> None:
    with pytest.raises(TypeError, match="date"):
        date_to_year_period("2020-01-01", "M")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceeds"):
        year_period_to_date((2020, 13), "M")
    with pytest.raises(ValueError, match="date_in_period"):
        year_period_to_date((2020, 1), "M", date_in_period="middle")  # type: ignore[arg-type]


def test_series_information_and_metadata() -> None:
    source = timeseries(
        [1, 2],
        start=(2020, 2),
        freq="Q",
        title="GDP",
        source="Provider",
        units="index",
        scale_factor=2,
    )
    expected = SeriesInfo(
        start=YearPeriod(2020, 2),
        end=YearPeriod(2020, 3),
        freq=Frequency.QUARTERLY,
        source="Provider",
        title="GDP",
        units="index",
        scale_factor=2,
    )

    assert series_info(source) == expected
    assert tsinfo(source, mode="start") == source.start
    assert tsinfo(source, mode="end") == source.end
    assert tsinfo(source, mode="freq") == source.freq
    assert tsinfo(source, mode="source") == "Provider"
    assert tsinfo(source, mode="title") == "GDP"
    assert tsinfo(source, mode="units") == "index"
    assert tsinfo(source, mode="factor") == 2
    with pytest.raises(ValueError, match="unknown information"):
        tsinfo(source, mode="owner")
    assert is_bimets(source)
    assert not is_bimets([1, 2])


@pytest.mark.source("bimets-R")
def test_tsinfo_supports_r_modes_and_multiple_series() -> None:
    first = timeseries([1, 2], start=(2020, 2), freq="Q", title="GDP")
    second = timeseries([3, 4, 5], start=(2019, 12), freq="M", title="CPI")

    assert tsinfo(first, second, mode="STARTY") == (2020, 2019)
    assert tsinfo(first, second, mode="STARTP") == (2, 12)
    assert tsinfo(first, second, mode="ENDY") == (2020, 2020)
    assert tsinfo(first, second, mode="ENDP") == (3, 2)
    assert tsinfo(first, second, mode="START2") == (first.start, second.start)
    assert tsinfo(first, second, mode="END2") == (first.end, second.end)
    assert tsinfo(first, mode="START") == 2020.5
    assert tsinfo(first, mode="END") == 2020.75
    assert tsinfo(first, second, mode="TITLE") == ("GDP", "CPI")


@pytest.mark.source("native")
def test_tsinfo_requires_a_series_and_string_mode() -> None:
    with pytest.raises(ValueError, match="at least one"):
        tsinfo(mode="FREQ")
    with pytest.raises(TypeError, match="mode"):
        tsinfo(timeseries([1]), mode=1)  # type: ignore[arg-type]


def test_magnitude_helpers_match_documented_definition() -> None:
    first = timeseries(np.arange(1, 11) * 0.1)
    second = timeseries(np.arange(1, 11) * 0.01, freq="Q")
    third = timeseries([*np.arange(1, 11) * 0.001, np.nan], freq="M")

    assert magnitude(timeseries([3, 4, np.nan])) == 5
    assert verify_magnitude([first, second, third], threshold=0.1) == [0, 1]
    with pytest.raises(ValueError, match="threshold"):
        verify_magnitude([first], threshold=-1)


@pytest.mark.source("native")
def test_common_ranges_and_tabulation() -> None:
    first = timeseries([1, 2, 3], start=(2020, 1), freq="Q", title="first")
    second = timeseries([10, 20, 30], start=(2020, 3), freq="Q")

    assert get_range(first, second) == (YearPeriod(2020, 3), YearPeriod(2020, 3))
    assert get_range(first, second, kind="outer") == (
        YearPeriod(2020, 1),
        YearPeriod(2021, 1),
    )
    table = tabulate(first, second, headers=["A", "B"])
    assert isinstance(table, pd.DataFrame)
    assert table.index.names == ["Date", "Prd."]
    assert list(table.index) == [
        ("2020 Q1", 1),
        ("2020 Q2", 2),
        ("2020 Q3", 3),
        ("2020 Q4", 4),
        ("2021 Q1", 1),
    ]
    np.testing.assert_allclose(table["A"], [1, 2, 3, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(table["B"], [np.nan, np.nan, 10, 20, 30], equal_nan=True)

    distant = timeseries([1], start=(2030, 1), freq="Q")
    assert get_range(first, distant) is None


@pytest.mark.source("bimets-R")
@pytest.mark.parametrize(
    ("freq", "start", "expected_dates", "expected_periods"),
    [
        ("Y", (2020, 1), ["2020", "2021"], [1, 1]),
        ("S", (2020, 1), ["2020", "2020"], [1, 2]),
        ("Q", (2020, 4), ["2020 Q4", "2021 Q1"], [4, 1]),
        ("M", (2020, 12), ["Dec 2020", "Jan 2021"], [12, 1]),
        ("W", (2020, 1), ["2020-01-07", "2020-01-14"], [1, 2]),
        ("D", (2020, 1), ["2020-01-01", "2020-01-02"], [1, 2]),
    ],
)
def test_tabulation_dates_are_frequency_aware(
    freq: str,
    start: tuple[int, int],
    expected_dates: list[str],
    expected_periods: list[int],
) -> None:
    series = timeseries([1, 2], start=start, freq=freq)

    table = TABIT(series)

    assert table.index.get_level_values("Date").tolist() == expected_dates
    assert table.index.get_level_values("Prd.").tolist() == expected_periods


@pytest.mark.source("bimets-R")
def test_tabulation_display_reuses_compact_series_value_formatting() -> None:
    source = timeseries([100, 101], start=(2023, 1), freq="Q")
    table = TABIT(source, source.lag(1), headers=("originale", "lag_1"))

    rendered = str(table)

    assert repr(table) == rendered
    assert type(table.round(0)) is type(table)
    assert "2023 Q1" in rendered
    assert "100" in rendered
    assert "100.0" not in rendered
    assert "NaN" not in rendered


def test_inspection_validation() -> None:
    source = timeseries([1], freq="Y")
    quarterly = timeseries([1], freq="Q")
    with pytest.raises(ValueError, match="at least one"):
        get_range()
    with pytest.raises(ValueError, match="kind"):
        get_range(source, kind="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same frequency"):
        get_range(source, quarterly)
    with pytest.raises(ValueError, match="headers"):
        tabulate(source, headers=["one", "two"])
    with pytest.raises(ValueError, match="precedes"):
        tabulate(source, start=(2021, 1), end=(2020, 1))


def test_paired_csv_round_trip(tmp_path: Path) -> None:
    source = {
        "annual": timeseries([1, np.nan, 3], start=(2000, 1), freq="Y"),
        "monthly": timeseries([10, 20], start=(2001, 2), freq="M"),
    }
    path = tmp_path / "paired.csv"
    assert bimets_to_csv(source, path) == path
    restored = csv_to_bimets(path)

    assert set(restored) == set(source)
    for name, expected in source.items():
        assert restored[name].start == expected.start
        assert restored[name].freq == expected.freq
        np.testing.assert_allclose(
            restored[name].values, expected.values, equal_nan=True
        )
    with pytest.raises(FileExistsError):
        bimets_to_csv(source, path)


def test_csv_imports_compatible_separator_and_decimal_metadata() -> None:
    restored = csv_to_bimets(FIXTURES / "compatible_paired.csv", decimal_separator=",")

    assert restored["gdp"].start == YearPeriod(2000, 1)
    assert restored["gdp"].freq == Frequency.QUARTERLY
    np.testing.assert_allclose(
        restored["gdp"].values, [1.25, np.nan, 3.75], equal_nan=True
    )
    assert restored["cpi"].start == YearPeriod(2001, 2)
    assert restored["cpi"].freq == Frequency.MONTHLY
    np.testing.assert_array_equal(restored["cpi"].values, [10, 20.5])


def test_csv_writes_separator_metadata_by_default(tmp_path: Path) -> None:
    path = tmp_path / "compatible.csv"

    bimets_to_csv({"gdp": timeseries([1])}, path)

    assert path.read_text(encoding="utf-8").startswith("sep=,\n")


def test_merged_csv_round_trip_and_custom_delimiters(tmp_path: Path) -> None:
    source = {
        "first": timeseries([1, 2], start=(2000, 1), freq="Q"),
        "second": timeseries([10, 20], start=(2000, 2), freq="Q"),
    }
    path = tmp_path / "merged.csv"
    bimets_to_csv(source, path, merged=True, delimiter=";", overwrite=True)
    restored = csv_to_bimets(path, merged=True, delimiter=";")

    assert restored["first"].start == YearPeriod(2000, 1)
    np.testing.assert_allclose(restored["first"].values, [1, 2, np.nan], equal_nan=True)
    np.testing.assert_allclose(
        restored["second"].values, [np.nan, 10, 20], equal_nan=True
    )


def test_csv_preserves_non_calendar_daily_padding_period(tmp_path: Path) -> None:
    source = timeseries([42], start=(2021, 366), freq="D")
    path = tmp_path / "daily.csv"

    bimets_to_csv(source, path)
    restored = csv_to_bimets(path)["series_1"]

    assert restored.start == source.start
    np.testing.assert_array_equal(restored.values, source.values)


@pytest.mark.source("bimets-R")
def test_csv_infers_frequency_and_fills_missing_periods(tmp_path: Path) -> None:
    path = tmp_path / "inferred.csv"
    path.write_text(
        "gdp,value\n2020/01/31,1\n2020/03/31,3\n",
        encoding="utf-8",
    )

    restored = csv_to_bimets(path)["gdp"]

    assert restored.freq is Frequency.MONTHLY
    assert restored.start == YearPeriod(2020, 1)
    np.testing.assert_allclose(restored.values, [1, np.nan, 3], equal_nan=True)


@pytest.mark.source("bimets-R")
def test_merged_csv_infers_frequency_without_frequency_header(tmp_path: Path) -> None:
    path = tmp_path / "plain.csv"
    path.write_text(
        "DATE,gdp,cpi\n2020/03/31,1,10\n2020/06/30,2,20\n2020/09/30,3,30\n",
        encoding="utf-8",
    )

    restored = csv_to_bimets(path, merged=True)

    assert all(series.freq is Frequency.QUARTERLY for series in restored.values())
    np.testing.assert_array_equal(restored["gdp"].values, [1, 2, 3])


@pytest.mark.source("native")
@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (("2020/03/31", "2020/03/31"), "unique increasing"),
        (("2020/06/30", "2020/03/31"), "unique increasing"),
        (("2020/01/01", "2020/03/31"), "unique increasing"),
    ],
)
def test_csv_rejects_duplicate_reversed_or_overlapping_periods(
    tmp_path: Path, dates: tuple[str, str], message: str
) -> None:
    path = tmp_path / "invalid-dates.csv"
    path.write_text(
        f"gdp,FREQ_4\n{dates[0]},1\n{dates[1]},2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        csv_to_bimets(path)


@pytest.mark.source("bimets-R")
def test_csv_supports_skipped_lines_and_custom_frequency_prefix(
    tmp_path: Path,
) -> None:
    source = {"gdp": timeseries([1, 2], start=(2020, 1), freq="Q")}
    exported = tmp_path / "custom-prefix.csv"
    bimets_to_csv(
        source,
        exported,
        freq_header_prefix="PERIODS_",
    )
    restored = csv_to_bimets(exported, freq_header_prefix="PERIODS_")
    assert restored["gdp"].freq is Frequency.QUARTERLY

    titled = tmp_path / "with-title.csv"
    titled.write_text(
        "sep=,\nGenerated table,\ngdp,FREQ_4\n2020/03/31,1\n",
        encoding="utf-8",
    )
    skipped = csv_to_bimets(titled, skip_lines=1)
    np.testing.assert_array_equal(skipped["gdp"].values, [1])


@pytest.mark.source("bimets-R")
def test_csv_export_range_metadata_names_and_plain_table(tmp_path: Path) -> None:
    gdp = BimetsSeries(
        [1, 2, 3, 4],
        start=(2020, 1),
        freq="Q",
        metadata={"title": "Fallback", "code": "GDP"},
    )
    path = tmp_path / "plain.csv"

    bimets_to_csv(
        [gdp],
        path,
        time_range=(2020, 2, 2020, 3),
        name_metadata_key="code",
        title_lines=("Generated", "Quarterly data"),
        plain_table=True,
    )

    text = path.read_text(encoding="utf-8")
    assert not text.startswith("sep=")
    assert "DATE;GDP" in text
    restored = csv_to_bimets(
        path,
        merged=True,
        delimiter=";",
        skip_lines=2,
    )["GDP"]
    assert restored.start == YearPeriod(2020, 2)
    np.testing.assert_array_equal(restored.values, [2, 3])


@pytest.mark.source("bimets-R")
def test_csv_export_append_writes_an_additional_table(tmp_path: Path) -> None:
    path = tmp_path / "append.csv"
    source = timeseries([1], title="GDP")
    bimets_to_csv(source, path)
    bimets_to_csv(source, path, append=True)

    assert path.read_text(encoding="utf-8").count("sep=,") == 2
    with pytest.raises(ValueError, match="mutually exclusive"):
        bimets_to_csv(source, path, overwrite=True, append=True)


@pytest.mark.source("bimets-R")
def test_csv_round_trip_supports_quarter_date_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "quarters.csv"
    source = timeseries([1, 2], start=(2020, 2), freq="Q", title="GDP")

    bimets_to_csv(source, path, date_format="%Y Q%q")
    restored = csv_to_bimets(path, date_format="%Y Q%q")["GDP"]

    assert restored.start == YearPeriod(2020, 2)
    np.testing.assert_array_equal(restored.values, [1, 2])


@pytest.mark.source("bimets-R")
def test_merged_csv_range_retains_missing_columns_inside_union(tmp_path: Path) -> None:
    path = tmp_path / "merged-range.csv"
    source = {
        "gdp": timeseries([1, 2, 3, 4], start=(2020, 1), freq="Q"),
        "late": timeseries([9], start=(2020, 4), freq="Q"),
    }

    bimets_to_csv(
        source,
        path,
        merged=True,
        time_range=(2020, 2, 2020, 3),
    )
    restored = csv_to_bimets(path, merged=True)

    np.testing.assert_array_equal(restored["gdp"].values, [2, 3])
    assert np.isnan(restored["late"].values).all()


def test_csv_validation(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        csv_to_bimets(empty)

    bad = tmp_path / "bad.csv"
    bad.write_text("name,FREQ_INVALID\n2000-12-31,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frequency header"):
        csv_to_bimets(bad)

    with pytest.raises(ValueError, match="skip_lines"):
        csv_to_bimets(bad, skip_lines=-1)
    with pytest.raises(TypeError, match="skip_lines"):
        csv_to_bimets(bad, skip_lines=True)
    with pytest.raises(ValueError, match="no header"):
        csv_to_bimets(bad, skip_lines=3)
    with pytest.raises(ValueError, match="freq_header_prefix"):
        csv_to_bimets(bad, freq_header_prefix="")
    with pytest.raises(ValueError, match="at least one series"):
        bimets_to_csv([], tmp_path / "none.csv")

    invalid_date = tmp_path / "invalid-date.csv"
    invalid_date.write_text("name,FREQ_4\nnot-a-date,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid CSV date"):
        csv_to_bimets(invalid_date)

    period_without_frequency = tmp_path / "period-without-frequency.csv"
    period_without_frequency.write_text("name,value\n2021-P366,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot infer frequency"):
        csv_to_bimets(period_without_frequency)

    single_date = tmp_path / "single-date.csv"
    single_date.write_text("name,value\n2020/01/01,1\n", encoding="utf-8")
    assert csv_to_bimets(single_date)["name"].freq is Frequency.DAILY

    with pytest.raises(ValueError, match="common frequency"):
        bimets_to_csv(
            [timeseries([1], freq="Y"), timeseries([1], freq="Q")],
            tmp_path / "mixed.csv",
            merged=True,
        )
