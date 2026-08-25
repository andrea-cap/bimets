from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bimets import BimetsDataset, Frequency, YearPeriod, timeseries


def sample_dataset() -> BimetsDataset:
    return BimetsDataset(
        {
            "gdp": timeseries([100, 102, 104], start=(2020, 1), freq="Q"),
            "cpi": timeseries([10, 20, 30], start=(2020, 2), freq="Q"),
        },
        metadata={"source": "Example"},
    )


def test_dataset_is_an_immutable_named_mapping() -> None:
    dataset = sample_dataset()

    assert dataset.names == ("gdp", "cpi")
    assert list(dataset) == ["gdp", "cpi"]
    assert len(dataset) == 2
    assert dataset["gdp"].start == YearPeriod(2020, 1)
    assert dataset.metadata == {"source": "Example"}
    assert dataset.homogeneous_frequency == Frequency.QUARTERLY
    assert "freq=4" in repr(dataset)
    with pytest.raises(TypeError):
        dataset.metadata["source"] = "Changed"  # type: ignore[index]


def test_dataset_selection_renaming_addition_and_combination() -> None:
    dataset = sample_dataset()
    unemployment = timeseries([5, 4], start=(2020, 1), freq="Q")

    assert dataset.select(["cpi"]).names == ("cpi",)
    assert dataset.drop("cpi").names == ("gdp",)
    assert dataset.rename({"gdp": "output"}).names == ("output", "cpi")
    added = dataset.with_series("unemployment", unemployment)
    assert added.names == ("gdp", "cpi", "unemployment")
    replaced = dataset.with_series("gdp", unemployment, replace=True)
    assert replaced["gdp"] is unemployment
    combined = dataset.combine({"unemployment": unemployment})
    assert combined.names == ("gdp", "cpi", "unemployment")


def test_dataset_collection_validation() -> None:
    dataset = sample_dataset()
    series = timeseries([1])

    with pytest.raises(ValueError, match="at least one name"):
        dataset.select([])
    with pytest.raises(KeyError, match="unknown"):
        dataset.drop("missing")
    with pytest.raises(ValueError, match="empty dataset"):
        dataset.drop(["gdp", "cpi"])
    with pytest.raises(KeyError, match="unknown"):
        dataset.rename({"missing": "value"})
    with pytest.raises(ValueError, match="duplicate"):
        dataset.rename({"gdp": "cpi"})
    with pytest.raises(KeyError, match="already contains"):
        dataset.with_series("gdp", series)
    with pytest.raises(KeyError, match="duplicate"):
        dataset.combine({"gdp": series})
    with pytest.raises(ValueError, match="at least one"):
        BimetsDataset({})
    with pytest.raises(ValueError, match="non-empty"):
        BimetsDataset({"": series})
    with pytest.raises(TypeError, match="BimetsSeries"):
        BimetsDataset({"gdp": [1, 2]})  # type: ignore[dict-item]


def test_dataset_ranges_and_alignment() -> None:
    dataset = sample_dataset()

    assert dataset.range() == (YearPeriod(2020, 2), YearPeriod(2020, 3))
    assert dataset.range(kind="outer") == (
        YearPeriod(2020, 1),
        YearPeriod(2020, 4),
    )
    inner = dataset.align()
    np.testing.assert_array_equal(inner["gdp"].values, [102, 104])
    np.testing.assert_array_equal(inner["cpi"].values, [10, 20])
    outer = dataset.align(kind="outer")
    np.testing.assert_allclose(
        outer["gdp"].values, [100, 102, 104, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        outer["cpi"].values, [np.nan, 10, 20, 30], equal_nan=True
    )


def test_dataset_alignment_validation() -> None:
    mixed = BimetsDataset(
        {
            "yearly": timeseries([1], freq="Y"),
            "quarterly": timeseries([1], freq="Q"),
        }
    )
    distant = BimetsDataset(
        {
            "first": timeseries([1], start=(2020, 1), freq="Q"),
            "second": timeseries([1], start=(2030, 1), freq="Q"),
        }
    )

    assert mixed.homogeneous_frequency is None
    assert "freq=mixed" in repr(mixed)
    with pytest.raises(ValueError, match="common frequency"):
        mixed.range()
    with pytest.raises(ValueError, match="common frequency"):
        mixed.to_frame()
    with pytest.raises(ValueError, match="do not intersect"):
        distant.align()


def test_dataset_map_and_pandas_conversions() -> None:
    dataset = sample_dataset()

    changed = dataset.map(lambda series: series.delta())
    np.testing.assert_array_equal(changed["gdp"].values, [2, 2])
    pandas_series = dataset.to_pandas()
    assert set(pandas_series) == {"gdp", "cpi"}
    assert all(isinstance(value, pd.Series) for value in pandas_series.values())
    restored = BimetsDataset.from_pandas(pandas_series)
    np.testing.assert_array_equal(restored["gdp"].values, [100, 102, 104])
    frame = dataset.to_frame()
    assert list(frame.columns) == ["gdp", "cpi"]
    np.testing.assert_allclose(
        frame["gdp"].to_numpy(), [100, 102, 104, np.nan], equal_nan=True
    )
    with pytest.raises(TypeError, match="must return BimetsSeries"):
        dataset.map(lambda series: series.values)  # type: ignore[arg-type, return-value]


@pytest.mark.source("native")
def test_dataframe_round_trip_preserves_table_contract() -> None:
    dataset = sample_dataset()

    frame = dataset.to_frame()
    restored = BimetsDataset.from_frame(frame)

    assert frame.attrs == {
        "bimets_frequency": 4,
        "bimets_metadata": {"source": "Example"},
    }
    assert restored.names == dataset.names
    assert restored.metadata == dataset.metadata
    assert restored.homogeneous_frequency == Frequency.QUARTERLY
    assert restored["gdp"].start == YearPeriod(2020, 1)
    assert restored["gdp"].end == YearPeriod(2020, 4)
    np.testing.assert_allclose(
        restored["gdp"].values,
        [100, 102, 104, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        restored["cpi"].values,
        [np.nan, 10, 20, 30],
        equal_nan=True,
    )

    period_frame = pd.DataFrame(
        {"rate": [1.5, 1.75]},
        index=pd.period_range("2024Q1", periods=2, freq="Q"),
    )
    inferred = BimetsDataset.from_frame(
        period_frame,
        metadata={"scenario": "baseline"},
    )
    assert inferred["rate"].start == YearPeriod(2024, 1)
    assert inferred.metadata == {"scenario": "baseline"}


@pytest.mark.source("native")
def test_dataframe_import_validation() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        BimetsDataset.from_frame([1, 2])
    with pytest.raises(ValueError, match="must not be empty"):
        BimetsDataset.from_frame(pd.DataFrame())

    duplicate_columns = pd.DataFrame(
        [[1.0, 2.0]],
        columns=["gdp", "gdp"],
        index=pd.period_range("2024Q1", periods=1, freq="Q"),
    )
    with pytest.raises(ValueError, match="duplicates"):
        BimetsDataset.from_frame(duplicate_columns)

    invalid_columns = pd.DataFrame(
        {1: [2.0]},
        index=pd.period_range("2024Q1", periods=1, freq="Q"),
    )
    with pytest.raises(ValueError, match="non-empty strings"):
        BimetsDataset.from_frame(invalid_columns)

    invalid_metadata = pd.DataFrame(
        {"gdp": [2.0]},
        index=pd.period_range("2024Q1", periods=1, freq="Q"),
    )
    invalid_metadata.attrs["bimets_metadata"] = "invalid"
    with pytest.raises(TypeError, match="must be a mapping"):
        BimetsDataset.from_frame(invalid_metadata)


@pytest.mark.source("native")
def test_assign_range_builds_an_immutable_scenario() -> None:
    dataset = sample_dataset()

    scenario = dataset.assign_range(
        {"gdp": [120, 125], "cpi": 40},
        start=(2020, 2),
        end=(2020, 3),
    )

    np.testing.assert_array_equal(scenario["gdp"].values, [100, 120, 125])
    np.testing.assert_array_equal(scenario["cpi"].values, [40, 40, 30])
    np.testing.assert_array_equal(dataset["gdp"].values, [100, 102, 104])
    assert scenario.metadata == dataset.metadata

    extended = dataset.assign_range(
        {"gdp": 90},
        start=(2019, 4),
        end=(2019, 4),
        extend=True,
    )
    assert extended["gdp"].start == YearPeriod(2019, 4)
    np.testing.assert_allclose(
        extended["gdp"].values,
        [90, 100, 102, 104],
        equal_nan=True,
    )
    assert extended["cpi"] is dataset["cpi"]


@pytest.mark.source("native")
def test_assign_range_validation() -> None:
    dataset = sample_dataset()

    with pytest.raises(ValueError, match="at least one variable"):
        dataset.assign_range({}, start=(2020, 1))
    with pytest.raises(KeyError, match="unknown"):
        dataset.assign_range({"missing": 1}, start=(2020, 1))
    with pytest.raises(ValueError, match="precedes"):
        dataset.assign_range({"gdp": 1}, start=(2020, 2), end=(2020, 1))
    with pytest.raises(ValueError, match="outside"):
        dataset.assign_range({"gdp": 1}, start=(2019, 4))
    with pytest.raises(ValueError, match="requires 2 values"):
        dataset.assign_range(
            {"gdp": [1]},
            start=(2020, 1),
            end=(2020, 2),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        dataset.assign_range(
            {"gdp": [[1, 2]]},  # type: ignore[list-item]
            start=(2020, 1),
            end=(2020, 2),
        )


def test_dataset_csv_round_trip(tmp_path: Path) -> None:
    dataset = sample_dataset()
    path = tmp_path / "dataset.csv"

    assert dataset.to_csv(path) == path
    restored = BimetsDataset.from_csv(path)

    assert restored.names == dataset.names
    for name in dataset:
        assert restored[name].start == dataset[name].start
        np.testing.assert_array_equal(restored[name].values, dataset[name].values)
