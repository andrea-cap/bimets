#!/usr/bin/env python3
"""Estimate and forecast the Klein model from the BIMETS concepts paper.

Source
------
Model, data, and forecast exercise are transcribed from section 3.1 of the
BIMETS concepts paper: https://doi.org/10.13140/RG.2.2.31160.83202.
"""

from pathlib import Path

from bimets import CSV2BIMETS, ESTIMATE, LOAD_MODEL, LOAD_MODEL_DATA, SIMULATE

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "klein-model.mdl"
DATA_PATH = EXAMPLE_DIR / "klein-data.csv"
DATA_COLUMNS = ("cn", "g", "i", "k", "p", "w1", "y", "t", "time", "w2")
FORECAST_RANGE = (1941, 1, 1944, 1)


def load_historical_data():
    """Load and validate the annual observations shared with the R example."""
    data = CSV2BIMETS(DATA_PATH, merged=True)
    if tuple(data) != DATA_COLUMNS:
        raise ValueError(f"expected CSV series {DATA_COLUMNS}, got {tuple(data)}")
    if any(
        (series.start.year, series.start.period, series.end.year, series.end.period)
        != (1920, 1, 1941, 1)
        for series in data.values()
    ):
        raise ValueError("expected annual observations for 1920--1941")
    return data


def main() -> None:
    """Run the paper's 1941--1944 GNP forecast and print comparable output."""
    model = LOAD_MODEL(model_text=MODEL_PATH.read_text(encoding="utf-8"))
    historical = load_historical_data()
    coefficients = ESTIMATE(LOAD_MODEL_DATA(model, historical))

    forecast_data = dict(historical)
    for name in ("w2", "t", "g"):
        forecast_data[name] = historical[name].extend(
            up_to=FORECAST_RANGE[2:], mode="constant"
        )
    forecast_data["time"] = historical["time"].extend(
        up_to=FORECAST_RANGE[2:], mode="linear"
    )

    result = SIMULATE(
        LOAD_MODEL_DATA(model, forecast_data),
        coefficients=coefficients,
        time_range=FORECAST_RANGE,
        simulation_type="FORECAST",
        convergence=1e-5,
        max_iterations=100,
    )

    print("year,y")
    years = range(result["y"].start.year, result["y"].start.year + len(result["y"]))
    for year, value in zip(years, result["y"].values, strict=True):
        print(f"{year},{value:.5f}")


if __name__ == "__main__":
    main()
