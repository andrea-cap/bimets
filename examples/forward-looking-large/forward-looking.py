#!/usr/bin/env python3
"""Large synthetic forward-looking workload.

Model and data have been generated specifically for this example. Python
and R scripts read the same MDL and CSV files so to make their results comparable.
"""

from pathlib import Path

from bimets import (
    CSV2BIMETS,
    LOAD_MODEL,
    SIMULATE,
    BimetsDataset,
    YearPeriod,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "forward-looking.mdl"
DATA_PATH = EXAMPLE_DIR / "forward-looking-data.csv"


def load_data() -> BimetsDataset:
    """Load the shared quarterly CSV as a BIMETS dataset."""
    return BimetsDataset(CSV2BIMETS(DATA_PATH, merged=True))


def metric(name: str, value: float) -> None:
    """Print a stable numerical metric."""
    print(f"{name},{value:.6f}")


SIMULATION_RANGE = (1960, 2, 2000, 1)
SHOCK_PERIOD = (1980, 2)


def solve(model, data):
    """Solve the extended sparse lead system."""
    return SIMULATE(
        model,
        data,
        coefficients={},
        time_range=SIMULATION_RANGE,
        algorithm="NEWTON",
        convergence=1e-8,
        max_iterations=100,
    )


def main() -> None:
    """Compare baseline and anticipated one-period policy shock paths."""
    model = LOAD_MODEL(model_text=MODEL_PATH.read_text(encoding="utf-8"))
    data = load_data()
    baseline = solve(model, data)
    shocked = solve(
        model,
        data.assign_range(
            {"x001": data["x001"].at_period(*SHOCK_PERIOD) + 1.0},
            start=SHOCK_PERIOD,
            end=SHOCK_PERIOD,
        ),
    )
    response = shocked["y001"] - baseline["y001"]
    shock = YearPeriod(*SHOCK_PERIOD)
    before_shock = shock.shift(-1, response.freq)
    metric("equations", len(model.endogenous))
    metric(
        "baseline_checksum", sum(series.values.sum() for series in baseline.values())
    )
    metric(
        "response_before_shock",
        response.at_period(before_shock.year, before_shock.period),
    )
    metric("response_at_shock", response.at_period(shock.year, shock.period))
    metric("response_last_period", response.values[-1])


if __name__ == "__main__":
    main()
