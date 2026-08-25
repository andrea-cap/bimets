#!/usr/bin/env python3
"""Small synthetic conditional-policy workload.

Model and data have been generated specifically for this example. Python
and R scripts read the same MDL and CSV files so to make their results comparable.
"""

from pathlib import Path

from bimets import (
    CSV2BIMETS,
    LOAD_MODEL,
    SIMULATE,
    BimetsDataset,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "conditional-policy.mdl"
DATA_PATH = EXAMPLE_DIR / "conditional-policy-data.csv"


def load_data() -> BimetsDataset:
    """Load the shared quarterly CSV as a BIMETS dataset."""
    return BimetsDataset(CSV2BIMETS(DATA_PATH, merged=True))


def metric(name: str, value: float) -> None:
    """Print a stable numerical metric."""
    print(f"{name},{value:.6f}")


SIMULATION_RANGE = (2001, 1, 2019, 4)
EXOGENIZATION_RANGE = (2006, 3, 2008, 1)


def run(model, data, adjustment, simulation_type: str):
    """Run one policy scenario with conditional identities and hybrid Newton."""
    return SIMULATE(
        model,
        data,
        coefficients={},
        time_range=SIMULATION_RANGE,
        simulation_type=simulation_type,
        algorithm="NEWTON",
        convergence=1e-7,
        max_iterations=100,
        constant_adjustments={"y001": adjustment},
        exogenize={"y002": EXOGENIZATION_RANGE},
        jacobian_drop="y001",
    )


def main() -> None:
    """Compare dynamic and static conditional policy simulations."""
    model = LOAD_MODEL(model_text=MODEL_PATH.read_text(encoding="utf-8"))
    data = load_data()
    adjustment = (
        data["y001"][[list(SIMULATION_RANGE[:2]), list(SIMULATION_RANGE[2:])]] * 0
        + 0.02
    )
    dynamic = run(model, data, adjustment, "DYNAMIC")
    static = run(model, data, adjustment, "STATIC")
    metric("equations", len(model.endogenous))
    metric("dynamic_aggregate_checksum", dynamic["aggregate"].values.sum())
    metric("static_aggregate_checksum", static["aggregate"].values.sum())
    metric("dynamic_last_aggregate", dynamic["aggregate"].values[-1])


if __name__ == "__main__":
    main()
