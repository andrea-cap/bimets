#!/usr/bin/env python3
"""Small synthetic stochastic-policy workload.

Model and data have been generated specifically for this example. Python
and R scripts read the same MDL and CSV files so to make their results comparable.
"""

from pathlib import Path

from bimets import (
    CSV2BIMETS,
    LOAD_MODEL,
    MULTMATRIX,
    OPTIMIZE,
    RENORM,
    SIMULATE,
    STOCHSIMULATE,
    BimetsDataset,
    StochasticDisturbance,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "stochastic-policy.mdl"
DATA_PATH = EXAMPLE_DIR / "stochastic-policy-data.csv"


def load_data() -> BimetsDataset:
    """Load the shared quarterly CSV as a BIMETS dataset."""
    return BimetsDataset(CSV2BIMETS(DATA_PATH, merged=True))


def metric(name: str, value: float) -> None:
    """Print a stable numerical metric."""
    print(f"{name},{value:.6f}")


SIMULATION_RANGE = (2010, 1, 2011, 4)
REPLICAS = 100


def main() -> None:
    """Run stochastic simulation, multipliers, targeting, and optimal control."""
    model = LOAD_MODEL(model_text=MODEL_PATH.read_text(encoding="utf-8"))
    data = load_data()
    common = dict(
        coefficients={},
        time_range=SIMULATION_RANGE,
        algorithm="NEWTON",
        convergence=1e-7,
        max_iterations=100,
    )

    stochastic = STOCHSIMULATE(
        model,
        data,
        disturbances={
            "y001": StochasticDisturbance("NORMAL", (0.0, 0.05)),
            "x001": StochasticDisturbance("UNIFORM", (-0.10, 0.10)),
        },
        replicas=REPLICAS,
        seed=123,
        **common,
    )
    multipliers = MULTMATRIX(
        model,
        data,
        targets=("aggregate", "y001"),
        instruments=("x001", "y001"),
        **common,
    )
    baseline = SIMULATE(model, data, **common)
    desired = baseline["aggregate"] * 1.01
    targeting = RENORM(
        model,
        data,
        targets={"aggregate": desired},
        instruments="x001",
        renormalization_iterations=6,
        renormalization_convergence=1e-6,
        **common,
    )
    optimum = OPTIMIZE(
        model,
        data,
        bounds={"x001": (0.5, 1.5)},
        restrictions="x001 >= 0.52 & x001 <= 1.48",
        objective_functions="aggregate-0.01*x001^2",
        replicas=REPLICAS,
        seed=321,
        **common,
    )
    metric("equations", len(model.endogenous))
    metric("stochastic_mean_checksum", stochastic["aggregate"].mean.values.sum())
    metric("stochastic_sd_checksum", stochastic["aggregate"].sd.values.sum())
    metric("multiplier_checksum", multipliers.matrix.sum())
    metric("renorm_instrument_checksum", targeting.instruments["x001"].values.sum())
    metric("optimum", float(optimum.objective_max))


if __name__ == "__main__":
    main()
