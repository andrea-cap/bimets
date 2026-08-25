#!/usr/bin/env python3
"""Small synthetic advanced-estimation workload.

Model and data have been generated specifically for this example. Python
and R scripts read the same MDL and CSV files so to make their results comparable.
"""

from pathlib import Path

from bimets import (
    CSV2BIMETS,
    ESTIMATE,
    LOAD_MODEL,
    BimetsDataset,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "advanced-estimation.mdl"
DATA_PATH = EXAMPLE_DIR / "advanced-estimation-data.csv"


def load_data() -> BimetsDataset:
    """Load the shared quarterly CSV as a BIMETS dataset."""
    return BimetsDataset(CSV2BIMETS(DATA_PATH, merged=True))


def metric(name: str, value: float) -> None:
    """Print a stable numerical metric."""
    print(f"{name},{value:.6f}")


CHOW_END = (1999, 4)


def main() -> None:
    """Estimate constrained PDL/AR equations and instrumented equations."""
    model = LOAD_MODEL(model_text=MODEL_PATH.read_text(encoding="utf-8"))
    data = load_data()
    pdl_equations = tuple(eq.name for eq in model.behaviorals if eq.pdls)
    iv_equations = tuple(eq.name for eq in model.behaviorals if eq.instruments)
    ols = ESTIMATE(model, data, equations=pdl_equations, method="OLS")
    iv = ESTIMATE(model, data, equations=iv_equations, method="IV")
    chow = ESTIMATE(
        model,
        data,
        equations=pdl_equations[0],
        method="OLS",
        chow_test=True,
        chow_end=CHOW_END,
    )[pdl_equations[0]].chow_test
    assert chow is not None

    metric("equations", len(model.endogenous))
    metric(
        "ols_coefficient_checksum",
        sum(sum(result.coefficients.values()) for result in ols.values()),
    )
    metric(
        "iv_coefficient_checksum",
        sum(sum(result.coefficients.values()) for result in iv.values()),
    )
    metric("chow_f_statistic", chow.f_statistic)


if __name__ == "__main__":
    main()
