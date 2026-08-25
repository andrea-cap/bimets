#!/usr/bin/env python3
"""Public FRB/US monetary-policy shock with bimets Python.

Source
------
The exercise is from the R Consortium article at
https://r-consortium.org/posts/us-federal-reserve-quarterly-model-in-r/.
The February 2024 model and data are extracted from the public ``FRB__MODEL``
and ``LONGBASE`` datasets distributed with BIMETS R.
"""

from pathlib import Path

from bimets import CSV2BIMETS, LOAD_MODEL, SIMULATE, BimetsDataset

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "frb-us-model.mdl"
DATA_PATH = EXAMPLE_DIR / "frb-us-data.csv"
SIMULATION_START = (2040, 1)
SIMULATION_END = (2045, 4)


def load_data(model) -> BimetsDataset:
    """Load and validate the quarterly data shared with the R script."""
    imported = CSV2BIMETS(DATA_PATH, merged=True)
    model_variables = set(model.endogenous).union(model.exogenous)
    missing = model_variables.difference(imported)
    if missing:
        raise ValueError(f"CSV is missing FRB/US model variables: {sorted(missing)}")
    if any(
        (series.start.year, series.start.period, series.end.year, series.end.period)
        != (2036, 1, 2045, 4)
        for series in imported.values()
    ):
        raise ValueError(
            "expected consecutive quarterly observations for 2036Q1--2045Q4"
        )
    return BimetsDataset(imported)


def main() -> None:
    """Apply the article's 100-basis-point shock and print real GDP."""
    model = LOAD_MODEL(model_text=MODEL_PATH.read_text(encoding="utf-8"))
    data = load_data(model).assign_range(
        {"dfpdbt": 0.0, "dfpsrp": 1.0},
        start=SIMULATION_START,
        end=SIMULATION_END,
    )

    residual_check = SIMULATE(
        model,
        data,
        coefficients={},
        time_range=(*SIMULATION_START, *SIMULATION_END),
        simulation_type="RESCHECK",
        zero_error_autocorrelation=True,
    )
    if residual_check.constant_adjustments is None:
        raise RuntimeError("RESCHECK did not produce constant adjustments")
    adjustments = residual_check.constant_adjustments.assign_range(
        {
            "rffintay": residual_check.constant_adjustments["rffintay"].at_period(
                *SIMULATION_START
            )
            + 1.0
        },
        start=SIMULATION_START,
    )

    result = SIMULATE(
        model,
        data,
        coefficients={},
        time_range=(*SIMULATION_START, *SIMULATION_END),
        algorithm="NEWTON",
        convergence=0.01,
        max_iterations=100,
        constant_adjustments=adjustments,
        backfill=12,
    )

    print("period,xgdp")
    output = result["xgdp"][[list(SIMULATION_START), list(SIMULATION_END)]]
    for position, value in enumerate(output.values):
        period = output.period_at(position)
        print(f"{period.year}Q{period.period},{value:.0f}")


if __name__ == "__main__":
    main()
