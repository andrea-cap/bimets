#!/usr/bin/env python3
"""Public forward-looking FRB/US policy shock with bimets Python.

Source
------
The MCE model, data, and exercise are from the public BIMETS FRB/US vignette,
section "Rational expectations":
https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf.
"""

from pathlib import Path

from bimets import CSV2BIMETS, LOAD_MODEL, SIMULATE, BimetsDataset

EXAMPLE_DIR = Path(__file__).resolve().parent
START = (2040, 1)
END = (2042, 1)


def main() -> None:
    """Apply the vignette's 100-basis-point monetary-policy shock."""
    model = LOAD_MODEL(
        model_text=(EXAMPLE_DIR / "frb-us-mce-model.mdl").read_text(encoding="utf-8")
    )
    imported = CSV2BIMETS(EXAMPLE_DIR / "frb-us-data.csv", merged=True)
    missing = set(model.endogenous).union(model.exogenous).difference(imported)
    if missing:
        raise ValueError(f"CSV is missing FRB/US model variables: {sorted(missing)}")
    data = (
        BimetsDataset(imported)
        .assign_range(
            {"dfpdbt": 0.0, "dfpsrp": 1.0, "drstar": 0.0}, start=START, end=END
        )
        .assign_range({"drstar": 1.0}, start=(2041, 1), end=END)
    )

    baseline = SIMULATE(
        model,
        data,
        coefficients={},
        time_range=(*START, *END),
        simulation_type="RESCHECK",
        zero_error_autocorrelation=True,
    )
    if baseline.constant_adjustments is None:
        raise RuntimeError("RESCHECK did not produce constant adjustments")
    adjustments = baseline.constant_adjustments.assign_range(
        {"rffintay": baseline.constant_adjustments["rffintay"].at_period(*START) + 1.0},
        start=START,
    )
    result = SIMULATE(
        model,
        data,
        coefficients={},
        time_range=(*START, *END),
        algorithm="NEWTON",
        convergence=0.01,
        max_iterations=100,
        constant_adjustments=adjustments,
        backfill=12,
    )

    print("period,xgdp,rff")
    output = result["xgdp"][[list(START), list(END)]]
    rates = result["rff"][[list(START), list(END)]]
    for position, (gdp, rate) in enumerate(
        zip(output.values, rates.values, strict=True)
    ):
        period = output.period_at(position)
        print(f"{period.year}Q{period.period},{gdp:.2f},{rate:.5f}")


if __name__ == "__main__":
    main()
