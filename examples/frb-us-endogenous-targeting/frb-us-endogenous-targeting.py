#!/usr/bin/env python3
"""Public FRB/US endogenous-targeting exercise.

Source
------
The model, data, targets, and instruments are from the public BIMETS FRB/US
vignette, section "Endogenous targeting":
https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf.
"""

from pathlib import Path

import numpy as np

from bimets import CSV2BIMETS, LOAD_MODEL, RENORM, SIMULATE, BimetsDataset, timeseries

EXAMPLE_DIR = Path(__file__).resolve().parent
START = (2021, 3)
END = (2022, 3)


def main() -> None:
    """Find five instrument add-factors that achieve five target paths."""
    model = LOAD_MODEL(
        model_text=(EXAMPLE_DIR / "frb-us-model.mdl").read_text(encoding="utf-8")
    )
    data = BimetsDataset(
        CSV2BIMETS(EXAMPLE_DIR / "frb-us-data.csv", merged=True)
    ).assign_range({"dfpdbt": 0.0, "dfpsrp": 1.0}, start=START, end=END)
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
    scenario = data.assign_range({"lurnat": 3.78}, start=START, end=END)
    gdp_growth = np.cumprod(
        (np.asarray([6.8, 5.2, 4.5, 3.4, 2.7]) / 100.0 + 1.0) ** 0.25
    )
    targets = {
        "xgdp": timeseries(
            scenario["xgdp"].at_period(2021, 2) * gdp_growth, start=START, freq=4
        ),
        "lur": timeseries([5.3, 4.9, 4.6, 4.4, 4.2], start=START, freq=4),
        "picxfe": timeseries([3.7, 2.2, 2.1, 2.1, 2.2], start=START, freq=4),
        "rff": timeseries([0.1] * 5, start=START, freq=4),
        "rg10": timeseries([1.4, 1.6, 1.6, 1.7, 1.9], start=START, freq=4),
    }
    result = RENORM(
        model,
        scenario,
        coefficients={},
        time_range=(*START, *END),
        targets=targets,
        instruments=("eco", "lhp", "picxfe", "rff", "rg10p"),
        algorithm="NEWTON",
        convergence=0.01,
        max_iterations=100,
        constant_adjustments=baseline.constant_adjustments,
        backfill=8,
    )
    if not result.converged:
        raise RuntimeError(f"RENORM did not converge: {result.unconverged_targets}")
    print("period,xgdp,lur,picxfe,rff,rg10")
    first = result.achieved_targets["xgdp"]
    for position in range(len(first)):
        period = first.period_at(position)
        values = [
            result.achieved_targets[name].values[position]
            for name in ("xgdp", "lur", "picxfe", "rff", "rg10")
        ]
        print(
            f"{period.year}Q{period.period},{values[0]:.1f},"
            + ",".join(f"{value:.3f}" for value in values[1:])
        )


if __name__ == "__main__":
    main()
