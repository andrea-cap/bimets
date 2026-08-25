#!/usr/bin/env python3
"""Public FRB/US persistent tracking-residual exercise.

Source
------
The model, data, shocks, and persistence rule are from the public BIMETS
FRB/US vignette, section "Auto-correlation on tracking residuals":
https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf.
"""

from pathlib import Path

import numpy as np

from bimets import CSV2BIMETS, LOAD_MODEL, SIMULATE, BimetsDataset

EXAMPLE_DIR = Path(__file__).resolve().parent
START = (2040, 1)
END = (2046, 1)
PERSISTENT_SHOCKS = {
    "eco": [-0.002, -0.0016, -0.0070, -0.0045],
    "ecd": [-0.0319, -0.0154, -0.0412, -0.0838],
    "eh": [-0.0512, -0.0501, -0.0124, -0.0723],
    "rbbbp": [0.3999, 2.7032, 0.3391, -0.7759],
    "lhp": [
        -0.0029,
        -0.0048,
        -0.0119,
        -0.0085,
        -0.0074,
        -0.0061,
        -0.0077,
        -0.0033,
        -0.0042,
    ],
}


def persistent_path(initial: list[float], periods: int, rho: float = 0.5) -> np.ndarray:
    """Extend an explicit shock path with first-order persistence."""
    values = np.empty(periods)
    values[: len(initial)] = initial
    for position in range(len(initial), periods):
        values[position] = rho * values[position - 1]
    return values


def main() -> None:
    """Apply the vignette's threshold and persistent residual scenario."""
    model = LOAD_MODEL(
        model_text=(EXAMPLE_DIR / "frb-us-model.mdl").read_text(encoding="utf-8")
    )
    data = BimetsDataset(
        CSV2BIMETS(EXAMPLE_DIR / "frb-us-data.csv", merged=True)
    ).assign_range(
        {
            "dfpdbt": 0.0,
            "dfpsrp": 1.0,
            "dmptay": 1.0,
            "dmpintay": 0.0,
            "dmptrsh": 1.0,
            "lurtrsh": 6.0,
            "pitrsh": 3.0,
        },
        start=START,
        end=END,
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
        {
            name: 0.0
            for name in (
                "rfftay",
                "rffrule",
                "rff",
                "dmptpi",
                "dmptlur",
                "dmptmax",
                "dmptr",
            )
        },
        start=START,
        end=END,
    )
    periods = 25
    for name, initial in PERSISTENT_SHOCKS.items():
        residual = adjustments[name][[list(START), list(END)]].values
        adjustments = adjustments.assign_range(
            {name: residual + persistent_path(initial, periods)}, start=START, end=END
        )
    adjustments = adjustments.assign_range({"dmptr": -1.0}, start=START)
    adjustments = adjustments.assign_range(
        {"dmptlur": [-1.0, -1.0, -1.0]}, start=START, end=(2040, 3)
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
    outputs = {
        name: result[name][[list(START), list(END)]]
        for name in ("lur", "picxfe", "rff")
    }
    print("period,lur,picxfe,rff")
    for position in range(periods):
        period = outputs["lur"].period_at(position)
        print(
            f"{period.year}Q{period.period},{outputs['lur'].values[position]:.1f},{outputs['picxfe'].values[position]:.1f},{outputs['rff'].values[position]:.1f}"
        )


if __name__ == "__main__":
    main()
