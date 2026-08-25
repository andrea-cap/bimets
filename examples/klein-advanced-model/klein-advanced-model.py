#!/usr/bin/env python3
"""Advanced Klein exercises from the BIMETS concepts paper.

Source
------
Model, annual data, stochastic forecast, and optimal-control exercise are
transcribed from sections 3.3, 3.7, and 3.10 of the public paper:
https://doi.org/10.13140/RG.2.2.31160.83202.
"""

from pathlib import Path

from bimets import (
    CSV2BIMETS,
    ESTIMATE,
    LOAD_MODEL,
    OPTIMIZE,
    STOCHSIMULATE,
    StochasticDisturbance,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "klein-advanced-model.mdl"
DATA_PATH = EXAMPLE_DIR / "klein-data.csv"


def extend_data(data, end):
    """Extend model inputs according to the rules used in the paper."""
    modes = {
        "w2": "constant",
        "g": "constant",
        "t": "linear",
        "k": "linear",
        "time": "linear",
    }
    return {
        name: series.extend(up_to=end, mode=modes.get(name, "missing"))
        for name, series in data.items()
    }


def main() -> None:
    """Estimate, stochastically forecast, and optimize the advanced model."""
    model = LOAD_MODEL(model_text=MODEL_PATH.read_text(encoding="utf-8"))
    historical = CSV2BIMETS(DATA_PATH, merged=True)
    estimates = ESTIMATE(model, historical)

    print("estimation,parameter,value")
    for equation in ("cn", "i", "w1"):
        result = estimates[equation]
        for name, value in result.coefficients.items():
            print(f"{equation},{name},{value:.7f}")
        for name, value in result.autoregressive_coefficients.items():
            print(f"{equation},{name},{value:.7f}")

    stochastic = STOCHSIMULATE(
        model,
        extend_data(historical, (1944, 1)),
        coefficients=estimates,
        time_range=(1941, 1, 1944, 1),
        disturbances={
            "cn": StochasticDisturbance(
                "NORMAL",
                (0.0, estimates["cn"].standard_error),
                time_range=(1942, 1, 1942, 1),
            ),
            "g": StochasticDisturbance("UNIFORM", (-1.0, 1.0)),
        },
        replicas=100,
        seed=123,
        simulation_type="FORECAST",
    )
    print("stochastic,year,y_mean,y_sd")
    for position, (mean, sd) in enumerate(
        zip(stochastic["y"].mean.values, stochastic["y"].sd.values, strict=True)
    ):
        print(f"stochastic,{1941 + position},{mean:.5f},{sd:.6f}")

    optimum = OPTIMIZE(
        model,
        extend_data(historical, (1942, 1)),
        coefficients=estimates,
        time_range=(1942, 1, 1942, 1),
        simulation_type="FORECAST",
        convergence=1e-4,
        max_iterations=1_000,
        bounds={"cn": (-5.0, 5.0), "g": (15.0, 25.0)},
        restrictions="g+(cn^2)/2 < 27 & g+cn > 17",
        objective_functions="(y-110)+(cn-90)*ABS(cn-90)-(g-20)^0.5",
        replicas=10_000,
        seed=123,
    )
    if optimum.objective_max is None or optimum.maximizing_replica is None:
        raise RuntimeError("the paper's optimization did not find a solution")
    replica = optimum.maximizing_replica
    cn_factor = optimum.stochastic.instrument_realizations["cn"][0, replica]
    government = optimum.stochastic.instrument_realizations["g"][0, replica]
    print("optimization,objective,cn_add_factor,g")
    print(f"optimization,{optimum.objective_max:.5f},{cn_factor:.6f},{government:.5f}")


if __name__ == "__main__":
    main()
