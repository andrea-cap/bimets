#!/usr/bin/env python3
"""Solve the public forward-looking Klein example with bimets Python.

Source
------
The model and terminal-value experiment come from the BIMETS R ``SIMULATE``
example and the original public repository:
https://github.com/andrea-luciani/bimets#rational-expectations.
"""

from pathlib import Path

from bimets import CSV2BIMETS, ESTIMATE, LOAD_MODEL, SIMULATE, BimetsDataset

EXAMPLE_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Estimate the behaviorals and solve investment over 1924--1930."""
    model = LOAD_MODEL(
        model_text=(EXAMPLE_DIR / "klein-rational-expectations.mdl").read_text(
            encoding="utf-8"
        )
    )
    data = BimetsDataset(
        CSV2BIMETS(EXAMPLE_DIR / "klein-data.csv", merged=True)
    ).assign_range({"i": 2.0}, start=(1931, 1))
    estimates = ESTIMATE(model, data)
    result = SIMULATE(
        model,
        data,
        coefficients=estimates,
        time_range=(1924, 1, 1930, 1),
        algorithm="NEWTON",
        convergence=1e-6,
        max_iterations=200,
    )

    print("year,i")
    output = result["i"][[[1924, 1], [1930, 1]]]
    for position, value in enumerate(output.values):
        print(f"{1924 + position},{value:.6f}")


if __name__ == "__main__":
    main()
