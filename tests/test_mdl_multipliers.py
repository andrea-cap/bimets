from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import bimets.mdl._stochastic as stochastic_module
from _paper_models import PAPER_DOI
from bimets import (
    MULTMATRIX,
    BimetsModel,
    MultiplierMatrixError,
    MultiplierMatrixResult,
    multiplier_matrix,
    timeseries,
)
from test_mdl_estimation import klein_data
from test_mdl_simulation import KLEIN_MODEL


def test_static_impact_and_dynamic_interim_multipliers() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=x+0.5*TSLAG(y)\nEND",
        name="lagged",
    )
    data = {
        "y": timeseries([0, 0, 0]),
        "x": timeseries([1, 1, 1]),
    }

    dynamic = multiplier_matrix(
        model,
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        targets="y",
        instruments="x",
    )
    static = model.bind(data).multiplier_matrix(
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        targets="y",
        instruments="x",
        simulation_type="STATIC",
    )

    np.testing.assert_allclose(
        dynamic.matrix, np.asarray([[1, 0], [0.5, 1]]), atol=1e-9
    )
    np.testing.assert_allclose(static.matrix, np.eye(2), atol=1e-9)
    assert isinstance(dynamic, MultiplierMatrixResult)
    assert dynamic.row_labels == ("y_1", "y_2")
    assert dynamic.column_labels == ("x_1", "x_2")
    assert dynamic.at("y", 2, "x", 1) == pytest.approx(0.5)
    assert dynamic.summary().shape == (2, 2)
    assert dynamic.matrix.flags.writeable is False
    with pytest.raises(IndexError, match="target_period"):
        dynamic.at("y", 0, "x", 1)
    with pytest.raises(IndexError, match="instrument_period"):
        dynamic.at("y", 1, "x", 3)
    with pytest.raises(KeyError, match="unknown"):
        dynamic.at("missing", 1, "x", 1)


@pytest.mark.source("bimets-R")
def test_multiplier_matrix_supports_fullnewton_and_backfilled_baseline() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    result = multiplier_matrix(
        model,
        {
            "y": timeseries([10.0, 0.0], start=(1999, 1)),
            "z": timeseries([2.0, 0.0], start=(1999, 1)),
            "x": timeseries([1.0, 2.0], start=(1999, 1)),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets="y",
        instruments="x",
        simulation_type="STATIC",
        algorithm="FULLNEWTON",
        convergence=1e-10,
        backfill=1,
        jacobian_drop="z",
    )

    np.testing.assert_allclose(result.matrix, [[1 / 0.875]], rtol=1e-8)
    np.testing.assert_allclose(result.baseline["y"].values, [10.0, 2 / 0.875])
    assert result.baseline.algorithm == "FULLNEWTON"


def test_endogenous_instrument_uses_its_constant_adjustment() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=2*x\nEND")
    result = model.multiplier_matrix(
        {"y": timeseries([0]), "x": timeseries([3])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets="y",
        instruments=("x", "y"),
        simulation_type="STATIC",
    )

    np.testing.assert_allclose(result.matrix, [[2, 1]], atol=1e-9)
    assert result.column_labels == ("x_1", "y_ADDFACTOR_1")


def test_endogenous_instrument_treats_undefined_adjustments_as_zero() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND")
    result = multiplier_matrix(
        model,
        {"y": timeseries([0, 0]), "x": timeseries([1, 1])},
        coefficients={},
        time_range=(2001, 1, 2001, 1),
        targets="y",
        instruments="y",
        constant_adjustments={"y": timeseries([2], start=(2000, 1))},
        simulation_type="STATIC",
    )

    np.testing.assert_allclose(result.matrix, [[1]], atol=1e-9)


@pytest.mark.source("bimets-R")
def test_klein_multipliers_match_original_bimets_examples() -> None:
    model = BimetsModel.from_text(KLEIN_MODEL, name="klein")
    data = klein_data()
    coefficients = model.estimate(data)

    impact = MULTMATRIX(
        model,
        data,
        coefficients=coefficients,
        time_range=(1941, 1, 1941, 1),
        targets=("cn", "y"),
        instruments=("w2", "g"),
        simulation_type="STATIC",
        convergence=1e-8,
    )
    interim = MULTMATRIX(
        model,
        data,
        coefficients=coefficients,
        time_range=(1940, 1, 1941, 1),
        targets=("cn", "y"),
        instruments=("w2", "g"),
        convergence=1e-8,
    )

    np.testing.assert_allclose(
        impact.matrix,
        [[0.4544079, 1.677342], [0.2537924, 3.661807]],
        atol=5e-6,
    )
    np.testing.assert_allclose(
        interim.matrix,
        [
            [0.4544079, 1.677342, 0, 0],
            [0.2537924, 3.661807, 0, 0],
            [-0.3850655, 1.889602, 0.4544079, 1.677342],
            [-0.6149874, 3.01788, 0.2537924, 3.661807],
        ],
        atol=5e-6,
    )


@pytest.mark.source(PAPER_DOI)
def test_klein_multipliers_match_paper_default_convergence() -> None:
    """Reproduce the impact and interim matrices in paper section 3.8."""
    model = BimetsModel.from_text(KLEIN_MODEL, name="klein-paper-multipliers")
    data = klein_data()
    coefficients = model.estimate(data)

    impact = MULTMATRIX(
        model,
        data,
        coefficients=coefficients,
        time_range=(1941, 1, 1941, 1),
        targets=("cn", "y"),
        instruments=("w2", "g"),
        simulation_type="STATIC",
    )
    interim = MULTMATRIX(
        model,
        data,
        coefficients=coefficients,
        time_range=(1940, 1, 1941, 1),
        targets=("cn", "y"),
        instruments=("w2", "g"),
    )

    np.testing.assert_allclose(
        impact.matrix,
        [[0.4540346, 1.671956], [0.2532000, 3.653260]],
        atol=5e-7,
    )
    np.testing.assert_allclose(
        interim.matrix,
        [
            [0.4478202, 1.582292, 0.0, 0.0],
            [0.2433382, 3.510971, 0.0, 0.0],
            [-0.3911001, 1.785042, 0.4540346, 1.671956],
            [-0.6251177, 2.843960, 0.2532000, 3.653260],
        ],
        atol=5e-7,
    )


def test_forward_looking_multiplier_contains_anticipation_effects() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+x\nEND")
    result = multiplier_matrix(
        model,
        {
            "y": timeseries([0, 0, 0, 0]),
            "x": timeseries([0, 1, 1, 0]),
        },
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        targets="y",
        instruments="x",
        algorithm="NEWTON",
        convergence=1e-10,
    )

    np.testing.assert_allclose(result.matrix, np.asarray([[1, 0.5], [0, 1]]), atol=1e-8)


@pytest.mark.source("native")
def test_backward_newton_shocks_share_one_multi_rhs_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    original = stochastic_module._simulate
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stochastic_module, "_simulate", counted)
    result = multiplier_matrix(
        model,
        {
            "y": timeseries([0.0, 0.0]),
            "z": timeseries([0.0, 0.0]),
            "x": timeseries([1.0, 1.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        targets="y",
        instruments="x",
        algorithm="NEWTON",
        convergence=1e-10,
    )

    np.testing.assert_allclose(
        result.matrix,
        [[1 / 0.875, 0], [0, 1 / 0.875]],
        atol=1e-8,
    )
    assert calls == 1  # deterministic baseline; shock columns use one Newton solve


@pytest.mark.source("bimets-R")
def test_shared_multiplier_supports_structural_exogenization() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=x+z\nIDENTITY> z\nEQ> z=q\nEND"
    )
    data = {
        "y": timeseries([0.0]),
        "z": timeseries([3.0]),
        "x": timeseries([2.0]),
        "q": timeseries([9.0]),
    }
    result = multiplier_matrix(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets="y",
        instruments="x",
        simulation_type="STATIC",
        exogenize="z",
    )

    np.testing.assert_allclose(result.matrix, [[1.0]], atol=1e-9)
    with pytest.raises(ValueError, match="targets cannot also be exogenized"):
        multiplier_matrix(
            model,
            data,
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            targets="z",
            instruments="x",
            exogenize="z",
        )


def test_instrument_below_one_uses_an_absolute_positive_shock() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x*x\nEND")
    result = multiplier_matrix(
        model,
        {"y": timeseries([0]), "x": timeseries([-2])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets="y",
        instruments="x",
        shock=1e-5,
        simulation_type="STATIC",
    )

    # BIMETS adds +shock rather than scaling a negative baseline value.
    assert result.matrix[0, 0] == pytest.approx(-3.99999, abs=1e-8)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"targets": ()}, ValueError, "targets"),
        ({"instruments": ()}, ValueError, "instruments"),
        ({"targets": ("y", "y")}, ValueError, "duplicates"),
        ({"instruments": ("x", "x")}, ValueError, "duplicates"),
        ({"targets": "missing"}, KeyError, "targets"),
        ({"instruments": "missing"}, KeyError, "instruments"),
        ({"shock": 0}, ValueError, "shock"),
        ({"simulation_type": "RESCHECK"}, ValueError, "simulation_type"),
    ],
)
def test_multiplier_input_validation(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {"targets": "y", "instruments": "x"}
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        multiplier_matrix(
            BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND"),
            {"y": timeseries([0]), "x": timeseries([1])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            **arguments,  # type: ignore[arg-type]
        )


def test_multiplier_failure_identifies_instrument_and_period() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=y*y+x\nEND")
    with pytest.raises(MultiplierMatrixError, match=r"instrument 'x'.*2000-1"):
        multiplier_matrix(
            model,
            {"y": timeseries([0]), "x": timeseries([0])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            targets="y",
            instruments="x",
            shock=1,
            max_iterations=5,
        )
