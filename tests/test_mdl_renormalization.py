from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from _paper_models import PAPER_DOI
from bimets import (
    RENORM,
    BimetsModel,
    RenormalizationError,
    RenormalizationResult,
    renormalize,
    timeseries,
)
from test_mdl_estimation import klein_data
from test_mdl_simulation import KLEIN_MODEL


def test_linear_targeting_function_model_and_bound_forms() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=2*x\nEND")
    data = {"y": timeseries([0]), "x": timeseries([1])}

    functional = renormalize(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets={"y": timeseries([10])},
        instruments="x",
        simulation_type="STATIC",
    )
    method = model.renormalize(
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets={"y": timeseries([10])},
        instruments="x",
        simulation_type="STATIC",
    )
    bound = model.bind(data).renormalize(
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets={"y": timeseries([10])},
        instruments="x",
        simulation_type="STATIC",
    )

    for result in (functional, method, bound):
        assert isinstance(result, RenormalizationResult)
        assert result.converged is True
        assert result.iterations == 1
        assert result.unconverged_targets == ()
        np.testing.assert_allclose(result.instruments["x"].values, [5])
        np.testing.assert_allclose(result.targets["y"].values, [10])
        np.testing.assert_allclose(result.data["x"].values, [5])
        assert result.summary().shape == (1, 3)
    assert data["x"].values.tolist() == [1]
    assert isinstance(functional.instruments, Mapping)
    with pytest.raises(TypeError):
        functional.instruments["x"] = timeseries([0])  # type: ignore[index]

    already_targeted = renormalize(
        model,
        {"y": timeseries([2]), "x": timeseries([1])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets={"y": timeseries([2])},
        instruments="x",
        simulation_type="STATIC",
    )
    assert already_targeted.iterations == 0


@pytest.mark.source("bimets-R")
def test_renormalization_supports_fullnewton_and_backfilled_simulation() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    result = renormalize(
        model,
        {
            "y": timeseries([10.0, 0.0], start=(1999, 1)),
            "z": timeseries([2.0, 0.0], start=(1999, 1)),
            "x": timeseries([1.0, 2.0], start=(1999, 1)),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        targets={"y": timeseries([4.0], start=(2000, 1))},
        instruments="x",
        simulation_type="STATIC",
        algorithm="FULLNEWTON",
        convergence=1e-10,
        backfill=1,
        jacobian_drop="z",
    )

    assert result.converged
    assert result.simulation.algorithm == "FULLNEWTON"
    np.testing.assert_allclose(result.simulation["y"].values, [10.0, 4.0])
    np.testing.assert_allclose(result.achieved_targets["y"].values, [4.0])


def test_dynamic_targeting_accounts_for_lag_propagation_and_zero_target() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x+0.5*TSLAG(y)\nEND")
    result = RENORM(
        model,
        {"y": timeseries([0, 0, 0]), "x": timeseries([1, 1, 1])},
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        targets={"y": timeseries([0, 4], start=(2001, 1))},
        instruments="x",
        renormalization_convergence=1e-10,
    )

    assert result.converged
    np.testing.assert_allclose(result.instruments["x"].values, [0, 4], atol=1e-9)
    np.testing.assert_allclose(result.achieved_targets["y"].values, [0, 4], atol=1e-10)


def test_forward_looking_targeting_accounts_for_anticipation() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+x\nEND")
    result = renormalize(
        model,
        {
            "y": timeseries([0, 0, 0, 0]),
            "x": timeseries([1, 1, 1, 0]),
        },
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        targets={"y": timeseries([2, 4], start=(2001, 1))},
        instruments="x",
        algorithm="NEWTON",
        convergence=1e-10,
        renormalization_convergence=1e-9,
    )

    assert result.converged
    np.testing.assert_allclose(result.instruments["x"].values, [0, 4], atol=1e-8)


def test_endogenous_instrument_updates_its_constant_adjustment() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=2*x\nEND")
    result = renormalize(
        model,
        {"y": timeseries([0, 0]), "x": timeseries([1, 1])},
        coefficients={},
        time_range=(2001, 1, 2001, 1),
        targets={"y": timeseries([7], start=(2001, 1))},
        instruments="y",
        constant_adjustments={"y": timeseries([0.25], start=(2000, 1))},
        simulation_type="STATIC",
    )

    np.testing.assert_allclose(result.instruments["y"].values, [5])
    adjustment = result.constant_adjustments["y"]
    assert not isinstance(adjustment, float)
    np.testing.assert_allclose(adjustment.values, [0.25, 5])
    np.testing.assert_allclose(result.data["y"].values, [0, 0])
    assert result.multiplier_matrix.column_labels == ("y_ADDFACTOR_1",)


@pytest.mark.source("bimets-R")
def test_klein_targeting_matches_original_bimets_at_tight_convergence() -> None:
    model = BimetsModel.from_text(KLEIN_MODEL, name="klein")
    data = klein_data()
    result = renormalize(
        model,
        data,
        coefficients=model.estimate(data),
        time_range=(1940, 1, 1941, 1),
        targets={
            "cn": timeseries([66, 78], start=(1940, 1)),
            "y": timeseries([77, 98], start=(1940, 1)),
        },
        instruments=("w2", "g"),
        convergence=1e-8,
    )

    assert result.converged
    np.testing.assert_allclose(
        result.instruments["w2"].values, [7.404308, 9.327926], atol=5e-6
    )
    np.testing.assert_allclose(
        result.instruments["g"].values, [16.102687, 22.651635], atol=5e-6
    )
    np.testing.assert_allclose(result.targets["cn"].values, [66, 78], atol=1e-6)
    np.testing.assert_allclose(result.targets["y"].values, [77, 98], atol=1e-6)


@pytest.mark.source(PAPER_DOI)
def test_klein_targeting_matches_paper_default_convergence() -> None:
    """Reproduce the instruments and verification in paper section 3.9."""
    model = BimetsModel.from_text(KLEIN_MODEL, name="klein-paper-renorm")
    data = klein_data()
    coefficients = model.estimate(data)
    result = RENORM(
        model,
        data,
        coefficients=coefficients,
        time_range=(1940, 1, 1941, 1),
        targets={
            "cn": timeseries([66, 78], start=(1940, 1)),
            "y": timeseries([77, 98], start=(1940, 1)),
        },
        instruments=("w2", "g"),
        max_iterations=100,
    )

    np.testing.assert_allclose(
        result.instruments["w2"].values, [7.413331, 9.343600], atol=5e-7
    )
    np.testing.assert_allclose(
        result.instruments["g"].values, [16.1069, 22.65985], atol=5e-6
    )

    verification = model.simulate(
        result.data,
        coefficients=coefficients,
        time_range=(1940, 1, 1941, 1),
        convergence=1e-5,
        max_iterations=100,
    )
    np.testing.assert_allclose(
        verification["cn"].values, [66.01116, 78.02538], atol=5e-6
    )
    np.testing.assert_allclose(
        verification["y"].values, [77.01772, 98.04121], atol=5e-6
    )


@pytest.mark.source("bimets-R")
def test_klein_endogenous_instrument_matches_original_add_factor_example() -> None:
    model = BimetsModel.from_text(KLEIN_MODEL, name="klein")
    data = klein_data()
    result = renormalize(
        model,
        data,
        coefficients=model.estimate(data),
        time_range=(1940, 1, 1941, 1),
        targets={
            "cn": timeseries([66, 78], start=(1940, 1)),
            "y": timeseries([77, 98], start=(1940, 1)),
        },
        instruments=("w2", "i"),
        constant_adjustments={"i": timeseries([0.1] * 22, start=(1920, 1))},
        convergence=1e-8,
    )

    np.testing.assert_allclose(
        result.instruments["w2"].values, [7.404308, 9.327926], atol=5e-6
    )
    np.testing.assert_allclose(
        result.instruments["i"].values, [0.702687, 0.430191], atol=5e-6
    )


def test_iteration_limit_returns_nonconverged_diagnostics() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x*x\nEND")
    with pytest.warns(RuntimeWarning, match="RENORM did not converge"):
        result = renormalize(
            model,
            {"y": timeseries([0]), "x": timeseries([1])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            targets={"y": timeseries([9])},
            instruments="x",
            simulation_type="STATIC",
            renormalization_iterations=1,
            renormalization_convergence=1e-10,
        )

    assert result.converged is False
    assert result.iterations == 1
    assert result.unconverged_targets == ("y",)
    assert result.achieved_targets["y"].values[0] == pytest.approx(25, abs=1e-3)


def test_singular_multiplier_matrix_has_contextual_error() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=x\nIDENTITY> z\nEQ> z=x+0*w\nEND"
    )
    with pytest.raises(RenormalizationError, match=r"singular.*iteration 1"):
        renormalize(
            model,
            {
                "y": timeseries([0]),
                "z": timeseries([0]),
                "x": timeseries([1]),
                "w": timeseries([1]),
            },
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            targets={"y": timeseries([2]), "z": timeseries([3])},
            instruments=("x", "w"),
            simulation_type="STATIC",
        )


def test_multiplier_failure_has_renormalization_context() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=y*y+x\nEND")
    with pytest.raises(
        RenormalizationError, match=r"multiplier calculation.*iteration 1"
    ):
        renormalize(
            model,
            {"y": timeseries([0]), "x": timeseries([0])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            targets={"y": timeseries([1])},
            instruments="x",
            shock=1,
            max_iterations=5,
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"targets": {}}, ValueError, "targets"),
        ({"targets": {1: timeseries([1])}}, TypeError, "target names"),
        ({"targets": {"y": [1]}}, TypeError, "target values"),
        ({"targets": {"missing": timeseries([1])}}, KeyError, "targets"),
        ({"targets": {"y": timeseries([1], freq="Q")}}, ValueError, "frequency"),
        ({"targets": {"y": timeseries([np.nan])}}, ValueError, "undefined"),
        ({"instruments": ()}, ValueError, "instruments"),
        ({"instruments": ("x", "y")}, ValueError, "same length"),
        ({"instruments": "missing"}, KeyError, "instruments"),
        ({"renormalization_iterations": 0}, ValueError, "iterations"),
        ({"renormalization_convergence": 0}, ValueError, "convergence"),
        ({"matrix_tolerance": np.inf}, ValueError, "matrix_tolerance"),
        ({"exogenize": "y"}, ValueError, "exogenized"),
        ({"exogenize": ["y"]}, ValueError, "exogenized"),
    ],
)
def test_renormalization_validation(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "targets": {"y": timeseries([2])},
        "instruments": "x",
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        renormalize(
            BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND"),
            {"y": timeseries([0]), "x": timeseries([1])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            **arguments,  # type: ignore[arg-type]
        )
