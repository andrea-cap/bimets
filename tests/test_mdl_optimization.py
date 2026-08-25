from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import bimets.mdl._stochastic as stochastic_module
from _paper_models import PAPER_ADVANCED_KLEIN, PAPER_DOI
from bimets import (
    BimetsModel,
    OptimizationBound,
    OptimizationError,
    OptimizationFunction,
    OptimizationRestriction,
    OptimizationResult,
    optimize_model,
    timeseries,
)
from bimets.mdl._random import RMersenneTwister
from bimets.mdl._simulation import SimulationConvergenceError
from bimets.mdl._simulation_batch import (
    _binary_columns,
    _initialize_columns,
    simulate_shared_columns,
)
from test_mdl_estimation import klein_data
from test_mdl_simulation import KLEIN_MODEL


def identity_model() -> BimetsModel:
    return BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=2*x\nEND")


@pytest.mark.source("native")
@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("&", True),
        ("|", True),
        ("==", False),
        ("!=", True),
        ("<", True),
        ("<=", True),
        (">", False),
        (">=", False),
        ("+", 3.0),
        ("-", -1.0),
        ("*", 2.0),
        ("/", 0.5),
        ("^", 1.0),
    ],
)
def test_compiled_column_binary_operators(operator: str, expected: object) -> None:
    result = _binary_columns(np.asarray([1.0]), operator, np.asarray([2.0]))

    assert result[0] == expected


def test_seeded_linear_optimum_function_and_method_forms() -> None:
    model = identity_model()
    data = {"y": timeseries([0]), "x": timeseries([0])}
    kwargs = {
        "coefficients": {},
        "time_range": (2000, 1, 2000, 1),
        "bounds": {"x": (0, 5)},
        "objective_functions": "y",
        "replicas": 20,
        "seed": 3,
        "simulation_type": "STATIC",
    }

    functional = optimize_model(model, data, **kwargs)  # type: ignore[arg-type]
    method = model.optimize(data, **kwargs)  # type: ignore[arg-type]
    bound = model.bind(data).optimize(**kwargs)  # type: ignore[arg-type]

    draws = RMersenneTwister(3).uniform(0, 5, (1, 20))[0]
    maximizing = int(np.argmax(draws))
    for result in (functional, method, bound):
        assert isinstance(result, OptimizationResult)
        assert result.maximizing_replica == maximizing
        assert result.objective_max == pytest.approx(2 * draws[maximizing])
        assert result.opt_fun_max == result.objective_max
        assert result.instruments["x"].values[0] == pytest.approx(draws[maximizing])
        assert result.data["x"].values[0] == pytest.approx(draws[maximizing])
        assert result.feasible_count == 20
        assert result.replicas == 20
        assert result.summary().shape == (20, 2)
        assert result.objective_results.flags.writeable is False
        assert result.objective_paths.flags.writeable is False
        assert result.feasible.flags.writeable is False
    assert data["x"].values.tolist() == [0]
    with pytest.raises(TypeError):
        functional.instruments["x"] = timeseries([1])  # type: ignore[index]
    with pytest.raises(ValueError, match="one-dimensional"):
        replace(functional, objective_results=np.ones((2, 2)))
    with pytest.raises(ValueError, match="periods by realizations"):
        replace(functional, objective_paths=np.ones((1, 2)))
    with pytest.raises(ValueError, match="feasible mask"):
        replace(functional, feasible=np.ones(2, dtype=bool))


@pytest.mark.source("bimets-R")
def test_optimization_supports_fullnewton_and_backfilled_baseline() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    result = optimize_model(
        model,
        {
            "y": timeseries([10.0, 0.0], start=(1999, 1)),
            "z": timeseries([2.0, 0.0], start=(1999, 1)),
            "x": timeseries([1.0, 0.0], start=(1999, 1)),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (1.0, 2.0)},
        objective_functions="y",
        replicas=3,
        seed=2,
        simulation_type="STATIC",
        algorithm="FULLNEWTON",
        convergence=1e-10,
        backfill=1,
        jacobian_drop="z",
    )

    assert result.stochastic.baseline.algorithm == "FULLNEWTON"
    np.testing.assert_allclose(
        result.stochastic.baseline["y"].values,
        [10.0, 0.0],
        atol=1e-12,
    )


def test_endogenous_instrument_restrictions_use_add_factor() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND")
    replicas = 100
    result = optimize_model(
        model,
        {"y": timeseries([0]), "x": timeseries([0])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"y": (-2, 2), "x": (0, 5)},
        objective_functions="y",
        restrictions="x+(y^2)/2 < 4 & x+y > 0",
        replicas=replicas,
        seed=12,
        simulation_type="STATIC",
    )

    rng = RMersenneTwister(12)
    add_factor = rng.uniform(-2, 2, (1, replicas))[0]
    exogenous = rng.uniform(0, 5, (1, replicas))[0]
    expected_feasible = (exogenous + add_factor**2 / 2 < 4) & (
        exogenous + add_factor > 0
    )
    expected_objective = exogenous + add_factor
    candidates = np.flatnonzero(expected_feasible)
    maximizing = int(candidates[np.argmax(expected_objective[candidates])])

    np.testing.assert_array_equal(result.feasible, expected_feasible)
    assert result.maximizing_replica == maximizing
    assert result.objective_max == pytest.approx(expected_objective[maximizing])
    assert result.instruments["y"].values[0] == pytest.approx(add_factor[maximizing])
    adjustment = result.constant_adjustments["y"]
    assert not isinstance(adjustment, float)
    assert adjustment.values[0] == pytest.approx(add_factor[maximizing])
    verification = model.simulate(
        result.data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        simulation_type="STATIC",
        constant_adjustments=result.constant_adjustments,
    )
    assert verification["y"].values[0] == pytest.approx(result.objective_max)


def test_time_ranged_bounds_and_objective_functions() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND")
    result = optimize_model(
        model,
        {
            "y": timeseries([0, 0], start=(2000, 1)),
            "x": timeseries([1, 1], start=(2000, 1)),
        },
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        bounds={"x": OptimizationBound(10, 20, time_range=(2001, 1, 2001, 1))},
        objective_functions={
            "first": OptimizationFunction("y", (2000, 1, 2000, 1)),
            "second": OptimizationFunction("2*y", (2001, 1, 2001, 1)),
        },
        replicas=10,
        seed=5,
    )

    draws = RMersenneTwister(5).uniform(10, 20, (1, 10))[0]
    maximizing = int(np.argmax(draws))
    np.testing.assert_allclose(result.instruments["x"].values, [1, draws[maximizing]])
    assert result.objective_path is not None
    np.testing.assert_allclose(result.objective_path.values, [1, 2 * draws[maximizing]])
    assert result.objective_max == pytest.approx(1 + 2 * draws[maximizing])


@pytest.mark.source("bimets-R")
def test_forecast_objective_extends_historical_endogenous_data() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND")
    result = optimize_model(
        model,
        {
            "y": timeseries([2], start=(2010, 1)),
            "x": timeseries([2, 4], start=(2010, 1)),
        },
        coefficients={},
        time_range=(2011, 1, 2011, 1),
        bounds={"y": (1, 3)},
        objective_functions="y",
        replicas=5,
        seed=7,
        simulation_type="FORECAST",
    )

    assert result.objective_path is not None
    assert result.objective_max == pytest.approx(4 + result.instruments["y"].values[0])
    assert result.objective_path.values[0] == pytest.approx(result.objective_max)
    adjustment = result.constant_adjustments["y"]
    assert not isinstance(adjustment, float)
    assert adjustment.end == result.objective_path.end


def test_no_feasible_or_finite_solution_returns_diagnostics() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND")
    impossible = optimize_model(
        model,
        {"y": timeseries([0]), "x": timeseries([0])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (0, 1)},
        objective_functions="y",
        restrictions="x < -1",
        replicas=5,
        seed=1,
    )
    nonfinite = optimize_model(
        model,
        {"y": timeseries([0]), "x": timeseries([0])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (-2, -1)},
        objective_functions="LOG(x)",
        replicas=5,
        seed=1,
    )

    for result in (impossible, nonfinite):
        assert result.objective_max is None
        assert result.objective_path is None
        assert result.objective_mean is None
        assert result.objective_standard_deviation is None
        assert result.maximizing_replica is None
        assert result.feasible_count == 0
        assert not result.instruments


def test_single_feasible_realization_has_undefined_sample_deviation() -> None:
    result = optimize_model(
        identity_model(),
        {"y": timeseries([0]), "x": timeseries([0])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (0, 1)},
        objective_functions="y",
        replicas=1,
        seed=1,
    )

    assert result.opt_fun_sd is not None
    assert math.isnan(result.opt_fun_sd)
    assert result.opt_fun_ave == result.objective_max


@pytest.mark.source("bimets-R")
def test_optimization_seed_reproduces_r_uniform_draws_and_matrix_order() -> None:
    rng = RMersenneTwister(1)

    actual = rng.uniform(0, 1, (2, 2))

    np.testing.assert_allclose(
        actual,
        [
            [0.2655086631421, 0.5728533633518964],
            [0.37212389963679016, 0.9082077899947762],
        ],
        rtol=0,
        atol=1e-15,
    )


def test_gauss_seidel_candidates_use_one_vectorized_matrix_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stochastic_module._simulate
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stochastic_module, "_simulate", counted)
    result = optimize_model(
        identity_model(),
        {"y": timeseries([0]), "x": timeseries([0])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (0, 5)},
        objective_functions="y",
        replicas=250,
        seed=3,
        simulation_type="STATIC",
    )

    assert result.replicas == 250
    assert calls == 1  # unperturbed baseline only; candidates are one matrix solve


@pytest.mark.source("native")
def test_vectorized_candidates_support_mdl_functions_and_conditions() -> None:
    model = BimetsModel.from_text(
        """MODEL
IDENTITY> transformed
EQ> transformed=+ABS(-x)+EXP(LOG(x))+TSLAG(x)+TSDELTA(x)+TSDELTALOG(x)+TSDELTAP(x)+MOVAVG(x,2)+MOVSUM(x,2)+pi
IDENTITY> selected
EQ> selected=x
IF> x>1.5
IDENTITY> selected
EQ> selected=-x
IF> x<=1.5
IDENTITY> selected
EQ> selected=0
IF> x>10
END"""
    )
    data = {
        "transformed": timeseries([0.0, 0.0], start=(1999, 1)),
        "selected": timeseries([0.0, 0.0], start=(1999, 1)),
        "x": timeseries([1.0, 1.0], start=(1999, 1)),
    }
    result = optimize_model(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (1.0, 2.0)},
        objective_functions="transformed+selected",
        replicas=8,
        seed=4,
        simulation_type="STATIC",
    )

    draws = result.stochastic.instrument_realizations["x"][0]
    transformed = (
        draws
        + draws
        + 1
        + (draws - 1)
        + np.log(draws)
        + 100 * (draws - 1)
        + (draws + 1) / 2
        + draws
        + 1
        + np.pi
    )
    selected = np.where(draws > 1.5, draws, -draws)
    np.testing.assert_allclose(result.objective_results, transformed + selected)


@pytest.mark.source("native")
def test_vectorized_candidates_support_all_lhs_transformations() -> None:
    model = BimetsModel.from_text(
        """MODEL
IDENTITY> log_y
EQ> LOG(log_y)=log_x
IDENTITY> exp_y
EQ> EXP(exp_y)=exp_x
IDENTITY> delta_y
EQ> TSDELTA(delta_y)=delta_x
IDENTITY> dlog_y
EQ> TSDELTALOG(dlog_y)=dlog_x
IDENTITY> pct_y
EQ> TSDELTAP(pct_y)=pct_x
END"""
    )
    data = {
        "log_y": timeseries([1.0, 1.0], start=(1999, 1)),
        "exp_y": timeseries([1.0, 1.0], start=(1999, 1)),
        "delta_y": timeseries([10.0, 10.0], start=(1999, 1)),
        "dlog_y": timeseries([10.0, 10.0], start=(1999, 1)),
        "pct_y": timeseries([10.0, 10.0], start=(1999, 1)),
        "log_x": timeseries([np.log(2), np.log(2)], start=(1999, 1)),
        "exp_x": timeseries([np.exp(2), np.exp(2)], start=(1999, 1)),
        "delta_x": timeseries([1.0, 1.0], start=(1999, 1)),
        "dlog_x": timeseries([0.1, 0.1], start=(1999, 1)),
        "pct_x": timeseries([10.0, 10.0], start=(1999, 1)),
    }
    result = optimize_model(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"delta_x": (1.0, 2.0)},
        objective_functions="log_y+exp_y+delta_y+dlog_y+pct_y",
        replicas=4,
        seed=9,
        simulation_type="STATIC",
    )

    draws = result.stochastic.instrument_realizations["delta_x"][0]
    expected = 2 + 2 + (10 + draws) + 10 * np.exp(0.1) + 11
    np.testing.assert_allclose(result.objective_results, expected)


@pytest.mark.source("native")
def test_vectorized_behavioral_pdl_and_ar_match_scalar_simulation() -> None:
    model = BimetsModel.from_text(
        """MODEL
BEHAVIORAL> y
EQ> y=a+b*x
COEFF> a b
PDL> b 2 4 N F
ERROR> AUTO(1)
END"""
    )
    data = {
        "y": timeseries([2.0, 3.0, 4.0, 5.0, 6.0], start=(2000, 1)),
        "x": timeseries([1.0, 2.0, 3.0, 4.0, 5.0], start=(2000, 1)),
    }
    coefficients = {
        "y": {
            "a": 1.0,
            "b": 0.5,
            "b__PDL__1": 0.25,
            "b__PDL__2": 0.125,
            "b__PDL__3": 0.0625,
            "RHO_1": 0.2,
        }
    }
    result = optimize_model(
        model,
        data,
        coefficients=coefficients,
        time_range=(2004, 1, 2004, 1),
        bounds={"x": (4.0, 6.0)},
        objective_functions="y",
        replicas=5,
        seed=6,
        simulation_type="DYNAMIC",
    )

    assert result.maximizing_replica is not None
    scalar = model.simulate(
        result.data,
        coefficients=coefficients,
        time_range=(2004, 1, 2004, 1),
        simulation_type="DYNAMIC",
    )
    assert result.objective_max == pytest.approx(scalar["y"].values[0])


@pytest.mark.source("native")
def test_vectorized_ar_residuals_support_transformed_lhs() -> None:
    model = BimetsModel.from_text(
        """MODEL
BEHAVIORAL> log_y
EQ> LOG(log_y)=a*log_x
COEFF> a
ERROR> AUTO(1)
BEHAVIORAL> exp_y
EQ> EXP(exp_y)=a*exp_x
COEFF> a
ERROR> AUTO(1)
BEHAVIORAL> delta_y
EQ> TSDELTA(delta_y)=a*delta_x
COEFF> a
ERROR> AUTO(1)
BEHAVIORAL> dlog_y
EQ> TSDELTALOG(dlog_y)=a*dlog_x
COEFF> a
ERROR> AUTO(1)
BEHAVIORAL> pct_y
EQ> TSDELTAP(pct_y)=a*pct_x
COEFF> a
ERROR> AUTO(1)
END"""
    )
    data = {
        "log_y": timeseries([2.0, 2.0, 2.0], start=(1998, 1)),
        "exp_y": timeseries([1.0, 1.0, 1.0], start=(1998, 1)),
        "delta_y": timeseries([10.0, 10.0, 10.0], start=(1998, 1)),
        "dlog_y": timeseries([10.0, 10.0, 10.0], start=(1998, 1)),
        "pct_y": timeseries([10.0, 10.0, 10.0], start=(1998, 1)),
        "log_x": timeseries([np.log(2)] * 3, start=(1998, 1)),
        "exp_x": timeseries([np.exp(1)] * 3, start=(1998, 1)),
        "delta_x": timeseries([1.0, 1.0, 1.0], start=(1998, 1)),
        "dlog_x": timeseries([0.1, 0.1, 0.1], start=(1998, 1)),
        "pct_x": timeseries([10.0, 10.0, 10.0], start=(1998, 1)),
    }
    coefficients = {
        name: {"a": 1.0, "RHO_1": 0.25}
        for name in ("log_y", "exp_y", "delta_y", "dlog_y", "pct_y")
    }
    result = optimize_model(
        model,
        data,
        coefficients=coefficients,
        time_range=(2000, 1, 2000, 1),
        bounds={"delta_x": (1.0, 2.0)},
        objective_functions="log_y+exp_y+delta_y+dlog_y+pct_y",
        replicas=4,
        seed=10,
        simulation_type="DYNAMIC",
    )

    assert result.maximizing_replica is not None
    scalar = model.simulate(
        result.data,
        coefficients=coefficients,
        time_range=(2000, 1, 2000, 1),
        simulation_type="DYNAMIC",
    )
    expected = sum(scalar[name].values[0] for name in model.endogenous)
    assert result.objective_max == pytest.approx(expected)


@pytest.mark.source("native")
def test_nonfinite_vectorized_candidate_reports_replica() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=LOG(x)\nEND")

    with pytest.raises(OptimizationError, match="non-finite value in replica"):
        optimize_model(
            model,
            {"y": timeseries([0.0]), "x": timeseries([1.0])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            bounds={"x": (-2.0, -1.0)},
            objective_functions="y",
            replicas=2,
            seed=3,
            simulation_type="STATIC",
        )


@pytest.mark.source("native")
def test_vectorized_static_initialization_uses_previous_finite_value() -> None:
    result = optimize_model(
        identity_model(),
        {
            "y": timeseries([2.0, np.nan], start=(1999, 1)),
            "x": timeseries([1.0, 1.0], start=(1999, 1)),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (1.0, 2.0)},
        objective_functions="y",
        replicas=3,
        seed=3,
        simulation_type="STATIC",
    )

    assert result.objective_max is not None


@pytest.mark.source("native")
def test_vectorized_solver_rejects_invalid_internal_initial_state() -> None:
    bound = identity_model().bind({"y": timeseries([np.nan]), "x": timeseries([1.0])})
    period = bound.data["y"].start
    historical = {"y": np.full((1, 2), np.nan), "x": np.ones((1, 2))}
    working = {name: values.copy() for name, values in historical.items()}

    with pytest.raises(ValueError, match="at least one simulation period"):
        simulate_shared_columns(
            bound,
            coefficients={},
            periods=(),
            instrument_realizations={},
            replicas=1,
            simulation_type="STATIC",
            convergence=0.01,
            max_iterations=10,
            zero_error_autocorrelation=False,
            constant_adjustments={},
        )
    with pytest.raises(ValueError, match="GAUSS-SEIDEL or NEWTON"):
        simulate_shared_columns(
            bound,
            coefficients={},
            periods=(period,),
            instrument_realizations={},
            replicas=1,
            simulation_type="STATIC",
            algorithm="FULLNEWTON",
            convergence=0.01,
            max_iterations=10,
            zero_error_autocorrelation=False,
            constant_adjustments={},
        )
    with pytest.raises(ValueError, match="requires a historical value"):
        _initialize_columns(
            period,
            0,
            "STATIC",
            bound,
            historical,
            working,
            frozenset({"y"}),
        )
    with pytest.raises(ValueError, match="initialize forecast"):
        _initialize_columns(
            period,
            0,
            "FORECAST",
            bound,
            historical,
            working,
            frozenset(),
        )
    with pytest.raises(ValueError, match="initialize endogenous"):
        _initialize_columns(
            period,
            0,
            "STATIC",
            bound,
            historical,
            working,
            frozenset(),
        )


@pytest.mark.source("native")
def test_shared_newton_builds_and_solves_a_multi_rhs_jacobian() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    bound = model.bind(
        {
            "y": timeseries([0.0]),
            "z": timeseries([0.0]),
            "x": timeseries([1.0]),
        }
    )
    period = bound.data["y"].start

    result = simulate_shared_columns(
        bound,
        coefficients={},
        periods=(period,),
        instrument_realizations={"x": np.asarray([[2.0]])},
        replicas=1,
        simulation_type="STATIC",
        algorithm="NEWTON",
        convergence=1e-10,
        max_iterations=20,
        zero_error_autocorrelation=False,
        constant_adjustments={},
    )

    np.testing.assert_allclose(result["y"], [[2.0 / 0.875]], atol=1e-10)
    np.testing.assert_allclose(result["z"], [[0.5 / 0.875]], atol=1e-10)


@pytest.mark.source("native")
def test_shared_newton_nonconvergence_can_raise_or_retain() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=y*y+x\nEND")
    bound = model.bind({"y": timeseries([1.0]), "x": timeseries([0.0])})
    period = bound.data["y"].start
    kwargs = {
        "coefficients": {},
        "periods": (period,),
        "instrument_realizations": {"x": np.asarray([[1.0]])},
        "replicas": 1,
        "simulation_type": "STATIC",
        "algorithm": "NEWTON",
        "convergence": 1e-14,
        "max_iterations": 3,
        "zero_error_autocorrelation": False,
        "constant_adjustments": {},
    }

    with pytest.raises(SimulationConvergenceError, match="shared Newton block"):
        simulate_shared_columns(bound, **kwargs)  # type: ignore[arg-type]
    with pytest.warns(RuntimeWarning, match="retaining the final iteration"):
        retained = simulate_shared_columns(
            bound,
            retain_final_iteration=True,
            **kwargs,  # type: ignore[arg-type]
        )
    assert retained["y"].shape == (1, 1)


@pytest.mark.source("bimets-R")
def test_klein_optimal_control_follows_original_bimets_example() -> None:
    model = BimetsModel.from_text(KLEIN_MODEL, name="klein-optimal-control")
    historical = klein_data()
    coefficients = model.estimate(historical)
    extended = {}
    for name, series in historical.items():
        if name in model.endogenous:
            mode = "missing"
        elif name in {"time", "t"}:
            mode = "linear"
        else:
            mode = "constant"
        extended[name] = series.extend(up_to=(1942, 1), mode=mode)

    result = optimize_model(
        model,
        extended,
        coefficients=coefficients,
        time_range=(1942, 1, 1942, 1),
        bounds={"cn": (-5, 5), "g": (15, 25)},
        restrictions="g+(cn^2)/2 < 27 & g+cn > 17",
        objective_functions="(y-110)+(cn-90)*ABS(cn-90)-(g-20)^0.5",
        replicas=100,
        seed=123,
        simulation_type="FORECAST",
        convergence=1e-6,
        max_iterations=500,
    )

    cn_factor = result.stochastic.instrument_realizations["cn"][0]
    government = result.stochastic.instrument_realizations["g"][0]
    consumption = result.stochastic["cn"].realizations[0]
    output = result.stochastic["y"].realizations[0]
    expected_restriction = (government + cn_factor**2 / 2 < 27) & (
        government + cn_factor > 17
    )
    with np.errstate(invalid="ignore"):
        expected_objective = (
            (output - 110)
            + (consumption - 90) * np.abs(consumption - 90)
            - np.power(government - 20, 0.5)
        )
    expected_feasible = expected_restriction & np.isfinite(expected_objective)
    candidates = np.flatnonzero(expected_feasible)
    maximizing = int(candidates[np.argmax(expected_objective[candidates])])

    np.testing.assert_array_equal(result.feasible, expected_feasible)
    np.testing.assert_allclose(
        result.objective_results, expected_objective, equal_nan=True
    )
    assert result.maximizing_replica == maximizing
    assert result.objective_max == pytest.approx(expected_objective[maximizing])
    assert result.instruments["cn"].values[0] == pytest.approx(cn_factor[maximizing])
    assert result.instruments["g"].values[0] == pytest.approx(government[maximizing])


def test_shared_candidate_nonconvergence_warns_and_retains_final_iteration() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=y*y\nEND")
    with pytest.warns(RuntimeWarning, match="retaining the final iteration"):
        result = optimize_model(
            model,
            {"y": timeseries([0])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            bounds={"y": (1, 2)},
            objective_functions="y",
            replicas=1,
            max_iterations=5,
        )
    assert result.objective_max is not None


def test_optimal_control_supports_rescheck_equation_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=x\nIDENTITY> z\nEQ> z=x+1\nEND"
    )
    original = stochastic_module._simulate
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stochastic_module, "_simulate", counted)
    result = optimize_model(
        model,
        {
            "y": timeseries([1]),
            "z": timeseries([2]),
            "x": timeseries([1]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        bounds={"x": (0, 2)},
        objective_functions="y",
        replicas=4,
        seed=2,
        simulation_type="RESCHECK",
        rescheck_equations="y",
    )

    assert tuple(result.stochastic) == ("y",)
    assert result.objective_max is not None
    assert calls == 1  # baseline only; RESCHECK candidates are vectorized


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: OptimizationBound(1, 1), ValueError, "less than"),
        (lambda: OptimizationBound(0, np.inf), ValueError, "finite"),
        (lambda: OptimizationFunction(""), ValueError, "cannot be empty"),
        (lambda: OptimizationRestriction(""), ValueError, "cannot be empty"),
    ],
)
def test_optimization_definition_validation(
    factory: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"bounds": {}}, ValueError, "bounds"),
        ({"bounds": {1: (0, 1)}}, ValueError, "instrument names"),
        ({"bounds": {"missing": (0, 1)}}, KeyError, "unknown"),
        ({"bounds": {"x": [0, 1]}}, TypeError, "bounds values"),
        ({"objective_functions": {}}, ValueError, "objective"),
        ({"objective_functions": "missing"}, KeyError, "unknown variables"),
        ({"objective_functions": "1"}, ValueError, "model variable"),
        ({"objective_functions": "y>0"}, ValueError, "numeric expression"),
        ({"objective_functions": {1: "y"}}, ValueError, "names"),
        ({"objective_functions": {"bad": object()}}, TypeError, "invalid type"),
        ({"objective_functions": object()}, TypeError, "invalid type"),
        (
            {
                "objective_functions": {
                    "one": OptimizationFunction("y"),
                    "two": OptimizationFunction("x"),
                }
            },
            ValueError,
            "overlap",
        ),
        ({"restrictions": ""}, ValueError, "cannot be empty"),
        ({"restrictions": "x+1"}, ValueError, "logical expression"),
        (
            {"bounds": {"x": OptimizationBound(0, 1, time_range=(2001, 1, 2001, 1))}},
            ValueError,
            "does not overlap",
        ),
        ({"replicas": True}, TypeError, "replicas"),
        ({"replicas": 0}, ValueError, "replicas"),
        ({"seed": True}, TypeError, "seed"),
    ],
)
def test_optimize_input_validation(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "bounds": {"x": (0, 1)},
        "objective_functions": "y",
        "replicas": 2,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        optimize_model(
            identity_model(),
            {"y": timeseries([0]), "x": timeseries([0])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.source(PAPER_DOI)
def test_advanced_optimal_control_matches_paper() -> None:
    """Reproduce the 10,000-replica control exercise in paper section 3.10."""
    model = BimetsModel.from_text(PAPER_ADVANCED_KLEIN)
    data = klein_data()
    coefficients = model.estimate(data)
    extended = {}
    for name, series in data.items():
        if name in {"w2", "g"}:
            mode = "constant"
        elif name in {"t", "k", "time"}:
            mode = "linear"
        else:
            mode = "missing"
        extended[name] = series.extend(up_to=(1942, 1), mode=mode)

    result = optimize_model(
        model,
        extended,
        coefficients=coefficients,
        time_range=(1942, 1, 1942, 1),
        simulation_type="FORECAST",
        convergence=1e-4,
        max_iterations=1_000,
        bounds={"cn": (-5, 5), "g": (15, 25)},
        restrictions="g+(cn^2)/2 < 27 & g+cn > 17",
        objective_functions="(y-110)+(cn-90)*ABS(cn-90)-(g-20)^0.5",
        replicas=10_000,
        seed=123,
    )

    assert result.objective_max == pytest.approx(210.5755, abs=5e-5)
    assert result.maximizing_replica == 6672
    maximizing = result.maximizing_replica
    cn_factor = result.stochastic.instrument_realizations["cn"][0, maximizing]
    government = result.stochastic.instrument_realizations["g"][0, maximizing]
    assert cn_factor == pytest.approx(2.032203, abs=5e-7)
    assert government == pytest.approx(24.89773, abs=5e-6)
    assert government + cn_factor**2 / 2 < 27
    assert government + cn_factor > 17

    verification = model.simulate(
        result.data,
        coefficients=coefficients,
        time_range=(1942, 1, 1942, 1),
        simulation_type="FORECAST",
        # A scalar verification has no slow candidate column keeping the
        # shared R-style iteration open, so use a tighter stopping threshold.
        convergence=1e-5,
        max_iterations=1_000,
        constant_adjustments=result.constant_adjustments,
    )
    output = verification["y"].values[0]
    consumption = verification["cn"].values[0]
    objective = (
        (output - 110)
        + (consumption - 90) * abs(consumption - 90)
        - (government - 20) ** 0.5
    )
    assert objective == pytest.approx(result.objective_max, abs=5e-3)
