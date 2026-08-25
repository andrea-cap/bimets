from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import bimets.mdl._sparse as sparse_backend
import bimets.mdl._stochastic as stochastic_module
from _paper_models import PAPER_ADVANCED_KLEIN, PAPER_DOI
from bimets import (
    STOCHSIMULATE,
    BimetsModel,
    StochasticDisturbance,
    StochasticSimulationError,
    StochasticSimulationResult,
    YearPeriod,
    stochastic_simulate,
    timeseries,
)
from bimets.mdl._random import RMersenneTwister
from bimets.mdl._simulation import SimulationConvergenceError
from bimets.mdl._simulation_batch import simulate_shared_columns
from bimets.mdl._sparse import FloatArray, ResidualFunction, SparseJacobian
from test_mdl_estimation import klein_data


def identity_model() -> BimetsModel:
    return BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND", name="identity")


def test_matrix_disturbances_replace_exogenous_and_add_to_endogenous() -> None:
    model = identity_model()
    data = {"y": timeseries([0, 0]), "x": timeseries([10, 20])}
    exogenous = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=float)
    endogenous = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    result = stochastic_simulate(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        disturbances={
            "x": StochasticDisturbance("MATRIX", exogenous),
            "y": StochasticDisturbance("MATRIX", endogenous),
        },
        constant_adjustments={"y": 1.0},
        replicas=3,
    )

    expected = exogenous + endogenous + 1
    assert isinstance(result, StochasticSimulationResult)
    np.testing.assert_allclose(result.baseline["y"].values, [11, 21])
    np.testing.assert_allclose(result["y"].realizations, expected)
    np.testing.assert_allclose(result["y"].mean.values, np.mean(expected, axis=1))
    np.testing.assert_allclose(
        result["y"].sd.values,
        np.std(expected, axis=1, ddof=1),
    )
    np.testing.assert_allclose(result.instrument_baseline["x"].values, [10, 20])
    np.testing.assert_allclose(result.instrument_realizations["x"], exogenous)
    np.testing.assert_allclose(result.instrument_realizations["y"], endogenous + 1)
    assert result["y"].realizations.flags.writeable is False
    assert result.instrument_realizations["x"].flags.writeable is False
    assert result.summary().shape == (2, 2)
    assert tuple(result) == ("y",)
    assert len(result) == 1
    assert "replicas=3" in repr(result)


def test_random_disturbances_are_seeded_iid_and_range_limited() -> None:
    model = identity_model()
    data = {"y": timeseries([0, 0, 0]), "x": timeseries([10, 10, 10])}
    disturbances = {
        "x": StochasticDisturbance("UNIF", (-1, 1), time_range=(2001, 1, 2001, 1)),
        "y": StochasticDisturbance("NORM", (0, 0.5)),
    }

    first = model.stochastic_simulate(
        data,
        coefficients={},
        time_range=(2000, 1, 2002, 1),
        disturbances=disturbances,
        replicas=8,
        seed=123,
    )
    second = model.bind(data).stochastic_simulate(
        coefficients={},
        time_range=(2000, 1, 2002, 1),
        disturbances=disturbances,
        replicas=8,
        seed=123,
    )

    np.testing.assert_array_equal(first["y"].realizations, second["y"].realizations)
    np.testing.assert_allclose(first.instrument_realizations["x"][[0, 2]], 10)
    assert np.any(first.instrument_realizations["x"][1] != 10)
    assert first.seed == 123
    assert first.replicas == 8


@pytest.mark.source("bimets-R")
def test_r_seed_reproduces_normal_then_uniform_matrix_draws() -> None:
    rng = RMersenneTwister(123)
    normal = rng.normal(0.0, 1.0, (2, 3))
    uniform = rng.uniform(1.0, 2.0, (2, 3))

    np.testing.assert_allclose(
        normal,
        [
            [-0.560475646552213, 1.55870831414912, 0.129287735160946],
            [-0.23017748948328, 0.070508391424576, 1.71506498688328],
        ],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        uniform,
        [
            [1.67757063545287, 1.10292468266562, 1.24608773435466],
            [1.57263340195641, 1.89982497040182, 1.04205953353085],
        ],
        atol=5e-15,
    )


@pytest.mark.source("bimets-R")
def test_r_seed_reproduces_replacement_sampling_and_matrix_order() -> None:
    indexes = RMersenneTwister(9).sample_with_replacement(176, (4, 5))

    np.testing.assert_array_equal(
        indexes,
        np.asarray(
            [
                [52, 82, 29, 36, 85],
                [5, 2, 43, 41, 64],
                [58, 139, 36, 34, 37],
                [151, 47, 17, 56, 174],
            ]
        ),
    )
    with pytest.raises(ValueError, match="population_size"):
        RMersenneTwister(1).sample_with_replacement(0, 1)


@pytest.mark.source("bimets-R")
def test_stochastic_seed_respects_r_distribution_and_variable_order() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x+z\nEND")
    result = stochastic_simulate(
        model,
        {
            "y": timeseries([0.0, 0.0]),
            "x": timeseries([0.0, 0.0]),
            "z": timeseries([0.0, 0.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        disturbances={
            "x": StochasticDisturbance("NORMAL", (0.0, 1.0)),
            "z": StochasticDisturbance("UNIFORM", (1.0, 2.0)),
        },
        replicas=3,
        seed=123,
    )

    np.testing.assert_allclose(
        result.instrument_realizations["x"],
        [
            [-0.560475646552213, 1.55870831414912, 0.129287735160946],
            [-0.23017748948328, 0.070508391424576, 1.71506498688328],
        ],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.instrument_realizations["z"],
        [
            [1.67757063545287, 1.10292468266562, 1.24608773435466],
            [1.57263340195641, 1.89982497040182, 1.04205953353085],
        ],
        atol=5e-15,
    )


@pytest.mark.source(PAPER_DOI)
def test_advanced_klein_stochastic_forecast_matches_paper() -> None:
    """Reproduce the seeded stochastic forecast in paper section 3.7."""
    model = BimetsModel.from_text(PAPER_ADVANCED_KLEIN, name="advanced-klein-paper")
    historical = klein_data()
    coefficients = model.estimate(historical)
    extended = {}
    for name, series in historical.items():
        if name in {"w2", "g"}:
            mode = "constant"
        elif name in {"t", "k", "time"}:
            mode = "linear"
        else:
            mode = "missing"
        extended[name] = series.extend(up_to=(1944, 1), mode=mode)

    result = STOCHSIMULATE(
        model,
        extended,
        coefficients=coefficients,
        time_range=(1941, 1, 1944, 1),
        disturbances={
            "cn": StochasticDisturbance(
                "NORMAL",
                (0.0, coefficients["cn"].standard_error),
                time_range=(1942, 1, 1942, 1),
            ),
            "g": StochasticDisturbance("UNIFORM", (-1.0, 1.0)),
        },
        replicas=100,
        seed=123,
        simulation_type="FORECAST",
    )

    np.testing.assert_allclose(
        result["y"].mean.values,
        [125.5045, 173.2946, 185.9602, 141.0807],
        atol=5e-5,
    )
    np.testing.assert_allclose(
        result["y"].sd.values,
        [4.250935, 9.2632, 11.87774, 11.6973],
        atol=5e-6,
    )
    np.testing.assert_allclose(
        result["y"].realizations[:, :5],
        [
            [121.3591, 123.7998, 120.3449, 121.0243, 123.0448],
            [170.2987, 174.5269, 170.0456, 169.3925, 168.6419],
            [186.5037, 187.3361, 185.0368, 177.2287, 186.3369],
            [145.1024, 139.5191, 139.5570, 135.5024, 151.5389],
        ],
        atol=5e-5,
    )


def test_stochastic_simulation_supports_forward_looking_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+x\nEND")
    data = {
        "y": timeseries([1, 2, 3, 4]),
        "x": timeseries([0, 1, 1, 0]),
    }
    shocks = np.asarray([[0, 1], [0, -1]], dtype=float)
    original = stochastic_module._simulate
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stochastic_module, "_simulate", counted)

    result = STOCHSIMULATE(
        model,
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        disturbances={"y": StochasticDisturbance("MATRIX", shocks)},
        replicas=2,
        algorithm="NEWTON",
        convergence=1e-10,
    )

    # Terminal y[2003] is fixed at 4. Solve backwards for each add-factor path.
    np.testing.assert_allclose(
        result["y"].realizations,
        np.asarray([[2.5, 3], [3, 2]]),
        atol=1e-10,
    )
    assert calls == 1


@pytest.mark.source("bimets-R")
def test_newton_stochastic_replicas_reuse_the_baseline_sparse_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    matrix = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]])
    original = sparse_backend.factorize_finite_difference_jacobian
    factorization_calls = 0

    def recording_factorization(
        residual: ResidualFunction,
        current: FloatArray,
        current_residual: FloatArray,
        *,
        relative_step: float,
        column_rows: Sequence[Sequence[int]] | None = None,
    ) -> SparseJacobian:
        nonlocal factorization_calls
        factorization_calls += 1
        return original(
            residual,
            current,
            current_residual,
            relative_step=relative_step,
            column_rows=column_rows,
        )

    monkeypatch.setattr(
        sparse_backend,
        "factorize_finite_difference_jacobian",
        recording_factorization,
    )
    result = stochastic_simulate(
        model,
        {
            "y": timeseries([0.0]),
            "z": timeseries([0.0]),
            "x": timeseries([1.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        disturbances={"x": StochasticDisturbance("MATRIX", matrix)},
        replicas=matrix.shape[1],
        algorithm="NEWTON",
        convergence=1e-10,
    )

    np.testing.assert_allclose(result["y"].realizations, matrix / 0.875)
    np.testing.assert_allclose(result["z"].realizations, matrix / 3.5)
    assert factorization_calls == 1


@pytest.mark.source("bimets-R")
def test_fullnewton_builds_an_independent_jacobian_for_each_realization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    matrix = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    original = sparse_backend.factorize_finite_difference_jacobian
    factorization_calls = 0

    def recording_factorization(
        residual: ResidualFunction,
        current: FloatArray,
        current_residual: FloatArray,
        *,
        relative_step: float,
        column_rows: Sequence[Sequence[int]] | None = None,
    ) -> SparseJacobian:
        nonlocal factorization_calls
        factorization_calls += 1
        return original(
            residual,
            current,
            current_residual,
            relative_step=relative_step,
            column_rows=column_rows,
        )

    monkeypatch.setattr(
        sparse_backend,
        "factorize_finite_difference_jacobian",
        recording_factorization,
    )
    result = stochastic_simulate(
        model,
        {
            "y": timeseries([0.0]),
            "z": timeseries([0.0]),
            "x": timeseries([1.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        disturbances={"x": StochasticDisturbance("MATRIX", matrix)},
        replicas=matrix.shape[1],
        algorithm="FULLNEWTON",
        convergence=1e-10,
    )

    np.testing.assert_allclose(result["y"].realizations, matrix / 0.875)
    assert result.baseline.algorithm == "FULLNEWTON"
    assert factorization_calls == matrix.shape[1] + 1


@pytest.mark.source("native")
def test_fullnewton_can_distribute_independent_realizations() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    matrix = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    disturbances = {"x": StochasticDisturbance("MATRIX", matrix)}
    data = {
        "y": timeseries([0.0]),
        "z": timeseries([0.0]),
        "x": timeseries([1.0]),
    }

    sequential = stochastic_simulate(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        disturbances=disturbances,
        replicas=matrix.shape[1],
        algorithm="FULLNEWTON",
        convergence=1e-10,
        workers=1,
    )
    parallel = model.stochastic_simulate(
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        disturbances=disturbances,
        replicas=matrix.shape[1],
        algorithm="FULLNEWTON",
        convergence=1e-10,
        workers=2,
    )

    np.testing.assert_allclose(parallel["y"].realizations, sequential["y"].realizations)
    np.testing.assert_allclose(parallel["z"].realizations, sequential["z"].realizations)


@pytest.mark.source("bimets-R")
def test_stochastic_backfill_applies_to_the_baseline_not_summary_ranges() -> None:
    model = identity_model()
    result = stochastic_simulate(
        model,
        {
            "y": timeseries([5.0, 0.0, 0.0], start=(1999, 1)),
            "x": timeseries([1.0, 2.0, 3.0], start=(1999, 1)),
        },
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        replicas=2,
        backfill=1,
    )

    np.testing.assert_allclose(result.baseline["y"].values, [5.0, 2.0, 3.0])
    assert result["y"].realizations.shape == (2, 2)
    assert result["y"].mean.start == YearPeriod(2000, 1)


@pytest.mark.source("bimets-R")
def test_stochastic_newton_forwards_jacobian_drop_to_realizations(
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
    result = stochastic_simulate(
        model,
        {
            "y": timeseries([0.0]),
            "z": timeseries([0.0]),
            "x": timeseries([1.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        disturbances={"x": StochasticDisturbance("MATRIX", np.asarray([[2.0, 3.0]]))},
        replicas=2,
        algorithm="NEWTON",
        convergence=1e-8,
        jacobian_drop="z",
    )

    np.testing.assert_allclose(result["y"].realizations, [[2 / 0.875, 3 / 0.875]])
    assert calls == 1


@pytest.mark.source("bimets-R")
def test_forward_stochastic_newton_supports_a_reduced_shared_jacobian() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    data = {
        "y": timeseries([0.0, 0.0, 0.0, 0.0]),
        "z": timeseries([0.0, 0.0, 0.0, 0.0]),
        "x": timeseries([0.0, 1.0, 1.0, 0.0]),
    }
    matrix = np.asarray([[1.0, 2.0], [2.0, 3.0]])

    shared = stochastic_simulate(
        model,
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        disturbances={"x": StochasticDisturbance("MATRIX", matrix)},
        replicas=2,
        algorithm="NEWTON",
        convergence=1e-9,
        jacobian_drop="z",
    )
    independent = stochastic_module._stochastic_simulate(
        model,
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        disturbances={"x": StochasticDisturbance("MATRIX", matrix)},
        replicas=2,
        algorithm="NEWTON",
        convergence=1e-9,
        jacobian_drop="z",
    )

    np.testing.assert_allclose(
        shared["y"].realizations,
        independent["y"].realizations,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        shared["z"].realizations,
        independent["z"].realizations,
        atol=1e-8,
    )


@pytest.mark.source("native")
def test_forward_shared_gauss_seidel_matches_the_analytical_solution() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+x\nEND")
    result = stochastic_simulate(
        model,
        {
            "y": timeseries([0.0, 0.0, 0.0, 4.0]),
            "x": timeseries([0.0, 1.0, 1.0, 0.0]),
        },
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        disturbances={
            "x": StochasticDisturbance("MATRIX", np.asarray([[1.0, 2.0], [1.0, 3.0]]))
        },
        replicas=2,
        algorithm="GAUSS-SEIDEL",
        convergence=1e-10,
    )

    np.testing.assert_allclose(result["y"].realizations, [[2.5, 4.5], [3.0, 5.0]])


@pytest.mark.source("native")
def test_forward_shared_newton_can_build_its_own_factorization() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+x\nEND")
    bound = model.bind(
        {
            "y": timeseries([0.0, 0.0, 0.0, 4.0]),
            "x": timeseries([0.0, 1.0, 1.0, 0.0]),
        }
    )
    result = simulate_shared_columns(
        bound,
        coefficients={},
        periods=(YearPeriod(2001, 1), YearPeriod(2002, 1)),
        instrument_realizations={"x": np.asarray([[2.0], [3.0]])},
        replicas=1,
        simulation_type="DYNAMIC",
        algorithm="NEWTON",
        convergence=1e-10,
        max_iterations=20,
        zero_error_autocorrelation=False,
        constant_adjustments={},
    )

    np.testing.assert_allclose(result["y"], [[4.5], [5.0]], atol=1e-10)

    undefined = model.bind(
        {
            "y": timeseries([np.nan, 4.0]),
            "x": timeseries([1.0, 0.0]),
        }
    )
    with pytest.raises(ValueError, match="is not defined"):
        simulate_shared_columns(
            undefined,
            coefficients={},
            periods=(YearPeriod(2000, 1),),
            instrument_realizations={},
            replicas=1,
            simulation_type="DYNAMIC",
            algorithm="NEWTON",
            convergence=0.01,
            max_iterations=10,
            zero_error_autocorrelation=False,
            constant_adjustments={},
        )


@pytest.mark.source("native")
def test_forward_shared_solver_handles_complete_exogenization_and_drop() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+x\nEND")
    data = {
        "y": timeseries([0.0, 2.0, 3.0, 4.0]),
        "x": timeseries([0.0, 1.0, 1.0, 0.0]),
    }
    replacement = timeseries([8.0, 9.0], start=(2001, 1))

    with pytest.warns(UserWarning, match="all feedback variables"):
        fixed = stochastic_simulate(
            model,
            data,
            coefficients={},
            time_range=(2001, 1, 2002, 1),
            disturbances={},
            replicas=2,
            algorithm="NEWTON",
            exogenize={"y": replacement},
        )
    np.testing.assert_allclose(fixed["y"].realizations, [[8.0, 8.0], [9.0, 9.0]])

    with pytest.warns(UserWarning, match="all feedback variables"):
        dropped = stochastic_simulate(
            model,
            data,
            coefficients={},
            time_range=(2001, 1, 2002, 1),
            disturbances={},
            replicas=2,
            algorithm="NEWTON",
            jacobian_drop="y",
            convergence=1e-10,
        )
    np.testing.assert_allclose(dropped["y"].realizations, [[2.5, 2.5], [3.0, 3.0]])


@pytest.mark.source("native")
def test_forward_shared_nonconvergence_identifies_the_replica() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x*y+TSLEAD(y)\nEND")
    with pytest.raises(StochasticSimulationError, match="replica 1"):
        stochastic_simulate(
            model,
            {
                "y": timeseries([4.0, 2.0, 1.0]),
                "x": timeseries([0.5, 0.5, 0.5]),
            },
            coefficients={},
            time_range=(2000, 1, 2001, 1),
            disturbances={
                "x": StochasticDisturbance("MATRIX", np.asarray([[2.0], [2.0]]))
            },
            replicas=1,
            algorithm="GAUSS-SEIDEL",
            max_iterations=2,
        )


@pytest.mark.source("native")
def test_forward_shared_newton_reports_and_can_retain_unconverged_iterations() -> None:
    singular = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=y+TSLEAD(y)\nEND").bind(
        {"y": timeseries([1.0, 1.0])}
    )
    with pytest.raises(
        SimulationConvergenceError, match="shared forward-looking Newton"
    ):
        simulate_shared_columns(
            singular,
            coefficients={},
            periods=(YearPeriod(2000, 1),),
            instrument_realizations={},
            replicas=1,
            simulation_type="DYNAMIC",
            algorithm="NEWTON",
            convergence=0.01,
            max_iterations=10,
            zero_error_autocorrelation=False,
            constant_adjustments={},
        )

    nonlinear = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=x*y+TSLEAD(y)\nEND"
    ).bind(
        {
            "y": timeseries([4.0, 2.0, 1.0]),
            "x": timeseries([0.5, 0.5, 0.5]),
        }
    )
    with pytest.warns(RuntimeWarning, match="retaining the final iteration"):
        retained = simulate_shared_columns(
            nonlinear,
            coefficients={},
            periods=(YearPeriod(2000, 1), YearPeriod(2001, 1)),
            instrument_realizations={"x": np.asarray([[2.0], [2.0]])},
            replicas=1,
            simulation_type="DYNAMIC",
            algorithm="NEWTON",
            convergence=1e-14,
            max_iterations=2,
            zero_error_autocorrelation=False,
            constant_adjustments={},
            retain_final_iteration=True,
        )
    assert retained["y"].shape == (2, 1)


@pytest.mark.source("native")
def test_backward_shared_hybrid_reports_outer_nonconvergence() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.99*z+x\nIDENTITY> z\nEQ> z=0.99*y\nEND"
    )
    bound = model.bind(
        {
            "y": timeseries([0.0]),
            "z": timeseries([0.0]),
            "x": timeseries([0.0]),
        }
    )
    kwargs = {
        "coefficients": {},
        "periods": (YearPeriod(2000, 1),),
        "instrument_realizations": {"x": np.asarray([[1.0]])},
        "replicas": 1,
        "simulation_type": "STATIC",
        "algorithm": "NEWTON",
        "convergence": 1e-12,
        "max_iterations": 2,
        "zero_error_autocorrelation": False,
        "constant_adjustments": {},
        "jacobian_drop": "z",
    }

    with pytest.raises(StochasticSimulationError, match="hybrid Newton block"):
        stochastic_simulate(
            bound,
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            disturbances={"x": StochasticDisturbance("MATRIX", np.asarray([[1.0]]))},
            replicas=1,
            simulation_type="STATIC",
            algorithm="NEWTON",
            convergence=1e-12,
            max_iterations=2,
            jacobian_drop="z",
        )
    with pytest.warns(RuntimeWarning, match="retaining the final iteration"):
        retained = simulate_shared_columns(
            bound,
            retain_final_iteration=True,
            **kwargs,  # type: ignore[arg-type]
        )
    assert retained["y"].shape == (1, 1)


def test_stochastic_rescheck_can_select_equations() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=x\nIDENTITY> z\nEQ> z=x+1\nEND"
    )
    result = model.stochastic_simulate(
        {
            "y": timeseries([1]),
            "z": timeseries([2]),
            "x": timeseries([1]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        simulation_type="RESCHECK",
        rescheck_equations="y",
        disturbances={"x": StochasticDisturbance("MATRIX", np.asarray([[2, 3]]))},
        replicas=2,
    )

    assert tuple(result) == ("y",)
    np.testing.assert_allclose(result["y"].realizations, [[2, 3]])

    for exogenize, expected in (
        ("y", [[1, 1]]),
        ({"y": timeseries([9])}, [[9, 9]]),
    ):
        exogenized = model.stochastic_simulate(
            {
                "y": timeseries([1]),
                "z": timeseries([2]),
                "x": timeseries([1]),
            },
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            simulation_type="RESCHECK",
            rescheck_equations="y",
            exogenize=exogenize,
            disturbances={"x": StochasticDisturbance("MATRIX", np.asarray([[2, 3]]))},
            replicas=2,
        )
        np.testing.assert_allclose(exogenized["y"].realizations, expected)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: StochasticDisturbance("bad", (0, 1)), ValueError, "distribution"),
        (
            lambda: StochasticDisturbance("NORM", [0, 1]),  # type: ignore[arg-type]
            TypeError,
            "tuple",
        ),
        (
            lambda: StochasticDisturbance("NORM", (0, np.inf)),
            ValueError,
            "finite",
        ),
        (lambda: StochasticDisturbance("NORM", (0, -1)), ValueError, "negative"),
        (lambda: StochasticDisturbance("UNIF", (2, 1)), ValueError, "lower"),
        (
            lambda: StochasticDisturbance("MATRIX", np.asarray([1, np.nan])),
            ValueError,
            "finite 2D",
        ),
    ],
)
def test_disturbance_validation(
    factory: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"replicas": True}, TypeError, "replicas"),
        ({"replicas": 0}, ValueError, "replicas"),
        ({"seed": True}, TypeError, "seed"),
        ({"workers": True}, TypeError, "workers"),
        ({"workers": 0}, ValueError, "workers"),
        ({"workers": 2}, ValueError, "FULLNEWTON"),
        (
            {"disturbances": {"missing": StochasticDisturbance("NORM", (0, 1))}},
            KeyError,
            "unknown",
        ),
        ({"disturbances": {"x": object()}}, TypeError, "StochasticDisturbance"),
        (
            {
                "replicas": 2,
                "disturbances": {"x": StochasticDisturbance("MATRIX", np.ones((2, 3)))},
            },
            ValueError,
            "shape",
        ),
    ],
)
def test_stochastic_simulation_input_validation(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        stochastic_simulate(
            identity_model(),
            {"y": timeseries([0, 0]), "x": timeseries([1, 1])},
            coefficients={},
            time_range=(2000, 1, 2001, 1),
            **kwargs,  # type: ignore[arg-type]
        )


def test_single_replica_has_undefined_sample_sd_and_failures_are_contextual() -> None:
    one = stochastic_simulate(
        identity_model(),
        {"y": timeseries([0]), "x": timeseries([1])},
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        replicas=1,
    )
    assert np.isnan(one["y"].standard_deviation.values[0])

    divergent = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=y*y\nEND")
    with pytest.raises(StochasticSimulationError, match="replica 1"):
        stochastic_simulate(
            divergent,
            {"y": timeseries([0])},
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            disturbances={"y": StochasticDisturbance("MATRIX", np.asarray([[1.0]]))},
            replicas=1,
            max_iterations=5,
        )
