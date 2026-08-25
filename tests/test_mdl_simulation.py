from __future__ import annotations

import pickle
from collections.abc import Sequence

import numpy as np
import pytest

from _paper_models import PAPER_DOI
from bimets import (
    BimetsModel,
    MdlTimeRange,
    SimulationBlock,
    SimulationConvergenceError,
    SimulationResult,
    YearPeriod,
    estimate,
    simulate,
    timeseries,
)
from bimets.mdl._simulation import (
    _binary_scalar,
    _reorder_simultaneous_component,
    _simulation_blocks,
)
from bimets.mdl._sparse import (
    FloatArray,
    ResidualFunction,
    SparseFactorizationError,
    SparseJacobian,
    SparseNewtonIterationLimit,
    factorize_finite_difference_jacobian,
    solve_sparse_newton,
)
from test_mdl_estimation import ADVANCED_KLEIN, klein_data

KLEIN_MODEL = """MODEL
BEHAVIORAL> cn
TSRANGE 1921 1 1941 1
EQ> cn = a1 + a2*p + a3*TSLAG(p,1) + a4*(w1+w2)
COEFF> a1 a2 a3 a4
BEHAVIORAL> i
TSRANGE 1921 1 1941 1
EQ> i = b1 + b2*p + b3*TSLAG(p,1) + b4*TSLAG(k,1)
COEFF> b1 b2 b3 b4
BEHAVIORAL> w1
TSRANGE 1921 1 1941 1
EQ> w1 = c1 + c2*(y+t-w2) + c3*TSLAG(y+t-w2,1) + c4*time
COEFF> c1 c2 c3 c4
IDENTITY> y
EQ> y = cn + i + g - t
IDENTITY> p
EQ> p = y - (w1+w2)
IDENTITY> k
EQ> k = TSLAG(k,1) + i
END"""

KLEIN_LEAD_MODEL = """MODEL
BEHAVIORAL> cn
TSRANGE 1921 1 1941 1
EQ> cn = a1 + a2*p + a3*TSLAG(p,1) + a4*(w1+w2)
COEFF> a1 a2 a3 a4
IDENTITY> i
EQ> i = (MOVAVG(i,2)+TSLEAD(i))/2
BEHAVIORAL> w1
TSRANGE 1921 1 1941 1
EQ> w1 = c1 + c2*(y+t-w2) + c3*TSLAG(y+t-w2,1) + c4*time
COEFF> c1 c2 c3 c4
IDENTITY> y
EQ> y = cn + i + g - t
IDENTITY> p
EQ> p = y - (w1+w2)
IDENTITY> k
EQ> k = TSLAG(k,1) + i
END"""


@pytest.mark.source("native")
def test_compiled_scalar_evaluation_covers_mdl_function_families() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\n"
        "EQ> y=+x+ABS(-x)+EXP(0)+LOG(EXP(1))+MOVAVG(x,2)+MOVSUM(x,2)"
        "+TSDELTA(x)+TSDELTALOG(x)+TSDELTAP(x)+pi^0\nEND"
    )
    result = model.simulate(
        {"y": timeseries([0, 0, 0]), "x": timeseries([1, 2, 4])},
        coefficients={},
        time_range=(2001, 1, 2002, 1),
    )

    np.testing.assert_allclose(
        result["y"].values,
        [112.5 + np.log(2), 122 + np.log(2)],
    )


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
def test_compiled_scalar_binary_operators(operator: str, expected: object) -> None:
    assert _binary_scalar(1.0, operator, 2.0) == expected


@pytest.mark.source("native")
def test_model_and_bound_data_support_process_serialization() -> None:
    model = BimetsModel.from_text(
        "MODEL\nBEHAVIORAL> y\nEQ> y=a*x\nCOEFF> a\nIDENTITY> z\nEQ> z=y\nEND",
        name="pickle",
    )
    bound = model.bind(
        {
            "y": timeseries([0], title="output"),
            "x": timeseries([1]),
            "z": timeseries([0]),
        }
    )

    restored = pickle.loads(pickle.dumps(bound))

    assert restored.model.raw_text == model.raw_text
    assert restored.model.name == "pickle"
    assert restored.model.behavioral("y").name == "y"
    assert restored.model.identity("z").name == "z"
    assert restored.data["y"].metadata == {"title": "output"}
    np.testing.assert_array_equal(restored.data["x"].values, [1])


@pytest.mark.source("native")
def test_sparse_newton_factorization_supports_multiple_replica_columns() -> None:
    size = 40
    residual_calls = 0

    def residual(values: np.ndarray) -> np.ndarray:
        nonlocal residual_calls
        residual_calls += 1
        output = 2.0 * values
        output[1:] -= values[:-1]
        return output

    current = np.linspace(1.0, 2.0, size)
    jacobian = factorize_finite_difference_jacobian(
        residual,
        current,
        residual(current),
        relative_step=1e-5,
        column_rows=tuple(
            (column, column + 1) if column + 1 < size else (column,)
            for column in range(size)
        ),
    )
    right_hand_sides = np.column_stack((np.ones(size), np.arange(1.0, size + 1.0)))
    solutions = jacobian.solve(right_hand_sides)

    assert jacobian.matrix.format == "csc"
    assert jacobian.matrix.nnz == 2 * size - 1
    assert jacobian.density < 0.05
    assert residual_calls == 3  # baseline plus two structurally colored groups
    np.testing.assert_allclose(
        jacobian.matrix @ solutions,
        right_hand_sides,
        atol=1e-10,
    )
    blocked = jacobian.solve_blocked(right_hand_sides, workspace_bytes=size * 16)
    np.testing.assert_allclose(blocked, solutions, atol=1e-12)
    np.testing.assert_allclose(
        jacobian.solve_blocked(right_hand_sides[:, 0]), solutions[:, 0], atol=1e-12
    )
    with pytest.raises(ValueError, match="align"):
        jacobian.solve_blocked(np.ones((size - 1, 2)))
    with pytest.raises(ValueError, match="positive"):
        jacobian.solve_blocked(right_hand_sides, workspace_bytes=0)


@pytest.mark.source("native")
def test_sparse_newton_backend_rejects_invalid_or_singular_systems() -> None:
    def identity(values: np.ndarray) -> np.ndarray:
        return values

    with pytest.raises(SparseFactorizationError, match="aligned"):
        factorize_finite_difference_jacobian(
            identity,
            np.asarray([1.0]),
            np.asarray([1.0, 2.0]),
            relative_step=1e-4,
        )
    with pytest.raises(SparseFactorizationError, match="finite"):
        factorize_finite_difference_jacobian(
            identity,
            np.asarray([np.nan]),
            np.asarray([0.0]),
            relative_step=1e-4,
        )
    with pytest.raises(SparseFactorizationError, match="one entry per column"):
        factorize_finite_difference_jacobian(
            identity,
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 2.0]),
            relative_step=1e-4,
            column_rows=((0,),),
        )
    with pytest.raises(SparseFactorizationError, match="valid indexes"):
        factorize_finite_difference_jacobian(
            identity,
            np.asarray([1.0]),
            np.asarray([1.0]),
            relative_step=1e-4,
            column_rows=((-1,),),
        )
    with pytest.raises(SparseFactorizationError, match="singular"):
        factorize_finite_difference_jacobian(
            lambda values: np.zeros_like(values),
            np.asarray([1.0, 2.0]),
            np.asarray([0.0, 0.0]),
            relative_step=1e-4,
        )
    with pytest.raises(SparseNewtonIterationLimit):
        solve_sparse_newton(
            lambda values: values**2 + 1.0,
            np.asarray([1.0]),
            relative_step=1e-4,
            convergence=1e-12,
            max_iterations=1,
        )


@pytest.mark.source("bimets-R")
def test_dynamic_klein_simulation_matches_original_bimets_example() -> None:
    model = BimetsModel.from_text(KLEIN_MODEL, name="klein")
    data = klein_data()
    coefficients = estimate(model, data)

    result = simulate(
        model,
        data,
        coefficients=coefficients,
        time_range=(1923, 1, 1941, 1),
        convergence=0.00001,
    )

    assert isinstance(result, SimulationResult)
    assert result.simulation_type == "DYNAMIC"
    assert result.algorithm == "GAUSS-SEIDEL"
    assert tuple(result) == model.endogenous
    np.testing.assert_allclose(
        result["cn"].values[[0, 1, 2, -2, -1]],
        [50.338, 55.6994, 56.7111, 66.7799, 75.451],
        atol=5e-4,
    )
    np.testing.assert_allclose(
        result["y"].values[[0, 1, 2, -2, -1]],
        [56.0305, 65.8526, 64.265, 76.8049, 93.4459],
        atol=5e-4,
    )
    assert any(block.simultaneous for block in result.blocks)
    assert result.summary().shape == (19, 6)


@pytest.mark.source("bimets-R")
def test_residual_check_reproduces_original_consumption_residual_sign() -> None:
    model = BimetsModel.from_text(KLEIN_MODEL)
    data = klein_data()
    coefficients = estimate(model, data)

    result = model.bind(data).simulate(
        coefficients=coefficients,
        time_range=(1923, 1, 1941, 1),
        simulation_type="RESCHECK",
    )

    np.testing.assert_allclose(
        result["cn"].values - data["cn"].project((1923, 1), (1941, 1)).values,
        [
            1.565741401,
            0.493503129,
            -0.007607907,
            -0.869096295,
            -1.338476868,
            -1.054978943,
            0.588557053,
            -0.282311734,
            0.229653489,
            0.322131892,
            -0.322281007,
            0.058010257,
            0.034662717,
            -1.616497310,
            0.435973632,
            -0.210054350,
            -0.989201310,
            -0.785077489,
            2.173448309,
        ],
        atol=5e-9,
    )
    assert result.constant_adjustments is not None
    np.testing.assert_allclose(
        result.constant_adjustments["cn"].values,
        -(result["cn"].values - data["cn"].project((1923, 1), (1941, 1)).values),
    )


def test_static_dynamic_lhs_conditions_adjustments_and_exogenization() -> None:
    model = BimetsModel.from_text(
        """MODEL
BEHAVIORAL> growth
EQ> TSDELTAP(growth)=a+b*x
COEFF> a b
IDENTITY> switch
EQ> switch=growth
IF> growth>115
IDENTITY> switch
EQ> switch=0
IF> growth<=115
END"""
    )
    data = {
        "growth": timeseries([100, 100, 100, 100]),
        "switch": timeseries([0, 0, 0, 0]),
        "x": timeseries([1, 1, 1, 1]),
    }
    coefficients = {"growth": {"a": 0, "b": 10}}

    dynamic = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(2001, 1, 2003, 1),
        constant_adjustments={"growth": 1.0},
    )
    static = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(2001, 1, 2003, 1),
        simulation_type="STATIC",
    )

    np.testing.assert_allclose(dynamic["growth"].values, [111, 123.21, 136.7631])
    np.testing.assert_allclose(dynamic["switch"].values, [0, 123.21, 136.7631])
    np.testing.assert_allclose(static["growth"].values, [110, 110, 110])
    fixed = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(2001, 1, 2003, 1),
        exogenize=("growth",),
    )
    np.testing.assert_allclose(fixed["growth"].values, 100)

    replacement = timeseries([0, 120, 130, 140])
    overridden = model.simulate(
        data,
        coefficients=coefficients,
        time_range=MdlTimeRange(2001, 1, 2003, 1),
        exogenize={"growth": replacement},
    )
    np.testing.assert_allclose(overridden["growth"].values, [120, 130, 140])
    assert overridden.data["switch"].values.tolist() == [120, 130, 140]
    assert len(overridden) == 2
    assert "SimulationResult" in repr(overridden)


@pytest.mark.source("bimets-R")
def test_complementary_conditions_on_different_endogenous_variables_retain_paths() -> (
    None
):
    """Reproduce the conditional Y/Z model described by BIMETS R's author."""
    model = BimetsModel.from_text(
        """MODEL
IDENTITY> Y
EQ> Y=Z
IF> cond>0
IDENTITY> Z
EQ> Z=Y+1
IF> cond<=0
END"""
    )
    data = {
        "Y": timeseries([5, 10, 20, 30]),
        "Z": timeseries([0, 0, 0, 0]),
        "cond": timeseries([0, 0, 0, 0]),
    }

    result = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2003, 1),
    )

    assert model.conditional_endogenous == ("Y", "Z")
    assert result.blocks[0].variables == ("Y", "Z")
    assert result.blocks[0].simultaneous
    np.testing.assert_allclose(result["Y"].values, [10, 20, 30])
    np.testing.assert_allclose(result["Z"].values, [11, 21, 31])
    np.testing.assert_allclose(data["Y"].values, [5, 10, 20, 30])

    residual_check = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2003, 1),
        simulation_type="RESCHECK",
        rescheck_equations=("Y",),
    )
    np.testing.assert_allclose(residual_check["Y"].values, [10, 20, 30])
    assert residual_check.constant_adjustments is not None
    np.testing.assert_allclose(
        residual_check.constant_adjustments["Y"].values, [0, 0, 0]
    )

    with pytest.raises(SimulationConvergenceError, match="singular Newton Jacobian"):
        model.simulate(
            data,
            coefficients={},
            time_range=(2001, 1, 2003, 1),
            algorithm="NEWTON",
        )

    dropped = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2003, 1),
        algorithm="NEWTON",
        jacobian_drop="Y",
    )
    np.testing.assert_allclose(dropped["Y"].values, [10, 20, 30])
    np.testing.assert_allclose(dropped["Z"].values, [11, 21, 31])


@pytest.mark.source("bimets-R")
def test_conditional_endogenous_requires_current_period_historical_data() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nIF> switch>0\nEND")
    data = {
        "y": timeseries([1, np.nan]),
        "x": timeseries([2, 2]),
        "switch": timeseries([0, 0]),
    }

    with pytest.raises(ValueError, match="requires a historical value"):
        model.simulate(
            data,
            coefficients={},
            time_range=(2001, 1, 2001, 1),
            simulation_type="FORECAST",
        )

    with pytest.raises(ValueError, match="has no historical value"):
        model.simulate(
            data,
            coefficients={},
            time_range=(2001, 1, 2001, 1),
            simulation_type="RESCHECK",
        )


def test_forecast_initialization_and_time_varying_adjustment() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=TSLAG(y)+x\nEND",
        name="forecast",
    )
    data = {
        "y": timeseries([1, np.nan, np.nan]),
        "x": timeseries([0, 2, 3]),
    }

    result = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        simulation_type="forecast",
        constant_adjustments={"y": timeseries([0, 0.5, 1])},
    )

    np.testing.assert_allclose(result["y"].values, [3.5, 7.5])
    assert result.simulation_type == "FORECAST"


@pytest.mark.source("bimets-R")
def test_forecast_prefers_available_current_values_as_iteration_seeds() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=LOG(z)+2\nIDENTITY> z\nEQ> z=y\nEND"
    )
    data = {
        "y": timeseries([-1.0, 2.0]),
        "z": timeseries([-1.0, 2.0]),
    }

    result = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2001, 1),
        simulation_type="FORECAST",
        convergence=1e-10,
    )

    assert result["y"].values[0] == pytest.approx(3.14619322062058)
    assert result["z"].values[0] == pytest.approx(3.14619322062058)


def test_all_supported_lhs_transformations_in_simulation_and_rescheck() -> None:
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
    expected_dlog = 10 * np.exp(0.1)
    data = {
        "log_y": timeseries([1, 2]),
        "exp_y": timeseries([1, 2]),
        "delta_y": timeseries([10, 11]),
        "dlog_y": timeseries([10, expected_dlog]),
        "pct_y": timeseries([10, 11]),
        "log_x": timeseries([np.log(2), np.log(2)]),
        "exp_x": timeseries([np.exp(2), np.exp(2)]),
        "delta_x": timeseries([1, 1]),
        "dlog_x": timeseries([0.1, 0.1]),
        "pct_x": timeseries([10, 10]),
    }

    result = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2001, 1),
    )
    assert result["log_y"].values[0] == pytest.approx(2)
    assert result["exp_y"].values[0] == pytest.approx(2)
    assert result["delta_y"].values[0] == pytest.approx(11)
    assert result["dlog_y"].values[0] == pytest.approx(expected_dlog)
    assert result["pct_y"].values[0] == pytest.approx(11)

    check = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2001, 1),
        simulation_type="RESCHECK",
    )
    assert check.constant_adjustments is not None
    for adjustment in check.constant_adjustments.values():
        np.testing.assert_allclose(adjustment.values, 0, atol=1e-12)


def test_simulation_validation_and_nonconvergence_errors() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=2*y+1\nEND")
    data = {"y": timeseries([1, 1])}

    with pytest.raises(SimulationConvergenceError, match="did not converge"):
        simulate(
            model,
            data,
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            max_iterations=3,
        )
    newton = simulate(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        algorithm="NEWTON",
    )
    np.testing.assert_allclose(newton["y"].values, -1)
    with pytest.raises(NotImplementedError, match="FULLNEWTON"):
        simulate(
            model,
            data,
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            algorithm="FULLNEWTON",
        )
    lead_model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=TSLEAD(y)\nEND")
    with pytest.raises(ValueError, match="DYNAMIC or RESCHECK"):
        simulate(
            lead_model,
            data,
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            simulation_type="STATIC",
        )


@pytest.mark.source("bimets-R")
def test_backfill_prepends_only_available_historical_endogenous_values() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND")
    data = {
        "y": timeseries([10, 20, 0, 0], start=(1998, 1)),
        "x": timeseries([1, 2, 30, 40], start=(1998, 1)),
    }

    result = model.simulate(
        data,
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        backfill=10,
    )
    bound_result = model.bind(data).simulate(
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        backfill=1,
    )

    assert result["y"].start == YearPeriod(1998, 1)
    np.testing.assert_allclose(result["y"].values, [10, 20, 30, 40])
    assert tuple(result.iterations) == (YearPeriod(2000, 1), YearPeriod(2001, 1))
    assert result.summary().shape == (4, 1)
    assert bound_result["y"].start == YearPeriod(1999, 1)
    np.testing.assert_allclose(bound_result["y"].values, [20, 30, 40])


@pytest.mark.source("bimets-R")
def test_jacobian_drop_uses_a_reduced_newton_system_for_feedback_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*y\nEND"
    )
    sizes: list[int] = []
    original = factorize_finite_difference_jacobian

    def recording_factorization(
        residual: ResidualFunction,
        current: FloatArray,
        current_residual: FloatArray,
        *,
        relative_step: float,
        column_rows: Sequence[Sequence[int]] | None = None,
    ) -> SparseJacobian:
        sizes.append(len(current))
        return original(
            residual,
            current,
            current_residual,
            relative_step=relative_step,
            column_rows=column_rows,
        )

    monkeypatch.setattr(
        "bimets.mdl._sparse.factorize_finite_difference_jacobian",
        recording_factorization,
    )
    result = simulate(
        model,
        {
            "y": timeseries([0.0]),
            "z": timeseries([0.0]),
            "x": timeseries([1.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        algorithm="NEWTON",
        convergence=1e-8,
        jacobian_drop="z",
    )

    np.testing.assert_allclose(result["y"].values, [1 / 0.875], rtol=1e-8)
    np.testing.assert_allclose(result["z"].values, [0.25 / 0.875], rtol=1e-8)
    assert sizes == [1]

    with pytest.warns(UserWarning, match="all feedback variables"):
        all_dropped = simulate(
            model,
            {
                "y": timeseries([0.0]),
                "z": timeseries([0.0]),
                "x": timeseries([1.0]),
            },
            coefficients={},
            time_range=(2000, 1, 2000, 1),
            algorithm="NEWTON",
            convergence=1e-8,
            jacobian_drop=("y", "z"),
        )
    np.testing.assert_allclose(all_dropped["y"].values, [1 / 0.875], rtol=1e-8)
    np.testing.assert_allclose(all_dropped["z"].values, [0.25 / 0.875], rtol=1e-8)


@pytest.mark.source("native")
def test_hybrid_newton_excludes_recursive_equations_from_reduced_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep recursive identities outside the reduced Newton unknown vector."""
    model = BimetsModel.from_text(
        """MODEL
IDENTITY> y1
EQ> y1=0.1*p1+0.1*aggregate+x1
IDENTITY> p1
EQ> p1=0.15*y1
IDENTITY> cap1
EQ> cap1=y1
IDENTITY> y2
EQ> y2=0.1*p2+0.1*aggregate+x2
IDENTITY> p2
EQ> p2=0.15*y2
IDENTITY> cap2
EQ> cap2=y2
IDENTITY> aggregate
EQ> aggregate=(cap1+cap2)/2
END"""
    )
    sizes: list[int] = []
    original = factorize_finite_difference_jacobian

    def recording_factorization(
        residual: ResidualFunction,
        current: FloatArray,
        current_residual: FloatArray,
        *,
        relative_step: float,
        column_rows: Sequence[Sequence[int]] | None = None,
    ) -> SparseJacobian:
        sizes.append(len(current))
        return original(
            residual,
            current,
            current_residual,
            relative_step=relative_step,
            column_rows=column_rows,
        )

    monkeypatch.setattr(
        "bimets.mdl._sparse.factorize_finite_difference_jacobian",
        recording_factorization,
    )
    data = {
        name: timeseries([value])
        for name, value in {
            "y1": 0.0,
            "p1": 0.0,
            "cap1": 0.0,
            "y2": 0.0,
            "p2": 0.0,
            "cap2": 0.0,
            "aggregate": 0.0,
            "x1": 1.0,
            "x2": 2.0,
        }.items()
    }

    result = simulate(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        algorithm="NEWTON",
        convergence=1e-8,
        jacobian_drop="y1",
    )

    assert model.endogenous == ("y1", "p1", "cap1", "y2", "p2", "cap2", "aggregate")
    assert result.blocks[0].feedback == ("y1", "y2")
    assert sizes == [1]
    np.testing.assert_allclose(
        [result["y1"].values[0], result["y2"].values[0]],
        [1.18730104, 2.20252947],
        rtol=1e-8,
    )


@pytest.mark.source("bimets-R")
def test_jacobian_drop_supports_forward_systems_and_warns_for_nonfeedback() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*z+x\nIDENTITY> z\nEQ> z=0.25*TSLEAD(y)\nEND"
    )
    data = {
        "y": timeseries([0.0, 0.0, 4.0]),
        "z": timeseries([0.0, 0.0, 0.0]),
        "x": timeseries([1.0, 1.0, 1.0]),
    }

    result = simulate(
        model,
        data,
        coefficients={},
        time_range=(2000, 1, 2001, 1),
        algorithm="NEWTON",
        convergence=1e-8,
        jacobian_drop="z",
    )

    np.testing.assert_allclose(result["y"].values, [1.1875, 1.5])
    np.testing.assert_allclose(result["z"].values, [0.375, 1.0])
    with pytest.warns(UserWarning, match="all feedback variables"):
        all_dropped = simulate(
            model,
            data,
            coefficients={},
            time_range=(2000, 1, 2001, 1),
            algorithm="NEWTON",
            convergence=1e-8,
            jacobian_drop=("y", "z"),
        )
    np.testing.assert_allclose(all_dropped["y"].values, [1.1875, 1.5])
    np.testing.assert_allclose(all_dropped["z"].values, [0.375, 1.0])

    with pytest.warns(UserWarning, match="not a model feedback variable"):
        simulate(
            model,
            data,
            coefficients={},
            time_range=(2000, 1, 2001, 1),
            algorithm="NEWTON",
            jacobian_drop="x",
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"simulation_type": "unknown"}, ValueError, "simulation_type"),
        ({"convergence": 0}, ValueError, "convergence"),
        ({"convergence": np.nan}, ValueError, "convergence"),
        ({"max_iterations": True}, TypeError, "max_iterations"),
        ({"max_iterations": 0}, ValueError, "max_iterations"),
        ({"jacobian_step": 0}, ValueError, "jacobian_step"),
        ({"backfill": True}, TypeError, "backfill"),
        ({"backfill": -1}, ValueError, "backfill"),
        ({"jacobian_drop": ("",)}, TypeError, "jacobian_drop"),
        (
            {"zero_error_autocorrelation": 1},
            TypeError,
            "zero_error_autocorrelation",
        ),
        ({"time_range": [2000, 1, 2000, 1]}, TypeError, "time_range"),
        ({"time_range": (2000, True, 2000, 1)}, TypeError, "components"),
        ({"time_range": (2000, 0, 2000, 1)}, ValueError, "positive"),
        ({"time_range": (2000, 2, 2000, 2)}, ValueError, "frequency"),
        ({"time_range": (2001, 1, 2000, 1)}, ValueError, "reversed"),
        ({"constant_adjustments": {"missing": 1}}, KeyError, "unknown"),
        ({"constant_adjustments": {"y": True}}, TypeError, "adjustments"),
        ({"exogenize": ("missing",)}, KeyError, "unknown"),
        ({"exogenize": {"y": 1}}, TypeError, "BimetsSeries"),
    ],
)
def test_simulation_public_input_validation(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=x\nEND")
    arguments: dict[str, object] = {
        "coefficients": {},
        "time_range": (2000, 1, 2000, 1),
    }
    arguments.update(kwargs)

    with pytest.raises(error, match=message):
        simulate(
            model,
            {"y": timeseries([1, 1]), "x": timeseries([2, 2])},
            **arguments,  # type: ignore[arg-type]
        )


def test_simulation_requires_coherent_model_data_and_coefficients() -> None:
    model = BimetsModel.from_text("MODEL\nBEHAVIORAL> y\nEQ> y=a*x\nCOEFF> a\nEND")
    data = {"y": timeseries([1]), "x": timeseries([2])}

    with pytest.raises(TypeError, match="data are required"):
        simulate(model, coefficients={}, time_range=(2000, 1, 2000, 1))
    with pytest.raises(TypeError, match="must be omitted"):
        simulate(
            model.bind(data),
            data,
            coefficients={"y": {"a": 1}},
            time_range=(2000, 1, 2000, 1),
        )
    with pytest.raises(KeyError, match="missing coefficients"):
        model.simulate(data, coefficients={}, time_range=(2000, 1, 2000, 1))
    with pytest.raises(KeyError, match="missing coefficient 'a'"):
        model.simulate(
            data,
            coefficients={"y": {}},
            time_range=(2000, 1, 2000, 1),
        )


def test_partial_and_total_exogenization_use_historical_or_override_values() -> None:
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=TSLAG(y)+1\nEND")
    data = {"y": timeseries([10, 20, 30, 40])}

    partial = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2003, 1),
        exogenize={"y": (2002, 1, 2002, 1)},
    )
    replacement = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2003, 1),
        exogenize={"y": timeseries([99], start=(2002, 1))},
    )
    total = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2003, 1),
        exogenize="y",
    )

    np.testing.assert_allclose(partial["y"].values, [11, 30, 31])
    np.testing.assert_allclose(replacement["y"].values, [11, 99, 100])
    np.testing.assert_allclose(total["y"].values, [20, 30, 40])


def test_autoregressive_errors_and_selected_rescheck_equations() -> None:
    model = BimetsModel.from_text(
        """MODEL
BEHAVIORAL> y
EQ> y=a*x
COEFF> a
ERROR> AUTO(1)
IDENTITY> z
EQ> z=y+x
END"""
    )
    data = {
        "y": timeseries([1, 3, 10]),
        "z": timeseries([2, 5, 14]),
        "x": timeseries([1, 2, 4]),
    }
    coefficients = {"y": {"a": 1, "RHO_1": 0.5}}

    dynamic = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(2001, 1, 2002, 1),
    )
    check = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(2001, 1, 2002, 1),
        simulation_type="RESCHECK",
        rescheck_equations="y",
    )
    uncorrected = model.simulate(
        data,
        coefficients={"y": {"a": 1}},
        time_range=(2001, 1, 2002, 1),
        simulation_type="RESCHECK",
        rescheck_equations=("y",),
        zero_error_autocorrelation=True,
    )
    exogenized = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(2001, 1, 2002, 1),
        simulation_type="RESCHECK",
        rescheck_equations="y",
        exogenize={"y": (2001, 1, 2001, 1)},
    )

    np.testing.assert_allclose(dynamic["y"].values, [2, 4])
    np.testing.assert_allclose(check["y"].values, [2, 4.5])
    assert tuple(check) == ("y",)
    assert check.constant_adjustments is not None
    np.testing.assert_allclose(check.constant_adjustments["y"].values, [1, 5.5])
    np.testing.assert_allclose(uncorrected["y"].values, [2, 4])
    np.testing.assert_allclose(exogenized["y"].values, [3, 4.5])
    assert exogenized.constant_adjustments is not None
    np.testing.assert_allclose(exogenized.constant_adjustments["y"].values, [0, 5.5])

    with pytest.raises(ValueError, match="only valid for RESCHECK"):
        model.simulate(
            data,
            coefficients=coefficients,
            time_range=(2001, 1, 2002, 1),
            rescheck_equations="y",
        )
    with pytest.raises(KeyError, match="unknown RESCHECK"):
        model.simulate(
            data,
            coefficients=coefficients,
            time_range=(2001, 1, 2002, 1),
            simulation_type="RESCHECK",
            rescheck_equations="missing",
        )


@pytest.mark.source("bimets-R")
def test_rescheck_inverts_log_lhs_when_historical_level_is_zero() -> None:
    model = BimetsModel.from_text(
        """MODEL
BEHAVIORAL> y
EQ> LOG(y)=c01*x
COEFF> c01
STORE> coeffs(1)
END"""
    )
    data = {
        "y": timeseries([0], start=(2000, 1)),
        "x": timeseries([2], start=(2000, 1)),
    }

    result = model.simulate(
        data,
        coefficients={"y": {"c01": 3}},
        time_range=(2000, 1, 2000, 1),
        simulation_type="RESCHECK",
    )

    np.testing.assert_allclose(result["y"].values, [403.4288], rtol=1e-6)
    assert result.constant_adjustments is not None
    assert result.constant_adjustments["y"].values[0] == -np.inf


def test_estimation_result_supplies_autoregressive_simulation_coefficients() -> None:
    model = BimetsModel.from_text(ADVANCED_KLEIN, name="advanced-klein")
    data = klein_data()
    estimated = model.estimate(data, equations="cn")
    equation = estimated["cn"]
    calibrated = dict(equation.coefficients) | dict(
        equation.autoregressive_coefficients
    )

    from_estimation = model.simulate(
        data,
        coefficients=estimated,
        time_range=(1925, 1, 1935, 1),
        simulation_type="RESCHECK",
        rescheck_equations="cn",
    )
    from_mapping = model.simulate(
        data,
        coefficients={"cn": calibrated},
        time_range=(1925, 1, 1935, 1),
        simulation_type="RESCHECK",
        rescheck_equations="cn",
    )

    np.testing.assert_allclose(
        from_estimation["cn"].values,
        from_mapping["cn"].values,
    )


@pytest.mark.parametrize("algorithm", ["GAUSS-SEIDEL", "NEWTON"])
def test_forward_looking_system_is_solved_over_all_periods(algorithm: str) -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+0.2*TSLAG(y)+x\nEND",
        name="rational-expectations",
    )
    data = {
        "y": timeseries([1, 10, 20, 4]),
        "x": timeseries([0, 2, 3, 0]),
    }

    result = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        algorithm=algorithm,
        convergence=1e-10,
    )

    np.testing.assert_allclose(
        result["y"].values,
        [5.222222222222222, 6.044444444444444],
        atol=1e-10,
    )
    assert result.blocks[0].variables == ("y",)
    assert result.blocks[0].simultaneous
    assert len(set(result.iterations.values())) == 1


def test_forward_rescheck_and_partial_exogenization() -> None:
    model = BimetsModel.from_text(
        "MODEL\nIDENTITY> y\nEQ> y=0.5*TSLEAD(y)+0.2*TSLAG(y)+x\nEND"
    )
    data = {
        "y": timeseries([1, 10, 20, 4]),
        "x": timeseries([0, 2, 3, 0]),
    }

    check = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        simulation_type="RESCHECK",
        rescheck_equations="y",
    )
    fixed = model.simulate(
        data,
        coefficients={},
        time_range=(2001, 1, 2002, 1),
        exogenize={"y": (2001, 1, 2001, 1)},
        convergence=1e-10,
    )

    np.testing.assert_allclose(check["y"].values, [12.2, 7])
    assert check.constant_adjustments is not None
    np.testing.assert_allclose(check.constant_adjustments["y"].values, [-2.2, 13])
    np.testing.assert_allclose(fixed["y"].values, [10, 7])


def test_forward_looking_simulation_requires_finite_initial_and_terminal_values() -> (
    None
):
    model = BimetsModel.from_text("MODEL\nIDENTITY> y\nEQ> y=TSLEAD(y)+x\nEND")
    with pytest.raises(ValueError, match="not defined at 2001-1"):
        model.simulate(
            {
                "y": timeseries([1, np.nan, 3]),
                "x": timeseries([0, 1, 1]),
            },
            coefficients={},
            time_range=(2001, 1, 2001, 1),
        )
    with pytest.raises(ValueError, match="non-finite"):
        model.simulate(
            {
                "y": timeseries([1, 2, np.nan]),
                "x": timeseries([0, 1, 1]),
            },
            coefficients={},
            time_range=(2001, 1, 2001, 1),
        )


@pytest.mark.source("bimets-R")
def test_forward_looking_klein_example_matches_original_bimets() -> None:
    model = BimetsModel.from_text(KLEIN_LEAD_MODEL, name="klein-lead")
    source = klein_data()
    data = dict(source)
    investment = source["i"].values.copy()
    investment[11] = 2.0  # terminal expectation for 1931
    data["i"] = timeseries(investment, start=(1920, 1))
    coefficients = model.estimate(data)

    result = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(1924, 1, 1930, 1),
    )

    np.testing.assert_allclose(
        result["i"].values,
        [3.594946, 2.792062, 2.390277, 2.189125, 2.08838, 2.037915, 2.012644],
        atol=5e-6,
    )


@pytest.mark.source(PAPER_DOI)
def test_klein_forecast_1941_to_1944_from_paper() -> None:
    """Reproduce the multi-equation forecast in paper section 3.5."""
    model = BimetsModel.from_text(KLEIN_MODEL)
    data = klein_data()
    coefficients = model.estimate(data)
    extended = {}
    for name, series in data.items():
        if name in {"w2", "t", "g"}:
            mode = "constant"
        elif name == "time":
            mode = "linear"
        else:
            mode = "missing"
        extended[name] = series.extend(up_to=(1944, 1), mode=mode)

    result = model.simulate(
        extended,
        coefficients=coefficients,
        time_range=(1941, 1, 1944, 1),
        simulation_type="FORECAST",
        convergence=1e-5,
        max_iterations=100,
    )

    np.testing.assert_allclose(
        result["y"].values,
        [95.41613, 106.8923, 107.4302, 100.7512],
        atol=5e-5,
    )


@pytest.mark.source(PAPER_DOI)
def test_klein_incidence_partition_from_paper() -> None:
    """Validate the recursive/simultaneous partition in paper section 3.6."""
    model = BimetsModel.from_text(KLEIN_MODEL)
    data = klein_data()
    result = model.simulate(
        data,
        coefficients=model.estimate(data),
        time_range=(1923, 1, 1923, 1),
        convergence=1e-5,
    )

    assert len(result.blocks) == 2
    simultaneous, post_recursive = result.blocks
    assert simultaneous.simultaneous is True
    assert simultaneous.variables == ("w1", "p", "cn", "i", "y")
    assert simultaneous.feedback == ("y",)
    assert post_recursive.simultaneous is False
    assert post_recursive.variables == ("k",)
    assert post_recursive.feedback == ()


def test_gauss_seidel_reorders_a_cycle_around_its_minimal_feedback_set() -> None:
    """Evaluate recursive equations before the reduced feedback variable."""
    model = BimetsModel.from_text(
        """MODEL
IDENTITY> A
EQ> A=1+B+C
IDENTITY> B
EQ> B=0.5*A
IDENTITY> C
EQ> C=0.2*B
END"""
    )
    result = model.simulate(
        {
            "A": timeseries([10.0]),
            "B": timeseries([5.0]),
            "C": timeseries([1.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        convergence=1e-8,
    )

    assert result.blocks == (
        SimulationBlock(
            variables=("C", "A", "B"),
            simultaneous=True,
            feedback=("B",),
        ),
    )
    np.testing.assert_allclose(
        [result[name].values[0] for name in ("A", "B", "C")],
        [2.5, 1.25, 0.25],
        rtol=1e-8,
    )


def test_exogenizing_the_only_feedback_variable_evaluates_the_block_once() -> None:
    """Avoid an iteration loop when incidence reduction leaves no active feedback."""
    model = BimetsModel.from_text(
        """MODEL
IDENTITY> A
EQ> A=1+B+C
IDENTITY> B
EQ> B=0.5*A
IDENTITY> C
EQ> C=0.2*B
END"""
    )
    result = model.simulate(
        {
            "A": timeseries([10.0]),
            "B": timeseries([1.0]),
            "C": timeseries([1.0]),
        },
        coefficients={},
        time_range=(2000, 1, 2000, 1),
        exogenize="B",
    )

    assert result.iterations[YearPeriod(2000, 1)] == 1
    np.testing.assert_allclose(
        [result[name].values[0] for name in ("A", "B", "C")],
        [2.2, 1.0, 0.2],
    )


@pytest.mark.parametrize(
    ("equations", "expected_feedback"),
    [
        (
            "A=B\nIDENTITY>B\nEQ>B=A+C+D\nIDENTITY>C\nEQ>C=A+B\nIDENTITY>D\nEQ>D=C+D",
            ("B", "D"),
        ),
        (
            "A=A+B\nIDENTITY>B\nEQ>B=B+C\nIDENTITY>C\nEQ>C=C+A",
            ("A", "B", "C"),
        ),
        (
            "A=B+C\nIDENTITY>B\nEQ>B=A+C\nIDENTITY>C\nEQ>C=A+B",
            ("A", "B", "C"),
        ),
    ],
)
def test_feedback_reduction_rules(
    equations: str, expected_feedback: tuple[str, ...]
) -> None:
    """Cover degree-one, self-loop and dense-cycle incidence reductions."""
    model = BimetsModel.from_text(f"MODEL\nIDENTITY>A\nEQ>{equations}\nEND")

    (block,) = _simulation_blocks(model)

    assert block.feedback == expected_feedback
    assert set(block.variables) == set(model.endogenous)


def test_feedback_reducer_preserves_an_already_recursive_component() -> None:
    """Keep the defensive empty-row/column reduction deterministic."""
    model = BimetsModel.from_text("MODEL\nIDENTITY>A\nEQ>A=1\nIDENTITY>B\nEQ>B=A\nEND")

    ordered, feedback = _reorder_simultaneous_component(model, ("A", "B"))

    assert ordered == ("A", "B")
    assert feedback == ()
