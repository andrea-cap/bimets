"""Deterministic simulation of backward- and forward-looking MDL models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd

from bimets.mdl._binding import BoundModel, bind_model_data
from bimets.mdl._estimation import ModelEstimationResult
from bimets.mdl._expression import (
    BinaryExpression,
    MdlExpression,
    Number,
    UnaryExpression,
    Variable,
    numeric_value,
    variable_offsets,
)
from bimets.mdl._model import (
    BehavioralEquation,
    BimetsModel,
    IdentityEquation,
    MdlEquation,
    MdlTimeRange,
)
from bimets.mdl._sparse import (
    NewtonWorkspace,
    SparseFactorizationError,
    SparseNewtonIterationLimit,
    solve_sparse_newton,
)
from bimets.timeseries import BimetsDataset, BimetsSeries, Frequency, YearPeriod

type CoefficientInput = ModelEstimationResult | Mapping[str, Mapping[str, float]]
type AdjustmentValue = float | BimetsSeries
type ExogenizationValue = bool | BimetsSeries | MdlTimeRange | tuple[int, int, int, int]


class SimulationConvergenceError(ValueError):
    """Raised when a simultaneous block does not converge."""


@dataclass(frozen=True, slots=True)
class SimulationBlock:
    """One topologically ordered block of endogenous equations.

    Attributes
    ----------
    variables : tuple of str
        Endogenous names in their simulation evaluation order.
    simultaneous : bool
        Whether the block is cyclic and therefore solved iteratively.
    feedback : tuple of str
        Minimal feedback variables used for the convergence check. Recursive
        blocks have an empty tuple.

    Notes
    -----
    A forward-looking simulation reports one simultaneous block containing
    the model endogenous names because all requested periods are solved as a
    single extended system.
    """

    variables: tuple[str, ...]
    simultaneous: bool
    feedback: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ExogenizationRule:
    start: YearPeriod
    end: YearPeriod
    values: BimetsSeries | None = None

    def applies(self, period: YearPeriod, freq: Frequency) -> bool:
        """Return whether this exogenization rule applies to a period."""
        ordinal = period.ordinal(freq)
        return self.start.ordinal(freq) <= ordinal <= self.end.ordinal(freq)


class SimulationResult(Mapping[str, BimetsSeries]):
    """Immutable result of a deterministic model simulation.

    Parameters
    ----------
    model_name : str
        Source model name.
    series : mapping of str to BimetsSeries
        Simulated endogenous series.
    simulation_type, algorithm : str
        Normalized simulation settings.
    convergence, max_iterations : float, int
        Requested convergence percentage and iteration limit.
    iterations : mapping of YearPeriod to int
        Maximum block iterations used in each simulated period.
    blocks : tuple of SimulationBlock
        Topologically ordered model blocks.
    constant_adjustments : BimetsDataset or None
        Tracking adjustments produced by a residual check.
    """

    __slots__ = (
        "_iterations",
        "_series",
        "algorithm",
        "blocks",
        "constant_adjustments",
        "convergence",
        "max_iterations",
        "model_name",
        "simulation_type",
    )

    def __init__(
        self,
        model_name: str,
        series: Mapping[str, BimetsSeries],
        *,
        simulation_type: str,
        algorithm: str,
        convergence: float,
        max_iterations: int,
        iterations: Mapping[YearPeriod, int],
        blocks: tuple[SimulationBlock, ...],
        constant_adjustments: BimetsDataset | None = None,
    ) -> None:
        self.model_name = model_name
        self.simulation_type = simulation_type
        self.algorithm = algorithm
        self.convergence = convergence
        self.max_iterations = max_iterations
        self.blocks = blocks
        self.constant_adjustments = constant_adjustments
        self._series = MappingProxyType(dict(series))
        self._iterations = MappingProxyType(dict(iterations))

    def __getitem__(self, name: str) -> BimetsSeries:
        return self._series[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._series)

    def __len__(self) -> int:
        return len(self._series)

    @property
    def iterations(self) -> Mapping[YearPeriod, int]:
        """Read-only per-period iteration counts."""
        return self._iterations

    @property
    def data(self) -> BimetsDataset:
        """Simulated endogenous variables as a dataset."""
        return BimetsDataset(self._series)

    def summary(self) -> pd.DataFrame:
        """Return simulated endogenous values as a pandas DataFrame."""
        return self.data.to_frame()

    def __repr__(self) -> str:
        return (
            f"SimulationResult(model_name={self.model_name!r}, "
            f"simulation_type={self.simulation_type!r}, "
            f"variables={tuple(self)!r})"
        )


def _simulate(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    simulation_type: str = "DYNAMIC",
    algorithm: str = "GAUSS-SEIDEL",
    convergence: float = 0.01,
    max_iterations: int = 100,
    jacobian_step: float = 1e-4,
    zero_error_autocorrelation: bool = False,
    constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
    exogenize: (str | Sequence[str] | Mapping[str, ExogenizationValue] | None) = None,
    rescheck_equations: str | Sequence[str] | None = None,
    backfill: int = 0,
    jacobian_drop: str | Sequence[str] | None = None,
    newton_workspace: NewtonWorkspace | None = None,
    allow_full_newton: bool = False,
) -> SimulationResult:
    """Run a deterministic simulation of an MDL model.

    Parameters
    ----------
    model : BimetsModel or BoundModel
        Parsed model, or a model already bound to data.
    data : BimetsDataset or mapping, optional
        Required when ``model`` is not a ``BoundModel``.
    coefficients : ModelEstimationResult or nested mapping
        Behavioral coefficients keyed first by equation and then by coefficient.
    time_range : MdlTimeRange or tuple of four int
        Inclusive simulation range.
    simulation_type : {"DYNAMIC", "STATIC", "FORECAST", "RESCHECK"}
        Treatment of lagged endogenous values and initialization.
    algorithm : {"GAUSS-SEIDEL", "NEWTON"}, default="GAUSS-SEIDEL"
        Numerical algorithm used for simultaneous blocks. ``FULLNEWTON`` is
        accepted only by vectorized parent operations such as stochastic
        simulation, multiplier matrices, renormalization, and optimization.
    convergence : float, default=0.01
        Maximum percentage change accepted for every value in the active
        convergence set.
    max_iterations : int, default=100
        Maximum solver iterations per block and period.
    jacobian_step : float, default=1e-4
        Relative finite-difference step used to build Newton Jacobians.
    zero_error_autocorrelation : bool, default=False
        Ignore coefficients declared by ``ERROR> AUTO(n)`` during simulation.
    constant_adjustments : mapping, optional
        Scalar or time-series add-factors applied before inversion of the LHS.
    exogenize : str, sequence, or mapping, optional
        Endogenous names fixed to historical values over the full simulation
        range. Mapping values may be ``True``, an exogenization time range, or
        a replacement series whose own range determines when it is active.
    rescheck_equations : str or sequence of str, optional
        Endogenous equations selected for ``RESCHECK``. By default all equations
        are evaluated. Invalid for other simulation types.
    backfill : int, default=0
        Maximum number of available historical observations to prepend to each
        returned endogenous solution. The simulation range and iteration
        counts are unchanged.
    jacobian_drop : str or sequence of str, optional
        Feedback variables evaluated by Gauss-Seidel but excluded from the
        Newton Jacobian. Names that are not feedback variables are ignored with
        a warning. If exclusions or exogenization leave no active Newton
        variables, a warning is emitted and the block uses Gauss-Seidel,
        matching BIMETS R.

    Returns
    -------
    SimulationResult
        Simulated endogenous series, block information, and iteration counts.

    Raises
    ------
    SimulationConvergenceError
        If a simultaneous block does not converge within ``max_iterations``.
    NotImplementedError
        If ``FULLNEWTON`` is requested through deterministic ``simulate()``.
    ValueError
        If a forward-looking model is requested with ``STATIC`` or ``FORECAST``.

    Notes
    -----
    Newton iteration constructs a sparse numerical Jacobian of the simultaneous
    block residuals. Structurally independent columns are perturbed together,
    and sparse LU factorization is used without allocating a dense square
    Jacobian. Recursive equations remain direct substitutions under either
    algorithm. For an ``AUTO(n)`` behavioral equation, each lagged
    error is evaluated as the historical or simulated transformed LHS minus
    the corresponding equation RHS, matching BIMETS simulation semantics.
    A model containing ``TSLEAD`` is solved as one system spanning every
    endogenous variable and requested period. Observations before and after
    the range provide the initial and terminal boundary conditions.

    Examples
    --------
    >>> from bimets import BimetsModel, simulate, timeseries
    >>> model = BimetsModel.from_text(
    ...     "MODEL\\nBEHAVIORAL> y\\nEQ> y=a+b*x\\nCOEFF> a b\\nEND",
    ...     name="linear",
    ... )
    >>> data = {"y": timeseries([0, 0, 0]), "x": timeseries([1, 2, 3])}
    >>> result = simulate(
    ...     model,
    ...     data,
    ...     coefficients={"y": {"a": 1, "b": 2}},
    ...     time_range=(2000, 1, 2002, 1),
    ... )
    >>> result["y"].values.tolist()
    [3.0, 5.0, 7.0]
    """
    bound = _resolve_bound_model(model, data)
    normalized_type = simulation_type.upper()
    if normalized_type not in {"DYNAMIC", "STATIC", "FORECAST", "RESCHECK"}:
        raise ValueError(
            "simulation_type must be 'DYNAMIC', 'STATIC', 'FORECAST', or 'RESCHECK'"
        )
    if bound.model.forward_looking and normalized_type not in {
        "DYNAMIC",
        "RESCHECK",
    }:
        raise ValueError(
            "forward-looking models support only DYNAMIC or RESCHECK simulations"
        )
    requested_algorithm = algorithm.upper()
    if requested_algorithm == "FULLNEWTON" and not allow_full_newton:
        raise NotImplementedError(
            "FULLNEWTON is available only in stochastic simulation, multiplier "
            "matrices, renormalization, and optimization"
        )
    if requested_algorithm not in {"GAUSS-SEIDEL", "NEWTON", "FULLNEWTON"}:
        raise ValueError("algorithm must be 'GAUSS-SEIDEL', 'NEWTON', or 'FULLNEWTON'")
    normalized_algorithm = (
        "NEWTON" if requested_algorithm == "FULLNEWTON" else requested_algorithm
    )
    rebuild_threshold = 0.6 if requested_algorithm == "FULLNEWTON" else 0.9
    relaxation_threshold = 0.45 if requested_algorithm == "FULLNEWTON" else 0.75
    if normalized_algorithm == "NEWTON" and newton_workspace is None:
        newton_workspace = NewtonWorkspace()
    if not math.isfinite(convergence) or convergence <= 0:
        raise ValueError("convergence must be a positive finite percentage")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(jacobian_step) or jacobian_step <= 0:
        raise ValueError("jacobian_step must be a positive finite number")
    if not isinstance(zero_error_autocorrelation, bool):
        raise TypeError("zero_error_autocorrelation must be a boolean")
    if isinstance(backfill, bool) or not isinstance(backfill, int):
        raise TypeError("backfill must be an integer")
    if backfill < 0:
        raise ValueError("backfill must be non-negative")
    start, end = _simulation_bounds(time_range, bound.freq)
    coefficient_values = _coefficient_mapping(coefficients)
    adjustments = _validate_adjustments(constant_adjustments, bound.model)
    exogenization = _validate_exogenization(
        exogenize, bound.model, bound.freq, start, end
    )
    selected_rescheck = _validate_rescheck_equations(
        rescheck_equations, normalized_type, bound.model
    )
    blocks = _simulation_blocks(bound.model)
    dropped_from_jacobian = _validate_jacobian_drop(
        jacobian_drop,
        bound.model,
        blocks,
    )
    required_end = (
        end.shift(bound.model.max_lead, bound.freq)
        if bound.model.forward_looking
        else end
    )
    historical, working, storage_start = _prepare_storage(
        bound,
        required_end,
        fill_trailing_endogenous=not bound.model.forward_looking,
    )
    conditional_endogenous = frozenset(bound.model.conditional_endogenous)

    if normalized_type == "RESCHECK":
        result = _residual_check(
            bound,
            coefficient_values,
            start,
            end,
            historical,
            storage_start,
            blocks,
            adjustments,
            exogenization,
            selected_rescheck,
            zero_error_autocorrelation,
        )
        return _backfilled_result(result, bound, start, backfill)

    if bound.model.forward_looking:
        result = _simulate_forward_looking(
            bound,
            coefficient_values,
            start,
            end,
            historical,
            working,
            storage_start,
            normalized_algorithm,
            convergence,
            max_iterations,
            jacobian_step,
            adjustments,
            exogenization,
            zero_error_autocorrelation,
            newton_workspace,
            rebuild_threshold,
            relaxation_threshold,
            dropped_from_jacobian,
        )
        result = _with_reported_algorithm(result, requested_algorithm)
        return _backfilled_result(result, bound, start, backfill)

    iteration_counts: dict[YearPeriod, int] = {}
    for period in _periods(start, end, bound.freq):
        position = _position(period, storage_start, bound.freq)
        _initialize_period(
            period,
            position,
            normalized_type,
            bound.model.endogenous,
            conditional_endogenous,
            historical,
            working,
        )
        maximum_used = 1
        for block in blocks:
            fixed = {
                name
                for name in block.variables
                if name in exogenization
                and exogenization[name].applies(period, bound.freq)
            }
            active = tuple(name for name in block.variables if name not in fixed)
            for name in block.variables:
                rule = exogenization.get(name)
                if rule is not None and name in fixed and rule.values is not None:
                    working[name][position] = rule.values.at_period(
                        period.year, period.period
                    )
            if not active:
                if normalized_algorithm == "NEWTON" and block.simultaneous:
                    _warn_empty_newton_feedback(
                        f"block {block.variables!r} at {period.year}-{period.period}"
                    )
                continue
            if not block.simultaneous:
                name = active[0]
                working[name][position] = _solve_equation(
                    name,
                    period,
                    position,
                    bound,
                    coefficient_values,
                    adjustments,
                    historical,
                    working,
                    normalized_type,
                    zero_error_autocorrelation,
                )
                continue
            solver_arguments = (
                block,
                active,
                period,
                position,
                bound,
                coefficient_values,
                adjustments,
                historical,
                working,
                normalized_type,
                convergence,
                max_iterations,
                zero_error_autocorrelation,
            )
            if normalized_algorithm == "GAUSS-SEIDEL":
                feedback = tuple(name for name in block.feedback if name in active)
                used = _solve_simultaneous_block(*solver_arguments, feedback)
            else:
                feedback = tuple(name for name in block.feedback if name in active)
                dropped = tuple(
                    name for name in feedback if name in dropped_from_jacobian
                )
                if not feedback:
                    _warn_empty_newton_feedback(
                        f"block {block.variables!r} at {period.year}-{period.period}"
                    )
                    used = _solve_simultaneous_block(*solver_arguments, feedback)
                elif not dropped:
                    used = _solve_newton_block(
                        *solver_arguments,
                        jacobian_step,
                        newton_workspace,
                        rebuild_threshold,
                        relaxation_threshold,
                        feedback,
                    )
                elif len(dropped) == len(feedback):
                    _warn_empty_newton_feedback(
                        f"block {block.variables!r} at {period.year}-{period.period}"
                    )
                    used = _solve_simultaneous_block(*solver_arguments, feedback)
                else:
                    used = _solve_hybrid_newton_block(
                        *solver_arguments,
                        jacobian_step,
                        newton_workspace,
                        rebuild_threshold,
                        relaxation_threshold,
                        frozenset(dropped),
                        feedback,
                    )
            maximum_used = max(maximum_used, used)
        iteration_counts[period] = maximum_used

    output = _output_series(
        bound.model.endogenous,
        working,
        storage_start,
        start,
        end,
        bound.freq,
    )
    result = SimulationResult(
        bound.model.name,
        output,
        simulation_type=normalized_type,
        algorithm=requested_algorithm,
        convergence=convergence,
        max_iterations=max_iterations,
        iterations=iteration_counts,
        blocks=blocks,
    )
    return _backfilled_result(result, bound, start, backfill)


def simulate(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    simulation_type: str = "DYNAMIC",
    algorithm: str = "GAUSS-SEIDEL",
    convergence: float = 0.01,
    max_iterations: int = 100,
    jacobian_step: float = 1e-4,
    zero_error_autocorrelation: bool = False,
    constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
    exogenize: (str | Sequence[str] | Mapping[str, ExogenizationValue] | None) = None,
    rescheck_equations: str | Sequence[str] | None = None,
    backfill: int = 0,
    jacobian_drop: str | Sequence[str] | None = None,
) -> SimulationResult:
    return _simulate(
        model,
        data,
        coefficients=coefficients,
        time_range=time_range,
        simulation_type=simulation_type,
        algorithm=algorithm,
        convergence=convergence,
        max_iterations=max_iterations,
        jacobian_step=jacobian_step,
        zero_error_autocorrelation=zero_error_autocorrelation,
        constant_adjustments=constant_adjustments,
        exogenize=exogenize,
        rescheck_equations=rescheck_equations,
        backfill=backfill,
        jacobian_drop=jacobian_drop,
    )


simulate.__doc__ = _simulate.__doc__
_simulate.__doc__ = None


def _with_reported_algorithm(
    result: SimulationResult, algorithm: str
) -> SimulationResult:
    """Copy a result while preserving the user-facing parent algorithm name."""
    if result.algorithm == algorithm:
        return result
    return SimulationResult(
        result.model_name,
        result,
        simulation_type=result.simulation_type,
        algorithm=algorithm,
        convergence=result.convergence,
        max_iterations=result.max_iterations,
        iterations=result.iterations,
        blocks=result.blocks,
        constant_adjustments=result.constant_adjustments,
    )


def _backfilled_result(
    result: SimulationResult,
    bound: BoundModel,
    simulation_start: YearPeriod,
    backfill: int,
) -> SimulationResult:
    """Prepend up to ``backfill`` available historical observations."""
    if backfill == 0:
        return result
    previous = simulation_start.shift(-1, bound.freq)
    requested = simulation_start.shift(-backfill, bound.freq)
    series: dict[str, BimetsSeries] = {}
    for name, solution in result.items():
        source = bound.data[name]
        start = (
            source.start
            if source.start.ordinal(bound.freq) > requested.ordinal(bound.freq)
            else requested
        )
        if start.ordinal(bound.freq) > previous.ordinal(bound.freq):
            series[name] = solution
            continue
        historical = source.project(start, previous)
        series[name] = BimetsSeries(
            np.concatenate((historical.values, solution.values)),
            start=start,
            freq=bound.freq,
            metadata=solution.metadata,
        )
    return SimulationResult(
        result.model_name,
        series,
        simulation_type=result.simulation_type,
        algorithm=result.algorithm,
        convergence=result.convergence,
        max_iterations=result.max_iterations,
        iterations=result.iterations,
        blocks=result.blocks,
        constant_adjustments=result.constant_adjustments,
    )


def _resolve_bound_model(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None,
) -> BoundModel:
    """Resolve bound model for internal processing."""
    if isinstance(model, BoundModel):
        if data is not None:
            raise TypeError("data must be omitted when simulating a BoundModel")
        return model
    if data is None:
        raise TypeError("data are required when simulating a BimetsModel")
    return bind_model_data(model, data)


def _simulation_bounds(
    value: MdlTimeRange | tuple[int, int, int, int], freq: Frequency
) -> tuple[YearPeriod, YearPeriod]:
    """Resolve and validate the simulation time bounds."""
    if isinstance(value, MdlTimeRange):
        components = value.as_tuple()
    elif isinstance(value, tuple) and len(value) == 4:
        components = value
    else:
        raise TypeError("time_range must be MdlTimeRange or a tuple of four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in components):
        raise TypeError("time_range components must be integers")
    if components[1] < 1 or components[3] < 1:
        raise ValueError("time_range periods must be positive")
    if components[1] > int(freq) or components[3] > int(freq):
        raise ValueError("time_range periods exceed data frequency")
    start = YearPeriod(components[0], components[1])
    end = YearPeriod(components[2], components[3])
    if end.ordinal(freq) < start.ordinal(freq):
        raise ValueError("time_range is reversed")
    return start, end


def _coefficient_mapping(
    values: CoefficientInput,
) -> Mapping[str, Mapping[str, float]]:
    """Build the coefficient values used during simulation."""
    if isinstance(values, ModelEstimationResult):
        return MappingProxyType(
            {
                name: MappingProxyType(
                    dict(result.coefficients) | dict(result.autoregressive_coefficients)
                )
                for name, result in values.items()
            }
        )
    output: dict[str, Mapping[str, float]] = {}
    for equation, coefficients in values.items():
        output[equation] = MappingProxyType(
            {name: float(value) for name, value in coefficients.items()}
        )
    return MappingProxyType(output)


def _validate_adjustments(
    values: Mapping[str, AdjustmentValue] | None, model: BimetsModel
) -> Mapping[str, AdjustmentValue]:
    """Validate adjustments for internal processing."""
    output = dict(values or {})
    unknown = set(output).difference(model.endogenous)
    if unknown:
        raise KeyError(f"unknown adjustment variables: {sorted(unknown)}")
    for value in output.values():
        if isinstance(value, bool) or not isinstance(value, (int, float, BimetsSeries)):
            raise TypeError("constant adjustments must be numeric or BimetsSeries")
    return MappingProxyType(output)


def _validate_exogenization(
    values: (str | Sequence[str] | Mapping[str, ExogenizationValue] | None),
    model: BimetsModel,
    freq: Frequency,
    simulation_start: YearPeriod,
    simulation_end: YearPeriod,
) -> Mapping[str, _ExogenizationRule]:
    """Validate exogenization for internal processing."""
    if values is None:
        return MappingProxyType({})
    if isinstance(values, Mapping):
        definitions = dict(values)
        names = set(definitions)
    else:
        names = {values} if isinstance(values, str) else set(values)
        definitions = {name: True for name in names}
    unknown = names.difference(model.endogenous)
    if unknown:
        raise KeyError(f"unknown endogenous variables to exogenize: {sorted(unknown)}")
    rules: dict[str, _ExogenizationRule] = {}
    for name, definition in definitions.items():
        if definition is True:
            rules[name] = _ExogenizationRule(simulation_start, simulation_end)
        elif isinstance(definition, BimetsSeries):
            if definition.freq != freq:
                raise ValueError(
                    f"exogenization series {name!r} has a different frequency"
                )
            rules[name] = _ExogenizationRule(
                definition.start, definition.end, definition
            )
        elif isinstance(definition, (MdlTimeRange, tuple)):
            start, end = _simulation_bounds(definition, freq)
            rules[name] = _ExogenizationRule(start, end)
        else:
            raise TypeError(
                "exogenization values must be True, BimetsSeries, MdlTimeRange, "
                "or a tuple of four integers"
            )
    return MappingProxyType(rules)


def _validate_rescheck_equations(
    values: str | Sequence[str] | None,
    simulation_type: str,
    model: BimetsModel,
) -> tuple[str, ...]:
    """Validate rescheck equations for internal processing."""
    if values is not None and simulation_type != "RESCHECK":
        raise ValueError("rescheck_equations is only valid for RESCHECK simulations")
    if values is None:
        return model.endogenous
    requested = (values,) if isinstance(values, str) else tuple(values)
    if not requested:
        raise ValueError("rescheck_equations cannot be empty")
    unknown = set(requested).difference(model.endogenous)
    if unknown:
        raise KeyError(f"unknown RESCHECK equations: {sorted(unknown)}")
    requested_set = set(requested)
    return tuple(name for name in model.endogenous if name in requested_set)


def _prepare_storage(
    bound: BoundModel,
    simulation_end: YearPeriod,
    *,
    fill_trailing_endogenous: bool,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    YearPeriod,
]:
    """Prepare storage for internal processing."""
    storage_start = min(series.start for series in bound.data.values())
    storage_end = max(
        simulation_end,
        *(series.end for series in bound.data.values()),
        key=lambda item: item.ordinal(bound.freq),
    )
    historical: dict[str, np.ndarray] = {}
    working: dict[str, np.ndarray] = {}
    for name, series in bound.data.items():
        values = series.project(storage_start, storage_end, extend=True).values.copy()
        historical[name] = values.copy()
        if fill_trailing_endogenous and name in bound.model.endogenous:
            finite = np.flatnonzero(np.isfinite(values))
            if finite.size:
                values[int(finite[-1]) + 1 :] = values[int(finite[-1])]
        working[name] = values
    return historical, working, storage_start


def _simulation_blocks(model: BimetsModel) -> tuple[SimulationBlock, ...]:
    """Return the equation blocks used by the simulation solver."""
    components = _strongly_connected_components(model)
    component_of = {
        name: index for index, component in enumerate(components) for name in component
    }
    incoming: list[set[int]] = [set() for _ in components]
    outgoing: list[set[int]] = [set() for _ in components]
    endogenous = set(model.endogenous)
    for name, dependencies in model.dependencies.items():
        target = component_of[name]
        for dependency in dependencies.intersection(endogenous):
            source = component_of[dependency]
            if source != target:
                incoming[target].add(source)
                outgoing[source].add(target)
    order: list[int] = []
    ready = [index for index, values in enumerate(incoming) if not values]
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in sorted(outgoing[current]):
            incoming[target].discard(current)
            if not incoming[target] and target not in ready and target not in order:
                ready.append(target)
    model_order = {name: index for index, name in enumerate(model.endogenous)}
    blocks: list[SimulationBlock] = []
    for index in order:
        variables = tuple(sorted(components[index], key=model_order.__getitem__))
        simultaneous = (
            len(variables) > 1 or variables[0] in model.dependencies[variables[0]]
        )
        if simultaneous:
            variables, feedback = _reorder_simultaneous_component(model, variables)
        else:
            feedback = ()
        blocks.append(SimulationBlock(variables, simultaneous, feedback))
    return tuple(blocks)


def _reorder_simultaneous_component(
    model: BimetsModel, variables: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a cyclic component into ordered equations and feedback variables.

    This is the reduction used by BIMETS R's ``.reorderEquations``: repeatedly
    remove empty rows/columns and degree-one nodes, then break any remaining
    cycle at its most connected node. Equations outside the resulting feedback
    set are topologically ordered and evaluated before the feedback equations.
    """
    names = list(variables)
    matrix = np.asarray(
        [
            [dependency in model.dependencies[name] for dependency in names]
            for name in names
        ],
        dtype=bool,
    )
    feedback: list[str] = []
    reduced_names = names.copy()
    reduced = matrix.copy()

    def remove(indices: np.ndarray) -> None:
        """Remove matching rows, columns and names from the reduced graph."""
        nonlocal reduced, reduced_names
        keep = np.ones(len(reduced_names), dtype=bool)
        keep[indices] = False
        reduced = reduced[np.ix_(keep, keep)]
        reduced_names = [
            name for name, selected in zip(reduced_names, keep, strict=True) if selected
        ]

    while reduced_names:
        changed = True
        while changed and reduced_names:
            changed = False
            for axis in (0, 1):
                empty = np.flatnonzero(np.sum(reduced, axis=axis) == 0)
                if empty.size:
                    remove(empty)
                    changed = True
                    if not reduced_names:
                        break
            if not reduced_names:
                break

            degree_one_columns = np.flatnonzero(np.sum(reduced, axis=0) == 1)
            if degree_one_columns.size:
                for index in degree_one_columns:
                    destinations = np.flatnonzero(reduced[:, index])
                    dependencies = np.flatnonzero(reduced[index, :])
                    reduced[np.ix_(destinations, dependencies)] = True
                feedback.extend(
                    reduced_names[index]
                    for index in degree_one_columns
                    if reduced[index, index]
                )
                remove(degree_one_columns)
                changed = True
                continue

            degree_one_rows = np.flatnonzero(np.sum(reduced, axis=1) == 1)
            if degree_one_rows.size:
                for index in degree_one_rows:
                    destinations = np.flatnonzero(reduced[index, :])
                    dependants = np.flatnonzero(reduced[:, index])
                    reduced[np.ix_(dependants, destinations)] = True
                feedback.extend(
                    reduced_names[index]
                    for index in degree_one_rows
                    if reduced[index, index]
                )
                remove(degree_one_rows)
                changed = True
                continue

            self_referential = np.flatnonzero(np.diag(reduced))
            if self_referential.size:
                feedback.extend(reduced_names[index] for index in self_referential)
                remove(self_referential)
                changed = True

        if reduced_names:
            connectivity = np.sum(reduced, axis=0) * np.sum(reduced, axis=1)
            selected = int(np.argmax(connectivity))
            feedback.append(reduced_names[selected])
            remove(np.asarray([selected]))

    feedback = list(dict.fromkeys(feedback))
    non_feedback = [name for name in names if name not in feedback]
    ordered: list[str] = []
    remaining = non_feedback.copy()
    while remaining:
        ready = [
            name
            for name in remaining
            if not model.dependencies[name].intersection(remaining)
        ]
        if not ready:
            raise RuntimeError("cannot reduce simultaneous incidence matrix")
        ordered.extend(ready)
        ready_set = set(ready)
        remaining = [name for name in remaining if name not in ready_set]
    return tuple([*ordered, *feedback]), tuple(feedback)


def _validate_jacobian_drop(
    values: str | Sequence[str] | None,
    model: BimetsModel,
    blocks: tuple[SimulationBlock, ...],
) -> frozenset[str]:
    """Validate and retain names that participate in a Newton feedback set."""
    if values is None:
        return frozenset()
    names = (values,) if isinstance(values, str) else tuple(values)
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("jacobian_drop must contain non-empty variable names")
    feedback = (
        set(model.endogenous)
        if model.forward_looking
        else {name for block in blocks if block.simultaneous for name in block.feedback}
    )
    selected: set[str] = set()
    for name in names:
        if name not in feedback:
            warnings.warn(
                f"{name!r} is not a model feedback variable and will be ignored",
                UserWarning,
                stacklevel=3,
            )
        else:
            selected.add(name)
    return frozenset(selected)


def _warn_empty_newton_feedback(scope: str) -> None:
    """Warn empty newton feedback for internal processing."""
    warnings.warn(
        "NEWTON was requested but all feedback variables were exogenized or "
        f"dropped from the Jacobian in {scope}; using Gauss-Seidel",
        UserWarning,
        stacklevel=3,
    )


def _strongly_connected_components(model: BimetsModel) -> list[tuple[str, ...]]:
    """Find simultaneous equation components in the model graph."""
    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    output: list[tuple[str, ...]] = []
    endogenous = set(model.endogenous)

    def visit(name: str) -> None:
        """Visit one node using Tarjan's component algorithm."""
        nonlocal index
        indexes[name] = index
        lowlinks[name] = index
        index += 1
        stack.append(name)
        on_stack.add(name)
        for dependency in model.dependencies[name].intersection(endogenous):
            if dependency not in indexes:
                visit(dependency)
                lowlinks[name] = min(lowlinks[name], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[name] = min(lowlinks[name], indexes[dependency])
        if lowlinks[name] == indexes[name]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == name:
                    break
            output.append(tuple(component))

    for name in model.endogenous:
        if name not in indexes:
            visit(name)
    return output


def _periods(
    start: YearPeriod, end: YearPeriod, freq: Frequency
) -> Iterator[YearPeriod]:
    """Return every year-period index in an inclusive range."""
    count = end.ordinal(freq) - start.ordinal(freq) + 1
    return (start.shift(offset, freq) for offset in range(count))


def _position(period: YearPeriod, start: YearPeriod, freq: Frequency) -> int:
    """Return a period offset relative to the simulation start."""
    return period.ordinal(freq) - start.ordinal(freq)


def _initialize_period(
    period: YearPeriod,
    position: int,
    simulation_type: str,
    endogenous: Sequence[str],
    conditional_endogenous: frozenset[str],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
) -> None:
    """Initialize period for internal processing."""
    for name in endogenous:
        historical_value = historical[name][position]
        if name in conditional_endogenous:
            if not math.isfinite(float(historical_value)):
                raise ValueError(
                    f"conditional endogenous variable {name!r} requires a "
                    f"historical value at {period}"
                )
            working[name][position] = historical_value
            continue
        if simulation_type == "FORECAST":
            if math.isfinite(float(historical_value)):
                working[name][position] = historical_value
            elif position > 0 and math.isfinite(float(working[name][position - 1])):
                working[name][position] = working[name][position - 1]
            else:
                raise ValueError(
                    f"cannot initialize forecast variable {name!r} at {period}"
                )
        elif math.isfinite(float(historical_value)):
            working[name][position] = historical_value
        elif position > 0 and math.isfinite(float(working[name][position - 1])):
            working[name][position] = working[name][position - 1]
        else:
            raise ValueError(
                f"cannot initialize endogenous variable {name!r} at {period}"
            )


def _solve_simultaneous_block(
    block: SimulationBlock,
    active: tuple[str, ...],
    period: YearPeriod,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    feedback: tuple[str, ...] | None = None,
) -> int:
    """Solve simultaneous block for internal processing."""
    convergence_names = active if feedback is None else feedback
    if not convergence_names:
        for name in active:
            working[name][position] = _solve_equation(
                name,
                period,
                position,
                bound,
                coefficients,
                adjustments,
                historical,
                working,
                simulation_type,
                zero_error_autocorrelation,
            )
        return 1
    for iteration in range(1, max_iterations + 1):
        previous = np.asarray([working[name][position] for name in convergence_names])
        for name in active:
            working[name][position] = _solve_equation(
                name,
                period,
                position,
                bound,
                coefficients,
                adjustments,
                historical,
                working,
                simulation_type,
                zero_error_autocorrelation,
            )
        current = np.asarray([working[name][position] for name in convergence_names])
        denominator = np.maximum(np.abs(previous), 1.0)
        percentage_change = 100.0 * np.abs(current - previous) / denominator
        if np.all(percentage_change < convergence):
            return iteration
    raise SimulationConvergenceError(
        f"block {block.variables!r} did not converge at "
        f"{period.year}-{period.period} in {max_iterations} iterations"
    )


def _solve_newton_block(
    block: SimulationBlock,
    active: tuple[str, ...],
    period: YearPeriod,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    newton_workspace: NewtonWorkspace | None,
    rebuild_threshold: float,
    relaxation_threshold: float,
    feedback: tuple[str, ...] | None = None,
) -> int:
    """Solve newton block for internal processing."""
    solve_names = active if feedback is None else feedback

    def residual(candidate: np.ndarray) -> np.ndarray:
        """Evaluate block residuals for a candidate Newton vector."""
        for name, value in zip(solve_names, candidate, strict=True):
            working[name][position] = value
        for name in active:
            working[name][position] = _solve_equation(
                name,
                period,
                position,
                bound,
                coefficients,
                adjustments,
                historical,
                working,
                simulation_type,
                zero_error_autocorrelation,
            )
        targets = np.asarray([working[name][position] for name in solve_names])
        result: np.ndarray = targets - candidate
        return result

    current = np.asarray([working[name][position] for name in solve_names], dtype=float)
    cache_key = (
        "backward",
        block.variables,
        solve_names,
    )
    column_rows = tuple(tuple(range(len(solve_names))) for _ in solve_names)
    try:
        solution, iterations = solve_sparse_newton(
            residual,
            current,
            relative_step=jacobian_step,
            convergence=convergence,
            max_iterations=max_iterations,
            workspace=newton_workspace,
            cache_key=cache_key,
            column_rows=column_rows,
            rebuild_threshold=rebuild_threshold,
            relaxation_threshold=relaxation_threshold,
        )
    except SparseFactorizationError as error:
        raise SimulationConvergenceError(
            f"singular Newton Jacobian for block {block.variables!r} at "
            f"{period.year}-{period.period}"
        ) from error
    except SparseNewtonIterationLimit as error:
        raise SimulationConvergenceError(
            f"block {block.variables!r} did not converge at "
            f"{period.year}-{period.period} in {max_iterations} Newton iterations"
        ) from error
    residual(solution)
    return iterations


def _solve_hybrid_newton_block(
    block: SimulationBlock,
    active: tuple[str, ...],
    period: YearPeriod,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    newton_workspace: NewtonWorkspace | None,
    rebuild_threshold: float,
    relaxation_threshold: float,
    dropped: frozenset[str],
    feedback: tuple[str, ...],
) -> int:
    """Alternate dropped Gauss-Seidel equations with a reduced Newton solve."""
    newton_active = tuple(name for name in active if name not in dropped)
    newton_feedback = tuple(name for name in feedback if name not in dropped)
    for iteration in range(1, max_iterations + 1):
        previous = np.asarray([working[name][position] for name in feedback])
        for name in active:
            if name in dropped:
                working[name][position] = _solve_equation(
                    name,
                    period,
                    position,
                    bound,
                    coefficients,
                    adjustments,
                    historical,
                    working,
                    simulation_type,
                    zero_error_autocorrelation,
                )
        _solve_newton_block(
            block,
            newton_active,
            period,
            position,
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            simulation_type,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
            jacobian_step,
            newton_workspace,
            rebuild_threshold,
            relaxation_threshold,
            newton_feedback,
        )
        current = np.asarray([working[name][position] for name in feedback])
        if _percentage_converged(previous, current, convergence):
            return iteration
    raise SimulationConvergenceError(
        f"block {block.variables!r} did not converge at "
        f"{period.year}-{period.period} in {max_iterations} hybrid iterations"
    )


def _solve_equation(
    name: str,
    period: YearPeriod,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    zero_error_autocorrelation: bool,
) -> float:
    """Solve one equation directly against the model's numeric arrays."""
    definition = bound.model._equation_definition(name)
    if isinstance(definition, IdentityEquation):
        equation = _select_identity_scalar(
            definition,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        if equation is None:
            return _value_at_scalar(
                name,
                position,
                position,
                bound,
                historical,
                working,
                simulation_type,
            )
        rhs = _evaluate_scalar(
            equation.rhs,
            position,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
    else:
        equation = definition.equation
        rhs = _behavioral_rhs_scalar(
            definition,
            coefficients,
            adjustments,
            position,
            period,
            bound,
            historical,
            working,
            simulation_type,
            zero_error_autocorrelation,
        )
    rhs += _adjustment_value(adjustments.get(name, 0.0), period)
    value = _invert_lhs_scalar(
        equation,
        rhs,
        position,
        bound,
        historical,
        working,
        simulation_type,
    )
    if not math.isfinite(value):
        raise ValueError(
            f"equation {name!r} produced a non-finite value at "
            f"{period.year}-{period.period}"
        )
    return value


type _CompiledArgument = float | int | str | tuple[str, int]
type _CompiledInstruction = tuple[str, _CompiledArgument]


@cache
def _compile_expression(
    expression: MdlExpression, offset: int = 0
) -> tuple[_CompiledInstruction, ...]:
    """Compile an immutable MDL tree into cached stack instructions."""
    if isinstance(expression, Number):
        return (("CONST", expression.value),)
    if isinstance(expression, Variable):
        if expression.name.lower() == "pi":
            return (("CONST", math.pi),)
        return (("VAR", (expression.name, offset)),)
    if isinstance(expression, UnaryExpression):
        instructions = _compile_expression(expression.operand, offset)
        return (
            instructions if expression.operator == "+" else (*instructions, ("NEG", 0))
        )
    if isinstance(expression, BinaryExpression):
        return (
            _compile_expression(expression.left, offset)
            + _compile_expression(expression.right, offset)
            + (("BINARY", expression.operator),)
        )

    periods = 1
    if len(expression.arguments) == 2:
        parsed = numeric_value(expression.arguments[1])
        assert parsed is not None
        periods = int(parsed)
    argument = expression.arguments[0]
    if expression.name == "TSLAG":
        return _compile_expression(argument, offset - periods)
    if expression.name == "TSLEAD":
        return _compile_expression(argument, offset + periods)
    if expression.name in {"ABS", "EXP", "LOG"}:
        return (*_compile_expression(argument, offset), (expression.name, 0))
    if expression.name in {"MOVAVG", "MOVSUM"}:
        instructions = tuple(
            instruction
            for lag in range(periods)
            for instruction in _compile_expression(argument, offset - lag)
        )
        return (*instructions, (expression.name, periods))
    current = _compile_expression(argument, offset)
    lagged = _compile_expression(argument, offset - periods)
    if expression.name == "TSDELTA":
        return current + lagged + (("BINARY", "-"),)
    if expression.name == "TSDELTALOG":
        return (*current, ("LOG", 0), *lagged, ("LOG", 0), ("BINARY", "-"))
    if expression.name == "TSDELTAP":
        return (
            current
            + lagged
            + (
                ("BINARY", "/"),
                ("CONST", 1.0),
                ("BINARY", "-"),
                ("CONST", 100.0),
                ("BINARY", "*"),
            )
        )
    raise AssertionError(f"unexpected MDL function {expression.name!r}")


def _evaluate_scalar(
    expression: MdlExpression,
    position: int,
    current_position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> float:
    """Evaluate a cached scalar instruction stream at one array position."""
    # NumPy's scalar ufuncs provide the R-compatible Inf/NaN semantics needed
    # by MDL.  One context per expression avoids paying for it at every node.
    with np.errstate(all="ignore"):
        stack: list[float | bool] = []
        for opcode, argument in _compile_expression(expression):
            if opcode == "CONST":
                stack.append(float(cast(float, argument)))
            elif opcode == "VAR":
                name, offset = cast(tuple[str, int], argument)
                stack.append(
                    _value_at_scalar(
                        str(name),
                        position + int(offset),
                        current_position,
                        bound,
                        historical,
                        working,
                        simulation_type,
                    )
                )
            elif opcode == "NEG":
                stack[-1] = -float(stack[-1])
            elif opcode == "BINARY":
                right = stack.pop()
                left = stack.pop()
                stack.append(_binary_scalar(left, str(argument), right))
            elif opcode == "ABS":
                stack[-1] = abs(float(stack[-1]))
            elif opcode == "EXP":
                stack[-1] = float(np.exp(float(stack[-1])))
            elif opcode == "LOG":
                stack[-1] = float(np.log(float(stack[-1])))
            else:
                periods = cast(int, argument)
                values = stack[-periods:]
                del stack[-periods:]
                total = sum(float(value) for value in values)
                stack.append(total / periods if opcode == "MOVAVG" else total)
        if len(stack) != 1:
            raise AssertionError("invalid compiled MDL scalar expression")
        return float(stack[0])


def _binary_scalar(
    left: float | bool, operator: str, right: float | bool
) -> float | bool:
    """Apply one MDL operator without constructing temporary time series."""
    if operator == "&":
        return bool(left) and bool(right)
    if operator == "|":
        return bool(left) or bool(right)
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "+":
        return float(left) + float(right)
    if operator == "-":
        return float(left) - float(right)
    if operator == "*":
        return float(left) * float(right)
    if operator == "/":
        return float(np.divide(float(left), float(right)))
    return float(np.power(float(left), float(right)))


def _value_at_scalar(
    name: str,
    position: int,
    current_position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> float:
    """Return one scalar variable value with static and RESCHECK semantics."""
    if position < 0 or position >= working[name].shape[0]:
        raise IndexError(f"variable {name!r} is unavailable at relative row {position}")
    if simulation_type == "RESCHECK" or (
        simulation_type == "STATIC"
        and name in bound.model.endogenous
        and position < current_position
    ):
        return float(historical[name][position])
    return float(working[name][position])


def _select_identity_scalar(
    identity: IdentityEquation,
    position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> MdlEquation | None:
    """Select the active identity alternative at one array position."""
    for alternative in identity.alternatives:
        if alternative.condition is None:
            return alternative.equation
        condition = _evaluate_scalar(
            alternative.condition,
            position,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        if bool(condition):
            return alternative.equation
    return None


def _behavioral_rhs_scalar(
    behavioral: BehavioralEquation,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    position: int,
    period: YearPeriod,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    zero_error_autocorrelation: bool,
) -> float:
    """Evaluate a behavioral RHS directly against numeric arrays."""
    output = _behavioral_base_rhs_scalar(
        behavioral,
        coefficients,
        position,
        position,
        bound,
        historical,
        working,
        simulation_type,
    )
    if behavioral.error is None or zero_error_autocorrelation:
        return output
    values = coefficients[behavioral.name]
    for lag in range(1, behavioral.error.order + 1):
        coefficient_name = f"RHO_{lag}"
        try:
            rho = float(values[coefficient_name])
        except KeyError as error:
            raise KeyError(
                f"missing autoregressive coefficient {coefficient_name!r} for "
                f"equation {behavioral.name!r}"
            ) from error
        lagged_position = position - lag
        lagged_period = period.shift(-lag, bound.freq)
        lagged_level = _value_at_scalar(
            behavioral.name,
            lagged_position,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        lagged_lhs = _lhs_value_scalar(
            behavioral.equation,
            lagged_level,
            lagged_position,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        lagged_rhs = _behavioral_base_rhs_scalar(
            behavioral,
            coefficients,
            lagged_position,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        lagged_rhs += _adjustment_value(
            adjustments.get(behavioral.name, 0.0), lagged_period
        )
        output += rho * (lagged_lhs - lagged_rhs)
    return output


@cache
def _behavioral_execution_plan(
    behavioral: BehavioralEquation,
) -> tuple[tuple[MdlExpression, tuple[str, ...], bool], ...]:
    """Precompute regressor coefficient names and PDL requirements."""
    definitions = {item.coefficient: item for item in behavioral.pdls}
    output: list[tuple[MdlExpression, tuple[str, ...], bool]] = []
    for coefficient, regressor in zip(
        behavioral.coefficients, behavioral.regressors, strict=True
    ):
        definition = definitions.get(coefficient)
        lag_count = definition.length if definition is not None else 1
        names = tuple(
            coefficient if lag == 0 else f"{coefficient}__PDL__{lag}"
            for lag in range(lag_count)
        )
        output.append((regressor, names, bool(variable_offsets(regressor))))
    return tuple(output)


def _behavioral_base_rhs_scalar(
    behavioral: BehavioralEquation,
    coefficients: Mapping[str, Mapping[str, float]],
    position: int,
    current_position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> float:
    """Evaluate a preprocessed behavioral RHS at one array position."""
    try:
        values = coefficients[behavioral.name]
    except KeyError as error:
        raise KeyError(
            f"missing coefficients for behavioral equation {behavioral.name!r}"
        ) from error
    output = 0.0
    for regressor, coefficient_names, is_series in _behavioral_execution_plan(
        behavioral
    ):
        for lag, coefficient_name in enumerate(coefficient_names):
            try:
                coefficient_value = float(values[coefficient_name])
            except KeyError as error:
                raise KeyError(
                    f"missing coefficient {coefficient_name!r} for "
                    f"equation {behavioral.name!r}"
                ) from error
            if lag and not is_series:
                raise TypeError("PDL simulation requires a series regressor")
            regressor_value = _evaluate_scalar(
                regressor,
                position - lag,
                current_position,
                bound,
                historical,
                working,
                simulation_type,
            )
            output += coefficient_value * regressor_value
    return output


def _invert_lhs_scalar(
    equation: MdlEquation,
    rhs: float,
    position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> float:
    """Recover a dependent scalar from its transformed left-hand side."""
    function = equation.lhs_function
    if function == "IDENTITY":
        return rhs
    if function == "LOG":
        return math.exp(rhs)
    if function == "EXP":
        return math.log(rhs)
    lagged = _value_at_scalar(
        equation.dependent,
        position - equation.lhs_periods,
        position,
        bound,
        historical,
        working,
        simulation_type,
    )
    if function == "TSDELTA":
        return lagged + rhs
    if function == "TSDELTALOG":
        return math.exp(math.log(lagged) + rhs)
    if function == "TSDELTAP":
        return lagged * (1.0 + rhs / 100.0)
    raise AssertionError(f"unexpected LHS function {function!r}")


def _lhs_value_scalar(
    equation: MdlEquation,
    level: float,
    position: int,
    current_position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> float:
    """Evaluate a transformed scalar left-hand side at one position."""
    function = equation.lhs_function
    if function == "IDENTITY":
        return level
    if function == "LOG":
        return math.log(level)
    if function == "EXP":
        return math.exp(level)
    lagged = _value_at_scalar(
        equation.dependent,
        position - equation.lhs_periods,
        current_position,
        bound,
        historical,
        working,
        simulation_type,
    )
    if function == "TSDELTA":
        return level - lagged
    if function == "TSDELTALOG":
        return math.log(level) - math.log(lagged)
    if function == "TSDELTAP":
        return 100.0 * (level / lagged - 1.0)
    raise AssertionError(f"unexpected LHS function {function!r}")


def _adjustment_value(value: AdjustmentValue, period: YearPeriod) -> float:
    """Return an adjustment value for one period."""
    if isinstance(value, BimetsSeries):
        position = period.ordinal(value.freq) - value.start.ordinal(value.freq)
        if position < 0 or position >= len(value):
            return 0.0
        return float(value[position])
    return float(value)


def _simulate_forward_looking(
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    start: YearPeriod,
    end: YearPeriod,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    storage_start: YearPeriod,
    algorithm: str,
    convergence: float,
    max_iterations: int,
    jacobian_step: float,
    adjustments: Mapping[str, AdjustmentValue],
    exogenization: Mapping[str, _ExogenizationRule],
    zero_error_autocorrelation: bool,
    newton_workspace: NewtonWorkspace | None,
    rebuild_threshold: float,
    relaxation_threshold: float,
    dropped_from_jacobian: frozenset[str],
) -> SimulationResult:
    """Simulate forward looking for internal processing."""
    periods = tuple(_periods(start, end, bound.freq))
    unknowns: list[tuple[str, YearPeriod, int]] = []
    for period in periods:
        position = _position(period, storage_start, bound.freq)
        for name in bound.model.endogenous:
            rule = exogenization.get(name)
            fixed = rule is not None and rule.applies(period, bound.freq)
            if fixed and rule is not None and rule.values is not None:
                value = rule.values.at_period(period.year, period.period)
            else:
                value = historical[name][position]
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"forward-looking endogenous variable {name!r} is not "
                    f"defined at {period.year}-{period.period}"
                )
            working[name][position] = value
            if not fixed:
                unknowns.append((name, period, position))

    if not unknowns:
        if algorithm == "NEWTON":
            _warn_empty_newton_feedback("the forward-looking system")
        iterations_used = 1
    elif algorithm == "GAUSS-SEIDEL":
        iterations_used = _solve_forward_gauss_seidel(
            tuple(unknowns),
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
        )
    elif not dropped_from_jacobian:
        iterations_used = _solve_forward_newton(
            tuple(unknowns),
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
            jacobian_step,
            newton_workspace,
            rebuild_threshold,
            relaxation_threshold,
        )
    elif all(name in dropped_from_jacobian for name, _, _ in unknowns):
        _warn_empty_newton_feedback("the forward-looking system")
        iterations_used = _solve_forward_gauss_seidel(
            tuple(unknowns),
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
        )
    else:
        iterations_used = _solve_forward_hybrid_newton(
            tuple(unknowns),
            dropped_from_jacobian,
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
            jacobian_step,
            newton_workspace,
            rebuild_threshold,
            relaxation_threshold,
        )

    output = _output_series(
        bound.model.endogenous,
        working,
        storage_start,
        start,
        end,
        bound.freq,
    )
    active_names = tuple(
        name
        for name in bound.model.endogenous
        if any(unknown_name == name for unknown_name, _, _ in unknowns)
    )
    result_blocks = (
        (SimulationBlock(active_names, simultaneous=True),) if active_names else ()
    )
    return SimulationResult(
        bound.model.name,
        output,
        simulation_type="DYNAMIC",
        algorithm=algorithm,
        convergence=convergence,
        max_iterations=max_iterations,
        iterations={period: iterations_used for period in periods},
        blocks=result_blocks,
    )


def _solve_forward_gauss_seidel(
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
) -> int:
    """Solve forward gauss seidel for internal processing."""
    for iteration in range(1, max_iterations + 1):
        previous = np.asarray(
            [working[name][position] for name, _, position in unknowns]
        )
        for name, period, position in unknowns:
            working[name][position] = _solve_equation(
                name,
                period,
                position,
                bound,
                coefficients,
                adjustments,
                historical,
                working,
                "DYNAMIC",
                zero_error_autocorrelation,
            )
        current = np.asarray(
            [working[name][position] for name, _, position in unknowns]
        )
        if _percentage_converged(previous, current, convergence):
            return iteration
    raise SimulationConvergenceError(
        "forward-looking system did not converge in "
        f"{max_iterations} Gauss-Seidel iterations"
    )


def _solve_forward_newton(
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    newton_workspace: NewtonWorkspace | None,
    rebuild_threshold: float,
    relaxation_threshold: float,
) -> int:
    """Solve forward newton for internal processing."""

    def residual(candidate: np.ndarray) -> np.ndarray:
        """Evaluate stacked forward residuals for a candidate vector."""
        for (name, _, position), value in zip(unknowns, candidate, strict=True):
            working[name][position] = value
        targets: np.ndarray = np.asarray(
            [
                _solve_equation(
                    name,
                    period,
                    position,
                    bound,
                    coefficients,
                    adjustments,
                    historical,
                    working,
                    "DYNAMIC",
                    zero_error_autocorrelation,
                )
                for name, period, position in unknowns
            ]
        )
        result: np.ndarray = targets - candidate
        return result

    current = np.asarray(
        [working[name][position] for name, _, position in unknowns], dtype=float
    )
    cache_key = (
        "forward",
        tuple((name, period.year, period.period) for name, period, _ in unknowns),
    )
    column_rows = _forward_jacobian_pattern(bound.model, unknowns)
    try:
        _, iterations = solve_sparse_newton(
            residual,
            current,
            relative_step=jacobian_step,
            convergence=convergence,
            max_iterations=max_iterations,
            workspace=newton_workspace,
            cache_key=cache_key,
            column_rows=column_rows,
            rebuild_threshold=rebuild_threshold,
            relaxation_threshold=relaxation_threshold,
        )
    except SparseFactorizationError as error:
        raise SimulationConvergenceError(
            "singular Newton Jacobian for the forward-looking system"
        ) from error
    except SparseNewtonIterationLimit as error:
        raise SimulationConvergenceError(
            f"forward-looking system did not converge in "
            f"{max_iterations} Newton iterations"
        ) from error
    return iterations


def _solve_forward_hybrid_newton(
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
    dropped: frozenset[str],
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, AdjustmentValue],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    newton_workspace: NewtonWorkspace | None,
    rebuild_threshold: float,
    relaxation_threshold: float,
) -> int:
    """Alternate dropped forward equations with a reduced extended Newton solve."""
    newton_unknowns = tuple(item for item in unknowns if item[0] not in dropped)
    for iteration in range(1, max_iterations + 1):
        previous = np.asarray(
            [working[name][position] for name, _, position in unknowns]
        )
        for name, period, position in unknowns:
            if name in dropped:
                working[name][position] = _solve_equation(
                    name,
                    period,
                    position,
                    bound,
                    coefficients,
                    adjustments,
                    historical,
                    working,
                    "DYNAMIC",
                    zero_error_autocorrelation,
                )
        _solve_forward_newton(
            newton_unknowns,
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
            jacobian_step,
            newton_workspace,
            rebuild_threshold,
            relaxation_threshold,
        )
        current = np.asarray(
            [working[name][position] for name, _, position in unknowns]
        )
        if _percentage_converged(previous, current, convergence):
            return iteration
    raise SimulationConvergenceError(
        "forward-looking system did not converge in "
        f"{max_iterations} hybrid Newton iterations"
    )


def _forward_jacobian_pattern(
    model: BimetsModel,
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the extended lead/lag Jacobian structure for sparse coloring."""
    column_for = {
        (name, position): column for column, (name, _, position) in enumerate(unknowns)
    }
    rows_by_column: list[set[int]] = [set() for _ in unknowns]
    for row, (dependent, _, position) in enumerate(unknowns):
        rows_by_column[row].add(row)
        for variable, offsets in _simulation_variable_offsets(model, dependent).items():
            for offset in offsets:
                column = column_for.get((variable, position + offset))
                if column is not None:
                    rows_by_column[column].add(row)
    return tuple(tuple(sorted(rows)) for rows in rows_by_column)


def _simulation_variable_offsets(
    model: BimetsModel,
    dependent: str,
) -> dict[str, frozenset[int]]:
    """Collect all temporal endogenous dependencies used during simulation."""
    collected: dict[str, set[int]] = {dependent: {0}}

    def include(expression: MdlExpression, shift: int = 0) -> None:
        """Merge endogenous offsets from one expression."""
        for name, offsets in variable_offsets(expression, shift).items():
            if name in model.endogenous:
                collected.setdefault(name, set()).update(offsets)

    definition = model._equation_definition(dependent)
    if isinstance(definition, IdentityEquation):
        for alternative in definition.alternatives:
            include(alternative.equation.rhs)
            if alternative.condition is not None:
                include(alternative.condition)
        equation = definition.alternatives[0].equation
    else:
        equation = definition.equation
        include(equation.rhs)
        pdl_lengths = {item.coefficient: item.length for item in definition.pdls}
        for coefficient, regressor in zip(
            definition.coefficients,
            definition.regressors,
            strict=True,
        ):
            for lag in range(pdl_lengths.get(coefficient, 1)):
                include(regressor, -lag)
        if definition.error is not None:
            base_offsets = {name: set(offsets) for name, offsets in collected.items()}
            for lag in range(1, definition.error.order + 1):
                for name, offsets in base_offsets.items():
                    collected.setdefault(name, set()).update(
                        offset - lag for offset in offsets
                    )
                collected[dependent].update({-lag, -lag - equation.lhs_periods})
    if equation.lhs_periods:
        collected[dependent].add(-equation.lhs_periods)
    return {name: frozenset(offsets) for name, offsets in collected.items()}


def _percentage_converged(
    previous: np.ndarray, current: np.ndarray, convergence: float
) -> bool:
    """Return the percentage change used by convergence checks."""
    denominator = np.maximum(np.abs(previous), 1.0)
    percentage_change = 100.0 * np.abs(current - previous) / denominator
    return bool(np.all(percentage_change < convergence))


def _residual_check(
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    start: YearPeriod,
    end: YearPeriod,
    historical: Mapping[str, np.ndarray],
    storage_start: YearPeriod,
    blocks: tuple[SimulationBlock, ...],
    adjustments: Mapping[str, AdjustmentValue],
    exogenization: Mapping[str, _ExogenizationRule],
    selected: tuple[str, ...],
    zero_error_autocorrelation: bool,
) -> SimulationResult:
    """Return the internally computed residual check."""
    names = selected
    simulated_values: dict[str, list[float]] = {name: [] for name in names}
    adjustment_values: dict[str, list[float]] = {name: [] for name in names}
    for period in _periods(start, end, bound.freq):
        position = _position(period, storage_start, bound.freq)
        for name in names:
            rule = exogenization.get(name)
            if rule is not None and rule.applies(period, bound.freq):
                value = (
                    float(historical[name][position])
                    if rule.values is None
                    else rule.values.at_period(period.year, period.period)
                )
                if not math.isfinite(value):
                    raise ValueError(
                        f"RESCHECK exogenized variable {name!r} has no finite "
                        f"value at {period.year}-{period.period}"
                    )
                simulated_values[name].append(value)
                adjustment_values[name].append(0.0)
                continue
            definition = bound.model._equation_definition(name)
            if isinstance(definition, IdentityEquation):
                equation = _select_identity_scalar(
                    definition,
                    position,
                    bound,
                    historical,
                    historical,
                    "RESCHECK",
                )
                if equation is None:
                    actual = float(historical[name][position])
                    if not math.isfinite(actual):
                        raise ValueError(
                            f"RESCHECK conditional identity {name!r} has no "
                            f"historical value at {period.year}-{period.period}"
                        ) from None
                    simulated_values[name].append(actual)
                    adjustment_values[name].append(0.0)
                    continue
                rhs = _evaluate_scalar(
                    equation.rhs,
                    position,
                    position,
                    bound,
                    historical,
                    historical,
                    "RESCHECK",
                )
            else:
                equation = definition.equation
                rhs = _behavioral_rhs_scalar(
                    definition,
                    coefficients,
                    adjustments,
                    position,
                    period,
                    bound,
                    historical,
                    historical,
                    "RESCHECK",
                    zero_error_autocorrelation,
                )
            rhs += _adjustment_value(adjustments.get(name, 0.0), period)
            simulated = _invert_lhs_scalar(
                equation,
                rhs,
                position,
                bound,
                historical,
                historical,
                "RESCHECK",
            )
            actual = float(historical[name][position])
            try:
                adjustment = (
                    _lhs_value_scalar(
                        equation,
                        actual,
                        position,
                        position,
                        bound,
                        historical,
                        historical,
                        "RESCHECK",
                    )
                    - rhs
                )
            except ValueError:
                if equation.lhs_function != "LOG" or actual != 0:
                    raise
                # R evaluates LOG(0) as -Inf. The diagnostic add-factor may
                # therefore be non-finite even though inversion of the model
                # equation yields a valid RESCHECK solution.
                adjustment = -math.inf
            if not all(math.isfinite(value) for value in (simulated, actual)):
                raise ValueError(
                    f"RESCHECK equation {name!r} has non-finite data or output at "
                    f"{period.year}-{period.period}"
                )
            simulated_values[name].append(simulated)
            adjustment_values[name].append(adjustment)
    simulated_series = {
        name: BimetsSeries(values, start=start, freq=bound.freq)
        for name, values in simulated_values.items()
    }
    adjustment_series = {
        name: BimetsSeries(values, start=start, freq=bound.freq)
        for name, values in adjustment_values.items()
    }
    return SimulationResult(
        bound.model.name,
        simulated_series,
        simulation_type="RESCHECK",
        algorithm="GAUSS-SEIDEL",
        convergence=0.0,
        max_iterations=1,
        iterations={period: 1 for period in _periods(start, end, bound.freq)},
        blocks=tuple(
            SimulationBlock(
                tuple(name for name in block.variables if name in names),
                block.simultaneous,
                tuple(name for name in block.feedback if name in names),
            )
            for block in blocks
            if any(name in names for name in block.variables)
        ),
        constant_adjustments=BimetsDataset(adjustment_series),
    )


def _output_series(
    names: Sequence[str],
    working: Mapping[str, np.ndarray],
    storage_start: YearPeriod,
    start: YearPeriod,
    end: YearPeriod,
    freq: Frequency,
) -> dict[str, BimetsSeries]:
    """Build output series from the simulation storage arrays."""
    first = _position(start, storage_start, freq)
    last = _position(end, storage_start, freq) + 1
    return {
        name: BimetsSeries(working[name][first:last], start=start, freq=freq)
        for name in names
    }
