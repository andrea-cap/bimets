"""Monte Carlo optimal control for MDL models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from bimets.mdl._binding import BoundModel
from bimets.mdl._evaluation import MdlValue, evaluate_expression
from bimets.mdl._expression import (
    BinaryExpression,
    MdlExpression,
    parse_expression,
    variable_names,
)
from bimets.mdl._model import BimetsModel, MdlTimeRange
from bimets.mdl._random import RMersenneTwister
from bimets.mdl._renormalization import (
    _full_adjustment_series,
    _replace_periods,
)
from bimets.mdl._simulation import (
    AdjustmentValue,
    CoefficientInput,
    ExogenizationValue,
    _adjustment_value,
    _periods,
    _resolve_bound_model,
    _simulation_bounds,
)
from bimets.mdl._stochastic import (
    StochasticDisturbance,
    StochasticSimulationError,
    StochasticSimulationResult,
    _stochastic_simulate,
)
from bimets.timeseries import (
    BimetsDataset,
    BimetsMask,
    BimetsSeries,
    YearPeriod,
)

type BoundSpec = OptimizationBound | tuple[float, float]
type FunctionSpec = str | OptimizationFunction
type RestrictionSpec = str | OptimizationRestriction


class OptimizationError(RuntimeError):
    """Raised when an optimal-control simulation cannot be completed."""


@dataclass(frozen=True, slots=True)
class OptimizationBound:
    """Uniform search bounds for one policy instrument.

    Parameters
    ----------
    lower, upper : float
        Finite endpoints with ``lower < upper``.
    time_range : MdlTimeRange or tuple of four int, optional
        Active range. The default selects the complete optimization range.
    """

    lower: float
    upper: float
    time_range: MdlTimeRange | tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("optimization bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("optimization lower bound must be less than upper bound")


@dataclass(frozen=True, slots=True)
class OptimizationFunction:
    """One time-ranged objective function to maximize.

    Parameters
    ----------
    expression : str
        Arithmetic MDL expression evaluated with simulated endogenous values
        and realized exogenous values.
    time_range : MdlTimeRange or tuple of four int, optional
        Active range. The default selects the complete optimization range.
    """

    expression: str
    time_range: MdlTimeRange | tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise ValueError("optimization function expression cannot be empty")


@dataclass(frozen=True, slots=True)
class OptimizationRestriction:
    """One time-ranged feasibility restriction.

    Parameters
    ----------
    expression : str
        Logical MDL expression. Endogenous names denote equation add-factors;
        exogenous names denote realized exogenous values.
    time_range : MdlTimeRange or tuple of four int, optional
        Active range. The default selects the complete optimization range.
    """

    expression: str
    time_range: MdlTimeRange | tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise ValueError("optimization restriction expression cannot be empty")


@dataclass(frozen=True, slots=True)
class _PreparedExpression:
    name: str
    expression: MdlExpression
    indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Immutable result of a Monte Carlo optimal-control search.

    Parameters
    ----------
    instruments : mapping of str to BimetsSeries
        Maximizing instrument paths, or an empty mapping if no feasible finite
        realization exists.
    objective_max : float or None
        Maximum summed objective value.
    objective_path : BimetsSeries or None
        Period-wise objective values for the maximizing realization.
    objective_mean, objective_standard_deviation : float or None
        Statistics across feasible finite realizations.
    objective_results : numpy.ndarray
        Summed objective for every unfiltered realization.
    objective_paths : numpy.ndarray
        Period-by-realization objective values, with missing values outside
        active objective ranges.
    feasible : numpy.ndarray
        Boolean mask combining restrictions and objective computability.
    data : BimetsDataset
        Model data with the maximizing exogenous instruments applied.
    constant_adjustments : mapping
        Adjustments with maximizing endogenous instruments applied.
    stochastic : StochasticSimulationResult
        Baseline and every simulated endogenous realization.
    maximizing_replica : int or None
        Zero-based maximizing realization index.
    seed : int or None
        Random-generator seed.
    """

    instruments: Mapping[str, BimetsSeries]
    objective_max: float | None
    objective_path: BimetsSeries | None
    objective_mean: float | None
    objective_standard_deviation: float | None
    objective_results: np.ndarray
    objective_paths: np.ndarray
    feasible: np.ndarray
    data: BimetsDataset
    constant_adjustments: Mapping[str, AdjustmentValue]
    stochastic: StochasticSimulationResult
    maximizing_replica: int | None
    seed: int | None

    def __post_init__(self) -> None:
        results = np.asarray(self.objective_results, dtype=float).copy()
        paths = np.asarray(self.objective_paths, dtype=float).copy()
        feasible = np.asarray(self.feasible, dtype=bool).copy()
        if results.ndim != 1:
            raise ValueError("objective_results must be one-dimensional")
        if paths.ndim != 2 or paths.shape[1] != results.size:
            raise ValueError("objective_paths must be periods by realizations")
        if feasible.shape != results.shape:
            raise ValueError("feasible mask and objective results differ")
        results.setflags(write=False)
        paths.setflags(write=False)
        feasible.setflags(write=False)
        object.__setattr__(self, "objective_results", results)
        object.__setattr__(self, "objective_paths", paths)
        object.__setattr__(self, "feasible", feasible)
        object.__setattr__(
            self, "instruments", MappingProxyType(dict(self.instruments))
        )
        object.__setattr__(
            self,
            "constant_adjustments",
            MappingProxyType(dict(self.constant_adjustments)),
        )

    @property
    def replicas(self) -> int:
        """Number of Monte Carlo realizations."""
        return int(self.objective_results.size)

    @property
    def feasible_count(self) -> int:
        """Number of finite realizations satisfying all restrictions."""
        return int(np.count_nonzero(self.feasible))

    @property
    def opt_fun_max(self) -> float | None:
        """Alias matching the BIMETS R ``optFunMax`` result name."""
        return self.objective_max

    @property
    def opt_fun_sd(self) -> float | None:
        """Alias matching the BIMETS R ``optFunSd`` result name."""
        return self.objective_standard_deviation

    @property
    def opt_fun_ave(self) -> float | None:
        """Alias matching the BIMETS R ``optFunAve`` result name."""
        return self.objective_mean

    def summary(self) -> pd.DataFrame:
        """Return objective values and feasibility by realization."""
        return pd.DataFrame(
            {
                "objective": self.objective_results,
                "feasible": self.feasible,
            },
            index=pd.RangeIndex(1, self.replicas + 1, name="replica"),
        )


def optimize_model(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    bounds: Mapping[str, BoundSpec],
    objective_functions: FunctionSpec | Mapping[str, FunctionSpec],
    restrictions: RestrictionSpec | Mapping[str, RestrictionSpec] | None = None,
    replicas: int = 100,
    seed: int | None = None,
    workers: int = 1,
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
) -> OptimizationResult:
    """Maximize time-ranged objective functions by Monte Carlo search.

    Parameters
    ----------
    model, data, coefficients, time_range
        See :func:`bimets.simulate`.
    bounds : mapping
        Instrument names mapped to :class:`OptimizationBound` or ``(lower,
        upper)`` tuples. Endogenous names control equation add-factors.
    objective_functions : str, OptimizationFunction, or mapping
        Arithmetic MDL expressions to maximize. Mapping values can use
        different non-overlapping active ranges.
    restrictions : str, OptimizationRestriction, or mapping, optional
        Logical MDL expressions. Different definitions must have
        non-overlapping active ranges.
    replicas : int, default=100
        Number of independent uniformly sampled candidate paths.
    seed : int, optional
        Seed using R-compatible ``set.seed`` semantics. This makes bounded
        candidate paths reproducible across BIMETS R and Python.
    workers : int, default=1
        Processes used only for independent ``FULLNEWTON`` candidates.
    simulation_type, algorithm, convergence, max_iterations, jacobian_step
        Deterministic solver settings.
    zero_error_autocorrelation, constant_adjustments, exogenize,
    rescheck_equations
        Additional deterministic simulation settings.
    backfill : int, default=0
        Historical observations prepended to deterministic baselines stored in
        the stochastic result.
    jacobian_drop : str or sequence of str, optional
        Feedback variables excluded from candidate Newton Jacobians.

    Returns
    -------
    OptimizationResult
        Maximizing instruments, objective statistics, all realizations, and
        adjusted inputs. No feasible solution is represented by ``None``
        optimum fields and an empty instrument mapping.

    Notes
    -----
    This implements the Monte Carlo algorithm used by BIMETS R ``OPTIMIZE``.
    With an explicit seed, bounded candidate paths reproduce R's default
    Mersenne Twister and column-major matrix filling. An omitted seed remains
    intentionally non-deterministic. During forecast searches, historical
    endogenous series are extended internally before candidate objective and
    restriction evaluation, as they are in BIMETS R simulation.

    Backward-looking Gauss-Seidel and common-Jacobian ``NEWTON`` searches
    evaluate the baseline and all candidates as columns of shared NumPy
    matrices. Newton reuses the baseline sparse factorization and solves the
    candidate right-hand sides together. Common exogenization and ``RESCHECK``
    also use this path. Forward-looking and reduced-Jacobian systems remain on
    the shared backend. If the iteration limit is reached, the final iteration
    is retained with a warning, as in BIMETS R. Only ``FULLNEWTON`` uses
    independent candidate solves and may distribute them with ``workers``.

    Examples
    --------
    >>> from bimets import BimetsModel, optimize_model, timeseries
    >>> model = BimetsModel.from_text("MODEL\\nIDENTITY> y\\nEQ> y=2*x\\nEND")
    >>> result = optimize_model(
    ...     model,
    ...     {"y": timeseries([0]), "x": timeseries([0])},
    ...     coefficients={},
    ...     time_range=(2000, 1, 2000, 1),
    ...     bounds={"x": (0, 5)},
    ...     objective_functions="y",
    ...     replicas=20,
    ...     seed=3,
    ...     simulation_type="STATIC",
    ... )
    >>> bool(result.objective_max == 2 * result.instruments["x"].values[0])
    True
    """
    bound = _resolve_bound_model(model, data)
    if isinstance(replicas, bool) or not isinstance(replicas, int):
        raise TypeError("replicas must be an integer")
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    prepared_bounds = _prepare_bounds(bound, bounds)
    start, end = _simulation_bounds(time_range, bound.freq)
    periods = tuple(_periods(start, end, bound.freq))
    functions = _prepare_expressions(
        objective_functions,
        OptimizationFunction,
        "objective",
        periods,
        bound,
    )
    prepared_restrictions = _prepare_expressions(
        {} if restrictions is None else restrictions,
        OptimizationRestriction,
        "restriction",
        periods,
        bound,
        allow_empty=True,
    )
    rng: RMersenneTwister | np.random.Generator = (
        RMersenneTwister(seed) if seed is not None else np.random.default_rng()
    )
    disturbances: dict[str, StochasticDisturbance] = {}
    endogenous = set(bound.model.endogenous)
    adjustments = dict(constant_adjustments or {})
    for name, definition in prepared_bounds.items():
        indexes = _active_indexes(definition.time_range, periods, bound)
        draws = rng.uniform(
            definition.lower, definition.upper, (len(indexes), replicas)
        )
        if name in endogenous:
            baseline = np.asarray(
                [
                    _adjustment_value(adjustments.get(name, 0.0), periods[index])
                    for index in indexes
                ]
            )
            draws = draws - baseline[:, None]
        active_range = (
            periods[indexes[0]].year,
            periods[indexes[0]].period,
            periods[indexes[-1]].year,
            periods[indexes[-1]].period,
        )
        disturbances[name] = StochasticDisturbance(
            "MATRIX", draws, time_range=active_range
        )
    try:
        stochastic = _stochastic_simulate(
            bound,
            coefficients=coefficients,
            time_range=time_range,
            disturbances=disturbances,
            replicas=replicas,
            workers=workers,
            simulation_type=simulation_type,
            algorithm=algorithm,
            convergence=convergence,
            max_iterations=max_iterations,
            jacobian_step=jacobian_step,
            zero_error_autocorrelation=zero_error_autocorrelation,
            constant_adjustments=adjustments,
            exogenize=exogenize,
            rescheck_equations=rescheck_equations,
            backfill=backfill,
            jacobian_drop=jacobian_drop,
            _shared_convergence=True,
            _retain_final_iteration=True,
        )
    except StochasticSimulationError as error:
        raise OptimizationError(f"candidate simulation failed: {error}") from error

    objective_results = np.empty(replicas, dtype=float)
    objective_paths = np.full((len(periods), replicas), np.nan)
    feasible = np.ones(replicas, dtype=bool)
    for replica in range(replicas):
        objective_data, restriction_data = _realization_data(
            bound,
            stochastic,
            adjustments,
            periods,
            replica,
        )
        feasible[replica] = _satisfies_restrictions(
            prepared_restrictions, restriction_data, periods
        )
        total, path = _objective_value(functions, objective_data, periods)
        objective_results[replica] = total
        objective_paths[:, replica] = path
    feasible &= np.isfinite(objective_results)
    candidates = np.flatnonzero(feasible)
    if candidates.size == 0:
        return OptimizationResult(
            instruments={},
            objective_max=None,
            objective_path=None,
            objective_mean=None,
            objective_standard_deviation=None,
            objective_results=objective_results,
            objective_paths=objective_paths,
            feasible=feasible,
            data=bound.data,
            constant_adjustments=adjustments,
            stochastic=stochastic,
            maximizing_replica=None,
            seed=seed,
        )

    maximizing = int(candidates[np.argmax(objective_results[candidates])])
    filtered = objective_results[feasible]
    optimal_data = dict(bound.data)
    optimal_adjustments: dict[str, AdjustmentValue] = dict(adjustments)
    instruments: dict[str, BimetsSeries] = {}
    for name in prepared_bounds:
        values = stochastic.instrument_realizations[name][:, maximizing]
        instruments[name] = BimetsSeries(values, start=periods[0], freq=bound.freq)
        if name in endogenous:
            base = _full_adjustment_series(
                bound.data[name], optimal_adjustments.get(name, 0.0)
            )
            if base.end.ordinal(base.freq) < periods[-1].ordinal(base.freq):
                base = base.extend(up_to=periods[-1], mode="missing")
            optimal_adjustments[name] = _replace_periods(base, periods, values)
        else:
            optimal_data[name] = _replace_periods(optimal_data[name], periods, values)
    objective_path = BimetsSeries(
        objective_paths[:, maximizing], start=periods[0], freq=bound.freq
    )
    return OptimizationResult(
        instruments=instruments,
        objective_max=float(objective_results[maximizing]),
        objective_path=objective_path,
        objective_mean=float(np.mean(filtered)),
        objective_standard_deviation=(
            float(np.std(filtered, ddof=1)) if filtered.size > 1 else math.nan
        ),
        objective_results=objective_results,
        objective_paths=objective_paths,
        feasible=feasible,
        data=BimetsDataset(optimal_data),
        constant_adjustments=optimal_adjustments,
        stochastic=stochastic,
        maximizing_replica=maximizing,
        seed=seed,
    )


def _prepare_bounds(
    bound: BoundModel, bounds: Mapping[str, BoundSpec]
) -> dict[str, OptimizationBound]:
    """Prepare bounds for internal processing."""
    if not isinstance(bounds, Mapping) or not bounds:
        raise ValueError("bounds must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in bounds):
        raise ValueError("bounded instrument names must be non-empty strings")
    variables = set(bound.model.endogenous).union(bound.model.exogenous)
    unknown = set(bounds).difference(variables)
    if unknown:
        raise KeyError(f"unknown bounded instruments: {sorted(unknown)}")
    output: dict[str, OptimizationBound] = {}
    for name, value in bounds.items():
        if isinstance(value, OptimizationBound):
            output[name] = value
        elif isinstance(value, tuple) and len(value) == 2:
            output[name] = OptimizationBound(float(value[0]), float(value[1]))
        else:
            raise TypeError(
                "bounds values must be OptimizationBound or two-number tuples"
            )
    return output


def _prepare_expressions(
    definitions: FunctionSpec
    | RestrictionSpec
    | Mapping[str, FunctionSpec | RestrictionSpec],
    expected_type: type[OptimizationFunction] | type[OptimizationRestriction],
    label: str,
    periods: tuple[YearPeriod, ...],
    bound: BoundModel,
    *,
    allow_empty: bool = False,
) -> tuple[_PreparedExpression, ...]:
    """Prepare expressions for internal processing."""
    if isinstance(definitions, str):
        items: Mapping[str, FunctionSpec | RestrictionSpec] = {label: definitions}
    elif isinstance(definitions, expected_type):
        items = {label: definitions}
    elif isinstance(definitions, Mapping):
        items = definitions
    else:
        raise TypeError(f"{label} definitions have an invalid type")
    if not items:
        if allow_empty:
            return ()
        raise ValueError(f"at least one {label} function is required")
    model_variables = set(bound.model.endogenous).union(bound.model.exogenous)
    occupied: set[int] = set()
    output: list[_PreparedExpression] = []
    for name, value in items.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} names must be non-empty strings")
        if isinstance(value, str):
            definition = expected_type(value)
        elif isinstance(value, expected_type):
            definition = value
        else:
            raise TypeError(f"{label} values have an invalid type")
        expression = parse_expression(definition.expression)
        logical = isinstance(expression, BinaryExpression) and expression.operator in {
            "&",
            "|",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }
        if label == "restriction" and not logical:
            raise ValueError(f"restriction {name!r} must be a logical expression")
        if label == "objective" and logical:
            raise ValueError(f"objective {name!r} must be a numeric expression")
        names = variable_names(expression)
        if not names:
            raise ValueError(f"{label} {name!r} must reference a model variable")
        unknown = names.difference(model_variables)
        if unknown:
            raise KeyError(f"unknown variables in {label} {name!r}: {sorted(unknown)}")
        indexes = _active_indexes(definition.time_range, periods, bound)
        overlap = occupied.intersection(indexes)
        if overlap:
            raise ValueError(f"{label} time ranges cannot overlap")
        occupied.update(indexes)
        output.append(_PreparedExpression(name, expression, indexes))
    return tuple(output)


def _active_indexes(
    time_range: MdlTimeRange | tuple[int, int, int, int] | None,
    periods: tuple[YearPeriod, ...],
    bound: BoundModel,
) -> tuple[int, ...]:
    """Return periods where an optimization expression is active."""
    if time_range is None:
        return tuple(range(len(periods)))
    start, end = _simulation_bounds(time_range, bound.freq)
    indexes = tuple(
        index
        for index, period in enumerate(periods)
        if start.ordinal(bound.freq)
        <= period.ordinal(bound.freq)
        <= end.ordinal(bound.freq)
    )
    if not indexes:
        raise ValueError("definition time range does not overlap optimization range")
    return indexes


def _realization_data(
    bound: BoundModel,
    stochastic: StochasticSimulationResult,
    adjustments: Mapping[str, AdjustmentValue],
    periods: tuple[YearPeriod, ...],
    replica: int,
) -> tuple[dict[str, BimetsSeries], dict[str, BimetsSeries]]:
    """Build model data for one optimization realization."""
    objective = dict(bound.data)
    for name in bound.model.endogenous:
        if name in stochastic:
            source = bound.data[name]
            if source.end.ordinal(source.freq) < periods[-1].ordinal(source.freq):
                source = source.extend(up_to=periods[-1], mode="missing")
            objective[name] = _replace_periods(
                source, periods, stochastic[name].realizations[:, replica]
            )
    for name, realizations in stochastic.instrument_realizations.items():
        if name in bound.model.exogenous:
            objective[name] = _replace_periods(
                bound.data[name], periods, realizations[:, replica]
            )
    restriction = dict(objective)
    for name in bound.model.endogenous:
        values = (
            stochastic.instrument_realizations[name][:, replica]
            if name in stochastic.instrument_realizations
            else np.asarray(
                [
                    _adjustment_value(adjustments.get(name, 0.0), period)
                    for period in periods
                ]
            )
        )
        base = _full_adjustment_series(bound.data[name], adjustments.get(name, 0.0))
        if base.end.ordinal(base.freq) < periods[-1].ordinal(base.freq):
            base = base.extend(up_to=periods[-1], mode="missing")
        restriction[name] = _replace_periods(base, periods, values)
    return objective, restriction


def _satisfies_restrictions(
    restrictions: tuple[_PreparedExpression, ...],
    data: Mapping[str, BimetsSeries],
    periods: tuple[YearPeriod, ...],
) -> bool:
    """Return whether satisfies restrictions."""
    for definition in restrictions:
        try:
            value = evaluate_expression(definition.expression, data)
            for index in definition.indexes:
                if _logical_at(value, periods[index]) is not True:
                    return False
        except (IndexError, KeyError, TypeError, ValueError):
            return False
    return True


def _objective_value(
    functions: tuple[_PreparedExpression, ...],
    data: Mapping[str, BimetsSeries],
    periods: tuple[YearPeriod, ...],
) -> tuple[float, np.ndarray]:
    """Evaluate and aggregate the optimization objective."""
    path = np.full(len(periods), np.nan)
    total = 0.0
    try:
        for definition in functions:
            value = evaluate_expression(definition.expression, data)
            for index in definition.indexes:
                current = _numeric_at(value, periods[index])
                path[index] = current
                total += current
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return math.nan, path
    return (total if math.isfinite(total) else math.nan), path


def _numeric_at(value: MdlValue, period: YearPeriod) -> float:
    """Extract a numeric optimization value at one period."""
    if isinstance(value, BimetsSeries):
        return value.at_period(period.year, period.period)
    if isinstance(value, (BimetsMask, bool)):
        raise TypeError("objective functions must be numeric")
    return float(value)


def _logical_at(value: MdlValue, period: YearPeriod) -> bool | None:
    """Extract a logical optimization value at one period."""
    if isinstance(value, BimetsMask):
        position = period.ordinal(value.freq) - value.start.ordinal(value.freq)
        if position < 0 or position >= len(value):
            raise IndexError("restriction period is outside expression range")
        return value[position]
    if isinstance(value, bool):
        return value
    raise TypeError("optimization restrictions must be logical")
