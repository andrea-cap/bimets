"""Stochastic simulation built on the deterministic MDL solver."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from types import MappingProxyType

import numpy as np
import pandas as pd

from bimets.mdl._binding import BoundModel
from bimets.mdl._model import BimetsModel, MdlTimeRange
from bimets.mdl._random import RMersenneTwister
from bimets.mdl._simulation import (
    AdjustmentValue,
    CoefficientInput,
    ExogenizationValue,
    SimulationConvergenceError,
    SimulationResult,
    _adjustment_value,
    _coefficient_mapping,
    _periods,
    _resolve_bound_model,
    _simulation_blocks,
    _simulation_bounds,
)
from bimets.mdl._simulation import _simulate as _simulate
from bimets.mdl._simulation_batch import simulate_shared_columns
from bimets.mdl._sparse import NewtonWorkspace
from bimets.timeseries import (
    BimetsDataset,
    BimetsSeries,
    Frequency,
    YearPeriod,
    get_dates,
)

type DisturbanceParameters = tuple[float, float] | np.ndarray


class StochasticSimulationError(RuntimeError):
    """Raised when one stochastic realization cannot be simulated."""


@dataclass(frozen=True, slots=True)
class StochasticDisturbance:
    """Definition of a stochastic perturbation.

    Parameters
    ----------
    distribution : {"NORMAL", "UNIFORM", "MATRIX"}
        Disturbance distribution. The BIMETS names ``NORM`` and ``UNIF`` are
        accepted as aliases.
    parameters : tuple of float or numpy.ndarray
        ``(mean, standard_deviation)`` for normal draws, ``(lower, upper)``
        for uniform draws, or a two-dimensional matrix with one row per active
        period and one column per realization.
    time_range : MdlTimeRange or tuple of four int, optional
        Inclusive perturbation range. The default applies it over the complete
        stochastic simulation range.

    Notes
    -----
    An endogenous disturbance is additive on the equation constant adjustment.
    A normal or uniform exogenous disturbance is additive on the data, whereas
    an exogenous matrix replaces the data in its active range.
    """

    distribution: str
    parameters: DisturbanceParameters
    time_range: MdlTimeRange | tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        aliases = {
            "NORM": "NORMAL",
            "NORMAL": "NORMAL",
            "UNIF": "UNIFORM",
            "UNIFORM": "UNIFORM",
            "MATRIX": "MATRIX",
        }
        try:
            normalized = aliases[self.distribution.upper()]
        except (AttributeError, KeyError) as error:
            raise ValueError(
                "distribution must be 'NORMAL', 'UNIFORM', or 'MATRIX'"
            ) from error
        object.__setattr__(self, "distribution", normalized)
        if normalized == "MATRIX":
            matrix = np.asarray(self.parameters, dtype=float)
            if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
                raise ValueError("MATRIX parameters must be a finite 2D array")
            matrix = matrix.copy()
            matrix.setflags(write=False)
            object.__setattr__(self, "parameters", matrix)
            return
        if not isinstance(self.parameters, tuple) or len(self.parameters) != 2:
            raise TypeError(
                "NORMAL and UNIFORM parameters must be a tuple of two numbers"
            )
        first, second = (float(value) for value in self.parameters)
        if not all(math.isfinite(value) for value in (first, second)):
            raise ValueError("disturbance parameters must be finite")
        if normalized == "NORMAL" and second < 0:
            raise ValueError("normal standard deviation cannot be negative")
        if normalized == "UNIFORM" and first > second:
            raise ValueError("uniform lower bound cannot exceed upper bound")
        object.__setattr__(self, "parameters", (first, second))


@dataclass(frozen=True, slots=True)
class StochasticSeriesResult:
    """Distribution summary and realizations for one endogenous variable.

    Attributes
    ----------
    mean, standard_deviation : BimetsSeries
        Period-wise sample statistics across perturbed realizations.
    realizations : numpy.ndarray
        Read-only matrix shaped ``(periods, replicas)``.
    """

    mean: BimetsSeries
    standard_deviation: BimetsSeries
    realizations: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.realizations, dtype=float).copy()
        if values.ndim != 2 or values.shape[0] != len(self.mean):
            raise ValueError("realizations must be a periods-by-replicas matrix")
        if (
            self.standard_deviation.start != self.mean.start
            or self.standard_deviation.end != self.mean.end
            or self.standard_deviation.freq != self.mean.freq
        ):
            raise ValueError("mean and standard deviation ranges differ")
        values.setflags(write=False)
        object.__setattr__(self, "realizations", values)

    @property
    def sd(self) -> BimetsSeries:
        """Alias matching the ``sd`` result name in BIMETS R."""
        return self.standard_deviation


class StochasticSimulationResult(Mapping[str, StochasticSeriesResult]):
    """Immutable result of a stochastic model simulation.

    Parameters
    ----------
    baseline : SimulationResult
        Unperturbed deterministic solution.
    series : mapping of str to StochasticSeriesResult
        Summaries and realization matrices for endogenous variables.
    instrument_baseline : mapping of str to BimetsSeries
        Unperturbed exogenous values or endogenous add-factors.
    instrument_realizations : mapping of str to numpy.ndarray
        Perturbed instruments shaped ``(periods, replicas)``.
    replicas : int
        Number of perturbed realizations.
    seed : int or None
        Random-generator seed. Seeded normal and uniform draws reproduce R.
    """

    __slots__ = (
        "_instrument_baseline",
        "_instrument_realizations",
        "_series",
        "baseline",
        "replicas",
        "seed",
    )

    def __init__(
        self,
        baseline: SimulationResult,
        series: Mapping[str, StochasticSeriesResult],
        *,
        instrument_baseline: Mapping[str, BimetsSeries],
        instrument_realizations: Mapping[str, np.ndarray],
        replicas: int,
        seed: int | None,
    ) -> None:
        self.baseline = baseline
        self.replicas = replicas
        self.seed = seed
        self._series = MappingProxyType(dict(series))
        self._instrument_baseline = MappingProxyType(dict(instrument_baseline))
        matrices: dict[str, np.ndarray] = {}
        for name, matrix in instrument_realizations.items():
            values = np.asarray(matrix, dtype=float).copy()
            values.setflags(write=False)
            matrices[name] = values
        self._instrument_realizations = MappingProxyType(matrices)

    def __getitem__(self, name: str) -> StochasticSeriesResult:
        return self._series[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._series)

    def __len__(self) -> int:
        return len(self._series)

    @property
    def instrument_baseline(self) -> Mapping[str, BimetsSeries]:
        """Read-only unperturbed values for disturbed variables."""
        return self._instrument_baseline

    @property
    def instrument_realizations(self) -> Mapping[str, np.ndarray]:
        """Read-only perturbed instrument matrices."""
        return self._instrument_realizations

    def summary(self) -> pd.DataFrame:
        """Return all endogenous means and standard deviations as a table."""
        columns: dict[tuple[str, str], np.ndarray] = {}
        for name, result in self._series.items():
            columns[(name, "mean")] = result.mean.values
            columns[(name, "standard_deviation")] = result.standard_deviation.values
        first = next(iter(self._series.values())).mean
        frame = pd.DataFrame(columns, index=get_dates(first))
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        return frame

    def __repr__(self) -> str:
        return (
            f"StochasticSimulationResult(replicas={self.replicas}, "
            f"variables={tuple(self)!r})"
        )


@dataclass(frozen=True, slots=True)
class _PreparedDisturbance:
    definition: StochasticDisturbance
    periods: tuple[YearPeriod, ...]
    indexes: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class _IndependentSimulationTask:
    """Pickleable configuration shared by independent FULLNEWTON workers."""

    bound: BoundModel
    coefficients: Mapping[str, Mapping[str, float]]
    time_range: MdlTimeRange | tuple[int, int, int, int]
    prepared: Mapping[str, _PreparedDisturbance]
    periods: tuple[YearPeriod, ...]
    simulation_type: str
    algorithm: str
    convergence: float
    max_iterations: int
    jacobian_step: float
    zero_error_autocorrelation: bool
    constant_adjustments: Mapping[str, AdjustmentValue]
    exogenize: str | Sequence[str] | Mapping[str, ExogenizationValue] | None
    rescheck_equations: str | Sequence[str] | None
    backfill: int
    jacobian_drop: str | Sequence[str] | None
    newton_workspace: NewtonWorkspace | None


_WORKER_TASK: _IndependentSimulationTask | None = None


def _stochastic_simulate(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    disturbances: Mapping[str, StochasticDisturbance] | None = None,
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
    _shared_convergence: bool = False,
    _retain_final_iteration: bool = False,
) -> StochasticSimulationResult:
    """Run repeated deterministic simulations with stochastic disturbances.

    Parameters
    ----------
    model, data, coefficients, time_range
        See :func:`bimets.simulate`.
    disturbances : mapping of str to StochasticDisturbance, optional
        Perturbations keyed by endogenous or exogenous model variable.
    replicas : int, default=100
        Number of perturbed model solutions.
    seed : int, optional
        Seed for R-compatible Mersenne Twister draws. With ``None``, an
        independent NumPy generator is used.
    workers : int, default=1
        Processes used for independent ``FULLNEWTON`` realizations. Shared
        Gauss-Seidel and Newton paths require the default single process.
    simulation_type, algorithm, convergence, max_iterations, jacobian_step
        Deterministic solver settings forwarded to :func:`bimets.simulate`.
    zero_error_autocorrelation, constant_adjustments, exogenize,
    rescheck_equations
        Additional deterministic simulation settings.
    backfill : int, default=0
        Historical observations prepended to the deterministic baseline.
    jacobian_drop : str or sequence of str, optional
        Feedback variables excluded from each Newton Jacobian.

    Returns
    -------
    StochasticSimulationResult
        Baseline, realization matrices, period-wise means and sample standard
        deviations, and disturbed instrument matrices.

    Notes
    -----
    Disturbances are independent and identically distributed across periods
    and replicas, including for equations declared with ``ERROR> AUTO(n)``.
    This matches the documented behavior of BIMETS R. Random values for the
    same explicit numeric seed match R's default Mersenne Twister, inversion
    normal generator, distribution order, and column-major matrix filling.
    Omitting the seed intentionally uses an independent NumPy generator.
    Gauss-Seidel and ``NEWTON`` realizations are evaluated as synchronized
    matrix columns for backward- and forward-looking models. With
    ``algorithm="NEWTON"``, sparse Jacobian factorizations built for the
    unperturbed baseline are reused for a multi-right-hand-side solve,
    following BIMETS R's matrix-column strategy. Reduced ``jacobian_drop``
    systems alternate vectorized Gauss-Seidel equations with shared Newton.
    Slow common-Jacobian convergence triggers column-specific relaxation.
    With ``algorithm="FULLNEWTON"``, the baseline and every realization build
    independent Jacobians; setting ``workers`` above one distributes only
    those independent realizations across processes.

    Examples
    --------
    >>> from bimets import BimetsModel, StochasticDisturbance, timeseries
    >>> model = BimetsModel.from_text("MODEL\\nIDENTITY> y\\nEQ> y=x\\nEND")
    >>> result = stochastic_simulate(
    ...     model,
    ...     {"y": timeseries([0, 0]), "x": timeseries([1, 1])},
    ...     coefficients={},
    ...     time_range=(2000, 1, 2001, 1),
    ...     disturbances={"x": StochasticDisturbance("UNIFORM", (-1, 1))},
    ...     replicas=3,
    ...     seed=7,
    ... )
    >>> result["y"].realizations.shape
    (2, 3)
    """
    bound = _resolve_bound_model(model, data)
    if isinstance(replicas, bool) or not isinstance(replicas, int):
        raise TypeError("replicas must be an integer")
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    start, end = _simulation_bounds(time_range, bound.freq)
    periods = tuple(_periods(start, end, bound.freq))
    definitions = dict(disturbances or {})
    unknown = set(definitions).difference(
        set(bound.model.endogenous).union(bound.model.exogenous)
    )
    if unknown:
        raise KeyError(f"unknown disturbed model variables: {sorted(unknown)}")
    if any(
        not isinstance(value, StochasticDisturbance) for value in definitions.values()
    ):
        raise TypeError("disturbances must contain StochasticDisturbance values")

    normalized_algorithm = algorithm.upper()
    if workers > 1 and normalized_algorithm != "FULLNEWTON":
        raise ValueError("workers above one are supported only with FULLNEWTON")
    newton_workspace = NewtonWorkspace() if normalized_algorithm == "NEWTON" else None
    baseline = _simulate(
        bound,
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
        newton_workspace=newton_workspace,
        allow_full_newton=True,
    )
    if newton_workspace is not None:
        newton_workspace.freeze()
    rng: RMersenneTwister | np.random.Generator = (
        RMersenneTwister(seed) if seed is not None else np.random.default_rng()
    )
    prepared = _prepare_disturbances(definitions, periods, replicas, rng, bound.freq)
    instrument_baseline, instrument_realizations = _instrument_values(
        bound,
        prepared,
        periods,
        replicas,
        constant_adjustments or {},
    )
    realization_values: dict[str, np.ndarray] = {
        name: np.empty((len(periods), replicas), dtype=float) for name in baseline
    }
    normalized_type = simulation_type.upper()
    use_columns = (
        _shared_convergence
        and normalized_algorithm in {"GAUSS-SEIDEL", "NEWTON"}
        and (
            normalized_type == "RESCHECK"
            or (
                not bound.model.forward_looking
                and normalized_type in {"DYNAMIC", "STATIC", "FORECAST"}
            )
            or (bound.model.forward_looking and normalized_type == "DYNAMIC")
        )
    )
    if use_columns:
        try:
            realization_values = simulate_shared_columns(
                bound,
                coefficients=coefficients,
                periods=periods,
                instrument_realizations=instrument_realizations,
                replicas=replicas,
                simulation_type=simulation_type,
                algorithm=algorithm,
                convergence=convergence,
                max_iterations=max_iterations,
                jacobian_step=jacobian_step,
                zero_error_autocorrelation=zero_error_autocorrelation,
                constant_adjustments=constant_adjustments or {},
                exogenize=exogenize,
                rescheck_equations=tuple(baseline),
                newton_workspace=newton_workspace,
                jacobian_drop=jacobian_drop,
                validated_jacobian_drop=_shared_jacobian_drop(bound, jacobian_drop),
                retain_final_iteration=_retain_final_iteration,
            )
        except (IndexError, KeyError, ValueError, SimulationConvergenceError) as error:
            raise StochasticSimulationError(
                f"shared-column stochastic simulation failed: {error}"
            ) from error
    else:
        _simulate_independent_realizations(
            bound,
            coefficients,
            time_range,
            prepared,
            periods,
            replicas,
            simulation_type,
            algorithm,
            convergence,
            max_iterations,
            jacobian_step,
            zero_error_autocorrelation,
            constant_adjustments,
            exogenize,
            rescheck_equations,
            backfill,
            jacobian_drop,
            newton_workspace,
            realization_values,
            workers,
        )

    series_results: dict[str, StochasticSeriesResult] = {}
    for name, values in realization_values.items():
        mean = BimetsSeries(np.mean(values, axis=1), start=start, freq=bound.freq)
        standard_deviation = BimetsSeries(
            np.std(values, axis=1, ddof=1)
            if replicas > 1
            else np.full(len(periods), np.nan),
            start=start,
            freq=bound.freq,
        )
        series_results[name] = StochasticSeriesResult(mean, standard_deviation, values)
    return StochasticSimulationResult(
        baseline,
        series_results,
        instrument_baseline=instrument_baseline,
        instrument_realizations=instrument_realizations,
        replicas=replicas,
        seed=seed,
    )


def _shared_jacobian_drop(
    bound: BoundModel,
    values: str | Sequence[str] | None,
) -> frozenset[str]:
    """Select already-validated dropped feedback names without warning twice."""
    if values is None:
        return frozenset()
    names = (values,) if isinstance(values, str) else tuple(values)
    feedback = (
        set(bound.model.endogenous)
        if bound.model.forward_looking
        else {
            name
            for block in _simulation_blocks(bound.model)
            if block.simultaneous
            for name in block.feedback
        }
    )
    return frozenset(name for name in names if name in feedback)


def _simulate_independent_realizations(
    bound: BoundModel,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    prepared: Mapping[str, _PreparedDisturbance],
    periods: tuple[YearPeriod, ...],
    replicas: int,
    simulation_type: str,
    algorithm: str,
    convergence: float,
    max_iterations: int,
    jacobian_step: float,
    zero_error_autocorrelation: bool,
    constant_adjustments: Mapping[str, AdjustmentValue] | None,
    exogenize: str | Sequence[str] | Mapping[str, ExogenizationValue] | None,
    rescheck_equations: str | Sequence[str] | None,
    backfill: int,
    jacobian_drop: str | Sequence[str] | None,
    newton_workspace: NewtonWorkspace | None,
    realization_values: dict[str, np.ndarray],
    workers: int,
) -> None:
    """Run isolated replicas sequentially or in explicit FULLNEWTON workers."""
    task = _IndependentSimulationTask(
        bound,
        {
            equation: dict(values)
            for equation, values in _coefficient_mapping(coefficients).items()
        },
        time_range,
        dict(prepared),
        periods,
        simulation_type,
        algorithm,
        convergence,
        max_iterations,
        jacobian_step,
        zero_error_autocorrelation,
        constant_adjustments or {},
        exogenize,
        rescheck_equations,
        backfill,
        jacobian_drop,
        newton_workspace,
    )
    results: Iterator[tuple[int, dict[str, np.ndarray]]]
    executor: ProcessPoolExecutor | None = None
    if workers == 1:
        results = (
            _simulate_independent_replica(task, replica) for replica in range(replicas)
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context(
                "fork" if "fork" in get_all_start_methods() else "spawn"
            ),
            initializer=_initialize_independent_worker,
            initargs=(task,),
        )
        results = executor.map(
            _simulate_independent_worker, range(replicas), chunksize=1
        )
    try:
        for replica, values in results:
            for name in realization_values:
                realization_values[name][:, replica] = values[name]
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)


def _initialize_independent_worker(task: _IndependentSimulationTask) -> None:
    """Install immutable FULLNEWTON configuration once in a child process."""
    global _WORKER_TASK
    _WORKER_TASK = task


def _simulate_independent_worker(
    replica: int,
) -> tuple[int, dict[str, np.ndarray]]:
    """Execute one replica using process-local initialized configuration."""
    if _WORKER_TASK is None:
        raise RuntimeError("independent simulation worker is not initialized")
    return _simulate_independent_replica(_WORKER_TASK, replica)


def _simulate_independent_replica(
    task: _IndependentSimulationTask,
    replica: int,
) -> tuple[int, dict[str, np.ndarray]]:
    """Solve one independent replica and return endogenous numeric paths."""
    replica_data = _perturbed_data(task.bound, task.prepared, replica)
    replica_adjustments = _perturbed_adjustments(
        task.prepared,
        replica,
        task.periods,
        task.bound.model.endogenous,
        task.bound.freq,
        task.constant_adjustments,
    )
    try:
        solution = _simulate(
            task.bound.model,
            replica_data,
            coefficients=task.coefficients,
            time_range=task.time_range,
            simulation_type=task.simulation_type,
            algorithm=task.algorithm,
            convergence=task.convergence,
            max_iterations=task.max_iterations,
            jacobian_step=task.jacobian_step,
            zero_error_autocorrelation=task.zero_error_autocorrelation,
            constant_adjustments=replica_adjustments,
            exogenize=task.exogenize,
            rescheck_equations=task.rescheck_equations,
            backfill=task.backfill,
            jacobian_drop=task.jacobian_drop,
            newton_workspace=task.newton_workspace,
            allow_full_newton=True,
        )
    except (IndexError, KeyError, ValueError, SimulationConvergenceError) as error:
        raise StochasticSimulationError(
            f"stochastic realization {replica + 1} failed: {error}"
        ) from error
    values = {
        name: np.asarray(
            [
                solution[name].at_period(period.year, period.period)
                for period in task.periods
            ]
        )
        for name in solution
    }
    return replica, values


def stochastic_simulate(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    disturbances: Mapping[str, StochasticDisturbance] | None = None,
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
) -> StochasticSimulationResult:
    """Run stochastic simulation using shared columns when semantically safe."""
    return _stochastic_simulate(
        model,
        data,
        coefficients=coefficients,
        time_range=time_range,
        disturbances=disturbances,
        replicas=replicas,
        seed=seed,
        workers=workers,
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
        _shared_convergence=True,
    )


# Keep the complete public NumPy-style documentation next to the implementation
# while presenting a signature without optimization-only controls.
stochastic_simulate.__doc__ = _stochastic_simulate.__doc__


def _prepare_disturbances(
    definitions: Mapping[str, StochasticDisturbance],
    periods: tuple[YearPeriod, ...],
    replicas: int,
    rng: RMersenneTwister | np.random.Generator,
    freq: Frequency,
) -> Mapping[str, _PreparedDisturbance]:
    """Prepare disturbances for internal processing."""
    period_ordinals = [period.ordinal(freq) for period in periods]
    prepared: dict[str, _PreparedDisturbance] = {}
    for name, definition in definitions.items():
        if definition.time_range is None:
            first, last = periods[0], periods[-1]
        else:
            first, last = _simulation_bounds(definition.time_range, freq)
        indexes = tuple(
            index
            for index, ordinal in enumerate(period_ordinals)
            if first.ordinal(freq) <= ordinal <= last.ordinal(freq)
        )
        active_periods = tuple(periods[index] for index in indexes)
        if definition.distribution == "MATRIX":
            values = np.asarray(definition.parameters, dtype=float)
            expected = (len(indexes), replicas)
            if values.shape != expected:
                raise ValueError(
                    f"MATRIX disturbance {name!r} must have shape {expected}"
                )
        else:
            first_parameter, second_parameter = definition.parameters
            size = (len(indexes), replicas)
            if definition.distribution == "NORMAL":
                if isinstance(rng, RMersenneTwister):
                    values = rng.normal(first_parameter, second_parameter, size)
                else:
                    values = rng.normal(first_parameter, second_parameter, size=size)
            else:
                if isinstance(rng, RMersenneTwister):
                    values = rng.uniform(first_parameter, second_parameter, size)
                else:
                    values = rng.uniform(first_parameter, second_parameter, size=size)
        prepared[name] = _PreparedDisturbance(
            definition, active_periods, indexes, np.asarray(values, dtype=float)
        )
    return MappingProxyType(prepared)


def _instrument_values(
    bound: BoundModel,
    prepared: Mapping[str, _PreparedDisturbance],
    periods: tuple[YearPeriod, ...],
    replicas: int,
    adjustments: Mapping[str, AdjustmentValue],
) -> tuple[dict[str, BimetsSeries], dict[str, np.ndarray]]:
    """Collect instrument values over the requested periods."""
    baseline: dict[str, BimetsSeries] = {}
    realizations: dict[str, np.ndarray] = {}
    endogenous = set(bound.model.endogenous)
    for name, disturbance in prepared.items():
        if name in endogenous:
            base_values = np.asarray(
                [
                    _adjustment_value(adjustments.get(name, 0.0), period)
                    for period in periods
                ]
            )
            values = np.repeat(base_values[:, None], replicas, axis=1)
            values[np.asarray(disturbance.indexes, dtype=int), :] += disturbance.values
        else:
            base_values = np.asarray(
                [
                    bound.data[name].at_period(period.year, period.period)
                    for period in periods
                ]
            )
            values = np.repeat(base_values[:, None], replicas, axis=1)
            indexes = np.asarray(disturbance.indexes, dtype=int)
            if disturbance.definition.distribution == "MATRIX":
                values[indexes, :] = disturbance.values
            else:
                values[indexes, :] += disturbance.values
        baseline[name] = BimetsSeries(base_values, start=periods[0], freq=bound.freq)
        realizations[name] = values
    return baseline, realizations


def _perturbed_data(
    bound: BoundModel,
    prepared: Mapping[str, _PreparedDisturbance],
    replica: int,
) -> dict[str, BimetsSeries]:
    """Apply one realization of stochastic shocks to model data."""
    data = dict(bound.data)
    endogenous = set(bound.model.endogenous)
    for name, disturbance in prepared.items():
        if name in endogenous:
            continue
        series = data[name]
        values = series.values.copy()
        for row, period in enumerate(disturbance.periods):
            position = period.ordinal(series.freq) - series.start.ordinal(series.freq)
            if position < 0 or position >= len(series):
                raise ValueError(
                    f"disturbance range for {name!r} exceeds its data range"
                )
            perturbation = disturbance.values[row, replica]
            if disturbance.definition.distribution == "MATRIX":
                values[position] = perturbation
            else:
                values[position] += perturbation
        data[name] = BimetsSeries(
            values,
            start=series.start,
            freq=series.freq,
            metadata=series.metadata,
        )
    return data


def _perturbed_adjustments(
    prepared: Mapping[str, _PreparedDisturbance],
    replica: int,
    periods: tuple[YearPeriod, ...],
    endogenous: Sequence[str],
    freq: Frequency,
    adjustments: Mapping[str, AdjustmentValue],
) -> dict[str, AdjustmentValue]:
    """Apply one realization of stochastic shocks to adjustments."""
    output: dict[str, AdjustmentValue] = dict(adjustments)
    endogenous_set = set(endogenous)
    for name, disturbance in prepared.items():
        if name not in endogenous_set:
            continue
        values = np.asarray(
            [
                _adjustment_value(adjustments.get(name, 0.0), period)
                for period in periods
            ]
        )
        values[np.asarray(disturbance.indexes, dtype=int)] += disturbance.values[
            :, replica
        ]
        output[name] = BimetsSeries(values, start=periods[0], freq=freq)
    return output
