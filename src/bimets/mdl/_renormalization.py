"""Endogenous targeting and model renormalization for MDL models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from bimets.mdl._binding import BoundModel
from bimets.mdl._model import BimetsModel, MdlTimeRange
from bimets.mdl._multipliers import (
    MultiplierMatrixError,
    MultiplierMatrixResult,
    _names,
    multiplier_matrix,
)
from bimets.mdl._simulation import (
    AdjustmentValue,
    CoefficientInput,
    ExogenizationValue,
    SimulationConvergenceError,
    SimulationResult,
    _adjustment_value,
    _periods,
    _resolve_bound_model,
    _simulation_bounds,
)
from bimets.timeseries import BimetsDataset, BimetsSeries, YearPeriod


class RenormalizationError(RuntimeError):
    """Raised when the endogenous-targeting system cannot be solved."""


@dataclass(frozen=True, slots=True)
class RenormalizationResult:
    """Immutable result of an endogenous-targeting operation.

    Parameters
    ----------
    instruments : mapping of str to BimetsSeries
        Instrument values over the targeting range. Endogenous names contain
        their calculated constant adjustments.
    desired_targets, achieved_targets : BimetsDataset
        Requested and simulated endogenous paths.
    data : BimetsDataset
        Complete model dataset with exogenous instruments replaced.
    constant_adjustments : mapping of str to float or BimetsSeries
        Adjustments with endogenous instruments replaced.
    simulation : SimulationResult
        Deterministic solution at the final iteration.
    multiplier_matrix : MultiplierMatrixResult
        Multiplier matrix calculated at the final iteration.
    converged : bool
        Whether all target norms are below the requested threshold.
    iterations : int
        Number of instrument corrections performed.
    unconverged_targets : tuple of str
        Target names still outside the convergence threshold.
    convergence : float
        Requested targeting convergence threshold.
    """

    instruments: Mapping[str, BimetsSeries]
    desired_targets: BimetsDataset
    achieved_targets: BimetsDataset
    data: BimetsDataset
    constant_adjustments: Mapping[str, AdjustmentValue]
    simulation: SimulationResult
    multiplier_matrix: MultiplierMatrixResult
    converged: bool
    iterations: int
    unconverged_targets: tuple[str, ...]
    convergence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instruments", MappingProxyType(dict(self.instruments))
        )
        object.__setattr__(
            self,
            "constant_adjustments",
            MappingProxyType(dict(self.constant_adjustments)),
        )

    @property
    def targets(self) -> BimetsDataset:
        """Achieved target paths, matching the BIMETS R result name."""
        return self.achieved_targets

    def summary(self) -> pd.DataFrame:
        """Return desired, achieved, and absolute target errors.

        Returns
        -------
        pandas.DataFrame
            Columns use a two-level ``(target, measure)`` index.
        """
        values: dict[tuple[str, str], np.ndarray] = {}
        for name in self.desired_targets:
            desired = self.desired_targets[name].values
            achieved = self.achieved_targets[name].values
            values[(name, "desired")] = desired
            values[(name, "achieved")] = achieved
            values[(name, "error")] = achieved - desired
        first = next(iter(self.desired_targets.values()))
        index = [first.period_at(position) for position in range(len(first))]
        frame = pd.DataFrame(values, index=index)
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        return frame


def renormalize(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    targets: Mapping[str, BimetsSeries],
    instruments: str | Sequence[str],
    simulation_type: str = "DYNAMIC",
    algorithm: str = "GAUSS-SEIDEL",
    workers: int = 1,
    convergence: float = 0.01,
    max_iterations: int = 100,
    jacobian_step: float = 1e-4,
    zero_error_autocorrelation: bool = False,
    constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
    exogenize: (str | Sequence[str] | Mapping[str, ExogenizationValue] | None) = None,
    backfill: int = 0,
    jacobian_drop: str | Sequence[str] | None = None,
    renormalization_iterations: int = 10,
    renormalization_convergence: float = 1e-4,
    shock: float = 1e-5,
    matrix_tolerance: float = 1e-12,
) -> RenormalizationResult:
    """Find instrument paths that achieve selected endogenous targets.

    Parameters
    ----------
    model, data, coefficients, time_range
        See :func:`bimets.simulate`.
    targets : mapping of str to BimetsSeries
        Desired paths for endogenous variables. Values are projected onto the
        complete targeting range and must be finite there.
    instruments : str or sequence of str
        Equally many exogenous variables or endogenous add-factors to adjust.
    simulation_type, algorithm, convergence, max_iterations, jacobian_step
        Settings forwarded to deterministic simulation and multiplier runs.
    workers : int, default=1
        Processes used only for independent ``FULLNEWTON`` multiplier shocks.
    zero_error_autocorrelation, constant_adjustments, exogenize
        Additional deterministic simulation settings.
    backfill : int, default=0
        Historical observations prepended to simulations returned in the
        result. Target comparisons remain restricted to ``time_range``.
    jacobian_drop : str or sequence of str, optional
        Feedback variables excluded from Newton Jacobians.
    renormalization_iterations : int, default=10
        Maximum number of instrument-correction steps.
    renormalization_convergence : float, default=1e-4
        Maximum Euclidean relative-error norm for every target. Absolute
        errors are used at observations whose desired target is zero.
    shock : float, default=1e-5
        Finite-difference shock used for multiplier matrices.
    matrix_tolerance : float, default=1e-12
        Relative singular-value threshold used before solving each multiplier
        system.

    Returns
    -------
    RenormalizationResult
        Final instruments, achieved targets, adjusted inputs, and diagnostics.

    Raises
    ------
    RenormalizationError
        If a multiplier calculation fails or its square matrix is singular.

    Notes
    -----
    This is the Python counterpart of BIMETS R ``RENORM``. The model and input
    data remain unchanged; adjusted data and constant adjustments are returned
    explicitly. A non-converged iteration limit emits a ``RuntimeWarning`` as
    in R and returns the final instruments with ``converged=False`` and full
    diagnostics.

    Examples
    --------
    >>> from bimets import BimetsModel, renormalize, timeseries
    >>> model = BimetsModel.from_text("MODEL\\nIDENTITY> y\\nEQ> y=2*x\\nEND")
    >>> result = renormalize(
    ...     model,
    ...     {"y": timeseries([0]), "x": timeseries([1])},
    ...     coefficients={},
    ...     time_range=(2000, 1, 2000, 1),
    ...     targets={"y": timeseries([10])},
    ...     instruments="x",
    ...     simulation_type="STATIC",
    ... )
    >>> result.converged, result.instruments["x"].values.round(6).tolist()
    (True, [5.0])
    """
    bound = _resolve_bound_model(model, data)
    target_names, desired_targets = _validate_targets(bound, targets, time_range)
    instrument_names = _names(instruments, "instruments")
    if len(target_names) != len(instrument_names):
        raise ValueError("targets and instruments must have the same length")
    model_variables = set(bound.model.endogenous).union(bound.model.exogenous)
    unknown_instruments = set(instrument_names).difference(model_variables)
    if unknown_instruments:
        raise KeyError(f"unknown model instruments: {sorted(unknown_instruments)}")
    if _exogenized_names(exogenize).intersection(target_names):
        raise ValueError("target variables cannot also be exogenized")
    if (
        not isinstance(renormalization_iterations, int)
        or isinstance(renormalization_iterations, bool)
        or renormalization_iterations <= 0
    ):
        raise ValueError("renormalization_iterations must be a positive integer")
    for value, label in (
        (renormalization_convergence, "renormalization_convergence"),
        (matrix_tolerance, "matrix_tolerance"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")

    start, end = _simulation_bounds(time_range, bound.freq)
    periods = tuple(_periods(start, end, bound.freq))
    current_data = dict(bound.data)
    current_adjustments: dict[str, AdjustmentValue] = dict(constant_adjustments or {})
    endogenous = set(bound.model.endogenous)
    for name in instrument_names:
        if name in endogenous:
            current_adjustments[name] = _full_adjustment_series(
                bound.data[name], current_adjustments.get(name, 0.0)
            )

    final_matrix: MultiplierMatrixResult | None = None
    unconverged = target_names
    for iteration in range(renormalization_iterations + 1):
        try:
            final_matrix = multiplier_matrix(
                bound.model,
                current_data,
                coefficients=coefficients,
                time_range=time_range,
                targets=target_names,
                instruments=instrument_names,
                shock=shock,
                simulation_type=simulation_type,
                algorithm=algorithm,
                workers=workers,
                convergence=convergence,
                max_iterations=max_iterations,
                jacobian_step=jacobian_step,
                zero_error_autocorrelation=zero_error_autocorrelation,
                constant_adjustments=current_adjustments,
                exogenize=exogenize,
                backfill=backfill,
                jacobian_drop=jacobian_drop,
            )
        except (MultiplierMatrixError, SimulationConvergenceError) as error:
            raise RenormalizationError(
                f"multiplier calculation failed at iteration {iteration + 1}: {error}"
            ) from error

        simulation = final_matrix.baseline
        unconverged = _unconverged_targets(
            desired_targets,
            simulation,
            target_names,
            renormalization_convergence,
        )
        if not unconverged or iteration == renormalization_iterations:
            if unconverged and iteration == renormalization_iterations:
                warnings.warn(
                    f"RENORM did not converge in {iteration} iterations; "
                    f"unconverged targets: {unconverged!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return _result(
                bound,
                current_data,
                current_adjustments,
                desired_targets,
                target_names,
                instrument_names,
                periods,
                simulation,
                final_matrix,
                iteration,
                unconverged,
                renormalization_convergence,
            )

        matrix = final_matrix.matrix
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if (
            singular_values.size == 0
            or singular_values[-1] <= singular_values[0] * matrix_tolerance
        ):
            raise RenormalizationError(
                f"multiplier matrix is singular at iteration {iteration + 1}"
            )
        desired_vector = _period_major_values(desired_targets, target_names, periods)
        achieved_vector = _period_major_values(simulation, target_names, periods)
        try:
            correction = np.linalg.solve(matrix, desired_vector - achieved_vector)
        except np.linalg.LinAlgError as error:
            raise RenormalizationError(
                f"cannot solve multiplier matrix at iteration {iteration + 1}"
            ) from error
        current_values = _instrument_values(
            current_data,
            current_adjustments,
            instrument_names,
            periods,
            endogenous,
        )
        adjusted_values = current_values + correction
        for instrument_index, name in enumerate(instrument_names):
            values = adjusted_values[instrument_index :: len(instrument_names)]
            if name in endogenous:
                adjustment = current_adjustments[name]
                assert isinstance(adjustment, BimetsSeries)
                current_adjustments[name] = _replace_periods(
                    adjustment, periods, values
                )
            else:
                current_data[name] = _replace_periods(
                    current_data[name], periods, values
                )

    raise AssertionError("renormalization loop terminated unexpectedly")


def _validate_targets(
    bound: BoundModel,
    targets: Mapping[str, BimetsSeries],
    time_range: MdlTimeRange | tuple[int, int, int, int],
) -> tuple[tuple[str, ...], BimetsDataset]:
    """Validate targets for internal processing."""
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must be a non-empty mapping")
    names = tuple(targets)
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("target names must be non-empty strings")
    unknown = set(names).difference(bound.model.endogenous)
    if unknown:
        raise KeyError(f"unknown endogenous targets: {sorted(unknown)}")
    start, end = _simulation_bounds(time_range, bound.freq)
    projected: dict[str, BimetsSeries] = {}
    for name, series in targets.items():
        if not isinstance(series, BimetsSeries):
            raise TypeError("target values must be BimetsSeries objects")
        if series.freq != bound.freq:
            raise ValueError(f"target {name!r} has a different frequency")
        value = series.project(start, end, extend=True)
        if not np.all(np.isfinite(value.values)):
            raise ValueError(f"target {name!r} is undefined in the targeting range")
        projected[name] = value
    return names, BimetsDataset(projected)


def _exogenized_names(
    exogenize: str | Sequence[str] | Mapping[str, ExogenizationValue] | None,
) -> set[str]:
    """Return the variable names exogenized during renormalization."""
    if exogenize is None:
        return set()
    if isinstance(exogenize, str):
        return {exogenize}
    return set(exogenize)


def _full_adjustment_series(
    reference: BimetsSeries, adjustment: AdjustmentValue
) -> BimetsSeries:
    """Expand an adjustment value across the full data range."""
    return BimetsSeries(
        [
            _adjustment_value(adjustment, reference.period_at(index))
            for index in range(len(reference))
        ],
        start=reference.start,
        freq=reference.freq,
    )


def _period_major_values(
    series: Mapping[str, BimetsSeries],
    names: Sequence[str],
    periods: Sequence[YearPeriod],
) -> np.ndarray:
    """Arrange target values in period-major order."""
    return np.asarray(
        [
            series[name].at_period(period.year, period.period)
            for period in periods
            for name in names
        ],
        dtype=float,
    )


def _unconverged_targets(
    desired: BimetsDataset,
    simulation: SimulationResult,
    names: tuple[str, ...],
    convergence: float,
) -> tuple[str, ...]:
    """Return target names that have not converged."""
    output: list[str] = []
    for name in names:
        target = desired[name].values
        achieved = (
            simulation[name].project(desired[name].start, desired[name].end).values
        )
        difference = target - achieved
        relative = np.divide(
            difference,
            target,
            out=difference.copy(),
            where=target != 0,
        )
        if np.linalg.norm(relative) >= convergence:
            output.append(name)
    return tuple(output)


def _instrument_values(
    data: Mapping[str, BimetsSeries],
    adjustments: Mapping[str, AdjustmentValue],
    instruments: tuple[str, ...],
    periods: tuple[YearPeriod, ...],
    endogenous: set[str],
) -> np.ndarray:
    """Collect instrument values over the requested periods."""
    return np.asarray(
        [
            (
                _adjustment_value(adjustments[name], period)
                if name in endogenous
                else data[name].at_period(period.year, period.period)
            )
            for period in periods
            for name in instruments
        ],
        dtype=float,
    )


def _replace_periods(
    series: BimetsSeries,
    periods: tuple[YearPeriod, ...],
    values: np.ndarray,
) -> BimetsSeries:
    """Replace periods for internal processing."""
    updated = series.values.copy()
    start = series.start.ordinal(series.freq)
    for period, value in zip(periods, values, strict=True):
        position = period.ordinal(series.freq) - start
        if position < 0 or position >= len(series):
            raise ValueError("instrument period exceeds its data range")
        updated[position] = value
    return BimetsSeries(
        updated,
        start=series.start,
        freq=series.freq,
        metadata=series.metadata,
    )


def _result(
    bound: BoundModel,
    data: Mapping[str, BimetsSeries],
    adjustments: Mapping[str, AdjustmentValue],
    desired: BimetsDataset,
    target_names: tuple[str, ...],
    instrument_names: tuple[str, ...],
    periods: tuple[YearPeriod, ...],
    simulation: SimulationResult,
    matrix: MultiplierMatrixResult,
    iterations: int,
    unconverged: tuple[str, ...],
    convergence: float,
) -> RenormalizationResult:
    """Assemble the renormalization result object."""
    endogenous = set(bound.model.endogenous)
    instrument_series: dict[str, BimetsSeries] = {}
    for name in instrument_names:
        values = [
            (
                _adjustment_value(adjustments[name], period)
                if name in endogenous
                else data[name].at_period(period.year, period.period)
            )
            for period in periods
        ]
        instrument_series[name] = BimetsSeries(
            values, start=periods[0], freq=bound.freq
        )
    achieved = BimetsDataset(
        {
            name: simulation[name].project(desired[name].start, desired[name].end)
            for name in target_names
        }
    )
    return RenormalizationResult(
        instruments=instrument_series,
        desired_targets=desired,
        achieved_targets=achieved,
        data=BimetsDataset(data),
        constant_adjustments=adjustments,
        simulation=simulation,
        multiplier_matrix=matrix,
        converged=not unconverged,
        iterations=iterations,
        unconverged_targets=unconverged,
        convergence=convergence,
    )
