"""Impact and interim multiplier matrices for MDL models."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bimets.mdl._binding import BoundModel
from bimets.mdl._model import BimetsModel, MdlTimeRange
from bimets.mdl._simulation import (
    AdjustmentValue,
    CoefficientInput,
    ExogenizationValue,
    SimulationResult,
    _adjustment_value,
    _periods,
    _resolve_bound_model,
    _simulation_bounds,
)
from bimets.mdl._stochastic import (
    StochasticDisturbance,
    StochasticSimulationError,
    _stochastic_simulate,
)
from bimets.timeseries import BimetsDataset, BimetsSeries, YearPeriod


class MultiplierMatrixError(RuntimeError):
    """Raised when a shocked simulation cannot be solved."""


@dataclass(frozen=True, slots=True)
class MultiplierMatrixResult:
    """Immutable impact or interim multiplier matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix shaped ``(periods * targets, periods * instruments)``.
    targets, instruments : tuple of str
        Selected variables in user order.
    periods : tuple of YearPeriod
        Simulation periods in chronological order.
    simulation_type : {"STATIC", "DYNAMIC", "FORECAST"}
        Static impact or dynamic/forecast interim calculation.
    shock : float
        Relative/absolute finite-difference shock setting.
    baseline : SimulationResult
        Unperturbed deterministic simulation.
    """

    matrix: np.ndarray
    targets: tuple[str, ...]
    instruments: tuple[str, ...]
    periods: tuple[YearPeriod, ...]
    simulation_type: str
    shock: float
    baseline: SimulationResult

    def __post_init__(self) -> None:
        values = np.asarray(self.matrix, dtype=float).copy()
        expected = (
            len(self.periods) * len(self.targets),
            len(self.periods) * len(self.instruments),
        )
        if values.shape != expected:
            raise ValueError(f"multiplier matrix must have shape {expected}")
        values.setflags(write=False)
        object.__setattr__(self, "matrix", values)

    @property
    def row_labels(self) -> tuple[str, ...]:
        """Labels ordered by period and then target."""
        return tuple(
            f"{target}_{period_index}"
            for period_index in range(1, len(self.periods) + 1)
            for target in self.targets
        )

    @property
    def column_labels(self) -> tuple[str, ...]:
        """Labels ordered by period and then instrument."""
        endogenous = set(self.baseline)
        return tuple(
            (
                f"{instrument}_ADDFACTOR_{period_index}"
                if instrument in endogenous
                else f"{instrument}_{period_index}"
            )
            for period_index in range(1, len(self.periods) + 1)
            for instrument in self.instruments
        )

    def at(
        self,
        target: str,
        target_period: int,
        instrument: str,
        instrument_period: int,
    ) -> float:
        """Return one multiplier using one-based period positions.

        Parameters
        ----------
        target, instrument : str
            Selected target and instrument names.
        target_period, instrument_period : int
            One-based positions in the requested simulation range.
        """
        try:
            row = (target_period - 1) * len(self.targets) + self.targets.index(target)
            column = (instrument_period - 1) * len(
                self.instruments
            ) + self.instruments.index(instrument)
        except (ValueError, IndexError) as error:
            raise KeyError("unknown multiplier target or instrument") from error
        if not 1 <= target_period <= len(self.periods):
            raise IndexError("target_period is outside the multiplier range")
        if not 1 <= instrument_period <= len(self.periods):
            raise IndexError("instrument_period is outside the multiplier range")
        return float(self.matrix[row, column])

    def summary(self) -> pd.DataFrame:
        """Return the multiplier matrix as a labeled pandas DataFrame."""
        return pd.DataFrame(
            self.matrix,
            index=self.row_labels,
            columns=self.column_labels,
        )


def multiplier_matrix(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    coefficients: CoefficientInput,
    time_range: MdlTimeRange | tuple[int, int, int, int],
    targets: str | Sequence[str],
    instruments: str | Sequence[str],
    shock: float = 1e-5,
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
) -> MultiplierMatrixResult:
    """Compute impact or interim multipliers by finite differences.

    Parameters
    ----------
    model, data, coefficients, time_range
        See :func:`bimets.simulate`.
    targets : str or sequence of str
        Endogenous variables whose responses form matrix rows.
    instruments : str or sequence of str
        Exogenous variables, or endogenous add-factors, to shock.
    shock : float, default=1e-5
        Positive BIMETS ``MM_SHOCK`` value. At each impulse period the actual
        increment is ``instrument * shock`` when the baseline instrument is at
        least one, and ``shock`` otherwise.
    simulation_type : {"DYNAMIC", "STATIC", "FORECAST"}, default="DYNAMIC"
        Dynamic/forecast interim or static impact multipliers.
    algorithm, convergence, max_iterations, jacobian_step
        Deterministic solver settings.
    workers : int, default=1
        Processes used only for independent ``FULLNEWTON`` shock columns.
    zero_error_autocorrelation, constant_adjustments, exogenize
        Additional deterministic simulation settings.
    backfill : int, default=0
        Historical observations prepended to the returned baseline. Multiplier
        matrix rows and columns remain restricted to ``time_range``.
    jacobian_drop : str or sequence of str, optional
        Feedback variables excluded from Newton Jacobians.

    Returns
    -------
    MultiplierMatrixResult
        Labeled immutable matrix and the baseline solution.

    Notes
    -----
    Every instrument-period shock is represented by one column of a common
    simulation matrix, matching BIMETS R. Gauss-Seidel uses shared
    convergence; ``NEWTON`` additionally reuses the baseline sparse Jacobian
    and solves all right-hand sides together. Lead and reduced-Jacobian systems
    use the same shared backend. Only ``FULLNEWTON`` retains independent
    per-column solves.

    Examples
    --------
    >>> from bimets import BimetsModel, multiplier_matrix, timeseries
    >>> model = BimetsModel.from_text("MODEL\\nIDENTITY> y\\nEQ> y=2*x\\nEND")
    >>> result = multiplier_matrix(
    ...     model,
    ...     {"y": timeseries([0]), "x": timeseries([3])},
    ...     coefficients={},
    ...     time_range=(2000, 1, 2000, 1),
    ...     targets="y",
    ...     instruments="x",
    ...     simulation_type="STATIC",
    ... )
    >>> result.matrix.round(6).tolist()
    [[2.0]]
    """
    bound = _resolve_bound_model(model, data)
    normalized_type = simulation_type.upper()
    if normalized_type not in {"DYNAMIC", "STATIC", "FORECAST"}:
        raise ValueError("simulation_type must be 'DYNAMIC', 'STATIC', or 'FORECAST'")
    if not math.isfinite(shock) or shock <= 0:
        raise ValueError("shock must be a positive finite number")
    target_names = _names(targets, "targets")
    instrument_names = _names(instruments, "instruments")
    unknown_targets = set(target_names).difference(bound.model.endogenous)
    if unknown_targets:
        raise KeyError(f"unknown endogenous targets: {sorted(unknown_targets)}")
    model_variables = set(bound.model.endogenous).union(bound.model.exogenous)
    unknown_instruments = set(instrument_names).difference(model_variables)
    if unknown_instruments:
        raise KeyError(f"unknown model instruments: {sorted(unknown_instruments)}")
    if exogenize is None:
        exogenized: set[str] = set()
    elif isinstance(exogenize, str):
        exogenized = {exogenize}
    else:
        exogenized = set(exogenize)
    invalid_targets = exogenized.intersection(target_names)
    if invalid_targets:
        raise ValueError(
            f"multiplier targets cannot also be exogenized: {sorted(invalid_targets)}"
        )
    start, end = _simulation_bounds(time_range, bound.freq)
    periods = tuple(_periods(start, end, bound.freq))
    replicas = len(periods) * len(instrument_names)
    endogenous = set(bound.model.endogenous)
    disturbances: dict[str, StochasticDisturbance] = {}
    increments = np.empty(replicas, dtype=float)
    for instrument_index, instrument in enumerate(instrument_names):
        if instrument in endogenous:
            candidate_values = np.zeros((len(periods), replicas), dtype=float)
        else:
            base_path = np.asarray(
                [
                    bound.data[instrument].at_period(period.year, period.period)
                    for period in periods
                ]
            )
            candidate_values = np.repeat(base_path[:, None], replicas, axis=1)
        for period_index, period in enumerate(periods):
            base_instrument = (
                _adjustment_value(
                    (constant_adjustments or {}).get(instrument, 0.0), period
                )
                if instrument in endogenous
                else bound.data[instrument].at_period(period.year, period.period)
            )
            increment = base_instrument * shock if base_instrument >= 1 else shock
            column = period_index * len(instrument_names) + instrument_index
            candidate_values[period_index, column] += increment
            increments[column] = increment
        disturbances[instrument] = StochasticDisturbance(
            "MATRIX",
            candidate_values,
            time_range=time_range,
        )
    try:
        simulations = _stochastic_simulate(
            bound,
            coefficients=coefficients,
            time_range=time_range,
            disturbances=disturbances,
            replicas=replicas,
            simulation_type=normalized_type,
            algorithm=algorithm,
            workers=workers,
            convergence=convergence,
            max_iterations=max_iterations,
            jacobian_step=jacobian_step,
            zero_error_autocorrelation=zero_error_autocorrelation,
            constant_adjustments=constant_adjustments,
            exogenize=exogenize,
            backfill=backfill,
            jacobian_drop=jacobian_drop,
            _shared_convergence=True,
        )
    except StochasticSimulationError as error:
        match = re.search(r"replica (\d+)", str(error))
        if match is not None:
            column = int(match.group(1)) - 1
            period_index, instrument_index = divmod(column, len(instrument_names))
            failed_period = periods[period_index]
            failed_instrument = instrument_names[instrument_index]
            raise MultiplierMatrixError(
                f"simulation for instrument {failed_instrument!r} at "
                f"{failed_period.year}-{failed_period.period} failed: {error}"
            ) from error
        raise MultiplierMatrixError(
            f"shared shock simulation failed: {error}"
        ) from error

    values = np.empty((len(periods) * len(target_names), replicas), dtype=float)
    for response_period_index, period in enumerate(periods):
        for target_index, target in enumerate(target_names):
            row = response_period_index * len(target_names) + target_index
            baseline_value = simulations.baseline[target].at_period(
                period.year, period.period
            )
            difference = (
                simulations[target].realizations[response_period_index, :]
                - baseline_value
            )
            values[row, :] = np.divide(
                difference,
                increments,
                out=np.zeros_like(difference),
                where=difference != 0,
            )
    return MultiplierMatrixResult(
        values,
        target_names,
        instrument_names,
        periods,
        normalized_type,
        shock,
        simulations.baseline,
    )


def _names(values: str | Sequence[str], label: str) -> tuple[str, ...]:
    """Normalize one or more names into a validated tuple."""
    output = (values,) if isinstance(values, str) else tuple(values)
    if not output:
        raise ValueError(f"{label} cannot be empty")
    if any(not isinstance(value, str) or not value for value in output):
        raise TypeError(f"{label} must contain non-empty strings")
    if len(set(output)) != len(output):
        raise ValueError(f"{label} cannot contain duplicates")
    return output
