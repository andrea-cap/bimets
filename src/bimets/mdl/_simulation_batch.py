"""Column-oriented simulation for synchronized stochastic replicas."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import numpy as np

from bimets.mdl._binding import BoundModel
from bimets.mdl._expression import MdlExpression
from bimets.mdl._model import BehavioralEquation, IdentityEquation, MdlEquation
from bimets.mdl._simulation import (
    AdjustmentValue,
    CoefficientInput,
    ExogenizationValue,
    SimulationConvergenceError,
    _adjustment_value,
    _behavioral_execution_plan,
    _coefficient_mapping,
    _compile_expression,
    _ExogenizationRule,
    _forward_jacobian_pattern,
    _periods,
    _simulation_blocks,
    _validate_exogenization,
    _validate_jacobian_drop,
    _warn_empty_newton_feedback,
)
from bimets.mdl._sparse import (
    NewtonWorkspace,
    SparseFactorizationError,
    SparseJacobian,
    factorize_finite_difference_jacobian,
)
from bimets.timeseries import YearPeriod


def simulate_shared_columns(
    bound: BoundModel,
    *,
    coefficients: CoefficientInput,
    periods: tuple[YearPeriod, ...],
    instrument_realizations: Mapping[str, np.ndarray],
    replicas: int,
    simulation_type: str,
    algorithm: str = "GAUSS-SEIDEL",
    convergence: float,
    max_iterations: int,
    jacobian_step: float = 1e-4,
    zero_error_autocorrelation: bool,
    constant_adjustments: Mapping[str, AdjustmentValue],
    exogenize: (str | Sequence[str] | Mapping[str, ExogenizationValue] | None) = None,
    rescheck_equations: tuple[str, ...] | None = None,
    newton_workspace: NewtonWorkspace | None = None,
    jacobian_drop: str | Sequence[str] | None = None,
    validated_jacobian_drop: frozenset[str] | None = None,
    retain_final_iteration: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate a baseline and replicas as synchronized matrix columns.

    Column zero is the unperturbed baseline and the remaining columns are
    stochastic candidates. Convergence is accepted only when every feedback
    value in every column satisfies the requested percentage threshold. This
    mirrors BIMETS R's internal matrix execution while keeping expression
    evaluation in vectorized NumPy operations.
    """
    if not periods:
        raise ValueError("at least one simulation period is required")
    normalized_type = simulation_type.upper()
    normalized_algorithm = algorithm.upper()
    if normalized_algorithm not in {"GAUSS-SEIDEL", "NEWTON"}:
        raise ValueError("shared columns support GAUSS-SEIDEL or NEWTON")
    columns = replicas + 1
    endogenous = set(bound.model.endogenous)
    model_names = tuple(
        dict.fromkeys((*bound.model.endogenous, *bound.model.exogenous))
    )
    model_data = {name: bound.data[name] for name in model_names}
    # Shared matrices need only the simulation window plus structural boundary
    # observations. Retaining complete historical/future input ranges would
    # multiply decades of unused data by every stochastic replica (notably in
    # FRB/US), without changing any equation value.
    storage_start = periods[0].shift(-max(bound.model.max_lag, 1), bound.freq)
    storage_end = periods[-1].shift(bound.model.max_lead, bound.freq)
    storage_periods = tuple(_periods(storage_start, storage_end, bound.freq))
    historical: dict[str, np.ndarray] = {}
    working: dict[str, np.ndarray] = {}
    for name, series in model_data.items():
        base = series.project(storage_start, storage_end, extend=True).values
        historical[name] = np.repeat(base[:, None], columns, axis=1)
        values = historical[name].copy()
        if name in endogenous:
            finite = np.flatnonzero(np.isfinite(base))
            if finite.size:
                values[int(finite[-1]) + 1 :, :] = base[int(finite[-1])]
        working[name] = values

    simulation_positions = np.asarray(
        [
            period.ordinal(bound.freq) - storage_start.ordinal(bound.freq)
            for period in periods
        ],
        dtype=np.intp,
    )
    for name, realizations in instrument_realizations.items():
        if name not in endogenous:
            working[name][simulation_positions, 1:] = realizations
            historical[name][simulation_positions, 1:] = realizations

    adjustment_matrices = _adjustment_matrices(
        bound,
        storage_periods,
        simulation_positions,
        instrument_realizations,
        constant_adjustments,
        columns,
    )
    coefficient_values = _coefficient_mapping(coefficients)
    blocks = _simulation_blocks(bound.model)
    exogenization = _validate_exogenization(
        exogenize,
        bound.model,
        bound.freq,
        periods[0],
        periods[-1],
    )
    if normalized_type == "RESCHECK":
        selected = (
            bound.model.endogenous if rescheck_equations is None else rescheck_equations
        )
        for period, position in zip(periods, simulation_positions, strict=True):
            for name in selected:
                rule = exogenization.get(name)
                if rule is not None and rule.applies(period, bound.freq):
                    working[name][position, :] = (
                        historical[name][position, :]
                        if rule.values is None
                        else rule.values.at_period(period.year, period.period)
                    )
                    continue
                working[name][position, :] = _solve_equation_columns(
                    name,
                    int(position),
                    bound,
                    coefficient_values,
                    adjustment_matrices,
                    historical,
                    working,
                    normalized_type,
                    zero_error_autocorrelation,
                )
        return {
            name: working[name][simulation_positions, 1:].copy() for name in selected
        }

    conditional = frozenset(bound.model.conditional_endogenous)
    dropped_from_jacobian = (
        _validate_jacobian_drop(jacobian_drop, bound.model, blocks)
        if validated_jacobian_drop is None
        else validated_jacobian_drop
    )
    if bound.model.forward_looking:
        _solve_forward_columns(
            periods,
            simulation_positions,
            bound,
            coefficient_values,
            adjustment_matrices,
            historical,
            working,
            normalized_algorithm,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
            jacobian_step,
            retain_final_iteration,
            newton_workspace,
            exogenization,
            dropped_from_jacobian,
        )
        return {
            name: working[name][simulation_positions, 1:].copy()
            for name in bound.model.endogenous
        }
    for period, position in zip(periods, simulation_positions, strict=True):
        _initialize_columns(
            period,
            int(position),
            normalized_type,
            bound,
            historical,
            working,
            conditional,
        )
        for block in blocks:
            fixed = {
                name
                for name in block.variables
                if name in exogenization
                and exogenization[name].applies(period, bound.freq)
            }
            for name in fixed:
                rule = exogenization[name]
                if rule.values is not None:
                    working[name][position, :] = rule.values.at_period(
                        period.year, period.period
                    )
                else:
                    working[name][position, :] = historical[name][position, :]
            active = tuple(name for name in block.variables if name not in fixed)
            if not active:
                continue
            if not block.simultaneous:
                name = active[0]
                working[name][position, :] = _solve_equation_columns(
                    name,
                    int(position),
                    bound,
                    coefficient_values,
                    adjustment_matrices,
                    historical,
                    working,
                    normalized_type,
                    zero_error_autocorrelation,
                )
                continue
            feedback = block.feedback
            feedback = tuple(name for name in feedback if name in active)
            if not feedback:
                if normalized_algorithm == "NEWTON":
                    _warn_empty_newton_feedback(
                        f"block {block.variables!r} at {period.year}-{period.period}"
                    )
                for name in active:
                    working[name][position, :] = _solve_equation_columns(
                        name,
                        int(position),
                        bound,
                        coefficient_values,
                        adjustment_matrices,
                        historical,
                        working,
                        normalized_type,
                        zero_error_autocorrelation,
                    )
                continue
            dropped = tuple(name for name in feedback if name in dropped_from_jacobian)
            if normalized_algorithm == "NEWTON" and not dropped:
                _solve_newton_columns(
                    active,
                    feedback,
                    block.variables,
                    period,
                    int(position),
                    bound,
                    coefficient_values,
                    adjustment_matrices,
                    historical,
                    working,
                    normalized_type,
                    convergence,
                    max_iterations,
                    zero_error_autocorrelation,
                    jacobian_step,
                    retain_final_iteration,
                    newton_workspace,
                )
                continue
            if normalized_algorithm == "NEWTON" and len(dropped) < len(feedback):
                _solve_hybrid_newton_columns(
                    active,
                    feedback,
                    block.variables,
                    frozenset(dropped),
                    period,
                    int(position),
                    bound,
                    coefficient_values,
                    adjustment_matrices,
                    historical,
                    working,
                    normalized_type,
                    convergence,
                    max_iterations,
                    zero_error_autocorrelation,
                    jacobian_step,
                    retain_final_iteration,
                    newton_workspace,
                )
                continue
            if normalized_algorithm == "NEWTON":
                _warn_empty_newton_feedback(
                    f"block {block.variables!r} at {period.year}-{period.period}"
                )
            for _iteration in range(1, max_iterations + 1):
                previous = np.stack(
                    [working[name][position, :].copy() for name in feedback]
                )
                for name in block.variables:
                    if name not in active:
                        continue
                    working[name][position, :] = _solve_equation_columns(
                        name,
                        int(position),
                        bound,
                        coefficient_values,
                        adjustment_matrices,
                        historical,
                        working,
                        normalized_type,
                        zero_error_autocorrelation,
                    )
                current = np.stack([working[name][position, :] for name in feedback])
                denominator = np.maximum(np.abs(previous), 1.0)
                change = 100.0 * np.abs(current - previous) / denominator
                if np.all(change < convergence):
                    break
            else:
                feedback_index, column_index = np.unravel_index(
                    int(np.nanargmax(change)), change.shape
                )
                column_label = (
                    "baseline" if column_index == 0 else f"replica {column_index}"
                )
                message = (
                    f"block {block.variables!r} did not converge at "
                    f"{period.year}-{period.period} in {max_iterations} iterations; "
                    f"largest change {change[feedback_index, column_index]:.6g}% "
                    f"for feedback {feedback[feedback_index]!r} in {column_label}"
                )
                if not retain_final_iteration:
                    raise SimulationConvergenceError(message)
                warnings.warn(
                    f"{message}; retaining the final iteration as BIMETS R does",
                    RuntimeWarning,
                    stacklevel=3,
                )
    return {
        name: working[name][simulation_positions, 1:].copy()
        for name in bound.model.endogenous
    }


def _solve_forward_columns(
    periods: tuple[YearPeriod, ...],
    positions: np.ndarray,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    algorithm: str,
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    retain_final_iteration: bool,
    newton_workspace: NewtonWorkspace | None,
    exogenization: Mapping[str, _ExogenizationRule],
    dropped: frozenset[str],
) -> None:
    """Solve a complete lead/lag horizon over synchronized matrix columns."""
    unknowns: list[tuple[str, YearPeriod, int]] = []
    for period, raw_position in zip(periods, positions, strict=True):
        position = int(raw_position)
        for name in bound.model.endogenous:
            rule = exogenization.get(name)
            fixed = rule is not None and rule.applies(period, bound.freq)
            if fixed and rule is not None and rule.values is not None:
                value = rule.values.at_period(period.year, period.period)
                working[name][position, :] = value
            else:
                working[name][position, :] = historical[name][position, :]
            if not np.all(np.isfinite(working[name][position, :])):
                raise ValueError(
                    f"forward-looking endogenous variable {name!r} is not "
                    f"defined at {period.year}-{period.period}"
                )
            if not fixed:
                unknowns.append((name, period, position))

    if not unknowns:
        if algorithm == "NEWTON":
            _warn_empty_newton_feedback("the shared forward-looking system")
        return
    stacked = tuple(unknowns)
    if algorithm == "GAUSS-SEIDEL" or all(name in dropped for name, _, _ in stacked):
        if algorithm == "NEWTON":
            _warn_empty_newton_feedback("the shared forward-looking system")
        _solve_forward_gauss_seidel_columns(
            stacked,
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
            retain_final_iteration,
        )
        return
    if dropped:
        _solve_forward_hybrid_newton_columns(
            stacked,
            dropped,
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            convergence,
            max_iterations,
            zero_error_autocorrelation,
            jacobian_step,
            retain_final_iteration,
            newton_workspace,
        )
        return
    _solve_forward_newton_columns(
        stacked,
        bound,
        coefficients,
        adjustments,
        historical,
        working,
        convergence,
        max_iterations,
        zero_error_autocorrelation,
        jacobian_step,
        retain_final_iteration,
        newton_workspace,
    )


def _solve_forward_gauss_seidel_columns(
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    retain_final_iteration: bool,
) -> None:
    """Run extended Gauss-Seidel with one convergence stop for all columns."""
    for _iteration in range(1, max_iterations + 1):
        previous = np.stack(
            [working[name][position, :].copy() for name, _, position in unknowns]
        )
        for name, _, position in unknowns:
            working[name][position, :] = _solve_equation_columns(
                name,
                position,
                bound,
                coefficients,
                adjustments,
                historical,
                working,
                "DYNAMIC",
                zero_error_autocorrelation,
            )
        current = np.stack(
            [working[name][position, :] for name, _, position in unknowns]
        )
        denominator = np.maximum(np.abs(previous), 1.0)
        change = 100.0 * np.abs(current - previous) / denominator
        if np.all(change < convergence):
            return
    _handle_forward_nonconvergence(
        "Gauss-Seidel",
        unknowns,
        change,
        max_iterations,
        retain_final_iteration,
    )


def _solve_forward_newton_columns(
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    retain_final_iteration: bool,
    newton_workspace: NewtonWorkspace | None,
) -> None:
    """Solve an extended system with one Jacobian and multiple right-hand sides."""
    current = np.stack([working[name][position, :] for name, _, position in unknowns])

    def residual(candidates: np.ndarray) -> np.ndarray:
        """Evaluate all extended residuals for synchronized candidates."""
        for (name, _, position), values in zip(unknowns, candidates, strict=True):
            working[name][position, :] = values
        targets = np.stack(
            [
                _solve_equation_columns(
                    name,
                    position,
                    bound,
                    coefficients,
                    adjustments,
                    historical,
                    working,
                    "DYNAMIC",
                    zero_error_autocorrelation,
                )
                for name, _, position in unknowns
            ]
        )
        result: np.ndarray = targets - candidates
        return result

    current_residual = residual(current)

    def baseline_residual(candidate: np.ndarray) -> np.ndarray:
        """Evaluate the baseline column used to construct the shared Jacobian."""
        candidates = current.copy()
        candidates[:, 0] = candidate
        return residual(candidates)[:, 0]

    cache_key = (
        "forward",
        tuple((name, period.year, period.period) for name, period, _ in unknowns),
    )
    jacobian = newton_workspace.get(cache_key) if newton_workspace is not None else None
    if jacobian is None:
        try:
            jacobian = factorize_finite_difference_jacobian(
                baseline_residual,
                current[:, 0],
                current_residual[:, 0],
                relative_step=jacobian_step,
                column_rows=_forward_jacobian_pattern(bound.model, unknowns),
            )
        except SparseFactorizationError as error:
            raise SimulationConvergenceError(
                "singular shared Newton Jacobian for the forward-looking system"
            ) from error
        if newton_workspace is not None:
            newton_workspace.store(cache_key, jacobian)

    current, percentage_change, converged = _iterate_shared_newton(
        residual,
        current,
        jacobian,
        convergence,
        max_iterations,
        "the forward-looking system",
    )
    if converged:
        residual(current)
        return
    if retain_final_iteration:
        residual(current)
    _handle_forward_nonconvergence(
        "Newton",
        unknowns,
        percentage_change,
        max_iterations,
        retain_final_iteration,
    )


def _solve_forward_hybrid_newton_columns(
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
    dropped: frozenset[str],
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    retain_final_iteration: bool,
    newton_workspace: NewtonWorkspace | None,
) -> None:
    """Alternate extended dropped equations with a reduced shared Newton solve."""
    newton_unknowns = tuple(item for item in unknowns if item[0] not in dropped)
    for _iteration in range(1, max_iterations + 1):
        previous = np.stack(
            [working[name][position, :].copy() for name, _, position in unknowns]
        )
        for name, _, position in unknowns:
            if name in dropped:
                working[name][position, :] = _solve_equation_columns(
                    name,
                    position,
                    bound,
                    coefficients,
                    adjustments,
                    historical,
                    working,
                    "DYNAMIC",
                    zero_error_autocorrelation,
                )
        _solve_forward_newton_columns(
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
            False,
            newton_workspace,
        )
        current = np.stack(
            [working[name][position, :] for name, _, position in unknowns]
        )
        denominator = np.maximum(np.abs(previous), 1.0)
        change = 100.0 * np.abs(current - previous) / denominator
        if np.all(change < convergence):
            return
    _handle_forward_nonconvergence(
        "hybrid Newton",
        unknowns,
        change,
        max_iterations,
        retain_final_iteration,
    )


def _handle_forward_nonconvergence(
    algorithm: str,
    unknowns: tuple[tuple[str, YearPeriod, int], ...],
    change: np.ndarray,
    max_iterations: int,
    retain_final_iteration: bool,
) -> None:
    """Report the largest unconverged extended-system matrix cell."""
    unknown_index, column_index = np.unravel_index(
        int(np.nanargmax(change)), change.shape
    )
    name, period, _ = unknowns[unknown_index]
    column_label = "baseline" if column_index == 0 else f"replica {column_index}"
    message = (
        f"shared forward-looking {algorithm} system did not converge in "
        f"{max_iterations} iterations; largest change "
        f"{change[unknown_index, column_index]:.6g}% for {name!r} at "
        f"{period.year}-{period.period} in {column_label}"
    )
    if not retain_final_iteration:
        raise SimulationConvergenceError(message)
    warnings.warn(
        f"{message}; retaining the final iteration as BIMETS R does",
        RuntimeWarning,
        stacklevel=4,
    )


def _iterate_shared_newton(
    residual: Callable[[np.ndarray], np.ndarray],
    current: np.ndarray,
    jacobian: SparseJacobian,
    convergence: float,
    max_iterations: int,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply common-Jacobian Newton updates with per-column relaxation."""
    relaxation = np.ones(current.shape[1], dtype=float)
    previous_norm: np.ndarray | None = None
    previous_ratio = np.full(current.shape[1], 0.5)
    percentage_change = np.full_like(current, np.inf)
    for _iteration in range(1, max_iterations + 1):
        current_residual = residual(current)
        change = jacobian.solve_blocked(-current_residual) * relaxation[None, :]
        previous = current.copy()
        current += change
        denominator = np.maximum(np.abs(previous), 1.0)
        percentage_change = 100.0 * np.abs(change) / denominator
        if np.all(percentage_change < convergence):
            return current, percentage_change, True
        current_norm = np.max(np.abs(current_residual), axis=0)
        if previous_norm is not None:
            ratios = np.divide(
                current_norm,
                previous_norm,
                out=np.zeros_like(current_norm),
                where=previous_norm > 0,
            )
            geometric = np.sqrt(np.maximum(ratios * previous_ratio, 0.0))
            if not np.all(np.isfinite(geometric)):
                raise SimulationConvergenceError(
                    f"non-finite shared Newton convergence rate for {scope}"
                )
            slow = geometric > 0.75
            relaxation[slow] = np.maximum(relaxation[slow] * 0.8, 0.5)
            previous_ratio = ratios
        previous_norm = current_norm
    return current, percentage_change, False


def _solve_hybrid_newton_columns(
    active: tuple[str, ...],
    feedback: tuple[str, ...],
    cache_variables: tuple[str, ...],
    dropped: frozenset[str],
    period: YearPeriod,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    retain_final_iteration: bool,
    newton_workspace: NewtonWorkspace | None,
) -> None:
    """Alternate shared dropped equations with a reduced multi-RHS Newton solve."""
    newton_active = tuple(name for name in active if name not in dropped)
    newton_feedback = tuple(name for name in feedback if name not in dropped)
    for _iteration in range(1, max_iterations + 1):
        previous = np.stack([working[name][position, :].copy() for name in feedback])
        for name in active:
            if name in dropped:
                working[name][position, :] = _solve_equation_columns(
                    name,
                    position,
                    bound,
                    coefficients,
                    adjustments,
                    historical,
                    working,
                    simulation_type,
                    zero_error_autocorrelation,
                )
        _solve_newton_columns(
            newton_active,
            newton_feedback,
            cache_variables,
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
            False,
            newton_workspace,
        )
        current = np.stack([working[name][position, :] for name in feedback])
        denominator = np.maximum(np.abs(previous), 1.0)
        change = 100.0 * np.abs(current - previous) / denominator
        if np.all(change < convergence):
            return

    feedback_index, column_index = np.unravel_index(
        int(np.nanargmax(change)), change.shape
    )
    column_label = "baseline" if column_index == 0 else f"replica {column_index}"
    message = (
        f"hybrid Newton block {cache_variables!r} did not converge at "
        f"{period.year}-{period.period} in {max_iterations} iterations; "
        f"largest change {change[feedback_index, column_index]:.6g}% "
        f"for feedback {feedback[feedback_index]!r} in {column_label}"
    )
    if not retain_final_iteration:
        raise SimulationConvergenceError(message)
    warnings.warn(
        f"{message}; retaining the final iteration as BIMETS R does",
        RuntimeWarning,
        stacklevel=4,
    )


def _solve_newton_columns(
    active: tuple[str, ...],
    feedback: tuple[str, ...],
    cache_variables: tuple[str, ...],
    period: YearPeriod,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    convergence: float,
    max_iterations: int,
    zero_error_autocorrelation: bool,
    jacobian_step: float,
    retain_final_iteration: bool,
    newton_workspace: NewtonWorkspace | None,
) -> None:
    """Solve all replica columns with one baseline Newton Jacobian."""
    current = np.stack([working[name][position, :] for name in feedback])

    def residual(candidates: np.ndarray) -> np.ndarray:
        """Evaluate the block residual matrix for synchronized candidates."""
        for name, values in zip(feedback, candidates, strict=True):
            working[name][position, :] = values
        for name in active:
            working[name][position, :] = _solve_equation_columns(
                name,
                position,
                bound,
                coefficients,
                adjustments,
                historical,
                working,
                simulation_type,
                zero_error_autocorrelation,
            )
        targets = np.stack([working[name][position, :] for name in feedback])
        result: np.ndarray = targets - candidates
        return result

    current_residual = residual(current)

    def baseline_residual(candidate: np.ndarray) -> np.ndarray:
        """Evaluate only the unperturbed column for Jacobian construction."""
        candidates = current.copy()
        candidates[:, 0] = candidate
        return residual(candidates)[:, 0]

    cache_key = ("backward", cache_variables, feedback)
    jacobian = newton_workspace.get(cache_key) if newton_workspace is not None else None
    if jacobian is None:
        try:
            jacobian = factorize_finite_difference_jacobian(
                baseline_residual,
                current[:, 0],
                current_residual[:, 0],
                relative_step=jacobian_step,
            )
        except SparseFactorizationError as error:
            raise SimulationConvergenceError(
                f"singular shared Newton Jacobian for block {active!r} at "
                f"{period.year}-{period.period}"
            ) from error

    current, percentage_change, converged = _iterate_shared_newton(
        residual,
        current,
        jacobian,
        convergence,
        max_iterations,
        f"block {active!r}",
    )
    if converged:
        residual(current)
        return

    feedback_index, column_index = np.unravel_index(
        int(np.nanargmax(percentage_change)), percentage_change.shape
    )
    column_label = "baseline" if column_index == 0 else f"replica {column_index}"
    message = (
        f"shared Newton block {active!r} did not converge at "
        f"{period.year}-{period.period} in {max_iterations} iterations; "
        f"largest change {percentage_change[feedback_index, column_index]:.6g}% "
        f"for feedback {feedback[feedback_index]!r} in {column_label}"
    )
    if not retain_final_iteration:
        raise SimulationConvergenceError(message)
    residual(current)
    warnings.warn(
        f"{message}; retaining the final iteration as BIMETS R does",
        RuntimeWarning,
        stacklevel=4,
    )


def _adjustment_matrices(
    bound: BoundModel,
    storage_periods: tuple[YearPeriod, ...],
    simulation_positions: np.ndarray,
    instrument_realizations: Mapping[str, np.ndarray],
    adjustments: Mapping[str, AdjustmentValue],
    columns: int,
) -> dict[str, np.ndarray]:
    """Build baseline-plus-replica add-factor matrices."""
    output: dict[str, np.ndarray] = {}
    for name in bound.model.endogenous:
        base = np.asarray(
            [
                _adjustment_value(adjustments.get(name, 0.0), period)
                for period in storage_periods
            ]
        )
        values = np.repeat(base[:, None], columns, axis=1)
        if name in instrument_realizations:
            values[simulation_positions, 1:] = instrument_realizations[name]
        output[name] = values
    return output


def _initialize_columns(
    period: YearPeriod,
    position: int,
    simulation_type: str,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    conditional: frozenset[str],
) -> None:
    """Initialize one period across every matrix column."""
    for name in bound.model.endogenous:
        historical_values = historical[name][position, :]
        if name in conditional:
            if not np.all(np.isfinite(historical_values)):
                raise ValueError(
                    f"conditional endogenous variable {name!r} requires a "
                    f"historical value at {period}"
                )
            working[name][position, :] = historical_values
        elif simulation_type == "FORECAST":
            if position == 0 or not np.all(np.isfinite(working[name][position - 1, :])):
                raise ValueError(
                    f"cannot initialize forecast variable {name!r} at {period}"
                )
            working[name][position, :] = working[name][position - 1, :]
        elif np.all(np.isfinite(historical_values)):
            working[name][position, :] = historical_values
        elif position > 0 and np.all(np.isfinite(working[name][position - 1, :])):
            working[name][position, :] = working[name][position - 1, :]
        else:
            raise ValueError(
                f"cannot initialize endogenous variable {name!r} at {period}"
            )


def _solve_equation_columns(
    name: str,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    zero_error_autocorrelation: bool,
) -> np.ndarray:
    """Evaluate and invert one equation for all synchronized columns."""
    definition = bound.model._equation_definition(name)
    if isinstance(definition, IdentityEquation):
        result = working[name][position, :].copy()
        selected = np.zeros(result.shape, dtype=bool)
        for alternative in definition.alternatives:
            if alternative.condition is None:
                mask = ~selected
            else:
                condition = _evaluate_columns(
                    alternative.condition,
                    position,
                    bound,
                    historical,
                    working,
                    simulation_type,
                )
                mask = ~selected & np.asarray(condition, dtype=bool)
            if not np.any(mask):
                continue
            rhs = _evaluate_columns(
                alternative.equation.rhs,
                position,
                bound,
                historical,
                working,
                simulation_type,
            )
            values = _invert_lhs_columns(
                alternative.equation,
                np.asarray(rhs, dtype=float) + adjustments[name][position, :],
                position,
                bound,
                historical,
                working,
                simulation_type,
            )
            result[mask] = values[mask]
            selected |= mask
    else:
        rhs = _behavioral_rhs_columns(
            definition,
            position,
            bound,
            coefficients,
            adjustments,
            historical,
            working,
            simulation_type,
            zero_error_autocorrelation,
        )
        result = _invert_lhs_columns(
            definition.equation,
            rhs + adjustments[name][position, :],
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
    if not np.all(np.isfinite(result)):
        replica = int(np.flatnonzero(~np.isfinite(result))[0])
        label = "baseline" if replica == 0 else f"replica {replica}"
        raise ValueError(f"equation {name!r} produced a non-finite value in {label}")
    return result


def _behavioral_rhs_columns(
    behavioral: BehavioralEquation,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    adjustments: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    zero_error_autocorrelation: bool,
) -> np.ndarray:
    """Evaluate a behavioral RHS, including autoregressive errors."""
    output = _behavioral_base_columns(
        behavioral,
        position,
        bound,
        coefficients,
        historical,
        working,
        simulation_type,
    )
    if behavioral.error is None or zero_error_autocorrelation:
        return output
    values = coefficients[behavioral.name]
    for lag in range(1, behavioral.error.order + 1):
        lagged = position - lag
        rho = float(values[f"RHO_{lag}"])
        level = _value_at(
            behavioral.name,
            lagged,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        lhs = _lhs_columns(
            behavioral.equation,
            level,
            lagged,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        rhs = _behavioral_base_columns(
            behavioral,
            lagged,
            bound,
            coefficients,
            historical,
            working,
            simulation_type,
            current_position=position,
        )
        output += rho * (lhs - rhs - adjustments[behavioral.name][lagged, :])
    return output


def _behavioral_base_columns(
    behavioral: BehavioralEquation,
    position: int,
    bound: BoundModel,
    coefficients: Mapping[str, Mapping[str, float]],
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    *,
    current_position: int | None = None,
) -> np.ndarray:
    """Evaluate the unadjusted behavioral RHS for all columns."""
    values = coefficients[behavioral.name]
    columns = next(iter(working.values())).shape[1]
    output = np.zeros(columns, dtype=float)
    current = position if current_position is None else current_position
    for regressor, coefficient_names, _ in _behavioral_execution_plan(behavioral):
        for lag, coefficient_name in enumerate(coefficient_names):
            regressor_value = _evaluate_columns(
                regressor,
                position - lag,
                bound,
                historical,
                working,
                simulation_type,
                current_position=current,
            )
            output += float(values[coefficient_name]) * regressor_value
    return output


def _evaluate_columns(
    expression: MdlExpression,
    position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
    *,
    current_position: int | None = None,
) -> np.ndarray:
    """Evaluate a compiled MDL instruction stream across matrix columns."""
    current = position if current_position is None else current_position
    columns = next(iter(working.values())).shape[1]
    with np.errstate(all="ignore"):
        stack: list[np.ndarray] = []
        for opcode, argument in _compile_expression(expression):
            if opcode == "CONST":
                stack.append(np.full(columns, float(cast(float, argument))))
            elif opcode == "VAR":
                name, offset = cast(tuple[str, int], argument)
                stack.append(
                    _value_at(
                        name,
                        position + offset,
                        current,
                        bound,
                        historical,
                        working,
                        simulation_type,
                    )
                )
            elif opcode == "NEG":
                stack[-1] = -stack[-1]
            elif opcode == "BINARY":
                right = stack.pop()
                left = stack.pop()
                stack.append(_binary_columns(left, str(argument), right))
            elif opcode == "ABS":
                stack[-1] = np.abs(stack[-1])
            elif opcode == "EXP":
                stack[-1] = np.exp(stack[-1])
            elif opcode == "LOG":
                stack[-1] = np.log(stack[-1])
            else:
                periods = cast(int, argument)
                values = stack[-periods:]
                del stack[-periods:]
                total = np.sum(np.stack(values), axis=0)
                stack.append(total / periods if opcode == "MOVAVG" else total)
        if len(stack) != 1:
            raise AssertionError("invalid compiled MDL column expression")
        return stack[0]


_COLUMN_BINARY_OPERATIONS: dict[str, np.ufunc] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "^": np.power,
    "&": np.logical_and,
    "|": np.logical_or,
    "==": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def _binary_columns(left: np.ndarray, operator: str, right: np.ndarray) -> np.ndarray:
    """Apply one MDL binary operator to matrix-column values."""
    return np.asarray(_COLUMN_BINARY_OPERATIONS[operator](left, right))


def _value_at(
    name: str,
    position: int,
    current_position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> np.ndarray:
    """Return one variable row with static-history semantics."""
    if position < 0 or position >= working[name].shape[0]:
        raise IndexError(f"variable {name!r} is unavailable at relative row {position}")
    if (
        simulation_type == "STATIC"
        and name in bound.model.endogenous
        and position < current_position
    ) or simulation_type == "RESCHECK":
        return historical[name][position, :]
    return working[name][position, :]


def _invert_lhs_columns(
    equation: MdlEquation,
    rhs: np.ndarray,
    position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> np.ndarray:
    """Recover dependent levels from a transformed LHS for all columns."""
    function = equation.lhs_function
    with np.errstate(all="ignore"):
        if function == "IDENTITY":
            return rhs
        if function == "LOG":
            return np.exp(rhs)
        if function == "EXP":
            return np.log(rhs)
        lagged = _value_at(
            equation.dependent,
            position - equation.lhs_periods,
            position,
            bound,
            historical,
            working,
            simulation_type,
        )
        if function == "TSDELTA":
            return np.asarray(lagged + rhs)
        if function == "TSDELTALOG":
            return np.asarray(np.exp(np.log(lagged) + rhs))
        if function == "TSDELTAP":
            return lagged * (1.0 + rhs / 100.0)
    raise AssertionError(f"unexpected LHS function {function!r}")


def _lhs_columns(
    equation: MdlEquation,
    level: np.ndarray,
    position: int,
    current_position: int,
    bound: BoundModel,
    historical: Mapping[str, np.ndarray],
    working: Mapping[str, np.ndarray],
    simulation_type: str,
) -> np.ndarray:
    """Evaluate a transformed LHS for all columns."""
    function = equation.lhs_function
    with np.errstate(all="ignore"):
        if function == "IDENTITY":
            return level
        if function == "LOG":
            return np.log(level)
        if function == "EXP":
            return np.exp(level)
        lagged = _value_at(
            equation.dependent,
            position - equation.lhs_periods,
            current_position,
            bound,
            historical,
            working,
            simulation_type,
        )
        if function == "TSDELTA":
            return np.asarray(level - lagged)
        if function == "TSDELTALOG":
            return np.log(level) - np.log(lagged)
        if function == "TSDELTAP":
            return np.asarray(100.0 * (level / lagged - 1.0))
    raise AssertionError(f"unexpected LHS function {function!r}")
