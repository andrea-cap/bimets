"""Sparse numerical linear algebra used by MDL simulation solvers."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

type FloatArray = npt.NDArray[np.float64]
type ResidualFunction = Callable[[FloatArray], FloatArray]

_DEFAULT_RHS_WORKSPACE_BYTES = 16 * 1024 * 1024


class SparseFactorizationError(RuntimeError):
    """Raised when a finite-difference Jacobian cannot be factorized."""


class SparseNewtonIterationLimit(RuntimeError):
    """Raised when a sparse Newton iteration does not converge."""


class _SparseSolver(Protocol):
    def solve(self, rhs: FloatArray) -> FloatArray:
        """Solve the factorized system for the supplied right-hand side."""
        ...


@dataclass(frozen=True, slots=True)
class SparseJacobian:
    """Factorized sparse finite-difference Jacobian.

    Parameters
    ----------
    matrix : scipy.sparse.csc_matrix
        Square Jacobian in compressed sparse column format.
    factorization : sparse linear solver
        Reusable sparse LU factorization.

    Notes
    -----
    The factorization accepts either one right-hand side or a matrix of right-
    hand sides. The latter mirrors the common-Jacobian, multiple-replica
    update used by BIMETS R's ``NEWTON`` stochastic solver.
    """

    matrix: csc_matrix
    factorization: _SparseSolver

    @property
    def density(self) -> float:
        """Fraction of structurally nonzero matrix entries."""
        rows, columns = self.matrix.shape
        return float(self.matrix.nnz / (rows * columns))

    def solve(self, rhs: FloatArray) -> FloatArray:
        """Solve the factorized system for one or more right-hand sides."""
        result = self.factorization.solve(np.asarray(rhs, dtype=float))
        return cast(FloatArray, np.asarray(result, dtype=float))

    def solve_blocked(
        self,
        rhs: FloatArray,
        *,
        workspace_bytes: int = _DEFAULT_RHS_WORKSPACE_BYTES,
    ) -> FloatArray:
        """Solve matrix right-hand sides in bounded column blocks.

        One-dimensional inputs use the direct sparse solve. Matrix inputs are
        split so the right-hand side and its solution consume approximately
        ``workspace_bytes`` per block. Blocks are processed within the same
        Newton iteration, preserving the shared convergence decision while
        limiting SuperLU's transient variables-times-periods-times-replicas
        workspace for large forward-looking simulations.
        """
        values = np.asarray(rhs, dtype=float)
        if values.ndim == 1:
            return self.solve(values)
        if values.ndim != 2 or values.shape[0] != self.matrix.shape[0]:
            raise ValueError("right-hand sides must align with the Jacobian rows")
        if workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be positive")
        bytes_per_column = max(values.shape[0] * values.itemsize * 2, 1)
        block_columns = max(1, workspace_bytes // bytes_per_column)
        if values.shape[1] <= block_columns:
            return self.solve(values)
        output = np.empty_like(values)
        for start in range(0, values.shape[1], block_columns):
            stop = min(start + block_columns, values.shape[1])
            output[:, start:stop] = self.solve(values[:, start:stop])
        return output


@dataclass(slots=True)
class NewtonWorkspace:
    """Cache baseline sparse Jacobians for a multi-replica simulation.

    The deterministic baseline records one factorization for each solved
    system. Calling :meth:`freeze` makes those factorizations read-only so all
    stochastic realizations start from the same baseline Jacobians, analogous
    to BIMETS R's ``NEWTON`` treatment of matrix columns.
    """

    _jacobians: dict[Hashable, SparseJacobian] = field(default_factory=dict)
    frozen: bool = False

    def get(self, key: Hashable) -> SparseJacobian | None:
        """Return a cached factorization, if available."""
        return self._jacobians.get(key)

    def store(self, key: Hashable, jacobian: SparseJacobian) -> None:
        """Record a baseline factorization while the workspace is mutable."""
        if not self.frozen:
            self._jacobians[key] = jacobian

    def freeze(self) -> None:
        """Keep baseline factorizations fixed for subsequent replicas."""
        self.frozen = True


def factorize_finite_difference_jacobian(
    residual: ResidualFunction,
    current: FloatArray,
    current_residual: FloatArray,
    *,
    relative_step: float,
    column_rows: Sequence[Sequence[int]] | None = None,
) -> SparseJacobian:
    """Build and factorize a sparse numerical Jacobian.

    Parameters
    ----------
    residual : callable
        Function mapping the current unknown vector to its residuals.
    current, current_residual : numpy.ndarray
        Current unknowns and the corresponding residual vector.
    relative_step : float
        Relative forward-difference shock. Values with magnitude below one use
        ``relative_step`` itself, matching the BIMETS R Jacobian shock rule.
    column_rows : sequence of sequences of int, optional
        Structural rows affected by each column. Structurally independent
        columns are perturbed together using greedy finite-difference coloring.

    Returns
    -------
    SparseJacobian
        CSC Jacobian and reusable sparse LU factorization.

    Raises
    ------
    SparseFactorizationError
        If dimensions are inconsistent, derivatives are non-finite, or the
        sparse matrix is singular.

    Notes
    -----
    Columns are accumulated directly into sparse coordinate arrays; a dense
    square Jacobian is never allocated. Exact zero derivatives are omitted.
    """
    values = np.asarray(current, dtype=float)
    baseline = np.asarray(current_residual, dtype=float)
    if values.ndim != 1 or baseline.shape != values.shape:
        raise SparseFactorizationError(
            "Newton unknowns and residuals must be one-dimensional and aligned"
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(baseline)):
        raise SparseFactorizationError("Newton state and residuals must be finite")

    size = len(values)
    structural_rows = _normalize_column_rows(column_rows, size)
    color_groups = _color_columns(structural_rows)
    steps = relative_step * np.maximum(np.abs(values), 1.0)
    row_indexes: list[int] = []
    column_indexes: list[int] = []
    derivatives: list[float] = []
    for columns in color_groups:
        perturbed = values.copy()
        perturbed[np.asarray(columns, dtype=int)] += steps[
            np.asarray(columns, dtype=int)
        ]
        difference = np.asarray(residual(perturbed), dtype=float) - baseline
        if difference.shape != values.shape or not np.all(np.isfinite(difference)):
            raise SparseFactorizationError(
                "non-finite or misaligned derivative in sparse Jacobian"
            )
        for column in columns:
            rows = np.fromiter(structural_rows[column], dtype=int)
            column_derivative = difference[rows] / steps[column]
            nonzero = column_derivative != 0.0
            selected_rows = rows[nonzero]
            row_indexes.extend(int(row) for row in selected_rows)
            column_indexes.extend([column] * len(selected_rows))
            derivatives.extend(float(value) for value in column_derivative[nonzero])

    matrix = csc_matrix(
        (derivatives, (row_indexes, column_indexes)),
        shape=(size, size),
        dtype=float,
    )
    matrix.eliminate_zeros()
    try:
        factorization = splu(matrix)
    except RuntimeError as error:
        raise SparseFactorizationError("sparse Newton Jacobian is singular") from error
    return SparseJacobian(matrix, factorization)


def solve_sparse_newton(
    residual: ResidualFunction,
    initial: FloatArray,
    *,
    relative_step: float,
    convergence: float,
    max_iterations: int,
    workspace: NewtonWorkspace | None = None,
    cache_key: Hashable | None = None,
    column_rows: Sequence[Sequence[int]] | None = None,
    rebuild_threshold: float = 0.9,
    relaxation_threshold: float = 0.75,
) -> tuple[FloatArray, int]:
    """Solve a nonlinear system with an adaptively refreshed sparse Jacobian.

    Parameters
    ----------
    residual, initial, relative_step
        Residual function, initial unknown vector, and Jacobian shock.
    convergence : float
        Maximum percentage update accepted for every unknown.
    max_iterations : int
        Iteration limit.
    workspace, cache_key : optional
        Shared baseline factorization storage for stochastic replicas.
    column_rows : sequence of sequences of int, optional
        Structural Jacobian pattern used for finite-difference coloring.
    rebuild_threshold, relaxation_threshold : float
        Geometric convergence-rate thresholds used to rebuild the Jacobian or
        relax an update. Defaults match BIMETS R ``NEWTON``; callers use the
        stricter ``0.6`` and ``0.45`` thresholds for ``FULLNEWTON``.

    Returns
    -------
    tuple of numpy.ndarray and int
        Converged unknowns and iterations used.

    Notes
    -----
    The Jacobian is reused while convergence is satisfactory. If two
    consecutive residual reductions indicate slow convergence, the update is
    relaxed or the Jacobian is rebuilt. The thresholds and minimum relaxation
    follow BIMETS R's ``NEWTON`` strategy.
    """
    current = np.asarray(initial, dtype=float).copy()
    jacobian = (
        workspace.get(cache_key)
        if workspace is not None and cache_key is not None
        else None
    )
    relaxation = 1.0
    previous_ratio = 0.5
    previous_norm: float | None = None

    for iteration in range(1, max_iterations + 1):
        current_residual = np.asarray(residual(current), dtype=float)
        current_norm = float(np.max(np.abs(current_residual), initial=0.0))
        if jacobian is None:
            jacobian = factorize_finite_difference_jacobian(
                residual,
                current,
                current_residual,
                relative_step=relative_step,
                column_rows=column_rows,
            )
            if workspace is not None and cache_key is not None:
                workspace.store(cache_key, jacobian)

        change = jacobian.solve(-current_residual) * relaxation
        previous = current.copy()
        current += change
        denominator = np.maximum(np.abs(previous), 1.0)
        percentage_change = 100.0 * np.abs(change) / denominator
        if np.all(percentage_change < convergence):
            residual(current)
            return current, iteration

        if previous_norm is not None and previous_norm > 0:
            ratio = current_norm / previous_norm
            geometric_ratio = math.sqrt(max(ratio * previous_ratio, 0.0))
            if not math.isfinite(geometric_ratio):
                raise SparseFactorizationError(
                    "non-finite convergence rate in sparse Newton iteration"
                )
            if geometric_ratio > rebuild_threshold:
                jacobian = None
            elif geometric_ratio > relaxation_threshold:
                relaxation = max(relaxation * 0.8, 0.5)
            previous_ratio = ratio
        previous_norm = current_norm

    raise SparseNewtonIterationLimit


def _normalize_column_rows(
    column_rows: Sequence[Sequence[int]] | None,
    size: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate and normalize sparse Jacobian row patterns."""
    if column_rows is None:
        return (
            tuple((row,) for row in range(size))
            if size == 1
            else tuple(tuple(range(size)) for _ in range(size))
        )
    if len(column_rows) != size:
        raise SparseFactorizationError(
            "Jacobian structural pattern must contain one entry per column"
        )
    normalized: list[tuple[int, ...]] = []
    for rows in column_rows:
        selected = tuple(sorted(set(rows)))
        if not selected or selected[0] < 0 or selected[-1] >= size:
            raise SparseFactorizationError(
                "Jacobian structural rows must be non-empty valid indexes"
            )
        normalized.append(selected)
    return tuple(normalized)


def _color_columns(
    column_rows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Group columns whose structural row sets do not overlap."""
    groups: list[list[int]] = []
    occupied: list[set[int]] = []
    for column, rows in enumerate(column_rows):
        selected = set(rows)
        for group, used_rows in zip(groups, occupied, strict=True):
            if selected.isdisjoint(used_rows):
                group.append(column)
                used_rows.update(selected)
                break
        else:
            groups.append([column])
            occupied.append(selected)
    return tuple(tuple(group) for group in groups)
