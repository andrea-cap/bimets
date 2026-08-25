# Solver architecture and execution strategies

[Back to the explanation index](README.md)

BIMETS models range from small recursive systems to large simultaneous or
forward-looking systems, sometimes evaluated over hundreds or thousands of
scenarios. The user selects the numerical algorithm; the library then chooses
the internal execution backend from the operation, model structure, and number
of paths. No single backend is optimal for every case. These strategies are
designed to preserve the numerical semantics of BIMETS R while using NumPy and
SciPy effectively.

## Choosing an algorithm

`GAUSS-SEIDEL`, the default, updates equations in incidence order. In a
backward-looking model, recursive blocks are evaluated once and only cyclic
blocks are iterated. `NEWTON` replaces the applicable fixed-point iterations
with finite-difference Jacobians and sparse linear algebra. `FULLNEWTON` gives
every scenario its own Jacobian; it is available to stochastic simulation,
multiplier matrices, renormalization, and optimal control, but not to a direct
deterministic `SIMULATE` call.

The algorithm is independent of the simulation type. `DYNAMIC`, `STATIC`,
`FORECAST`, and `RESCHECK` determine which observed or simulated values an
equation reads; the selected algorithm determines how simultaneous equations
are solved. Forward-looking solution paths require `DYNAMIC`; `RESCHECK`
remains available as an equation diagnostic.

Convergence uses the largest scaled update in the solver's active convergence
set: `100 * abs(current - previous) / max(abs(previous), 1)`. A block converges
only when every checked value is below the requested `convergence` threshold.

## Scalar deterministic simulation

A direct `SIMULATE` call solves one path. Recursive equations follow their
incidence order, while simultaneous blocks use either Gauss-Seidel or sparse
finite-difference Newton. Backward-looking incidence reduction limits
convergence checks and Newton unknowns to the feedback variables. Lead models
solve one extended variable-by-period system over the complete simulation
horizon because a value in one period can affect equations in earlier periods.

MDL expressions are parsed and compiled once into compact execution plans.
During iteration these plans read and write the model arrays directly, avoiding
the repeated construction of Python mappings and temporary series objects.
Sparse Jacobians use structural coloring to combine independent
finite-difference perturbations, followed by reusable sparse LU factorization.

With `NEWTON`, `jacobian_drop` removes selected feedback variables from the
Newton unknown vector, not from the model. Their equations continue to be
updated by Gauss-Seidel around the reduced Newton solve. If every feedback
variable in a block is excluded, the solver emits a warning and uses
Gauss-Seidel for that block.

## Shared-column backend

Stochastic simulation, multiplier matrices, renormalization, and optimal
control naturally create many structurally identical model paths. For
Gauss-Seidel and ordinary `NEWTON`, these paths are stored as columns of shared
NumPy arrays and advanced in the same solver iteration. Column zero is the
unperturbed baseline and the remaining columns are realizations, shocks, or
candidates. This reproduces BIMETS R's shared convergence rule: the complete
matrix must satisfy the stopping criterion before the operation converges.

The column solver consumes the same cached MDL instruction plans as the scalar
solver, applying NumPy operations to complete rows of realizations. Parsing,
expression compilation, PDL execution-plan construction, and operator-table
construction therefore remain outside the iterative evaluation path.

With ordinary `NEWTON`, the preliminary deterministic baseline supplies cached
sparse Jacobians and factorizations. The shared solver applies each
factorization to all column residuals as multiple right-hand sides. Large
right-hand-side matrices are processed in bounded blocks to control temporary
memory, but convergence is still assessed over all columns together.
Column-local relaxation limits the effect of a path that is poorly represented
by the common Jacobian without slowing every other column by the same amount.

The shared backend supports:

- backward- and forward-looking systems;
- common exogenizations, `RESCHECK` paths, and add factors;
- reduced Newton systems selected with `jacobian_drop`;
- stochastic realizations, multiplier shocks, renormalization iterations, and
  optimization candidates.

Only the simulation interval and the lag/lead boundary observations needed by
the model are retained in shared working storage. Public result arrays remain
fully allocated because callers can inspect every realization.

## FULLNEWTON and multiprocessing

`FULLNEWTON` deliberately does not share a baseline Jacobian. Every realization
or candidate builds and adapts its own Jacobian. This can be more robust when
paths traverse substantially different nonlinear regions, but performs more
work and uses more memory. It is available to the parent operations above,
while a single deterministic `SIMULATE` follows BIMETS R and rejects it.

Independent `FULLNEWTON` solves can use `workers` processes. Immutable model
configuration is initialized once in each worker and only independent work is
distributed. Process startup, serialization, and memory traffic can outweigh
parallel execution for small systems, so `workers=1` is the default. Values
above one are rejected for Gauss-Seidel and ordinary `NEWTON`: their shared
matrix and multi-right-hand-side kernels are normally more efficient and their
global convergence semantics cannot be replaced by independent processes.

## Strategy summary

| Operation | Gauss-Seidel / `NEWTON` | `FULLNEWTON` |
|---|---|---|
| Direct deterministic simulation | Scalar path | Not accepted |
| Stochastic simulation | Shared realization columns | Independent replicas |
| Multiplier matrix | Shared shock columns | Independent shocks |
| Renormalization | Shared multiplier columns | Independent multiplier shocks |
| Optimal control | Shared candidate columns | Independent candidates |
| Multiprocessing | Not used | Optional through `workers` |

The shared backend is an algorithmic optimization rather than a
parallelization technique: it removes repeated Python solver invocations and
reuses numerical structure. Multiprocessing is reserved for cases in which
that structure intentionally cannot be shared.
