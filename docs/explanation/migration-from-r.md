# Key differences from BIMETS R

[Back to the documentation index](../README.md)

This port is an implementation of BIMETS domain model and preserves its concepts related to time-series and numerical behavior, while adapting the public interface to the Python ecosystem.

The public sources and parallel examples used to check these differences are
documented in [Compatibility and numerical validation](conformance.md).

| Topic | BIMETS R | Python port |
|---|---|---|
| Canonical type | Base R `ts` or `xts`, selected through global configuration | The dedicated `BimetsSeries` type |
| Construction | `TIMESERIES(...)` / `TSERIES(...)` | `timeseries(...)`, returning a `BimetsSeries` |
| Date-valued construction start | `START` can be an R `Date` | `timeseries()` accepts a `YearPeriod` or `(year, period)` tuple; convert dates explicitly with `date_to_year_period()` |
| Mutability | Observations can be replaced or ranges extended through assignment | Series and datasets are immutable; `series.with_values()` and `dataset.assign_range()` return new objects |
| Positional indexing | One-based | Zero-based, following Python conventions |
| Year-period indexing | `[[year, period]]`, `[[start]]`, and inclusive `[[start, end]]` | The same read syntax is supported with Python lists, alongside `at_period()`, `period_at()`, and `project()` |
| Date indexing | Overloaded R indexing operators | ISO date, year, closed-range, and open-range selectors work through `[]`; `at_date()` and `between_dates()` are explicit alternatives |
| Missing values | R `NA` | NumPy `NaN` |
| Supported years | R `ts` uses a numeric time index and is not constrained by Python calendar objects | Calendar-compatible years 1--9999; the former artificial 1800--2199 restriction has been removed |
| Comparison result | A logical `ts`, with `NA` where either operand is missing | An indexed `BimetsMask`, with `None` for missing truth values |
| Named collections | Named lists of `ts`/`xts` objects | An immutable `BimetsDataset` mapping |
| Tabular model data | A data frame can be split into mutable named series | `BimetsDataset.from_frame()` imports regular pandas tables; `to_frame()` aligns an outer range and preserves common frequency and dataset metadata in `DataFrame.attrs` |
| Scenario ranges | `series[[start, end]] <- value` mutates selected observations | `series.with_values([start, end], value)` and `dataset.assign_range(...)` return immutable scenarios; scalars broadcast and fixed-range sequences have exact range length |
| Binary series operations | Align compatible series on their common range; a disjoint range can produce an empty result | Align on the common range; disjoint ranges raise `ValueError` because `BimetsSeries` cannot be empty |
| Metadata | R attributes such as `Title`, `Units`, `Source`, and `ScaleFac` | A read-only `metadata` mapping with lowercase keys |
| Metadata propagation | Depends on the R operation and object conversion | Selection, projection, pandas adapters, and frequency conversion preserve metadata; arithmetic and mathematical transformations intentionally return derived series without inherited metadata |
| CSV dates | Missing frequency tags are inferred through `xts`; irregular inputs are regularized | Missing frequency tags are inferred from date spacing; gaps become `NaN`, while duplicate, overlapping, or decreasing periods are rejected explicitly |
| Calendar inspection | `GETYEARPERIOD` returns named vectors or a joined matrix; `TSINFO` returns vectors and numeric start/end values | `get_year_periods()` defaults to typed values and optionally returns named arrays or a joined matrix; `tsinfo()` returns a scalar or tuple, with uppercase `START`/`END` retaining R fractional-year semantics |
| External series type | `xts`/`zoo` | `pandas.Series` through `to_pandas()` and `from_pandas()` |
| Configuration | Global options such as `BIMETS_CONF_CCT` and `BIMETS_CONF_DIP` | Explicit arguments such as `index=` and `date_in_period=` |
| Function naming | Uppercase names and arguments such as `TSLAG`, `FUN`, and `MV` | Canonical Python functions and aliases whose spelling exactly matches a public R function are exported; R aliases retain Python keyword names such as `method` and `skip_missing` |
| Moving-sum naming | The paper calls the public operation `MOVSUM`; current BIMETS documentation uses `MOVTOT` | The public operation is `movtot()`/`series.moving_sum()`; MDL expressions retain `MOVSUM` |
| Series and tabular display | Base `print(ts)` uses matrices for quarterly/monthly series and a `Time Series` block for other frequencies; `TABIT()` prints `Date`/`Prd.` rows | `print(series)` reproduces those frequency-dependent layouts, while `repr(series)` supplies a compact technical view; `tabulate()`/`TABIT()` reuse the formatter in a pandas `DataFrame` and do not print as a side effect |
| Compliance checks | Repeated runtime checks controlled by `avoidCompliance` | Invariants are enforced when a `BimetsSeries` is constructed |
| MDL representation | Nested mutable lists containing generated R expressions | An immutable `BimetsModel` containing typed expression-tree nodes |
| MDL execution | Generated expressions are parsed and evaluated by R | A typed expression tree is evaluated directly without Python `eval()` |
| Model data | `LOAD_MODEL_DATA` mutates the model list | `model.bind(data)` returns an immutable `BoundModel` and validates variables and frequency |
| Current estimation | OLS, restrictions, AR errors, PDL, IV, and Chow tests | The same estimator families and stability test, returned as immutable result objects |
| Linear restrictions | Uses the scaled augmented system described in the paper | Solves exact restrictions in an SVD-derived null space, avoiding the conditioning loss of repeated normal equations; published Klein results are reproduced without augmentation scaling |
| Estimation range arguments | `TSRANGE` and `forceTSRANGE` | `time_range` and `force_time_range`; local MDL ranges have the same default precedence |
| Numerical tolerance | `tol=1e-28` by default in `ESTIMATE()` | `tol=1e-12` by default, used as a relative singularity threshold suitable for NumPy floating-point linear algebra |
| Chow predictive power | Runs a `RESCHECK` simulation of the model | Evaluates the selected equation using observed out-of-sample regressors; this avoids an implicit full-model simulation but differs when regressors are endogenous |
| Instrument override | `IV` and `forceIV`, with MDL declarations taking precedence unless forced | Supplying `instruments=` explicitly always overrides `IV>` declarations |
| Missing estimation values | The declared sample must be fully defined | The same rule is enforced; rows are not silently dropped |
| Simulation result | `SIMULATE()` mutates and returns the model, storing values under `$simulation` | `simulate()` leaves model and data unchanged and returns an immutable `SimulationResult` |
| Forecast initialization | Uses available current observations as solver starting values and propagates the previous solution where future data are missing | Uses the same current-value-first rule; returned values are always solved forecasts rather than the initial observations |
| Simulation coefficients | Estimated coefficients are stored inside each behavioral equation | An estimation result or calibrated coefficient mapping is passed explicitly to `simulate()` |
| Conditional identities | Every `IF>` identity remains endogenous; if no condition is true, its current-period data value is retained and must be defined throughout the simulation range | The same runtime evaluation, historical fallback, structural endogeneity, and forecast-data requirement are enforced; `model.conditional_endogenous` exposes the affected subset |
| Deterministic solver | Gauss-Seidel and Newton; full-Newton is used only by vectorized parent operations | Gauss-Seidel and sparse finite-difference Newton; direct `simulate()` rejects `FULLNEWTON` like R, while stochastic simulation, multipliers, renormalization, and optimization accept it |
| Jacobian exclusion | `JacobianDrop` removes selected feedback variables from Newton and retains their Gauss-Seidel updates; an empty active Jacobian produces a warning | `jacobian_drop` solves the same reduced Newton/remaining Gauss-Seidel partition for backward and forward-looking systems; backward convergence uses the reduced feedback set and an all-excluded block warns before using Gauss-Seidel |
| Historical solution output | `BackFill` prepends up to the requested number of available endogenous observations | `backfill` has the same range semantics and leaves simulation periods and iteration counts unchanged |
| Backward-looking equation ordering | Builds `vpre`, `vsim`, `vfeed`, and `vpost`, seeking a small feedback-variable set inside the simultaneous block | Uses the same incidence reduction inside topologically ordered strongly connected components; `SimulationBlock.variables` gives the evaluation order and `SimulationBlock.feedback` the reduced convergence set |
| Residual check | Supports equation selection, autoregressive errors, and `ZeroErrorAC` | The corresponding features are available as `rescheck_equations` and `zero_error_autocorrelation`; results and tracking adjustments are returned rather than stored in the model |
| Exogenization | A named list maps endogenous variables to `TRUE` or a `TSRANGE`, always using historical values | A name/sequence fixes the full range; a mapping accepts a time range or a replacement `BimetsSeries` |
| Rational expectations | Builds explicit `__LEAD__n` variables, reorders the extended incidence matrix, and checks convergence on feedback variables | Evaluates `TSLEAD` directly in one multi-period system and checks every active endogenous-period value; generated lead names are not exposed |
| Forward-looking scalability | Uses optimized equation reordering and numerical Jacobians | Uses a global Gauss-Seidel iteration or a structurally colored sparse Newton Jacobian over the extended variable-period system |
| Stochastic execution | Solves all realizations together as columns of internal matrices; `NEWTON` shares the unperturbed column's Jacobian while `FULLNEWTON` builds one per column | Gauss-Seidel and `NEWTON` use synchronized NumPy columns for backward, forward-looking, exogenized, residual-check, and reduced-Jacobian systems; `FULLNEWTON` preserves an independent sparse Jacobian for every realization and can distribute those independent solves with explicit `workers=` |
| Stochastic result | Baseline and realizations are stored in `$simulation_MM`; summaries in `$stochastic_simulation` | The immutable result exposes `baseline` separately and groups each endogenous mean, standard deviation, and realization matrix |
| Random seed | Uses R's default Mersenne Twister and inversion normal generator | Explicitly seeded `OPTIMIZE` and `STOCHSIMULATE` reproduce R uniform/normal draws, extraction order, and column-major filling; an omitted seed uses an independent NumPy generator |
| Equation evaluation | Parses and evaluates model expressions through R internals | Python compiles each MDL expression once into a cached execution plan shared by scalar and column-oriented solvers, then evaluates it directly against simulation arrays; syntax and numerical semantics are unchanged |
| Multiplier execution | Evaluates all instrument-period shocks together as columns of shared simulation matrices | Uses the same column layout; backward Gauss-Seidel shares convergence and backward `NEWTON` reuses the baseline sparse Jacobian with a multi-RHS solve |
| Multiplier result | Stores `MM_MATRIX` and related metadata in the mutable model object | Returns an immutable `MultiplierMatrixResult` with labels, lookup helpers, and a separate baseline |
| Multiplier convergence | The common solver stopping error affects all SIMD columns | Shared backward and forward-looking solvers use the same global stopping rule, including reduced Jacobians; `FULLNEWTON` uses the independent-column path |
| Renormalization state | Mutates and returns the model with results under `$renorm` | Returns an immutable `RenormalizationResult`; adjusted data and add-factors are explicit and the inputs remain unchanged |
| Renormalization failure | A missed iteration limit emits a warning and only exposes achieved targets and unconverged names | Emits a warning and returns the final instruments plus full diagnostics with `converged=False`; singular multiplier systems raise `RenormalizationError` |
| Renormalization iterations | Delays its first convergence check and can perform one extra multiplier pass | Checks the current baseline at every pass, including the initial state, and reports the number of actual instrument corrections |
| Optimal-control execution | Simulates candidates together as columns and applies one convergence stop to the complete matrix | Gauss-Seidel and common-Jacobian `NEWTON` use shared NumPy columns for backward, forward-looking, exogenized, `RESCHECK`, and reduced-Jacobian searches; `FULLNEWTON` uses independent candidate solves |
| Optimal-control result | Mutates the model and stores `$optimize`, `$simulation_MM`, and `$INSTRUMENT_MM` | Returns an immutable `OptimizationResult` containing adjusted inputs, objective diagnostics, instrument realizations, and stochastic simulations |
| No feasible optimum | Emits a warning and leaves optimum fields empty | Returns `None` optimum fields and an empty instrument mapping while preserving every objective value and feasibility decision |
| Legacy `STORE>` | Accepted and retained as unused `storeVarName` and `storePosition` fields | Accepted and validated, then discarded without affecting execution |

Both functional and object-oriented forms are available where they are useful:

```python
change = tsdeltap(gdp, lag=1)
same_change = gdp.delta_percent(lag=1)
```

R-specific facilities intentionally omitted from the port and their Python
alternatives are listed directly in the table above.

See [Compatibility and numerical validation](conformance.md) for reproducible
R/Python comparisons and
[Solver architecture and execution strategies](solver-strategies.md) for
implementation and performance details.
