# Public API

[Back to the documentation index](../README.md)

The public API is organized as follows:

- construction and conversion: `timeseries`, `to_pandas`, `from_pandas`;
- arithmetic and comparison: Python numeric and comparison operators on
  `BimetsSeries`, with logical operations on `BimetsMask`;
- named collections: `BimetsDataset`, including selection, renaming,
  alignment, mapping, immutable range assignment, DataFrame round-trips, and
  CSV exchange;
- calendar and indexing: `normalize_year_period`, `num_periods`, `get_dates`,
  `get_year_periods`, `date_to_year_period`, `year_period_to_date`;
- frequency conversion: `convert_frequency`, `annual`, `semiannual`,
  `quarterly`, `monthly`, `daily`;
- transformations: `tslag`, `tslead`, `tsdelta`, `tsdeltalog`, `tsdeltap`;
- ranges: `tsproject`, `tstrim`, `tsextend`;
- cumulative and moving-window operations: `cumsum`, `cumprod`, `movavg`,
  `movtot`;
- combination: `tsmerge`, `tsjoin`;
- index numbers: `indexnum`;
- inspection: `series_info`, `tsinfo`, `get_range`, `tabulate`, `magnitude`,
  `verify_magnitude`, `is_bimets`;
- CSV exchange: `bimets_to_csv`, `csv_to_bimets`;
- MDL loading and evaluation: `load_model`, `parse_mdl`, `parse_expression`,
  and `evaluate_expression`;
- MDL data and estimation: `bind_model_data` and `estimate`, also available as
  `model.bind()`, `model.estimate()`, and `bound_model.estimate()`;
- deterministic MDL simulation: `simulate`, also available as
  `model.simulate()` and `bound_model.simulate()`;
- stochastic MDL simulation: `stochastic_simulate`, also
  available as `model.stochastic_simulate()` and
  `bound_model.stochastic_simulate()`;
- MDL multiplier matrices: `multiplier_matrix`, also available as
  `model.multiplier_matrix()` and `bound_model.multiplier_matrix()`;
- endogenous targeting: `renormalize`, also available as
  `model.renormalize()` and `bound_model.renormalize()`;
- optimal control: `optimize_model`, also available as
  `model.optimize()` and `bound_model.optimize()`;
- MDL structures: `BimetsModel`, `BehavioralEquation`, `IdentityEquation`,
  `MdlEquation`, `MdlTimeRange`, `PdlDefinition`, `CoefficientRestriction`,
  `BoundModel`, `ModelEstimationResult`, `EquationEstimationResult`,
  `ChowTestResult`, `SimulationResult`, `SimulationBlock`,
  `SimulationConvergenceError`, `StochasticDisturbance`,
  `StochasticSimulationResult`, `StochasticSeriesResult`,
  `MultiplierMatrixResult`, `MultiplierMatrixError`, and the public expression
  nodes; stochastic failures use `StochasticSimulationError`;
  endogenous-targeting results use `RenormalizationResult` and
  `RenormalizationError`; optimal control uses `OptimizationBound`,
  `OptimizationFunction`, `OptimizationRestriction`, `OptimizationResult`, and
  `OptimizationError`.

## Public typing aliases

Public signatures use these aliases to describe accepted inputs and expression
results:

| Alias | Meaning |
|---|---|
| `MdlExpression` | Any immutable MDL expression node |
| `MdlValue` | A numeric or logical scalar, `BimetsSeries`, or `BimetsMask` |
| `CoefficientInput` | A `ModelEstimationResult` or nested coefficient mapping |
| `AdjustmentValue` | A scalar or `BimetsSeries` add-factor |
| `ExogenizationValue` | A boolean, replacement series, or year-period range |
| `DisturbanceParameters` | A two-number distribution tuple or NumPy matrix |

## BIMETS R function aliases

Original BIMETS R names that identify the same operation are public aliases of
the canonical Python functions. Aliases are the same callable objects, not
separate implementations:

| BIMETS R names | Canonical Python function |
|---|---|
| `TIMESERIES`, `TSERIES` | `timeseries` |
| `ANNUAL`, `YEARLY` | `annual` |
| `SEMIANNUAL`, `QUARTERLY`, `MONTHLY`, `DAILY` | matching lowercase function |
| `CUMSUM`, `CUMULO` | `cumsum` |
| `CUMPROD` | `cumprod` |
| `TSDELTA`, `DELTA` | `tsdelta` |
| `TSDELTALOG` | `tsdeltalog` |
| `TSDELTAP`, `DELTAP` | `tsdeltap` |
| `TSEXTEND`, `EXTEND` | `tsextend` |
| `MOVAVG`, `MAVE` | `movavg` |
| `MOVTOT`, `MOVSUM`, `MTOT`, `MSUM` | `movtot` |
| `GETDATE` | `get_dates` |
| `GETRANGE` | `get_range` |
| `GETYEARPERIOD`, `TSDATES` | `get_year_periods` |
| `date2yp` | `date_to_year_period` |
| `normalizeYP` | `normalize_year_period` |
| `INDEXNUM`, `NUMPERIOD` | `indexnum`, `num_periods` |
| `TABIT`, `TSINFO`, `TSLOOK` | `tabulate`, `tsinfo`, `series_info` |
| `TSJOIN`, `TSLAG`, `TSLEAD`, `TSMERGE` | matching lowercase function |
| `TSPROJECT`, `TSTRIM` | matching lowercase function |
| `VERIFY_MAGNITUDE` | `verify_magnitude` |
| `BIMETS2CSV`, `CSV2BIMETS` | `bimets_to_csv`, `csv_to_bimets` |
| `LOAD_MODEL`, `LOAD_MODEL_DATA` | `load_model`, `bind_model_data` |
| `ESTIMATE`, `SIMULATE`, `STOCHSIMULATE` | `estimate`, `simulate`, `stochastic_simulate` |
| `MULTMATRIX`, `RENORM`, `OPTIMIZE` | `multiplier_matrix`, `renormalize`, `optimize_model` |

They retain Python signatures and return types. For example:

```python
from bimets import TIMESERIES, TSLAG

series = TIMESERIES([1, 2, 3], start=(2020, 1), freq="Q")
lagged = TSLAG(series, periods=1)
```

R-specific keyword names such as `START`, `FREQ`, `FUN`, and `ignoreNA` are
not accepted. Existing methods are unaffected by the compatibility aliases.

## R-specific facilities without aliases

Some BIMETS R names either are not valid Python identifiers or expose
R-specific data structures and global state. Their Python dispositions are:

| BIMETS R facility | Python disposition |
|---|---|
| `is.bimets` | `is_bimets()` |
| `as.bimets` | Explicit `timeseries()`, `BimetsSeries`, or `from_pandas()` construction |
| `frequency.xts` | The immutable `series.freq` property |
| `print.BIMETS_MODEL`, `summary.BIMETS_MODEL` | `repr()` and typed result `summary()` methods |
| `fromBIMETStoTS`, `fromBIMETStoXTS`, `fromTStoXTS`, `fromXTStoTS` | `to_pandas()` and `from_pandas()`; Python does not expose R `ts`, `xts`, or `zoo` objects |
| `ym2yp`, `yq2yp` | pandas `PeriodIndex` conversion and explicit calendar helpers |
| `A1D`, `INTS`, `LOCS` | Standard NumPy reshaping and Python/NumPy indexing |
| `ELIMELS`, `NOELS`, `NAMELIST` | Standard Python lists, comprehensions, mappings, and datasets |
| `getBIMETSconf`, `setBIMETSconf` | Explicit function arguments replace mutable global configuration |

See [Key differences from BIMETS R](../explanation/migration-from-r.md) for the
complete conceptual and behavioral comparison.

The main public types are `BimetsSeries`, `BimetsMask`, `BimetsDataset`,
`BimetsModel`, `BoundModel`, `ModelEstimationResult`, `YearPeriod`, `Frequency`,
`SimulationResult`, `StochasticSimulationResult`, `MultiplierMatrixResult`,
`RenormalizationResult`, `OptimizationResult`, and `SeriesInfo`.
