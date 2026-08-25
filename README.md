# bimets — Time-Series Analysis and Econometric Modeling in Python

[![CI](https://github.com/andrea-cap/bimets/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/andrea-cap/bimets/actions/workflows/ci.yml)
[![PyPI downloads](https://api.pepy.tech/badge/bimets)](https://pepy.tech/project/bimets)

**bimets** is a Python port of the
[BIMETS R package](https://github.com/andrea-luciani/bimets) for regular
time-series analysis and econometric modeling. It covers time-series
manipulation, parsing of Model Description Language (MDL), estimation, and deterministic and stochastic simulation. 

**bimets** has only three dependencies: *NumPy* and *SciPy*, for numerical computation, and *pandas* for data interoperability.

The library keeps all the underlying mathematical behavior of
BIMETS but introduces some design differences due to the Python ecosystem, such as immutable objects, explicit inputs, NumPy-based storage and pandas interoperability.

See [Migration from R](https://github.com/andrea-cap/bimets/blob/master/docs/explanation/migration-from-r.md)
for a detailed comparison.

## Main features

**bimets** provides tools for working with regular time series and for estimating and simulating multi-equation econometric models:

- **Work with regular time series.** Create annual, semiannual, quarterly,
  monthly, weekly, and daily series; inspect their calendars; perform aligned
  arithmetic; apply lag, lead, difference, growth, moving-window, extension,
  aggregation, and disaggregation operations.
- **Organize and exchange data.** Store named series in immutable datasets,
  convert them to and from pandas objects, and read or write BIMETS-compatible
  CSV files.
- **Define models in MDL.** Load and safely parse BIMETS Model Description
  Language models, including identities, behavioral equations, conditional alternatives, lags, leads, transformations, and coefficient
  declarations.
- **Estimate behavioral equations.** Use ordinary least squares (OLS) or
  instrumental variables (IV), with support for coefficient restrictions,
  polynomial distributed lags (PDLs), autoregressive errors, and Chow stability
  tests.
- **Run deterministic simulations.** Solve static, dynamic, forecast, and
  residual-check simulations for backward- or forward-looking models using
  Gauss-Seidel or Newton algorithms, with exogenizations and add factors where
  needed.
- **Explore alternative scenarios.** Perform stochastic simulation, calculate
  multiplier matrices, target endogenous variables through renormalization,
  and run Monte Carlo optimal-control searches.
- **Move from BIMETS R incrementally.** Use familiar uppercase function aliases
  alongside the idiomatic Python API, with documented compatibility behavior
  and intentional differences.

## Installation

```console
pip install bimets
```

## Quick start

Create a quarterly series with the user-oriented `timeseries()`
constructor, which returns a `BimetsSeries`:

```python
from bimets import timeseries

gdp = timeseries(
    [100.0, 102.0, 105.0, 107.0],
    start=(2020, 1),
    freq="Q",
    title="GDP",
)

growth = gdp.delta_percent()
growth.values
# array([2.        , 2.94117647, 1.9047619 ])

print(gdp)
#      Qtr1 Qtr2 Qtr3 Qtr4
# 2020  100  102  105  107
```

For users coming from BIMETS R, the uppercase `TIMESERIES()` constructor is
intentionally provided as a familiar entry point. The same example can be
written using the original BIMETS function names:

```python
from bimets import TIMESERIES, TSDELTAP

gdp = TIMESERIES(
    [100.0, 102.0, 105.0, 107.0],
    start=(2020, 1),
    freq="Q",
    title="GDP",
)

growth = TSDELTAP(gdp, lag=1)
growth.values
# array([2.        , 2.94117647, 1.9047619 ])

print(gdp)
#      Qtr1 Qtr2 Qtr3 Qtr4
# 2020  100  102  105  107
```

`TIMESERIES()` is an alias of `timeseries()`, so both constructors return the
same immutable `BimetsSeries` and accept the same arguments.

Printed series follow R's frequency-dependent layout: quarterly and monthly
series are arranged by year and cycle, while other frequencies use a compact
`Time Series` block. For debugging, `repr(gdp)` instead shows a concise value
preview with the range, frequency, and metadata. `TABIT()` and `tabulate()`
reuse the display rules in an aligned `Date`/`Prd.` table.

Named model data can be exchanged with pandas and updated immutably over an
inclusive year-period range:

```python
from bimets import BimetsDataset

data = BimetsDataset({"gdp": gdp})
frame = data.to_frame()
restored = BimetsDataset.from_frame(frame)
scenario = restored.assign_range(
    {"gdp": [110.0, 112.0]},
    start=(2020, 3),
    end=(2020, 4),
)
```

`assign_range()` leaves `data` and `restored` unchanged. Scalars are broadcast;
sequences must contain one value per selected period.

Individual series support BIMETS-compatible year-period and date indexing,
with immutable replacements through `with_values()`:

```python
gdp[[2020, 2]]  # 102.0
gdp["2020-04/2020-09"]  # inclusive date range

revised_gdp = gdp.with_values([[2020, 2], [2020, 3]], [103, 106])
```

The [time-series tutorial](https://github.com/andrea-cap/bimets/blob/master/docs/tutorial/02.timeseries.md) covers
calendar inspection, cumulative ranges, conversion, tabular display, and CSV
exchange.

Operations of `BimetsSeries` are available as functions and, where natural, as methods:

```python
from bimets import tsdeltap

functional = tsdeltap(gdp, lag=1)
method = gdp.delta_percent(lag=1)
```

Compatible BIMETS R function names are also exported with their original,
case-sensitive spelling. Most are uppercase; `date2yp` and `normalizeYP`
retain their mixed-case names as in the R implementation. These aliases reference the canonical Python
functions and retain their Python signatures. See the [public API
inventory](https://github.com/andrea-cap/bimets/blob/master/docs/reference/api.md)
for the complete alias mapping and the
[time-series API reference](https://github.com/andrea-cap/bimets/blob/master/docs/reference/timeseries.md)
for constructor,
function, and method signatures.

A small MDL model can be parsed and simulated directly:

```python
from bimets import BimetsModel, simulate, timeseries

model = BimetsModel.from_text(
    """MODEL
IDENTITY> y
EQ> y = x + 0.5 * TSLAG(y)
END""",
    name="dynamic-example",
)
data = {
    # The 1999 observation supplies the initial value for TSLAG(y).
    "y": timeseries([0, 0, 0, 0], start=(1999, 1)),
    "x": timeseries([1, 1, 1], start=(2000, 1)),
}

result = simulate(
    model,
    data,
    coefficients={},
    time_range=(2000, 1, 2002, 1),
)
result["y"].values.tolist()
# [1.0, 1.5, 1.75]
```

## Documentation

| Topic | Documentation |
|---|---|
| Index of detailed documentation | [Documentation](https://github.com/andrea-cap/bimets/blob/master/docs/README.md) |
| All tutorials | [Tutorial index](https://github.com/andrea-cap/bimets/blob/master/docs/tutorial/README.md) |
| Time-series construction, access, and indexing | [Time-series tutorial](https://github.com/andrea-cap/bimets/blob/master/docs/tutorial/02.timeseries.md) |
| Time-series manipulation | [Manipulating time series](https://github.com/andrea-cap/bimets/blob/master/docs/tutorial/03.manipulating-timeseries.md) |
| MDL, estimation, and simulation guides | [MDL](https://github.com/andrea-cap/bimets/blob/master/docs/tutorial/05.mdl.md) · [Estimation](https://github.com/andrea-cap/bimets/blob/master/docs/tutorial/06.estimation.md) · [Simulation](https://github.com/andrea-cap/bimets/blob/master/docs/tutorial/07.simulation.md) |
| API documentation | [API reference](https://github.com/andrea-cap/bimets/blob/master/docs/reference/README.md) |
| Public symbols and R aliases | [API inventory](https://github.com/andrea-cap/bimets/blob/master/docs/reference/api.md) |
| Conceptual and API differences from R | [Migration from BIMETS R](https://github.com/andrea-cap/bimets/blob/master/docs/explanation/migration-from-r.md) |
| Solver architecture, vectorization, and multiprocessing | [Solver strategies](https://github.com/andrea-cap/bimets/blob/master/docs/explanation/solver-strategies.md) |
| Compatibility and numerical validation | [Conformance page](https://github.com/andrea-cap/bimets/blob/master/docs/explanation/conformance.md) |
| Reproducible public examples | [`examples/`](https://github.com/andrea-cap/bimets/tree/master/examples) |


## Origin, copyright, and license

This project is a Python port of
[BIMETS](https://github.com/andrea-luciani/bimets), originally developed by
Andrea Luciani and Roberto Stok. The original BIMETS package is copyright
2021–2031 Bank of Italy and is distributed under the GNU General Public
License, version 3 or later.

The port is based on BIMETS 4.1.2. It includes adaptations of concepts, interfaces, documentation examples, test cases, and model definitions from the original package, together with a new Python implementation. Copyright in original or adapted material remains with its respective holders; copyright in new contributions remains with the
respective contributors unless otherwise agreed.

The complete Python distribution is licensed under the
[GNU General Public License, version 3 or later](https://github.com/andrea-cap/bimets/blob/master/LICENSE).
See [NOTICE](https://github.com/andrea-cap/bimets/blob/master/NOTICE) for the
full attribution and modification notice.

The names BIMETS and Bank of Italy are used for attribution and identification
of compatibility. This project does not imply endorsement by the original
authors or by Bank of Italy.
