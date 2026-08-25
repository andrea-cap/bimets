# Compatibility and numerical validation

[Back to the documentation index](../README.md)

## Compatibility scope

The compatibility baseline is BIMETS R 4.1.2. The evidence on this page comes
from paired R and Python scripts in [`examples/`](../../examples/).

Each example contains one R script and one Python script. Model examples read
the same external MDL and CSV inputs. The scripts are executed against BIMETS R
4.1.2 and the current bimets Python source, and their final numerical outputs
are compared directly.

In this context, compatibility means equivalent mathematical behavior where Python and R perform a conceptual operation. It does not require identical object representation,
mutability, display, exception text, or process-global random state.

The examples fall into two groups:

- **Public examples** reproduce material extracted from Andrea Luciani,
  *bimets: Time Series and Econometric Modeling in R*,
  [doi:10.13140/RG.2.2.31160.83202](https://doi.org/10.13140/RG.2.2.31160.83202),
  the original BIMETS repository, the
  [FRB/US BIMETS vignette](https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf),
  and the
  [R Consortium FRB/US article](https://r-consortium.org/posts/us-federal-reserve-quarterly-model-in-r/).
- **Synthetic examples** use models and observations generated specifically for
  the examples and their cross-language validation.

API mappings, intentional differences, and R-specific facilities not ported are
documented in [Key differences from BIMETS R](migration-from-r.md). Canonical
Python names and BIMETS R compatibility aliases are listed in the
[public API inventory](../reference/api.md).

## Numerical validation

The complete paired-example set currently contains one time-series workflow and fifteen model workflows:

| Example group | Main functionality exercised | Comparison result |
|---|---|---|
| `timeseries` | Construction, indexing, immutable updates, transformations, frequency conversion, pandas/xts interoperability, and CSV round trips | Numerical content agrees; CSV content is identical after newline normalization |
| `klein-base-model` | OLS estimation and deterministic forecast | Exact matching |
| `klein-advanced-model` | AR errors, coefficient restrictions, PDL, stochastic forecast, and optimal control | Exact matching |
| `klein-rational-expectations` | Forward-looking equations and terminal values | Exact matching |
| `frb-us-policy-shock` | RESCHECK, tracking residuals, dynamic Newton simulation, and policy shock | Exact matching |
| `frb-us-mce-policy-shock` | Forward-looking FRB/US MCE simulation | Exact matching |
| `frb-us-tracking-residuals` | Persistent residual shocks and policy thresholds | Exact matching |
| `frb-us-endogenous-targeting` | Multi-target, multi-instrument RENORM | Exact matching |
| `advanced-estimation-small/large` | Restrictions, PDL, AR errors, and large estimation systems | Exact matching |
| `conditional-policy-small/large` | Conditional equations, dynamic/static paths, and Newton simulation | Exact matching |
| `forward-looking-small/large` | Computationally larger lead systems | Exact matching |
| `stochastic-policy-small` | Seeded stochastic simulation, multiplier matrices, RENORM, and optimization | Exact matching |
| `stochastic-policy-large` | Large seeded stochastic and policy-control workflow | Maximum observed difference: `1e-6` in the RENORM checksum |

Scripts in `timeseries` intentionally use their host language's natural object and display representations. Their complete console output is therefore not
expected to be textually identical. The generated CSV observations and the displayed numerical operations agree; the raw CSV files differ only because Python writes CRLF line endings while R writes LF line endings.

These examples provide reproducible conformance evidence for the paths they
exercise. They should not be interpreted as proof that every possible model,
calendar, missing-value pattern, or solver configuration behaves identically.

## Some performance measures

The following wall-clock measurements were collected by timing each script
externally on the development machine. They include interpreter startup and
file I/O. Each value is from a single run on the same machine, so the table is
an indicative comparison rather than a statistically controlled benchmark. R
execution time is the `100%` baseline: values below `100%` indicate that Python
completed faster, while values above `100%` indicate that Python took longer.

| Example | Python execution time relative to R |
|---|---:|
| `advanced-estimation-small` | 34.9% |
| `advanced-estimation-large` | 17.9% |
| `conditional-policy-small` | 28.1% |
| `conditional-policy-large` | 116.9% |
| `forward-looking-small` | 10.4% |
| `forward-looking-large` | 0.5% |
| `stochastic-policy-small` | 31.4% |
| `stochastic-policy-large` | 239.8% |
| `klein-base-model` | 56.0% |
| `klein-advanced-model` | 117.5% |
| `klein-rational-expectations` | 68.4% |
| `frb-us-policy-shock` | 25.3% |
| `frb-us-mce-policy-shock` | 22.8% |
| `frb-us-tracking-residuals` | 20.1% |
| `frb-us-endogenous-targeting` | 24.7% |
| `timeseries` | 36.4% |

Python is faster in most of these single runs, with the largest difference in
the forward-looking large workload. BIMETS R remains faster in the current
`conditional-policy-large`, `stochastic-policy-large`, and
`klein-advanced-model` executions. These measurements describe the present
examples and environment only; they are not general performance guarantees.
