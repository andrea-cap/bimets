#!/usr/bin/env Rscript

# Small synthetic stochastic-policy workload.
#
# Model and data have been generated specifically for this example. Python
# and R scripts read the same MDL and CSV files so to make their results comparable.

suppressPackageStartupMessages(library(bimets))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
example_dir <- dirname(script_path)
model_path <- file.path(example_dir, "stochastic-policy.mdl")
data_path <- file.path(example_dir, "stochastic-policy-data.csv")

model_data <- CSV2BIMETS(data_path, mergedList = TRUE)
model_text <- paste(readLines(model_path, warn = FALSE), collapse = "
")

metric <- function(name, value) cat(sprintf("%s,%.6f
", name, value))

simulation_range <- c(2010, 1, 2011, 4)
base_model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
base_model <- LOAD_MODEL_DATA(base_model, model_data, quietly = TRUE)

stochastic <- STOCHSIMULATE(
  base_model,
  TSRANGE = simulation_range,
  simAlgo = "NEWTON",
  simConvergence = 1e-7,
  simIterLimit = 100,
  StochStructure = list(
    y001 = list(TSRANGE = TRUE, TYPE = "NORM", PARS = c(0, 0.05)),
    x001 = list(TSRANGE = TRUE, TYPE = "UNIF", PARS = c(-0.10, 0.10))
  ),
  StochReplica = 100,
  StochSeed = 123,
  quietly = TRUE
)
multipliers <- MULTMATRIX(
  base_model,
  TSRANGE = simulation_range,
  simAlgo = "NEWTON",
  simConvergence = 1e-7,
  simIterLimit = 100,
  TARGET = c("aggregate", "y001"),
  INSTRUMENT = c("x001", "y001"),
  quietly = TRUE
)
baseline <- SIMULATE(
  base_model,
  TSRANGE = simulation_range,
  simAlgo = "NEWTON",
  simConvergence = 1e-7,
  simIterLimit = 100,
  quietly = TRUE
)
desired <- baseline$simulation$aggregate * 1.01
targeting <- RENORM(
  base_model,
  TSRANGE = simulation_range,
  simAlgo = "NEWTON",
  simConvergence = 1e-7,
  simIterLimit = 100,
  TARGET = list(aggregate = desired),
  INSTRUMENT = "x001",
  renormIterLimit = 6,
  renormConvergence = 1e-6,
  quietly = TRUE
)
optimum <- OPTIMIZE(
  base_model,
  TSRANGE = simulation_range,
  simAlgo = "NEWTON",
  simConvergence = 1e-7,
  simIterLimit = 100,
  OptimizeBounds = list(x001 = list(TSRANGE = TRUE, BOUNDS = c(0.5, 1.5))),
  OptimizeRestrictions = list(r = list(
    TSRANGE = TRUE, INEQUALITY = "x001 >= 0.52 & x001 <= 1.48"
  )),
  OptimizeFunctions = list(f = list(
    TSRANGE = TRUE, FUNCTION = "aggregate-0.01*x001^2"
  )),
  StochReplica = 100,
  StochSeed = 321,
  quietly = TRUE
)
metric("equations", length(base_model$vendog))
metric("stochastic_mean_checksum", sum(stochastic$stochastic_simulation$aggregate$mean))
metric("stochastic_sd_checksum", sum(stochastic$stochastic_simulation$aggregate$sd))
metric("multiplier_checksum", sum(multipliers$MultiplierMatrix))
metric("renorm_instrument_checksum", sum(targeting$renorm$INSTRUMENT$x001))
metric("optimum", optimum$optimize$optFunMax)
