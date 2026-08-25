#!/usr/bin/env Rscript

# Large synthetic conditional-policy workload.
#
# Model and data have been generated specifically for this example. Python
# and R scripts read the same MDL and CSV files so to make their results comparable.

suppressPackageStartupMessages(library(bimets))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
example_dir <- dirname(script_path)
model_path <- file.path(example_dir, "conditional-policy.mdl")
data_path <- file.path(example_dir, "conditional-policy-data.csv")

model_data <- CSV2BIMETS(data_path, mergedList = TRUE)
model_text <- paste(readLines(model_path, warn = FALSE), collapse = "
")

metric <- function(name, value) cat(sprintf("%s,%.6f
", name, value))

simulation_range <- c(1941, 1, 1999, 4)
exogenization <- list(y002 = c(1960, 1, 1965, 1))
adjustment_range <- window(
  model_data$y001,
  start = simulation_range[1:2],
  end = simulation_range[3:4]
)
adjustments <- list(y001 = adjustment_range * 0 + 0.02)

run <- function(simulation_type) {
  model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
  model <- LOAD_MODEL_DATA(model, model_data, quietly = TRUE)
  model <- SIMULATE(
    model,
    TSRANGE = simulation_range,
    simType = simulation_type,
    simAlgo = "NEWTON",
    simConvergence = 1e-7,
    simIterLimit = 100,
    ConstantAdjustment = adjustments,
    Exogenize = exogenization,
    JacobianDrop = "y001",
    quietly = TRUE
  )
  model
}

dynamic <- run("DYNAMIC")
static <- run("STATIC")
metric("equations", length(dynamic$vendog))
metric("dynamic_aggregate_checksum", sum(dynamic$simulation$aggregate))
metric("static_aggregate_checksum", sum(static$simulation$aggregate))
metric("dynamic_last_aggregate", tail(dynamic$simulation$aggregate, 1))
