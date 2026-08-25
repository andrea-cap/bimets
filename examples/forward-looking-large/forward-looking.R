#!/usr/bin/env Rscript

# Large synthetic forward-looking workload.
#
# Model and data have been generated specifically for this example. Python
# and R scripts read the same MDL and CSV files so to make their results comparable.

suppressPackageStartupMessages(library(bimets))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
example_dir <- dirname(script_path)
model_path <- file.path(example_dir, "forward-looking.mdl")
data_path <- file.path(example_dir, "forward-looking-data.csv")

model_data <- CSV2BIMETS(data_path, mergedList = TRUE)
model_text <- paste(readLines(model_path, warn = FALSE), collapse = "
")

metric <- function(name, value) cat(sprintf("%s,%.6f
", name, value))

simulation_range <- c(1960, 2, 2000, 1)
shock_period <- c(1980, 2)

solve <- function(data) {
  model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
  model <- LOAD_MODEL_DATA(model, data, quietly = TRUE)
  model <- SIMULATE(
    model,
    TSRANGE = simulation_range,
    simAlgo = "NEWTON",
    simConvergence = 1e-8,
    simIterLimit = 100,
    quietly = TRUE
  )
  model
}

baseline <- solve(model_data)
shocked_data <- model_data
shocked_data$x001[[shock_period]] <- shocked_data$x001[[shock_period]] + 1
shocked <- solve(shocked_data)
response_series <- shocked$simulation$y001 - baseline$simulation$y001
shock_response <- tail(window(response_series, end = shock_period), 2)
metric("equations", length(baseline$vendog))
metric("baseline_checksum", sum(unlist(baseline$simulation[names(baseline$simulation) != "__SIM_PARAMETERS__"])))
metric("response_before_shock", shock_response[[1]])
metric("response_at_shock", shock_response[[2]])
metric("response_last_period", tail(response_series, 1))
