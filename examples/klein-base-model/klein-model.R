#!/usr/bin/env Rscript

# Estimate and forecast the Klein model from the BIMETS concepts paper.
#
# Source: model, data, and forecast exercise are transcribed from section 3.1
# of https://doi.org/10.13140/RG.2.2.31160.83202.

suppressPackageStartupMessages(library(bimets))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
example_dir <- dirname(script_path)
model_path <- file.path(example_dir, "klein-model.mdl")
data_path <- file.path(example_dir, "klein-data.csv")
model_text <- paste(readLines(model_path, warn = FALSE), collapse = "\n")
forecast_range <- c(1941, 1, 1944, 1)

data_columns <- c("cn", "g", "i", "k", "p", "w1", "y", "t", "time", "w2")
klein_data <- CSV2BIMETS(data_path, mergedList = TRUE)

if (!identical(names(klein_data), data_columns)) {
  stop(sprintf(
    "expected CSV series %s, got %s",
    paste(data_columns, collapse = ", "),
    paste(names(klein_data), collapse = ", ")
  ))
}
if (any(vapply(klein_data, length, integer(1)) != length(klein_data[[1]])) ||
    !identical(start(klein_data[[1]]), c(1920, 1)) ||
    !identical(end(klein_data[[1]]), c(1941, 1))) {
  stop("expected annual observations for 1920--1941")
}

model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
model <- LOAD_MODEL_DATA(model, klein_data, quietly = TRUE)
model <- ESTIMATE(model, quietly = TRUE)

model$modelData <- within(model$modelData, {
  w2 <- TSEXTEND(w2, UPTO = forecast_range[3:4], EXTMODE = "CONSTANT")
  t <- TSEXTEND(t, UPTO = forecast_range[3:4], EXTMODE = "CONSTANT")
  g <- TSEXTEND(g, UPTO = forecast_range[3:4], EXTMODE = "CONSTANT")
  time <- TSEXTEND(time, UPTO = forecast_range[3:4], EXTMODE = "LINEAR")
})

model <- SIMULATE(
  model,
  simType = "FORECAST",
  TSRANGE = forecast_range,
  simConvergence = 1e-5,
  simIterLimit = 100,
  quietly = TRUE
)

cat("year,y\n")
years <- seq.int(start(model$simulation$y)[[1]], end(model$simulation$y)[[1]])
cat(sprintf("%d,%.5f\n", years, as.numeric(model$simulation$y)), sep = "")
