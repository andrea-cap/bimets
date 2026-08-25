#!/usr/bin/env Rscript

# Public FRB/US monetary-policy shock with BIMETS R.
#
# Source: the exercise is from
# https://r-consortium.org/posts/us-federal-reserve-quarterly-model-in-r/.
# The February 2024 model and data are extracted from the public FRB__MODEL
# and LONGBASE datasets distributed with BIMETS R.

suppressPackageStartupMessages(library(bimets))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
example_dir <- dirname(script_path)
model_path <- file.path(example_dir, "frb-us-model.mdl")
data_path <- file.path(example_dir, "frb-us-data.csv")

model_text <- paste(readLines(model_path, warn = FALSE), collapse = "\n")
model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
model_data <- CSV2BIMETS(data_path, mergedList = TRUE)
if (any(vapply(model_data, length, integer(1)) != length(model_data[[1]])) ||
    !identical(start(model_data[[1]]), c(2036, 1)) ||
    !identical(end(model_data[[1]]), c(2045, 4))) {
  stop("expected consecutive quarterly observations for 2036Q1--2045Q4")
}
missing_variables <- setdiff(c(model$vendog, model$vexog), names(model_data))
if (length(missing_variables) > 0) {
  stop(sprintf(
    "CSV is missing FRB/US model variables: %s",
    paste(missing_variables, collapse = ", ")
  ))
}

model <- LOAD_MODEL_DATA(model, model_data, quietly = TRUE)

start <- c(2040, 1)
end <- c(2045, 4)
model$modelData$dfpdbt[[start, end]] <- 0
model$modelData$dfpsrp[[start, end]] <- 1

model <- SIMULATE(
  model,
  simType = "RESCHECK",
  TSRANGE = c(start, end),
  ZeroErrorAC = TRUE,
  quietly = TRUE
)
adjustments <- model$ConstantAdjustmentRESCHECK
adjustments$rffintay[[start]] <- adjustments$rffintay[[start]] + 1

model <- SIMULATE(
  model,
  simAlgo = "NEWTON",
  simConvergence = 0.01,
  simIterLimit = 100,
  TSRANGE = c(start, end),
  ConstantAdjustment = adjustments,
  BackFill = 12,
  quietly = TRUE
)

values <- as.numeric(window(model$simulation$xgdp, start = start, end = end))
offsets <- seq.int(0, length(values) - 1)
periods <- sprintf(
  "%dQ%d",
  start[[1]] + (start[[2]] - 1 + offsets) %/% frequency(model$simulation$xgdp),
  (start[[2]] - 1 + offsets) %% frequency(model$simulation$xgdp) + 1
)
cat("period,xgdp\n")
cat(sprintf("%s,%.0f\n", periods, values), sep = "")
