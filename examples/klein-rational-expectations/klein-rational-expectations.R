#!/usr/bin/env Rscript

# Solve the public forward-looking Klein example with BIMETS R.
#
# Source: model and terminal-value experiment are from the BIMETS R SIMULATE
# example and the original public repository:
# https://github.com/andrea-luciani/bimets#rational-expectations.

suppressPackageStartupMessages(library(bimets))
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
example_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
model_text <- paste(readLines(file.path(example_dir, "klein-rational-expectations.mdl"), warn = FALSE), collapse = "\n")
model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
model <- LOAD_MODEL_DATA(model, CSV2BIMETS(file.path(example_dir, "klein-data.csv"), mergedList = TRUE), quietly = TRUE)
model$modelData$i[[1931, 1]] <- 2
model <- ESTIMATE(model, quietly = TRUE)
model <- SIMULATE(model, simAlgo = "NEWTON", simConvergence = 1e-6, simIterLimit = 200, TSRANGE = c(1924, 1, 1930, 1), quietly = TRUE)

values <- as.numeric(window(model$simulation$i, start = c(1924, 1), end = c(1930, 1)))
cat("year,i\n")
cat(sprintf("%d,%.6f\n", 1923 + seq_along(values), values), sep = "")
