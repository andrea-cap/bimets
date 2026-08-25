#!/usr/bin/env Rscript

# Small synthetic advanced-estimation workload.
#
# Model and data have been generated specifically for this example. Python
# and R scripts read the same MDL and CSV files so to make their results comparable.

suppressPackageStartupMessages(library(bimets))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
example_dir <- dirname(script_path)
model_path <- file.path(example_dir, "advanced-estimation.mdl")
data_path <- file.path(example_dir, "advanced-estimation-data.csv")

model_data <- CSV2BIMETS(data_path, mergedList = TRUE)
model_text <- paste(readLines(model_path, warn = FALSE), collapse = "
")

metric <- function(name, value) cat(sprintf("%s,%.6f
", name, value))

model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
model <- LOAD_MODEL_DATA(model, model_data, quietly = TRUE)
pdl_equations <- names(Filter(function(eq) length(eq$pdlRaw) > 0, model$behaviorals))
iv_equations <- names(Filter(function(eq) length(eq$IV) > 0, model$behaviorals))

model <- ESTIMATE(model, eqList = pdl_equations, estTech = "OLS", quietly = TRUE)
ols_checksum <- sum(unlist(lapply(
  model$behaviorals[pdl_equations], function(eq) eq$coefficients
)))

model <- ESTIMATE(model, eqList = iv_equations, estTech = "IV", quietly = TRUE)
iv_checksum <- sum(unlist(lapply(
  model$behaviorals[iv_equations], function(eq) eq$coefficients
)))

model <- ESTIMATE(
  model,
  eqList = pdl_equations[[1]],
  estTech = "OLS",
  CHOWTEST = TRUE,
  CHOWPAR = c(1999, 4),
  quietly = TRUE
)
chow <- model$behaviorals[[pdl_equations[[1]]]]$ChowTest$Fvalue

metric("equations", length(model$vendog))
metric("ols_coefficient_checksum", ols_checksum)
metric("iv_coefficient_checksum", iv_checksum)
metric("chow_f_statistic", chow)
