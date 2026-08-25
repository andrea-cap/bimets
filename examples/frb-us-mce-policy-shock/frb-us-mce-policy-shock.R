#!/usr/bin/env Rscript

# Public forward-looking FRB/US policy shock with BIMETS R.
#
# Source: MCE model, data, and exercise are from the public BIMETS FRB/US
# vignette, section "Rational expectations":
# https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf.

suppressPackageStartupMessages(library(bimets))
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
example_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
model_text <- paste(readLines(file.path(example_dir, "frb-us-mce-model.mdl"), warn = FALSE), collapse = "\n")
model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
model_data <- CSV2BIMETS(file.path(example_dir, "frb-us-data.csv"), mergedList = TRUE)
model <- LOAD_MODEL_DATA(model, model_data, quietly = TRUE)
start <- c(2040, 1)
end <- c(2042, 1)
model$modelData$dfpdbt[[start, end]] <- 0
model$modelData$dfpsrp[[start, end]] <- 1
model$modelData$drstar[[start, end]] <- 0
model$modelData$drstar[[c(2041, 1), end]] <- 1
model <- SIMULATE(model, simType = "RESCHECK", TSRANGE = c(start, end), ZeroErrorAC = TRUE, quietly = TRUE)
adjustments <- model$ConstantAdjustmentRESCHECK
adjustments$rffintay[[start]] <- adjustments$rffintay[[start]] + 1
model <- SIMULATE(model, simAlgo = "NEWTON", simConvergence = 0.01, simIterLimit = 100, TSRANGE = c(start, end), ConstantAdjustment = adjustments, BackFill = 12, quietly = TRUE)

xgdp <- as.numeric(window(model$simulation$xgdp, start = start, end = end))
rff <- as.numeric(window(model$simulation$rff, start = start, end = end))
offsets <- seq.int(0, length(xgdp) - 1)
periods <- sprintf("%dQ%d", start[[1]] + (start[[2]] - 1 + offsets) %/% 4, (start[[2]] - 1 + offsets) %% 4 + 1)
cat("period,xgdp,rff\n")
cat(sprintf("%s,%.2f,%.5f\n", periods, xgdp, rff), sep = "")
