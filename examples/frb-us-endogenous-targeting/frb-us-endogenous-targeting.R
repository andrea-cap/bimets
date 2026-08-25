#!/usr/bin/env Rscript

# Public FRB/US endogenous-targeting exercise with BIMETS R.
#
# Source: model, data, targets, and instruments are from the public BIMETS
# FRB/US vignette, section "Endogenous targeting":
# https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf.

suppressPackageStartupMessages(library(bimets))
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
example_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
model <- LOAD_MODEL(modelText = paste(readLines(file.path(example_dir, "frb-us-model.mdl"), warn = FALSE), collapse = "\n"), quietly = TRUE)
model <- LOAD_MODEL_DATA(model, CSV2BIMETS(file.path(example_dir, "frb-us-data.csv"), mergedList = TRUE), quietly = TRUE)
start <- c(2021, 3)
end <- c(2022, 3)
model$modelData$dfpdbt[[start, end]] <- 0
model$modelData$dfpsrp[[start, end]] <- 1
model <- SIMULATE(model, simType = "RESCHECK", TSRANGE = c(start, end), ZeroErrorAC = TRUE, quietly = TRUE)
model$modelData$lurnat[[start, end]] <- 3.78
targets <- list(
  xgdp = TSERIES(model$modelData$xgdp[[2021, 2]] * CUMPROD((c(6.8, 5.2, 4.5, 3.4, 2.7) / 100 + 1)^0.25), START = start, FREQ = 4),
  lur = TSERIES(c(5.3, 4.9, 4.6, 4.4, 4.2), START = start, FREQ = 4),
  picxfe = TSERIES(c(3.7, 2.2, 2.1, 2.1, 2.2), START = start, FREQ = 4),
  rff = TSERIES(rep(0.1, 5), START = start, FREQ = 4),
  rg10 = TSERIES(c(1.4, 1.6, 1.6, 1.7, 1.9), START = start, FREQ = 4)
)
model <- RENORM(model, simAlgo = "NEWTON", simConvergence = 0.01, simIterLimit = 100, TSRANGE = c(start, end), ConstantAdjustment = model$ConstantAdjustmentRESCHECK, TARGET = targets, INSTRUMENT = c("eco", "lhp", "picxfe", "rff", "rg10p"), BackFill = 8, quietly = TRUE)
values <- lapply(c("xgdp", "lur", "picxfe", "rff", "rg10"), function(name) as.numeric(model$renorm$TARGET[[name]]))
offsets <- seq.int(0, length(values[[1]]) - 1)
periods <- sprintf("%dQ%d", start[[1]] + (start[[2]] - 1 + offsets) %/% 4, (start[[2]] - 1 + offsets) %% 4 + 1)
cat("period,xgdp,lur,picxfe,rff,rg10\n")
cat(sprintf("%s,%.1f,%.3f,%.3f,%.3f,%.3f\n", periods, values[[1]], values[[2]], values[[3]], values[[4]], values[[5]]), sep = "")
