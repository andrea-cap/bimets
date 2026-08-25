#!/usr/bin/env Rscript

# Public FRB/US persistent tracking-residual exercise with BIMETS R.
#
# Source: model, data, shocks, and persistence rule are from the public BIMETS
# FRB/US vignette, section "Auto-correlation on tracking residuals":
# https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf.

suppressPackageStartupMessages(library(bimets))
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
example_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
model <- LOAD_MODEL(modelText = paste(readLines(file.path(example_dir, "frb-us-model.mdl"), warn = FALSE), collapse = "\n"), quietly = TRUE)
model <- LOAD_MODEL_DATA(model, CSV2BIMETS(file.path(example_dir, "frb-us-data.csv"), mergedList = TRUE), quietly = TRUE)
start <- c(2040, 1)
end <- c(2046, 1)
for (name in c("dfpdbt", "dmpintay")) model$modelData[[name]][[start, end]] <- 0
for (name in c("dfpsrp", "dmptay", "dmptrsh")) model$modelData[[name]][[start, end]] <- 1
model$modelData$lurtrsh[[start, end]] <- 6
model$modelData$pitrsh[[start, end]] <- 3
model <- SIMULATE(model, simType = "RESCHECK", TSRANGE = c(start, end), ZeroErrorAC = TRUE, quietly = TRUE)
trac <- model$ConstantAdjustmentRESCHECK
for (name in c("rfftay", "rffrule", "rff", "dmptpi", "dmptlur", "dmptmax", "dmptr")) trac[[name]][[start, end]] <- 0
aerr <- list(
  eco = TSERIES(c(-0.002, -0.0016, -0.0070, -0.0045), START = start, FREQ = 4),
  ecd = TSERIES(c(-0.0319, -0.0154, -0.0412, -0.0838), START = start, FREQ = 4),
  eh = TSERIES(c(-0.0512, -0.0501, -0.0124, -0.0723), START = start, FREQ = 4),
  rbbbp = TSERIES(c(0.3999, 2.7032, 0.3391, -0.7759), START = start, FREQ = 4),
  lhp = TSERIES(c(-0.0029, -0.0048, -0.0119, -0.0085, -0.0074, -0.0061, -0.0077, -0.0033, -0.0042), START = start, FREQ = 4)
)
for (name in names(aerr)) {
  aerr[[name]] <- TSEXTEND(aerr[[name]], UPTO = end, EXTMODE = "MYRATE", FACTOR = 0.5)
}
aerr$dmptr <- TSERIES(-1, START = start, FREQ = 4)
aerr$dmptlur <- TSERIES(c(-1, -1, -1), START = start, FREQ = 4)
for (name in names(aerr)) trac[[name]] <- trac[[name]] + aerr[[name]]
model <- SIMULATE(model, simAlgo = "NEWTON", simConvergence = 0.01, simIterLimit = 100, TSRANGE = c(start, end), ConstantAdjustment = trac, BackFill = 12, quietly = TRUE)
values <- lapply(c("lur", "picxfe", "rff"), function(name) as.numeric(window(model$simulation[[name]], start = start, end = end)))
offsets <- seq.int(0, length(values[[1]]) - 1)
periods <- sprintf("%dQ%d", start[[1]] + (start[[2]] - 1 + offsets) %/% 4, (start[[2]] - 1 + offsets) %% 4 + 1)
cat("period,lur,picxfe,rff\n")
cat(sprintf("%s,%.1f,%.1f,%.1f\n", periods, values[[1]], values[[2]], values[[3]]), sep = "")
