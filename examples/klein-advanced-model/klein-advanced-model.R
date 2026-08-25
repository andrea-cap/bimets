#!/usr/bin/env Rscript

# Advanced Klein exercises with BIMETS R.
#
# Source: model, annual data, stochastic forecast, and optimal-control exercise
# are transcribed from sections 3.3, 3.7, and 3.10 of the public paper:
# https://doi.org/10.13140/RG.2.2.31160.83202.

suppressPackageStartupMessages(library(bimets))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
example_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
model_text <- paste(readLines(file.path(example_dir, "klein-advanced-model.mdl"), warn = FALSE), collapse = "\n")
historical <- CSV2BIMETS(file.path(example_dir, "klein-data.csv"), mergedList = TRUE)

load_estimated_model <- function(up_to = NULL) {
  model <- LOAD_MODEL(modelText = model_text, quietly = TRUE)
  model <- LOAD_MODEL_DATA(model, historical, quietly = TRUE)
  model <- ESTIMATE(model, quietly = TRUE)
  if (!is.null(up_to)) {
    for (name in names(model$modelData)) {
      mode <- if (name %in% c("w2", "g")) "CONSTANT" else if (name %in% c("t", "k", "time")) "LINEAR" else "MISSING"
      model$modelData[[name]] <- TSEXTEND(model$modelData[[name]], UPTO = up_to, EXTMODE = mode)
    }
  }
  model
}

model <- load_estimated_model()
cat("estimation,parameter,value\n")
for (equation in c("cn", "i", "w1")) {
  result <- model$behaviorals[[equation]]
  values <- setNames(as.numeric(result$coefficients), rownames(result$coefficients))
  if (!is.null(result$errorCoefficients)) {
    values <- c(values, setNames(as.numeric(result$errorCoefficients), rownames(result$errorCoefficients)))
  }
  for (name in names(values)) cat(sprintf("%s,%s,%.7f\n", equation, name, values[[name]]))
}

model <- load_estimated_model(c(1944, 1))
stoch_structure <- list(
  cn = list(TSRANGE = c(1942, 1, 1942, 1), TYPE = "NORM", PARS = c(0, model$behaviorals$cn$statistics$StandardErrorRegression)),
  g = list(TSRANGE = TRUE, TYPE = "UNIF", PARS = c(-1, 1))
)
model <- STOCHSIMULATE(model, simType = "FORECAST", TSRANGE = c(1941, 1, 1944, 1), StochStructure = stoch_structure, StochReplica = 100, StochSeed = 123, quietly = TRUE)
cat("stochastic,year,y_mean,y_sd\n")
for (position in seq_along(model$stochastic_simulation$y$mean)) {
  cat(sprintf("stochastic,%d,%.5f,%.6f\n", 1940 + position, model$stochastic_simulation$y$mean[[position]], model$stochastic_simulation$y$sd[[position]]))
}

model <- load_estimated_model(c(1942, 1))
model <- OPTIMIZE(
  model,
  simType = "FORECAST", TSRANGE = c(1942, 1, 1942, 1),
  simConvergence = 1e-4, simIterLimit = 1000,
  StochReplica = 10000, StochSeed = 123,
  OptimizeBounds = list(cn = list(TSRANGE = TRUE, BOUNDS = c(-5, 5)), g = list(TSRANGE = TRUE, BOUNDS = c(15, 25))),
  OptimizeRestrictions = list(restriction = list(TSRANGE = TRUE, INEQUALITY = "g+(cn^2)/2<27 & g+cn>17")),
  OptimizeFunctions = list(objective = list(TSRANGE = TRUE, FUNCTION = "(y-110)+(cn-90)*ABS(cn-90)-(g-20)^0.5")),
  quietly = TRUE
)
cat("optimization,objective,cn_add_factor,g\n")
cat(sprintf("optimization,%.5f,%.6f,%.5f\n", model$optimize$optFunMax, as.numeric(model$optimize$INSTRUMENT$cn)[[1]], as.numeric(model$optimize$INSTRUMENT$g)[[1]]))
