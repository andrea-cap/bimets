#!/usr/bin/env Rscript
library(bimets)

section <- function(title) {
  cat("\n", paste(rep("=", 78), collapse = ""), "\n", sep = "")
  cat(title, "\n")
  cat(paste(rep("=", 78), collapse = ""), "\n", sep = "")
}

show_object <- function(label, object) {
  cat("\n---", label, "---\n")
  print(object)
  invisible(object)
}

# ---- Series creation and metadata --------------------------------------------

section("SERIES CREATION AND METADATA")

annual_series <- TSERIES(
  c(100, 105, 111, 118, 126),
  START = c(2020, 1),
  FREQ = "A",
  TITLE = "Annual index",
  UNITS = "index",
  SOURCE = "synthetic",
  SCALEFAC = 0
)

semiannual_series <- TSERIES(
  1:8,
  START = c(2023, 1),
  FREQ = "S"
)

quarterly_series <- TSERIES(
  seq(100, 123),
  START = c(2023, 1),
  FREQ = "Q",
  TITLE = "Quarterly index",
  UNITS = "index"
)

monthly_series <- TSERIES(
  seq(100, 123),
  START = c(2024, 1),
  FREQ = "M",
  TITLE = "Monthly index",
  UNITS = "index"
)

weekly_series <- TSERIES(
  1:10,
  START = c(2025, 1),
  FREQ = "W"
)

daily_series <- TSERIES(
  1:20,
  START = as.Date("2025-01-01"),
  FREQ = "D"
)

show_object("Annual series", annual_series)
show_object("Quarterly series", quarterly_series)
show_object("Monthly series", monthly_series)

# TIMESERIES is an alias of TSERIES.
alias_series <- TIMESERIES(
  c(10, 20, 30),
  START = c(2022, 1),
  FREQ = "A"
)
show_object("Series created with TIMESERIES()", alias_series)

# ---- Inspection and compliance ----------------------------------------------

section("INSPECTION AND COMPLIANCE")

show_object("R class", class(monthly_series))
show_object("BIMETS-compliant?", is.bimets(monthly_series))
show_object("Range and frequency with TSLOOK()", TSLOOK(monthly_series))

show_object(
  "Frequencies with TSINFO(..., MODE = 'FREQ')",
  TSINFO(
    annual_series,
    quarterly_series,
    monthly_series,
    MODE = "FREQ"
  )
)

show_object(
  "Initial period with TSINFO(..., MODE = 'START2')",
  TSINFO(
    annual_series,
    quarterly_series,
    monthly_series,
    MODE = "START2"
  )
)

show_object(
  "Annual-series title",
  TSINFO(annual_series, MODE = "TITLE")
)

show_object(
  "Monthly-series dates",
  GETDATE(monthly_series)
)

show_object(
  "Monthly-series year-period values",
  GETYEARPERIOD(monthly_series)
)

cat("\nTABIT with multiple series:\n")
TABIT(
  quarterly_series,
  TSLAG(quarterly_series, 1),
  TSRANGE = c(2023, 1, 2024, 2),
  headers = c("original", "lag_1")
)

# ---- Indexing and modification ----------------------------------------------

section("INDEXING AND MODIFICATION")

# By position
show_object("First three observations", monthly_series[1:3])

modified_series <- monthly_series
modified_series[1] <- 999
show_object("Update by position", modified_series[1:3])

# By year-period
show_object(
  "Observation at 2024-03",
  monthly_series[[2024, 3]]
)

show_object(
  "Range 2024-03 / 2024-06",
  monthly_series[[c(2024, 3), c(2024, 6)]]
)

modified_series <- monthly_series
modified_series[[2024, 3]] <- NA
modified_series[[c(2024, 4), c(2024, 6)]] <- c(200, 201, 202)
show_object("Updates by year-period", modified_series)

# Assignment beyond the current end extends the series.
index_extended_series <- monthly_series
index_extended_series[[2026, 1]] <- c(200, 201, 202)
show_object(
  "Extension through assignment",
  TSLOOK(index_extended_series)
)

# By date
show_object(
  "Daily observation on January 5",
  daily_series["2025-01-05"]
)

show_object(
  "Daily range",
  daily_series["2025-01-05/2025-01-10"]
)

modified_daily_series <- daily_series
modified_daily_series["2025-01-05"] <- NA
modified_daily_series["2025-01-07/2025-01-09"] <- 50
show_object("Update by date", modified_daily_series)

# ---- Lags, leads, and differences -------------------------------------------

section("LAGS, LEADS, AND DIFFERENCES")

lag_1 <- TSLAG(monthly_series, L = 1)
lead_1 <- TSLEAD(monthly_series, L = 1)

delta_1 <- TSDELTA(monthly_series, L = 1)
second_order_delta <- TSDELTA(monthly_series, L = 1, O = 2)

percentage_delta <- TSDELTAP(monthly_series, L = 1)
annual_percentage_delta <- TSDELTAP(
  monthly_series,
  L = 12
)

log_delta <- TSDELTALOG(monthly_series, L = 1)

TABIT(
  monthly_series,
  lag_1,
  lead_1,
  delta_1,
  percentage_delta,
  log_delta,
  TSRANGE = c(2024, 1, 2024, 8),
  headers = c(
    "original",
    "lag_1",
    "lead_1",
    "delta",
    "delta_pct",
    "log_delta"
  ),
  digits = 4
)

show_object("Second-order difference", second_order_delta)
show_object("Twelve-month percentage change", annual_percentage_delta)

# ---- Moving and cumulative operations --------------------------------------

section("MOVING AND CUMULATIVE OPERATIONS")

series_with_na <- TSERIES(
  c(1, 2, 3, 4, NA, 6, 7, 8, 9, 10),
  START = c(2024, 1),
  FREQ = "M"
)

trailing_moving_average <- MOVAVG(
  series_with_na,
  L = 3,
  DIRECTION = "BACK",
  ignoreNA = TRUE
)

centered_moving_average <- MOVAVG(
  series_with_na,
  L = 3,
  DIRECTION = "CENTER",
  ignoreNA = TRUE
)

moving_sum <- MOVSUM(
  series_with_na,
  L = 3,
  DIRECTION = "BACK",
  ignoreNA = TRUE
)

# MOVTOT is equivalent to MOVSUM.
moving_sum_alias <- MOVTOT(
  series_with_na,
  L = 3,
  ignoreNA = TRUE
)

cumulative_sum <- CUMSUM(
  series_with_na,
  ignoreNA = TRUE
)

factor_series <- TSERIES(
  c(1.02, 1.01, 0.99, 1.03, 1.02),
  START = c(2024, 1),
  FREQ = "Q"
)

cumulative_product <- CUMPROD(factor_series)

TABIT(
  series_with_na,
  trailing_moving_average,
  centered_moving_average,
  moving_sum,
  cumulative_sum,
  headers = c(
    "original",
    "trailing_average",
    "centered_average",
    "moving_sum",
    "cumulative_sum"
  ),
  digits = 3
)

show_object("MOVSUM and MOVTOT return the same values", all.equal(
  as.numeric(moving_sum),
  as.numeric(moving_sum_alias)
))

TABIT(
  factor_series,
  cumulative_product,
  headers = c("factor", "cumulative_product"),
  digits = 5
)

# ---- Projection, extension, trim, join, and merge ---------------------------

section("PROJECTION, EXTENSION, TRIM, JOIN, AND MERGE")

projection <- TSPROJECT(
  monthly_series,
  TSRANGE = c(2024, 4, 2024, 9)
)
show_object("TSPROJECT: April-September 2024", projection)

missing_extension <- TSEXTEND(
  annual_series,
  BACKTO = c(2018, 1),
  UPTO = c(2027, 1),
  EXTMODE = "MISSING"
)

constant_extension <- TSEXTEND(
  annual_series,
  UPTO = c(2027, 1),
  EXTMODE = "CONSTANT"
)

rate_extension <- TSEXTEND(
  annual_series,
  UPTO = c(2027, 1),
  EXTMODE = "MYRATE",
  FACTOR = 5
)

TABIT(
  missing_extension,
  constant_extension,
  rate_extension,
  headers = c("missing", "constant", "increase_5")
)

series_to_trim <- TSERIES(
  c(NA, NA, 10, 20, 30, NA),
  START = c(2020, 1),
  FREQ = "A"
)

trimmed_series <- TSTRIM(series_to_trim)
TABIT(
  series_to_trim,
  trimmed_series,
  headers = c("with_missing_ends", "trimmed")
)

# TSJOIN uses the first series up to the beginning of the second one, then
# continues with the second series.
historical_series <- TSERIES(
  c(10, 11, 12, 13, 14, 15),
  START = c(2020, 1),
  FREQ = "A"
)

new_series <- TSERIES(
  c(100, 101, 102, 103),
  START = c(2023, 1),
  FREQ = "A"
)

joined_series <- TSJOIN(historical_series, new_series)
TABIT(
  historical_series,
  new_series,
  joined_series,
  headers = c("historical_series", "new_series", "join")
)

# TSMERGE senza fun prende il primo valore non NA in ordine di argomento.
first_series <- TSERIES(
  c(1, 2, NA, NA, 5, 6),
  START = c(2020, 1),
  FREQ = "A"
)

second_series <- TSERIES(
  c(30, 40, 50, 60),
  START = c(2022, 1),
  FREQ = "A"
)

priority_merge <- TSMERGE(first_series, second_series)
sum_merge <- TSMERGE(first_series, second_series, fun = "SUM", MV = FALSE)
average_merge <- TSMERGE(first_series, second_series, fun = "AVE", MV = FALSE)
maximum_merge <- TSMERGE(first_series, second_series, fun = "MAX", MV = FALSE)
minimum_merge <- TSMERGE(first_series, second_series, fun = "MIN", MV = FALSE)

TABIT(
  first_series,
  second_series,
  priority_merge,
  sum_merge,
  average_merge,
  maximum_merge,
  minimum_merge,
  headers = c(
    "first_series",
    "second_series",
    "priority",
    "sum",
    "average",
    "max",
    "min"
  )
)

# ---- Rebasing ----------------------------------------------------------------

section("REBASING")

index_base_2024 <- INDEXNUM(
  monthly_series,
  BASEYEAR = 2024
)

TABIT(
  monthly_series,
  index_base_2024,
  headers = c("original", "base_2024_equals_100"),
  digits = 3
)

# ---- Aggregation and disaggregation -----------------------------------------

section("AGGREGATION AND DISAGGREGATION")

# Aggregate monthly observations to quarterly frequency.
quarterly_average <- QUARTERLY(monthly_series, "AVE")
quarterly_sum <- QUARTERLY(monthly_series, "SUM")
quarterly_stock <- QUARTERLY(monthly_series, "STOCK")

TABIT(
  quarterly_average,
  quarterly_sum,
  quarterly_stock,
  headers = c("average", "sum", "stock")
)

# Aggregate monthly observations to annual frequency.
annual_average <- YEARLY(monthly_series, "AVE")
annual_sum <- YEARLY(monthly_series, "SUM")
annual_stock <- YEARLY(monthly_series, "STOCK")

TABIT(
  annual_average,
  annual_sum,
  annual_stock,
  headers = c("average", "sum", "stock")
)

# Aggregate to semiannual frequency.
semiannual_average <- SEMIANNUAL(monthly_series, "AVE")
show_object("Monthly -> semiannual using average", semiannual_average)

# Disaggregate by interpolation.
simple_annual_series <- TSERIES(
  c(100, 110, 125),
  START = c(2022, 1),
  FREQ = "A"
)

annual_to_quarterly <- QUARTERLY(
  simple_annual_series,
  "INTERP_CENTER"
)

annual_to_monthly <- MONTHLY(
  simple_annual_series,
  "INTERP_CENTER"
)

monthly_to_daily <- DAILY(
  TSERIES(
    c(100, 103, 107),
    START = c(2025, 1),
    FREQ = "M"
  ),
  "INTERP_CENTER"
)


show_object(
  "First observations of the interpolated monthly series",
  annual_to_monthly[1:8]
)

show_object(
  "First observations of the interpolated daily series",
  monthly_to_daily[1:10]
)

# Aggregate a daily series to monthly frequency.
long_daily_series <- TSERIES(
  1:90,
  START = as.Date("2025-01-01"),
  FREQ = "D"
)

daily_to_monthly_sum <- MONTHLY(long_daily_series, "SUM")
daily_to_monthly_average <- MONTHLY(long_daily_series, "AVE")
daily_to_monthly_stock <- MONTHLY(long_daily_series, "STOCK")

TABIT(
  daily_to_monthly_sum,
  daily_to_monthly_average,
  daily_to_monthly_stock,
  headers = c("sum", "average", "stock")
)

# ---- Conversion between classes ---------------------------------------------

section("TS / XTS CONVERSION")

xts_series <- fromBIMETStoXTS(monthly_series)
show_object("Class after fromBIMETStoXTS()", class(xts_series))
show_object("First xts rows", head(xts_series))

back_to_bimets_series <- as.bimets(xts_series)
show_object(
  "BIMETS-compliant again?",
  is.bimets(back_to_bimets_series)
)

base_ts_series <- fromBIMETStoTS(as.bimets(xts_series))
show_object("Class after fromBIMETStoTS()", class(base_ts_series))

# ---- CSV import/export -------------------------------------------------------

section("CSV IMPORT / EXPORT")

csv_file <- file.path(getwd(), "bimets_series.csv")

series_list <- list(
  monthly = monthly_series,
  quarterly = quarterly_series
)

BIMETS2CSV(
  series_list,
  cellSeparator = ";",
  decimalSeparator = ",",
  dateFormat = "%Y%m%d",
  filePath = csv_file,
  overWrite = TRUE
)

imported_series <- CSV2BIMETS(
  csv_file,
  cellSeparator = ";",
  decimalSeparator = ",",
  dateFormat = "%Y%m%d"
)

cat("\nCSV created at:\n", csv_file, "\n")
show_object("Names of imported series", names(imported_series))

for (name in names(imported_series)) {
  cat("\nImported series:", name, "\n")
  print(TSLOOK(imported_series[[name]]))
}

# ---- Final checks ------------------------------------------------------------

section("FINAL CHECKS")

stopifnot(
  is.bimets(annual_series),
  is.bimets(monthly_series),
  is.bimets(quarterly_average),
  is.bimets(annual_to_monthly),
  file.exists(csv_file)
)

cat(
  "\nExecution completed.\n",
  "Output available at:\n",
  csv_file,
  "\n",
  sep = ""
)
