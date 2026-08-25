"""Single-equation estimators for MDL behavioral equations."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bimets.mdl._binding import BoundModel, bind_model_data
from bimets.mdl._evaluation import MdlValue, evaluate_expression
from bimets.mdl._expression import MdlExpression, parse_expression
from bimets.mdl._model import BehavioralEquation, BimetsModel, MdlTimeRange
from bimets.timeseries import (
    BimetsDataset,
    BimetsMask,
    BimetsSeries,
    YearPeriod,
    get_year_periods,
)


@dataclass(frozen=True, slots=True)
class ChowTestResult:
    """Structural-stability and predictive-power results.

    Attributes
    ----------
    f_statistic, f_probability : float
        Chow statistic and upper-tail F probability.
    numerator_degrees_of_freedom, denominator_degrees_of_freedom : int
        Degrees of freedom of the F distribution.
    base_start, base_end, extended_end : YearPeriod
        Base estimation bounds and final extended bound.
    actual, predicted, errors, standard_errors, t_statistics : BimetsSeries
        Out-of-sample predictive-power series.
    """

    f_statistic: float
    f_probability: float
    numerator_degrees_of_freedom: int
    denominator_degrees_of_freedom: int
    base_start: YearPeriod
    base_end: YearPeriod
    extended_end: YearPeriod
    actual: BimetsSeries
    predicted: BimetsSeries
    errors: BimetsSeries
    standard_errors: BimetsSeries
    t_statistics: BimetsSeries

    def summary(self) -> pd.DataFrame:
        """Return predictive-power observations as a pandas DataFrame."""
        return pd.DataFrame(
            {
                "actual": self.actual.values,
                "predicted": self.predicted.values,
                "error": self.errors.values,
                "standard_error": self.standard_errors.values,
                "t_statistic": self.t_statistics.values,
            },
            index=get_year_periods(self.actual),
        )


@dataclass(frozen=True, slots=True)
class EquationEstimationResult:
    """Estimation result for one behavioral equation.

    Attributes
    ----------
    name, method : str
        Equation name and estimation method (``"OLS"`` or ``"IV"``).
    coefficients : mapping of str to float
        Estimated coefficients in expanded declaration order.
    coefficient_standard_errors, t_statistics, coefficient_p_values : mapping
        of str to float
        Coefficient diagnostics.
    fitted_values, residuals : BimetsSeries
        Fitted values on the original scale and estimation residuals. With an
        autoregressive error, ``residuals`` contains transformed residuals.
    residuals_without_error_correction : BimetsSeries or None
        Original-scale residuals for a Cochrane--Orcutt estimation.
    autoregressive_coefficients, autoregressive_standard_errors,
        autoregressive_t_statistics, autoregressive_p_values : mapping
        Estimated ``RHO_1``, ..., ``RHO_n`` values and diagnostics.
    sample_start, sample_end : YearPeriod
        Inclusive estimation bounds.
    observations, degrees_of_freedom : int
        Sample size and residual degrees of freedom.
    residual_sum_squares : float
        Sum of squared estimation residuals.
    standard_error, standard_error_not_centered : float
        Centered and uncentered regression standard errors.
    r_squared, adjusted_r_squared : float
        Coefficients of determination.
    durbin_watson : float
        Durbin--Watson statistic calculated from estimation residuals.
    log_likelihood, f_statistic, f_probability, aic, bic : float
        Regression diagnostics following the original BIMETS definitions.
    restriction_f_statistic, restriction_f_probability : float or None
        F test comparing restricted and unrestricted estimates.
    covariance : numpy.ndarray
        Read-only coefficient covariance matrix.
    regressor_matrix, dependent_values : numpy.ndarray
        Read-only original-scale estimation sample used by the equation.
    chow_test : ChowTestResult or None
        Optional structural-stability analysis.
    """

    name: str
    method: str
    coefficients: Mapping[str, float]
    coefficient_standard_errors: Mapping[str, float]
    t_statistics: Mapping[str, float]
    coefficient_p_values: Mapping[str, float]
    fitted_values: BimetsSeries
    residuals: BimetsSeries
    residuals_without_error_correction: BimetsSeries | None
    autoregressive_coefficients: Mapping[str, float]
    autoregressive_standard_errors: Mapping[str, float]
    autoregressive_t_statistics: Mapping[str, float]
    autoregressive_p_values: Mapping[str, float]
    autoregressive_iterations: int
    sample_start: YearPeriod
    sample_end: YearPeriod
    observations: int
    degrees_of_freedom: int
    residual_sum_squares: float
    standard_error: float
    standard_error_not_centered: float
    r_squared: float
    adjusted_r_squared: float
    durbin_watson: float
    log_likelihood: float
    f_statistic: float
    f_probability: float
    aic: float
    bic: float
    restriction_f_statistic: float | None
    restriction_f_probability: float | None
    covariance: NDArray[np.float64]
    regressor_matrix: NDArray[np.float64]
    dependent_values: NDArray[np.float64]
    chow_test: ChowTestResult | None = None

    def __post_init__(self) -> None:
        for name in (
            "coefficients",
            "coefficient_standard_errors",
            "t_statistics",
            "coefficient_p_values",
            "autoregressive_coefficients",
            "autoregressive_standard_errors",
            "autoregressive_t_statistics",
            "autoregressive_p_values",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("covariance", "regressor_matrix", "dependent_values"):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def summary(self) -> pd.DataFrame:
        """Return coefficient estimates and diagnostics as a pandas DataFrame."""
        return pd.DataFrame(
            {
                "coefficient": self.coefficients,
                "standard_error": self.coefficient_standard_errors,
                "t_statistic": self.t_statistics,
                "p_value": self.coefficient_p_values,
            }
        )


class ModelEstimationResult(Mapping[str, EquationEstimationResult]):
    """Immutable collection of behavioral-equation estimates."""

    __slots__ = ("_equations", "method", "model_name")

    def __init__(
        self,
        model_name: str,
        equations: Mapping[str, EquationEstimationResult],
        *,
        method: str = "OLS",
    ) -> None:
        self.model_name = model_name
        self.method = method
        self._equations = MappingProxyType(dict(equations))

    def __getitem__(self, name: str) -> EquationEstimationResult:
        return self._equations[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._equations)

    def __len__(self) -> int:
        return len(self._equations)

    def __repr__(self) -> str:
        return (
            f"ModelEstimationResult(model_name={self.model_name!r}, "
            f"method={self.method!r}, equations={tuple(self)!r})"
        )

    def summary(self) -> pd.DataFrame:
        """Return all coefficient results as a pandas DataFrame."""
        return pd.concat(
            {name: result.summary() for name, result in self.items()},
            names=("equation", "coefficient_name"),
        )


@dataclass(slots=True)
class _RegressionFit:
    beta: NDArray[np.float64]
    covariance_factor: NDArray[np.float64]
    unrestricted_beta: NDArray[np.float64] | None


@dataclass(slots=True)
class _AutoregressiveFit:
    coefficients: NDArray[np.float64]
    covariance_factor: NDArray[np.float64]
    innovations: NDArray[np.float64]


def estimate(
    model: BimetsModel | BoundModel,
    data: BimetsDataset | Mapping[str, BimetsSeries] | None = None,
    *,
    equations: str | Sequence[str] | None = None,
    method: str = "OLS",
    instruments: Sequence[str | MdlExpression] | None = None,
    center_covariance: bool = True,
    time_range: MdlTimeRange | tuple[int, int, int, int] | None = None,
    force_time_range: bool = False,
    tol: float = 1e-12,
    chow_test: bool = False,
    chow_end: YearPeriod | tuple[int, int] | None = None,
) -> ModelEstimationResult:
    """Estimate selected MDL behavioral equations.

    Parameters
    ----------
    model : BimetsModel or BoundModel
        Parsed model, or a model already associated with its data.
    data : BimetsDataset or mapping, optional
        Required when ``model`` is not already bound.
    equations : str or sequence of str, optional
        Equation names to estimate. By default all behaviorals are estimated.
    method : {"OLS", "IV"}, default="OLS"
        Ordinary least squares or instrumental variables (2SLS).
    instruments : sequence of str or MdlExpression, optional
        Instrument expressions overriding declarations from ``IV>``. The same
        override is applied to every selected equation.
    center_covariance : bool, default=True
        Subtract the residual mean in the variance used by covariance matrices,
        matching the original BIMETS default.
    time_range : MdlTimeRange or tuple of four int, optional
        Call-level estimation range. A range declared in MDL takes precedence
        unless ``force_time_range`` is true.
    force_time_range : bool, default=False
        Give ``time_range`` precedence over an MDL ``TSRANGE``.
    tol : float, default=1e-12
        Relative singularity threshold used before matrix inversion.
    chow_test : bool, default=False
        Perform structural-stability and predictive-power analysis.
    chow_end : YearPeriod or tuple of int, optional
        Final bound of the extended Chow sample. If omitted, use the latest
        common period for the equation inputs.

    Returns
    -------
    ModelEstimationResult
        Immutable estimates keyed by equation name.

    Notes
    -----
    Linear restrictions, Almon PDL expansions, and ``ERROR> AUTO(n)`` are
    applied when declared. Autoregressive errors use the BIMETS
    Cochrane--Orcutt settings: at most 20 iterations and an absolute rho
    convergence threshold of 0.005.

    Examples
    --------
    >>> from bimets import BimetsModel, estimate, timeseries
    >>> model = BimetsModel.from_text(
    ...     "MODEL\\nBEHAVIORAL> y\\nEQ> y=a+b*x\\nCOEFF> a b\\nEND",
    ...     name="linear",
    ... )
    >>> result = estimate(model, {
    ...     "y": timeseries([1, 3, 5, 7]),
    ...     "x": timeseries([0, 1, 2, 3]),
    ... })
    >>> {name: round(value, 12) for name, value in result["y"].coefficients.items()}
    {'a': 1.0, 'b': 2.0}
    """
    normalized_method = method.upper()
    if normalized_method not in {"OLS", "IV"}:
        raise ValueError("method must be 'OLS' or 'IV'")
    if not isinstance(center_covariance, bool):
        raise TypeError("center_covariance must be boolean")
    if not isinstance(force_time_range, bool):
        raise TypeError("force_time_range must be boolean")
    if not isinstance(chow_test, bool):
        raise TypeError("chow_test must be boolean")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tolerance must be a positive finite number")
    parsed_time_range = _parse_time_range(time_range)
    if force_time_range and parsed_time_range is None:
        raise ValueError("force_time_range requires time_range")
    if isinstance(model, BoundModel):
        if data is not None:
            raise TypeError("data must be omitted when estimating a BoundModel")
        bound = model
    else:
        if data is None:
            raise TypeError("data are required when estimating a BimetsModel")
        bound = bind_model_data(model, data)

    parsed_instruments = _parse_instruments(instruments)
    selected = _select_equations(bound.model, equations)
    results: dict[str, EquationEstimationResult] = {}
    for equation in selected:
        result = _estimate_equation(
            equation,
            bound,
            method=normalized_method,
            instruments=parsed_instruments,
            center_covariance=center_covariance,
            time_range=parsed_time_range,
            force_time_range=force_time_range,
            tol=tol,
        )
        if chow_test:
            analysis = _run_chow_test(
                equation,
                bound,
                result,
                method=normalized_method,
                instruments=parsed_instruments,
                center_covariance=center_covariance,
                tol=tol,
                chow_end=chow_end,
            )
            result = replace(result, chow_test=analysis)
        results[equation.name] = result
    return ModelEstimationResult(bound.model.name, results, method=normalized_method)


def _parse_time_range(
    value: MdlTimeRange | tuple[int, int, int, int] | None,
) -> MdlTimeRange | None:
    """Parse time range for internal processing."""
    if value is None or isinstance(value, MdlTimeRange):
        return value
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("time_range must be MdlTimeRange or a tuple of four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("time_range components must be integers")
    if value[1] < 1 or value[3] < 1:
        raise ValueError("time_range periods must be positive")
    return MdlTimeRange(*value)


def _parse_instruments(
    instruments: Sequence[str | MdlExpression] | None,
) -> tuple[MdlExpression, ...] | None:
    """Parse instruments for internal processing."""
    if instruments is None:
        return None
    if not instruments:
        raise ValueError("instruments must not be empty")
    return tuple(
        parse_expression(item) if isinstance(item, str) else item
        for item in instruments
    )


def _select_equations(
    model: BimetsModel, names: str | Sequence[str] | None
) -> tuple[BehavioralEquation, ...]:
    """Select equations for internal processing."""
    if names is None:
        return model.behaviorals
    requested = (names,) if isinstance(names, str) else tuple(names)
    if not requested:
        raise ValueError("at least one equation must be selected")
    if len(set(requested)) != len(requested):
        raise ValueError("equation selection contains duplicates")
    return tuple(model.behavioral(name) for name in requested)


def _estimate_equation(
    equation: BehavioralEquation,
    bound: BoundModel,
    *,
    method: str,
    instruments: tuple[MdlExpression, ...] | None,
    center_covariance: bool,
    time_range: MdlTimeRange | None,
    force_time_range: bool,
    tol: float,
) -> EquationEstimationResult:
    """Estimate equation for internal processing."""
    dependent = _evaluate_lhs(equation, bound.data)
    regressors = [evaluate_expression(item, bound.data) for item in equation.regressors]
    start, end = _estimation_bounds(
        equation,
        dependent,
        regressors,
        bound,
        time_range=time_range,
        force_time_range=force_time_range,
    )
    error_order = equation.error.order if equation.error is not None else 0
    extended_start = start.shift(-error_order, bound.freq)

    coefficient_names, columns = _expanded_regressors(
        equation,
        regressors,
        extended_start,
        end,
        periods_per_year=int(bound.freq),
    )
    x_full = np.column_stack(columns)
    y_full = _sample_values(dependent, extended_start, end, equation.name)
    _require_finite(y_full, f"dependent variable {equation.name!r}")
    _require_finite(x_full, f"regressors for equation {equation.name!r}")

    selected_instruments = instruments
    if selected_instruments is None:
        selected_instruments = equation.instruments or None
    z_full = (
        _instrument_matrix(
            selected_instruments,
            bound.data,
            extended_start,
            end,
            len(y_full),
            equation.name,
        )
        if method == "IV"
        else None
    )
    if method == "IV" and z_full is None:
        raise ValueError(
            f"IV estimation of equation {equation.name!r} requires instruments"
        )

    restriction_matrix, restriction_values = _restriction_system(
        equation, coefficient_names
    )
    observations = end.ordinal(bound.freq) - start.ordinal(bound.freq) + 1
    restriction_count = len(restriction_values)
    degrees_of_freedom = (
        observations - len(coefficient_names) + restriction_count - error_order
    )
    if degrees_of_freedom <= 0:
        raise ValueError(
            f"equation {equation.name!r} needs positive residual degrees of freedom"
        )

    rho_fit = _AutoregressiveFit(
        np.empty(0, dtype=np.float64),
        np.empty((0, 0), dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    iterations = 0
    if error_order:
        fit, rho_fit, iterations, corrected_y, corrected_x = _cochrane_orcutt(
            y_full,
            x_full,
            z_full,
            restriction_matrix,
            restriction_values,
            error_order,
            equation.name,
            tol,
        )
        residual_values = corrected_y - corrected_x @ fit.beta
        original_y = y_full[error_order:]
        original_x = x_full[error_order:]
        original_residuals = original_y - original_x @ fit.beta
        fitted = original_x @ fit.beta
    else:
        fit = _fit_regression(
            y_full,
            x_full,
            z_full,
            restriction_matrix,
            restriction_values,
            equation.name,
            tol,
        )
        residual_values = y_full - x_full @ fit.beta
        original_residuals = None
        original_y = y_full
        original_x = x_full
        fitted = x_full @ fit.beta

    ssr = float(residual_values @ residual_values)
    centered_ssr = ssr - float(np.sum(residual_values) ** 2 / observations)
    centered_ssr = max(centered_ssr, 0.0)
    standard_error = math.sqrt(centered_ssr / degrees_of_freedom)
    standard_error_not_centered = math.sqrt(ssr / degrees_of_freedom)
    covariance_scale = (
        standard_error**2 if center_covariance else standard_error_not_centered**2
    )
    covariance = covariance_scale * fit.covariance_factor
    coefficient_errors = np.sqrt(np.abs(np.diag(covariance)))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = fit.beta / coefficient_errors
    p_values = np.asarray(
        [
            _student_t_two_sided_probability(value, degrees_of_freedom)
            for value in t_statistics
        ]
    )

    has_constant = any(
        not isinstance(value, BimetsSeries)
        and not isinstance(value, (BimetsMask, bool))
        for value in regressors
    )
    center = float(np.mean(original_y)) if has_constant else 0.0
    total_sum_squares = float((original_y - center) @ (original_y - center))
    r_squared = 1.0 - ssr / total_sum_squares if total_sum_squares else math.nan
    adjusted = (
        1.0
        - (1.0 - r_squared)
        * (
            observations - 1
            if has_constant or error_order or restriction_count
            else observations
        )
        / degrees_of_freedom
        if math.isfinite(r_squared)
        else math.nan
    )
    log_likelihood = (
        -observations
        / 2.0
        * (math.log(2.0 * math.pi) + 1.0 + math.log(ssr / observations))
        if ssr > 0
        else math.inf
    )
    effective_parameters = observations - degrees_of_freedom
    f_denominator = effective_parameters - int(has_constant)
    f_statistic = (
        r_squared / (1.0 - r_squared) * degrees_of_freedom / f_denominator
        if f_denominator > 0 and 0 <= r_squared < 1
        else math.nan
    )
    f_probability = (
        _f_survival(f_statistic, f_denominator, degrees_of_freedom)
        if math.isfinite(f_statistic)
        else math.nan
    )
    parameter_count_for_ic = effective_parameters + 1
    aic = -2.0 * log_likelihood + 2.0 * parameter_count_for_ic
    bic = -2.0 * log_likelihood + math.log(observations) * parameter_count_for_ic
    durbin_watson = (
        float(np.diff(residual_values) @ np.diff(residual_values)) / ssr
        if ssr > 0
        else math.nan
    )
    restriction_f = _restriction_f_statistic(
        fit,
        original_y if not error_order else corrected_y,
        original_x if not error_order else corrected_x,
        ssr,
        restriction_count,
        observations - len(coefficient_names) - error_order,
    )
    restriction_probability = (
        _f_survival(
            restriction_f,
            restriction_count,
            observations - len(coefficient_names),
        )
        if restriction_f is not None and math.isfinite(restriction_f)
        else (0.0 if restriction_f == math.inf else None)
    )
    coefficient_values = _named_values(coefficient_names, fit.beta)
    coefficient_standard_errors = _named_values(coefficient_names, coefficient_errors)
    coefficient_t_statistics = _named_values(coefficient_names, t_statistics)
    coefficient_p_values = _named_values(coefficient_names, p_values)
    rho_names = tuple(f"RHO_{index}" for index in range(1, error_order + 1))
    if error_order:
        rho_ssr = float(rho_fit.innovations @ rho_fit.innovations)
        centered_rho_ssr = rho_ssr - float(
            np.sum(rho_fit.innovations) ** 2 / observations
        )
        rho_scale = (
            max(centered_rho_ssr, 0.0) / degrees_of_freedom
            if center_covariance
            else rho_ssr / degrees_of_freedom
        )
        rho_covariance = rho_scale * rho_fit.covariance_factor
        rho_errors = np.sqrt(np.abs(np.diag(rho_covariance)))
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_t_statistics = rho_fit.coefficients / rho_errors
        rho_p_values = np.asarray(
            [
                _student_t_two_sided_probability(value, degrees_of_freedom)
                for value in rho_t_statistics
            ]
        )
    else:
        rho_errors = np.empty(0, dtype=np.float64)
        rho_t_statistics = np.empty(0, dtype=np.float64)
        rho_p_values = np.empty(0, dtype=np.float64)
    return EquationEstimationResult(
        name=equation.name,
        method=method,
        coefficients=coefficient_values,
        coefficient_standard_errors=coefficient_standard_errors,
        t_statistics=coefficient_t_statistics,
        coefficient_p_values=coefficient_p_values,
        fitted_values=BimetsSeries(fitted, start=start, freq=bound.freq),
        residuals=BimetsSeries(residual_values, start=start, freq=bound.freq),
        residuals_without_error_correction=(
            BimetsSeries(original_residuals, start=start, freq=bound.freq)
            if original_residuals is not None
            else None
        ),
        autoregressive_coefficients=_named_values(rho_names, rho_fit.coefficients),
        autoregressive_standard_errors=_named_values(rho_names, rho_errors),
        autoregressive_t_statistics=_named_values(rho_names, rho_t_statistics),
        autoregressive_p_values=_named_values(rho_names, rho_p_values),
        autoregressive_iterations=iterations,
        sample_start=start,
        sample_end=end,
        observations=observations,
        degrees_of_freedom=degrees_of_freedom,
        residual_sum_squares=ssr,
        standard_error=standard_error,
        standard_error_not_centered=standard_error_not_centered,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        durbin_watson=durbin_watson,
        log_likelihood=log_likelihood,
        f_statistic=f_statistic,
        f_probability=f_probability,
        aic=aic,
        bic=bic,
        restriction_f_statistic=restriction_f,
        restriction_f_probability=restriction_probability,
        covariance=covariance,
        regressor_matrix=original_x,
        dependent_values=original_y,
    )


def _expanded_regressors(
    equation: BehavioralEquation,
    regressors: Sequence[MdlValue],
    start: YearPeriod,
    end: YearPeriod,
    *,
    periods_per_year: int,
) -> tuple[tuple[str, ...], list[NDArray[np.float64]]]:
    """Expand regression terms, including polynomial distributed lags."""
    definitions = {item.coefficient: item for item in equation.pdls}
    names: list[str] = []
    columns: list[NDArray[np.float64]] = []
    observations = end.ordinal(periods_per_year) - start.ordinal(periods_per_year) + 1
    for coefficient, value in zip(equation.coefficients, regressors, strict=True):
        definition = definitions.get(coefficient)
        lag_count = definition.length if definition is not None else 1
        for lag in range(lag_count):
            name = coefficient if lag == 0 else f"{coefficient}__PDL__{lag}"
            names.append(name)
            lagged = value
            if lag:
                if not isinstance(value, BimetsSeries):
                    raise TypeError(
                        f"PDL coefficient {coefficient!r} requires a series"
                    )
                lagged = value.lag(lag)
            if isinstance(lagged, BimetsSeries):
                columns.append(_sample_values(lagged, start, end, name))
            else:
                if isinstance(lagged, (BimetsMask, bool)):
                    raise TypeError(f"regressor {name!r} must be numeric")
                if observations == 0:
                    raise ValueError("at least one series regressor is required")
                columns.append(np.full(observations, lagged, dtype=np.float64))
    return tuple(names), columns


def _instrument_matrix(
    instruments: tuple[MdlExpression, ...] | None,
    data: Mapping[str, BimetsSeries],
    start: YearPeriod,
    end: YearPeriod,
    observations: int,
    equation_name: str,
) -> NDArray[np.float64] | None:
    """Build the instrument matrix for an estimation sample."""
    if instruments is None:
        return None
    columns = [
        _regressor_values(
            evaluate_expression(expression, data),
            start,
            end,
            observations,
            f"instrument {index}",
        )
        for index, expression in enumerate(instruments, start=1)
    ]
    matrix = np.column_stack(columns)
    _require_finite(matrix, f"instruments for equation {equation_name!r}")
    return matrix


def _restriction_system(
    equation: BehavioralEquation, coefficient_names: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the linear coefficient-restriction system."""
    positions = {name: index for index, name in enumerate(coefficient_names)}
    rows: list[NDArray[np.float64]] = []
    values: list[float] = []
    for restriction in equation.restrictions:
        row = np.zeros(len(coefficient_names), dtype=np.float64)
        for term in restriction.terms:
            name = (
                term.coefficient
                if term.lag == 0
                else f"{term.coefficient}__PDL__{term.lag}"
            )
            row[positions[name]] += term.multiplier
        rows.append(row)
        values.append(restriction.target)

    for definition in equation.pdls:
        indexes = [
            positions[
                definition.coefficient
                if lag == 0
                else f"{definition.coefficient}__PDL__{lag}"
            ]
            for lag in range(definition.length)
        ]
        difference_order = definition.degree + 1
        pattern = np.array(
            [
                (-1.0) ** offset * math.comb(difference_order, offset)
                for offset in range(difference_order + 1)
            ]
        )
        for start in range(definition.length - difference_order):
            row = np.zeros(len(coefficient_names), dtype=np.float64)
            row[indexes[start : start + difference_order + 1]] = pattern
            rows.append(row)
            values.append(0.0)
        if definition.zero_nearest:
            row = np.zeros(len(coefficient_names), dtype=np.float64)
            row[indexes[0]] = 1.0
            rows.append(row)
            values.append(0.0)
        if definition.zero_farthest:
            row = np.zeros(len(coefficient_names), dtype=np.float64)
            row[indexes[-1]] = 1.0
            rows.append(row)
            values.append(0.0)
    matrix = (
        np.vstack(rows)
        if rows
        else np.empty((0, len(coefficient_names)), dtype=np.float64)
    )
    return matrix, np.asarray(values, dtype=np.float64)


def _fit_regression(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    z: NDArray[np.float64] | None,
    restrictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    equation_name: str,
    tol: float,
) -> _RegressionFit:
    """Fit regression for internal processing."""
    try:
        if z is None:
            effective_x = x
        else:
            instrument_cross_product = z.T @ z
            _require_invertible(instrument_cross_product, tol)
            effective_x = z @ np.linalg.solve(instrument_cross_product, z.T @ x)
        cross_product = effective_x.T @ effective_x
        _require_invertible(cross_product, tol)
        inverse = np.linalg.inv(cross_product)
        unrestricted = np.linalg.solve(cross_product, effective_x.T @ y)
        if not len(targets):
            return _RegressionFit(unrestricted, inverse, None)
        beta, covariance_factor = _constrained_least_squares(
            effective_x,
            y,
            restrictions,
            targets,
            equation_name,
            tol,
        )
        return _RegressionFit(beta, covariance_factor, unrestricted)
    except np.linalg.LinAlgError as error:
        kind = "instrument or regressor" if z is not None else "regressor"
        raise ValueError(
            f"{kind} matrix for equation {equation_name!r} is singular"
        ) from error


def _constrained_least_squares(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    restrictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    equation_name: str,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve exact linear restrictions in a numerically stable null space.

    Forming the classic restricted normal-equation correction squares the
    condition number twice: once in ``X'X`` and again in its restriction
    projection. Long PDLs in production models can therefore select a visibly
    different solution. Parameterizing the feasible coefficients as a
    particular solution plus the null space of the restrictions preserves the
    same least-squares problem without that additional loss of precision.
    """
    _, singular_values, right = np.linalg.svd(restrictions, full_matrices=True)
    if singular_values.size == 0:
        raise ValueError(f"restriction matrix for equation {equation_name!r} is empty")
    threshold = tol * singular_values[0]
    rank = int(np.sum(singular_values > threshold))
    if rank != restrictions.shape[0]:
        raise ValueError(
            f"restriction matrix for equation {equation_name!r} is singular"
        )
    particular = np.linalg.lstsq(restrictions, targets, rcond=tol)[0]
    null_space = right[rank:].T
    if null_space.shape[1] == 0:
        return particular, np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)

    reduced_x = x @ null_space
    reduced_y = y - x @ particular
    reduced_beta, _, reduced_rank, _ = np.linalg.lstsq(reduced_x, reduced_y, rcond=tol)
    if reduced_rank != null_space.shape[1]:
        raise ValueError(f"regressor matrix for equation {equation_name!r} is singular")
    reduced_cross_product = reduced_x.T @ reduced_x
    reduced_inverse = np.linalg.solve(
        reduced_cross_product,
        np.eye(reduced_cross_product.shape[0]),
    )
    beta = particular + null_space @ reduced_beta
    covariance_factor = null_space @ reduced_inverse @ null_space.T
    return beta, covariance_factor


def _cochrane_orcutt(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    z: NDArray[np.float64] | None,
    restrictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    order: int,
    equation_name: str,
    tol: float,
) -> tuple[
    _RegressionFit,
    _AutoregressiveFit,
    int,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fit an equation with iterative Cochrane-Orcutt correction."""
    fit = _fit_regression(y, x, z, restrictions, targets, equation_name, tol)
    rho_fit = _estimate_rho(y - x @ fit.beta, order, equation_name, tol)
    converged = False
    iterations = 1
    transformed_y, transformed_x = _ar_transform(y, x, rho_fit.coefficients, order)
    transformed_z = z[order:] if z is not None else None
    for current_iteration in range(2, 21):
        iterations = current_iteration
        fit = _fit_regression(
            transformed_y,
            transformed_x,
            transformed_z,
            restrictions,
            targets,
            equation_name,
            tol,
        )
        if converged:
            break
        next_rho_fit = _estimate_rho(y - x @ fit.beta, order, equation_name, tol)
        converged = bool(
            np.all(np.abs(next_rho_fit.coefficients - rho_fit.coefficients) < 0.005)
        )
        rho_fit = next_rho_fit
        transformed_y, transformed_x = _ar_transform(y, x, rho_fit.coefficients, order)
    return fit, rho_fit, iterations, transformed_y, transformed_x


def _estimate_rho(
    residuals: NDArray[np.float64],
    order: int,
    equation_name: str,
    tol: float,
) -> _AutoregressiveFit:
    """Estimate rho for internal processing."""
    dependent = residuals[order:]
    regressors = np.column_stack(
        [residuals[order - lag : len(residuals) - lag] for lag in range(1, order + 1)]
    )
    try:
        cross_product = regressors.T @ regressors
        _require_invertible(cross_product, tol)
        inverse = np.linalg.inv(cross_product)
        coefficients = np.linalg.solve(cross_product, regressors.T @ dependent)
        innovations = dependent - regressors @ coefficients
        return _AutoregressiveFit(coefficients, inverse, innovations)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"residual lag matrix for equation {equation_name!r} is singular"
        ) from error


def _require_invertible(matrix: NDArray[np.float64], tol: float) -> None:
    """Validate invertible for internal processing."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= tol * singular_values[0]:
        raise np.linalg.LinAlgError("matrix is numerically singular")


def _ar_transform(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    rho: NDArray[np.float64],
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the autoregressive transformation to a sample."""
    transformed_y = y[order:].copy()
    transformed_x = x[order:].copy()
    for lag, coefficient in enumerate(rho, start=1):
        transformed_y -= coefficient * y[order - lag : len(y) - lag]
        transformed_x -= coefficient * x[order - lag : len(x) - lag]
    return transformed_y, transformed_x


def _restriction_f_statistic(
    fit: _RegressionFit,
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    restricted_ssr: float,
    restriction_count: int,
    unrestricted_dof: int,
) -> float | None:
    """Compute the F statistic for coefficient restrictions."""
    if fit.unrestricted_beta is None or restriction_count == 0 or unrestricted_dof <= 0:
        return None
    unrestricted_residuals = y - x @ fit.unrestricted_beta
    unrestricted_ssr = float(unrestricted_residuals @ unrestricted_residuals)
    if unrestricted_ssr == 0:
        return math.inf if restricted_ssr > 0 else 0.0
    return (
        (restricted_ssr - unrestricted_ssr)
        / restriction_count
        / (unrestricted_ssr / unrestricted_dof)
    )


def _run_chow_test(
    equation: BehavioralEquation,
    bound: BoundModel,
    base: EquationEstimationResult,
    *,
    method: str,
    instruments: tuple[MdlExpression, ...] | None,
    center_covariance: bool,
    tol: float,
    chow_end: YearPeriod | tuple[int, int] | None,
) -> ChowTestResult:
    """Run chow test for internal processing."""
    final_period = (
        _normalize_chow_end(chow_end, bound)
        if chow_end is not None
        else _latest_equation_period(equation, bound, instruments, method)
    )
    if final_period.ordinal(bound.freq) <= base.sample_end.ordinal(bound.freq):
        raise ValueError(
            f"chow_end for equation {equation.name!r} must follow the base sample"
        )
    extended_range = MdlTimeRange(
        base.sample_start.year,
        base.sample_start.period,
        final_period.year,
        final_period.period,
    )
    extended = _estimate_equation(
        equation,
        bound,
        method=method,
        instruments=instruments,
        center_covariance=center_covariance,
        time_range=extended_range,
        force_time_range=True,
        tol=tol,
    )
    numerator_degrees = extended.degrees_of_freedom - base.degrees_of_freedom
    if numerator_degrees <= 0 or base.residual_sum_squares <= 0:
        raise ValueError(
            f"equation {equation.name!r} has insufficient information for a Chow test"
        )
    statistic = (
        (extended.residual_sum_squares / base.residual_sum_squares - 1.0)
        * extended.degrees_of_freedom
        / numerator_degrees
    )
    probability = _f_survival(statistic, numerator_degrees, extended.degrees_of_freedom)

    offset = base.observations
    out_x = extended.regressor_matrix[offset:]
    actual_values = extended.dependent_values[offset:]
    beta = np.asarray(tuple(base.coefficients.values()), dtype=np.float64)
    predicted_values = out_x @ beta
    error_values = actual_values - predicted_values
    covariance_scale = (
        base.standard_error**2
        if center_covariance
        else base.standard_error_not_centered**2
    )
    if covariance_scale > 0:
        covariance_factor = base.covariance / covariance_scale
        forecast_variances = np.einsum("ij,jk,ik->i", out_x, covariance_factor, out_x)
        standard_error_values = base.standard_error * np.sqrt(
            np.maximum(1.0 + forecast_variances, 0.0)
        )
    else:
        standard_error_values = np.zeros(len(actual_values), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = error_values / standard_error_values
    prediction_start = base.sample_end.shift(1, bound.freq)
    return ChowTestResult(
        f_statistic=statistic,
        f_probability=probability,
        numerator_degrees_of_freedom=numerator_degrees,
        denominator_degrees_of_freedom=extended.degrees_of_freedom,
        base_start=base.sample_start,
        base_end=base.sample_end,
        extended_end=final_period,
        actual=BimetsSeries(actual_values, start=prediction_start, freq=bound.freq),
        predicted=BimetsSeries(
            predicted_values, start=prediction_start, freq=bound.freq
        ),
        errors=BimetsSeries(error_values, start=prediction_start, freq=bound.freq),
        standard_errors=BimetsSeries(
            standard_error_values,
            start=prediction_start,
            freq=bound.freq,
        ),
        t_statistics=BimetsSeries(t_values, start=prediction_start, freq=bound.freq),
    )


def _normalize_chow_end(
    value: YearPeriod | tuple[int, int], bound: BoundModel
) -> YearPeriod:
    """Normalize chow end for internal processing."""
    if isinstance(value, YearPeriod):
        year, period = value.year, value.period
    elif isinstance(value, tuple) and len(value) == 2:
        year, period = value
    else:
        raise TypeError("chow_end must be YearPeriod or a tuple of two integers")
    if any(
        isinstance(item, bool) or not isinstance(item, int) for item in (year, period)
    ):
        raise TypeError("chow_end components must be integers")
    if period < 1 or period > int(bound.freq):
        raise ValueError("chow_end period exceeds data frequency")
    return YearPeriod(year, period)


def _latest_equation_period(
    equation: BehavioralEquation,
    bound: BoundModel,
    instruments: tuple[MdlExpression, ...] | None,
    method: str,
) -> YearPeriod:
    """Return the latest period supported by an equation sample."""
    values: list[MdlValue] = [_evaluate_lhs(equation, bound.data)]
    values.extend(evaluate_expression(item, bound.data) for item in equation.regressors)
    if method == "IV":
        selected = instruments if instruments is not None else equation.instruments
        values.extend(evaluate_expression(item, bound.data) for item in selected)
    return min(item.end for item in values if isinstance(item, BimetsSeries))


def _named_values(
    names: Sequence[str], values: NDArray[np.float64]
) -> dict[str, float]:
    """Associate names with numeric values."""
    return dict(zip(names, (float(value) for value in values), strict=True))


def _student_t_two_sided_probability(value: float, degrees: int) -> float:
    """Return a two-sided Student t probability."""
    if math.isnan(value):
        return math.nan
    if math.isinf(value):
        return 0.0
    x = degrees / (degrees + value * value)
    return _regularized_beta(x, degrees / 2.0, 0.5)


def _f_survival(
    value: float, numerator_degrees: int, denominator_degrees: int
) -> float:
    """Return the survival probability of an F statistic."""
    if value < 0 or numerator_degrees <= 0 or denominator_degrees <= 0:
        return math.nan
    if math.isinf(value):
        return 0.0
    x = denominator_degrees / (denominator_degrees + numerator_degrees * value)
    return _regularized_beta(x, denominator_degrees / 2.0, numerator_degrees / 2.0)


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta using a stable continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(x, a, b) / a
    return 1.0 - front * _beta_continued_fraction(1.0 - x, b, a) / b


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """Evaluate the continued fraction used by the beta function."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), tiny) * (1.0 if d >= 0 else -1.0)
    result = d
    for iteration in range(1, 201):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        d = d if abs(d) >= tiny else tiny
        c = 1.0 + coefficient / c
        c = c if abs(c) >= tiny else tiny
        d = 1.0 / d
        result *= d * c
        coefficient = (
            -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        )
        d = 1.0 + coefficient * d
        d = d if abs(d) >= tiny else tiny
        c = 1.0 + coefficient / c
        c = c if abs(c) >= tiny else tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return result


def _require_finite(values: NDArray[np.float64], description: str) -> None:
    """Validate finite for internal processing."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{description} is not finite on the estimation range")


def _evaluate_lhs(
    behavioral: BehavioralEquation, data: Mapping[str, BimetsSeries]
) -> BimetsSeries:
    """Evaluate lhs for internal processing."""
    equation = behavioral.equation
    source = data[equation.dependent]
    function = equation.lhs_function
    if function == "IDENTITY":
        return source
    if function in {"LOG", "EXP"}:
        from bimets.mdl._expression import FunctionCall, Variable

        value = evaluate_expression(
            FunctionCall(function, (Variable(equation.dependent),)), data
        )
        assert isinstance(value, BimetsSeries)
        return value
    transformations = {
        "TSDELTA": source.delta,
        "TSDELTALOG": source.delta_log,
        "TSDELTAP": source.delta_percent,
    }
    return transformations[function](equation.lhs_periods)


def _estimation_bounds(
    equation: BehavioralEquation,
    dependent: BimetsSeries,
    regressors: Sequence[MdlValue],
    bound: BoundModel,
    *,
    time_range: MdlTimeRange | None,
    force_time_range: bool,
) -> tuple[YearPeriod, YearPeriod]:
    """Resolve and validate the estimation sample bounds."""
    declared = (
        time_range
        if force_time_range or equation.estimation_range is None
        else equation.estimation_range
    )
    if declared is not None:
        if declared.start_period > int(bound.freq) or declared.end_period > int(
            bound.freq
        ):
            raise ValueError(
                f"TSRANGE periods for equation {equation.name!r} exceed data frequency"
            )
        start = YearPeriod(declared.start_year, declared.start_period)
        end = YearPeriod(declared.end_year, declared.end_period)
        if end.ordinal(bound.freq) < start.ordinal(bound.freq):
            raise ValueError(f"TSRANGE for equation {equation.name!r} is reversed")
        return start, end

    series = [dependent]
    series.extend(item for item in regressors if isinstance(item, BimetsSeries))
    start = max(item.start for item in series)
    end = min(item.end for item in series)
    if end.ordinal(bound.freq) < start.ordinal(bound.freq):
        raise ValueError(f"equation {equation.name!r} has no common estimation range")
    return start, end


def _sample_values(
    series: BimetsSeries, start: YearPeriod, end: YearPeriod, name: str
) -> NDArray[np.float64]:
    """Return the internally computed sample values."""
    expected = end.ordinal(series.freq) - start.ordinal(series.freq) + 1
    projected = series.project(start, end, extend=True)
    if len(projected) != expected:
        raise ValueError(f"series {name!r} does not cover the estimation range")
    return projected.values


def _regressor_values(
    value: MdlValue,
    start: YearPeriod,
    end: YearPeriod,
    observations: int,
    name: str,
) -> NDArray[np.float64]:
    """Return the internally computed regressor values."""
    if isinstance(value, BimetsSeries):
        return _sample_values(value, start, end, name)
    if isinstance(value, (BimetsMask, bool)):
        raise TypeError(f"regressor {name!r} must be numeric")
    return np.full(observations, value, dtype=np.float64)
