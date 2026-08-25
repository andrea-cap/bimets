"""Association between parsed MDL models and time-series data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bimets.mdl._expression import MdlExpression
from bimets.mdl._model import BimetsModel, MdlTimeRange
from bimets.timeseries import BimetsDataset, BimetsSeries, Frequency, YearPeriod

if TYPE_CHECKING:
    from bimets.mdl._estimation import ModelEstimationResult
    from bimets.mdl._multipliers import MultiplierMatrixResult
    from bimets.mdl._optimization import (
        BoundSpec,
        FunctionSpec,
        OptimizationResult,
        RestrictionSpec,
    )
    from bimets.mdl._renormalization import RenormalizationResult
    from bimets.mdl._simulation import (
        AdjustmentValue,
        CoefficientInput,
        ExogenizationValue,
        SimulationResult,
    )
    from bimets.mdl._stochastic import (
        StochasticDisturbance,
        StochasticSimulationResult,
    )


@dataclass(frozen=True, slots=True)
class BoundModel:
    """A parsed model associated with a validated dataset.

    Attributes
    ----------
    model : BimetsModel
        Parsed model definition.
    data : BimetsDataset
        Immutable collection containing every model variable.
    freq : Frequency
        Common frequency of all bound series.

    Examples
    --------
    >>> from bimets import BimetsDataset, BimetsModel, timeseries
    >>> model = BimetsModel.from_text(
    ...     "MODEL\\nBEHAVIORAL> y\\nEQ> y=a+b*x\\nCOEFF> a b\\nEND"
    ... )
    >>> data = BimetsDataset({
    ...     "y": timeseries([1, 3, 5]),
    ...     "x": timeseries([0, 1, 2]),
    ... })
    >>> bound = model.bind(data)
    >>> bound.freq
    <Frequency.YEARLY: 1>
    """

    model: BimetsModel
    data: BimetsDataset
    freq: Frequency

    def estimate(
        self,
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
        """Estimate selected behavioral equations using OLS or IV.

        Parameters
        ----------
        equations : str or sequence of str, optional
            Behavioral equation names. The default estimates all of them.
        method : {"OLS", "IV"}, default="OLS"
            Estimation technique.
        instruments : sequence of str or MdlExpression, optional
            Instrument expressions overriding ``IV>`` declarations.
        center_covariance : bool, default=True
            Center residuals when calculating covariance matrices.
        time_range : MdlTimeRange or tuple of four int, optional
            Call-level estimation range.
        force_time_range : bool, default=False
            Override an MDL ``TSRANGE`` with ``time_range``.
        tol : float, default=1e-12
            Relative numerical singularity threshold.
        chow_test : bool, default=False
            Perform structural-stability analysis.
        chow_end : YearPeriod or tuple of int, optional
            Final extended period used by the Chow test.

        Returns
        -------
        ModelEstimationResult
            Per-equation estimates for the selected method and model
            declarations.
        """
        from bimets.mdl._estimation import estimate

        return estimate(
            self,
            equations=equations,
            method=method,
            instruments=instruments,
            center_covariance=center_covariance,
            time_range=time_range,
            force_time_range=force_time_range,
            tol=tol,
            chow_test=chow_test,
            chow_end=chow_end,
        )

    def simulate(
        self,
        *,
        coefficients: CoefficientInput,
        time_range: MdlTimeRange | tuple[int, int, int, int],
        simulation_type: str = "DYNAMIC",
        algorithm: str = "GAUSS-SEIDEL",
        convergence: float = 0.01,
        max_iterations: int = 100,
        jacobian_step: float = 1e-4,
        zero_error_autocorrelation: bool = False,
        constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
        exogenize: (
            str | Sequence[str] | Mapping[str, ExogenizationValue] | None
        ) = None,
        rescheck_equations: str | Sequence[str] | None = None,
        backfill: int = 0,
        jacobian_drop: str | Sequence[str] | None = None,
    ) -> SimulationResult:
        """Run a deterministic simulation using the bound data.

        Parameters are equivalent to :func:`bimets.simulate`, except that model
        data are already available on this object.
        """
        from bimets.mdl._simulation import simulate

        return simulate(
            self,
            coefficients=coefficients,
            time_range=time_range,
            simulation_type=simulation_type,
            algorithm=algorithm,
            convergence=convergence,
            max_iterations=max_iterations,
            jacobian_step=jacobian_step,
            zero_error_autocorrelation=zero_error_autocorrelation,
            constant_adjustments=constant_adjustments,
            exogenize=exogenize,
            rescheck_equations=rescheck_equations,
            backfill=backfill,
            jacobian_drop=jacobian_drop,
        )

    def stochastic_simulate(
        self,
        *,
        coefficients: CoefficientInput,
        time_range: MdlTimeRange | tuple[int, int, int, int],
        disturbances: Mapping[str, StochasticDisturbance] | None = None,
        replicas: int = 100,
        seed: int | None = None,
        workers: int = 1,
        simulation_type: str = "DYNAMIC",
        algorithm: str = "GAUSS-SEIDEL",
        convergence: float = 0.01,
        max_iterations: int = 100,
        jacobian_step: float = 1e-4,
        zero_error_autocorrelation: bool = False,
        constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
        exogenize: (
            str | Sequence[str] | Mapping[str, ExogenizationValue] | None
        ) = None,
        rescheck_equations: str | Sequence[str] | None = None,
        backfill: int = 0,
        jacobian_drop: str | Sequence[str] | None = None,
    ) -> StochasticSimulationResult:
        """Run stochastic simulations using the bound data.

        Parameters are equivalent to :func:`bimets.stochastic_simulate`, except
        that model data are already available on this object.
        """
        from bimets.mdl._stochastic import stochastic_simulate

        return stochastic_simulate(
            self,
            coefficients=coefficients,
            time_range=time_range,
            disturbances=disturbances,
            replicas=replicas,
            seed=seed,
            workers=workers,
            simulation_type=simulation_type,
            algorithm=algorithm,
            convergence=convergence,
            max_iterations=max_iterations,
            jacobian_step=jacobian_step,
            zero_error_autocorrelation=zero_error_autocorrelation,
            constant_adjustments=constant_adjustments,
            exogenize=exogenize,
            rescheck_equations=rescheck_equations,
            backfill=backfill,
            jacobian_drop=jacobian_drop,
        )

    def multiplier_matrix(
        self,
        *,
        coefficients: CoefficientInput,
        time_range: MdlTimeRange | tuple[int, int, int, int],
        targets: str | Sequence[str],
        instruments: str | Sequence[str],
        shock: float = 1e-5,
        simulation_type: str = "DYNAMIC",
        algorithm: str = "GAUSS-SEIDEL",
        workers: int = 1,
        convergence: float = 0.01,
        max_iterations: int = 100,
        jacobian_step: float = 1e-4,
        zero_error_autocorrelation: bool = False,
        constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
        exogenize: (
            str | Sequence[str] | Mapping[str, ExogenizationValue] | None
        ) = None,
        backfill: int = 0,
        jacobian_drop: str | Sequence[str] | None = None,
    ) -> MultiplierMatrixResult:
        """Compute multiplier matrices using the bound data.

        Parameters are equivalent to :func:`bimets.multiplier_matrix`, except
        that model data are already available on this object.
        """
        from bimets.mdl._multipliers import multiplier_matrix

        return multiplier_matrix(
            self,
            coefficients=coefficients,
            time_range=time_range,
            targets=targets,
            instruments=instruments,
            shock=shock,
            simulation_type=simulation_type,
            algorithm=algorithm,
            workers=workers,
            convergence=convergence,
            max_iterations=max_iterations,
            jacobian_step=jacobian_step,
            zero_error_autocorrelation=zero_error_autocorrelation,
            constant_adjustments=constant_adjustments,
            exogenize=exogenize,
            backfill=backfill,
            jacobian_drop=jacobian_drop,
        )

    def renormalize(
        self,
        *,
        coefficients: CoefficientInput,
        time_range: MdlTimeRange | tuple[int, int, int, int],
        targets: Mapping[str, BimetsSeries],
        instruments: str | Sequence[str],
        simulation_type: str = "DYNAMIC",
        algorithm: str = "GAUSS-SEIDEL",
        workers: int = 1,
        convergence: float = 0.01,
        max_iterations: int = 100,
        jacobian_step: float = 1e-4,
        zero_error_autocorrelation: bool = False,
        constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
        exogenize: (
            str | Sequence[str] | Mapping[str, ExogenizationValue] | None
        ) = None,
        backfill: int = 0,
        jacobian_drop: str | Sequence[str] | None = None,
        renormalization_iterations: int = 10,
        renormalization_convergence: float = 1e-4,
        shock: float = 1e-5,
        matrix_tolerance: float = 1e-12,
    ) -> RenormalizationResult:
        """Find instrument paths using the bound model data.

        Parameters are equivalent to :func:`bimets.renormalize`, except that
        model data are already available on this object.
        """
        from bimets.mdl._renormalization import renormalize

        return renormalize(
            self,
            coefficients=coefficients,
            time_range=time_range,
            targets=targets,
            instruments=instruments,
            simulation_type=simulation_type,
            algorithm=algorithm,
            workers=workers,
            convergence=convergence,
            max_iterations=max_iterations,
            jacobian_step=jacobian_step,
            zero_error_autocorrelation=zero_error_autocorrelation,
            constant_adjustments=constant_adjustments,
            exogenize=exogenize,
            backfill=backfill,
            jacobian_drop=jacobian_drop,
            renormalization_iterations=renormalization_iterations,
            renormalization_convergence=renormalization_convergence,
            shock=shock,
            matrix_tolerance=matrix_tolerance,
        )

    def optimize(
        self,
        *,
        coefficients: CoefficientInput,
        time_range: MdlTimeRange | tuple[int, int, int, int],
        bounds: Mapping[str, BoundSpec],
        objective_functions: FunctionSpec | Mapping[str, FunctionSpec],
        restrictions: RestrictionSpec | Mapping[str, RestrictionSpec] | None = None,
        replicas: int = 100,
        seed: int | None = None,
        workers: int = 1,
        simulation_type: str = "DYNAMIC",
        algorithm: str = "GAUSS-SEIDEL",
        convergence: float = 0.01,
        max_iterations: int = 100,
        jacobian_step: float = 1e-4,
        zero_error_autocorrelation: bool = False,
        constant_adjustments: Mapping[str, AdjustmentValue] | None = None,
        exogenize: (
            str | Sequence[str] | Mapping[str, ExogenizationValue] | None
        ) = None,
        rescheck_equations: str | Sequence[str] | None = None,
        backfill: int = 0,
        jacobian_drop: str | Sequence[str] | None = None,
    ) -> OptimizationResult:
        """Perform Monte Carlo optimal control using the bound data.

        Parameters are equivalent to :func:`bimets.optimize_model`, except
        that model data are already available on this object.
        """
        from bimets.mdl._optimization import optimize_model

        return optimize_model(
            self,
            coefficients=coefficients,
            time_range=time_range,
            bounds=bounds,
            objective_functions=objective_functions,
            restrictions=restrictions,
            replicas=replicas,
            seed=seed,
            workers=workers,
            simulation_type=simulation_type,
            algorithm=algorithm,
            convergence=convergence,
            max_iterations=max_iterations,
            jacobian_step=jacobian_step,
            zero_error_autocorrelation=zero_error_autocorrelation,
            constant_adjustments=constant_adjustments,
            exogenize=exogenize,
            rescheck_equations=rescheck_equations,
            backfill=backfill,
            jacobian_drop=jacobian_drop,
        )


def bind_model_data(
    model: BimetsModel,
    data: BimetsDataset | Mapping[str, BimetsSeries],
) -> BoundModel:
    """Validate and bind time-series data to an MDL model.

    Parameters
    ----------
    model : BimetsModel
        Parsed model definition.
    data : BimetsDataset or mapping of str to BimetsSeries
        Dataset containing all endogenous and exogenous model variables.

    Returns
    -------
    BoundModel
        Model and immutable homogeneous-frequency dataset.

    Raises
    ------
    TypeError
        If ``model`` or a dataset value has the wrong type.
    ValueError
        If frequencies differ.
    KeyError
        If model variables are missing.
    """
    if not isinstance(model, BimetsModel):
        raise TypeError("model must be a BimetsModel")
    dataset = data if isinstance(data, BimetsDataset) else BimetsDataset(data)
    missing = set(model.endogenous).union(model.exogenous).difference(dataset)
    if missing:
        raise KeyError(f"dataset is missing model variables: {sorted(missing)}")
    freq = dataset.homogeneous_frequency
    if freq is None:
        raise ValueError("all model data series must have the same frequency")
    return BoundModel(model=model, data=dataset, freq=freq)
