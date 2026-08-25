"""Immutable public data structures produced by the MDL parser."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from bimets.mdl._expression import MdlExpression

if TYPE_CHECKING:
    from bimets.mdl._binding import BoundModel
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
    from bimets.timeseries import BimetsDataset, BimetsSeries, YearPeriod


def _restore_model(raw_text: str, name: str) -> BimetsModel:
    """Reparse immutable model text during process deserialization."""
    return BimetsModel.from_text(raw_text, name=name)


@dataclass(frozen=True, slots=True)
class MdlTimeRange:
    """An inclusive MDL year-period range.

    Attributes
    ----------
    start_year, start_period : int
        First year and one-based period.
    end_year, end_period : int
        Last year and one-based period.

    Notes
    -----
    The parser preserves declared periods; validation against an actual series
    frequency is deferred until model data are bound.
    """

    start_year: int
    start_period: int
    end_year: int
    end_period: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return bounds as ``(start_year, start_period, end_year, end_period)``.

        Returns
        -------
        tuple of int
            Four declared range components.
        """
        return (
            self.start_year,
            self.start_period,
            self.end_year,
            self.end_period,
        )


@dataclass(frozen=True, slots=True)
class MdlEquation:
    """A parsed equation with a normalized left-hand side.

    Attributes
    ----------
    dependent : str
        Endogenous variable defined by the equation.
    lhs_function : str
        Normalized left-hand-side transformation, or ``IDENTITY``.
    lhs_periods : int
        Period argument of a temporal left-hand-side transformation.
    rhs : MdlExpression
        Parsed right-hand side.
    source : str
        Original equation text.
    line : int
        One-based source line.
    """

    dependent: str
    lhs_function: str
    lhs_periods: int
    rhs: MdlExpression
    source: str
    line: int


@dataclass(frozen=True, slots=True)
class AutoregressiveError:
    """An ``ERROR> AUTO(n)`` declaration.

    Attributes
    ----------
    order : int
        Positive autoregressive order ``n``.
    """

    order: int


@dataclass(frozen=True, slots=True)
class PdlDefinition:
    """An Almon polynomial distributed-lag declaration.

    Attributes
    ----------
    coefficient : str
        Coefficient expanded by the declaration.
    degree : int
        Polynomial degree.
    length : int
        Number of distributed lags.
    zero_nearest, zero_farthest : bool
        Whether the nearest or farthest endpoint is constrained to zero.
    """

    coefficient: str
    degree: int
    length: int
    zero_nearest: bool = False
    zero_farthest: bool = False


@dataclass(frozen=True, slots=True)
class RestrictionTerm:
    """One term in a linear coefficient restriction.

    Attributes
    ----------
    coefficient : str
        Referenced coefficient.
    multiplier : float
        Numeric term multiplier.
    lag : int, default=0
        PDL lag index, when present.
    """

    coefficient: str
    multiplier: float
    lag: int = 0


@dataclass(frozen=True, slots=True)
class CoefficientRestriction:
    """A linear coefficient restriction equal to a numeric target.

    Attributes
    ----------
    terms : tuple of RestrictionTerm
        Left-hand-side linear terms.
    target : float
        Right-hand-side value.
    source : str
        Original declaration text.
    """

    terms: tuple[RestrictionTerm, ...]
    target: float
    source: str


@dataclass(frozen=True, slots=True)
class BehavioralEquation:
    """A behavioral equation and its estimation declarations.

    Attributes
    ----------
    name : str
        Endogenous variable name.
    equation : MdlEquation
        Parsed equation.
    coefficients : tuple of str
        Declared coefficient names.
    regressors : tuple of MdlExpression
        Expressions paired with coefficients.
    estimation_range : MdlTimeRange or None
        Optional ``TSRANGE`` declaration.
    error : AutoregressiveError or None
        Optional autoregressive error declaration.
    restrictions : tuple of CoefficientRestriction
        Linear coefficient restrictions.
    pdls : tuple of PdlDefinition
        Polynomial distributed-lag declarations.
    instruments : tuple of MdlExpression
        Instrumental variables.
    """

    name: str
    equation: MdlEquation
    coefficients: tuple[str, ...]
    regressors: tuple[MdlExpression, ...]
    estimation_range: MdlTimeRange | None = None
    error: AutoregressiveError | None = None
    restrictions: tuple[CoefficientRestriction, ...] = ()
    pdls: tuple[PdlDefinition, ...] = ()
    instruments: tuple[MdlExpression, ...] = ()

    @property
    def expanded_coefficients(self) -> tuple[str, ...]:
        """Coefficient names after expanding PDL declarations.

        Returns
        -------
        tuple of str
            Declared names followed by generated lag names for each PDL.
        """
        definitions = {item.coefficient: item for item in self.pdls}
        output: list[str] = []
        for coefficient in self.coefficients:
            output.append(coefficient)
            definition = definitions.get(coefficient)
            if definition is not None:
                output.extend(
                    f"{coefficient}__PDL__{lag}" for lag in range(1, definition.length)
                )
        return tuple(output)


@dataclass(frozen=True, slots=True)
class IdentityAlternative:
    """One unconditional or conditional definition of an identity.

    Attributes
    ----------
    equation : MdlEquation
        Parsed identity equation.
    condition : MdlExpression or None
        Parsed ``IF>`` condition.
    condition_source : str or None
        Original condition text.
    """

    equation: MdlEquation
    condition: MdlExpression | None = None
    condition_source: str | None = None


@dataclass(frozen=True, slots=True)
class IdentityEquation:
    """An identity with one or more conditional alternatives.

    Attributes
    ----------
    name : str
        Endogenous variable name.
    alternatives : tuple of IdentityAlternative
        Definitions evaluated in declaration order.
    """

    name: str
    alternatives: tuple[IdentityAlternative, ...]

    @property
    def conditional(self) -> bool:
        """Whether at least one alternative has an ``IF>`` condition."""
        return any(item.condition is not None for item in self.alternatives)


@dataclass(frozen=True, slots=True)
class BimetsModel:
    """An immutable model definition produced from BIMETS MDL.

    Attributes
    ----------
    name : str
        User-facing model name or source identifier.
    raw_text : str
        Original MDL document.
    clean_lines : tuple of str
        Normalized operational lines retained for inspection.
    behaviorals : tuple of BehavioralEquation
        Behavioral equations in declaration order.
    identities : tuple of IdentityEquation
        Identity equations in declaration order.
    endogenous, exogenous : tuple of str
        Discovered model variables.
    max_lag, max_lead : int
        Largest temporal offsets used by the model.
    dependencies : mapping of str to frozenset of str
        Current-period endogenous dependencies.

    Examples
    --------
    >>> from bimets import BimetsModel
    >>> text = "MODEL\\nIDENTITY> y\\nEQ> y = x\\nEND"
    >>> model = BimetsModel.from_text(text, name="demo")
    >>> model.endogenous, model.exogenous
    (('y',), ('x',))
    >>> model.identity("y").conditional
    False
    """

    name: str
    raw_text: str
    clean_lines: tuple[str, ...]
    behaviorals: tuple[BehavioralEquation, ...]
    identities: tuple[IdentityEquation, ...]
    endogenous: tuple[str, ...]
    exogenous: tuple[str, ...]
    max_lag: int
    max_lead: int
    dependencies: Mapping[str, frozenset[str]]
    _equation_by_name: Mapping[str, BehavioralEquation | IdentityEquation] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dependencies", MappingProxyType(dict(self.dependencies))
        )
        behavioral_by_name = {item.name: item for item in self.behaviorals}
        identity_by_name = {item.name: item for item in self.identities}
        object.__setattr__(
            self,
            "_equation_by_name",
            MappingProxyType(identity_by_name | behavioral_by_name),
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize canonical source rather than internal mapping proxies."""
        return _restore_model, (self.raw_text, self.name)

    @property
    def forward_looking(self) -> bool:
        """Whether the model contains at least one lead reference."""
        return self.max_lead > 0

    @property
    def conditional_endogenous(self) -> tuple[str, ...]:
        """Endogenous identities containing at least one ``IF>`` condition.

        These variables remain structurally endogenous. During simulation,
        when none of their alternatives is active, they retain the value
        supplied in the model data for the current period, following BIMETS R.

        Returns
        -------
        tuple of str
            Conditional identity names in model declaration order.
        """
        return tuple(
            identity.name for identity in self.identities if identity.conditional
        )

    @property
    def coefficient_count(self) -> int:
        """Number of explicitly declared behavioral coefficients."""
        return sum(len(item.coefficients) for item in self.behaviorals)

    def behavioral(self, name: str) -> BehavioralEquation:
        """Return a behavioral equation by endogenous name.

        Parameters
        ----------
        name : str
            Exact parsed equation name.

        Returns
        -------
        BehavioralEquation
            Matching declaration.

        Raises
        ------
        KeyError
            If the model has no matching behavioral equation.
        """
        definition = self._equation_by_name[name]
        if not isinstance(definition, BehavioralEquation):
            raise KeyError(name)
        return definition

    def identity(self, name: str) -> IdentityEquation:
        """Return an identity by endogenous name.

        Parameters
        ----------
        name : str
            Exact parsed identity name.

        Returns
        -------
        IdentityEquation
            Matching declaration.

        Raises
        ------
        KeyError
            If the model has no matching identity.
        """
        definition = self._equation_by_name[name]
        if not isinstance(definition, IdentityEquation):
            raise KeyError(name)
        return definition

    def _equation_definition(self, name: str) -> BehavioralEquation | IdentityEquation:
        """Return a pre-indexed equation definition for solver dispatch."""
        return self._equation_by_name[name]

    def bind(self, data: BimetsDataset | Mapping[str, BimetsSeries]) -> BoundModel:
        """Associate this model with homogeneous time-series data.

        Parameters
        ----------
        data : BimetsDataset or mapping of str to BimetsSeries
            Dataset containing every endogenous and exogenous model variable.

        Returns
        -------
        BoundModel
            Validated model-data association.
        """
        from bimets.mdl._binding import bind_model_data

        return bind_model_data(self, data)

    def estimate(
        self,
        data: BimetsDataset | Mapping[str, BimetsSeries],
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
        data : BimetsDataset or mapping of str to BimetsSeries
            Model data.
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
            data,
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
        data: BimetsDataset | Mapping[str, BimetsSeries],
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
        """Run a deterministic simulation of this model.

        Parameters are equivalent to :func:`bimets.simulate`.
        """
        from bimets.mdl._simulation import simulate

        return simulate(
            self,
            data,
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
        data: BimetsDataset | Mapping[str, BimetsSeries],
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
        """Run stochastic simulations of this model.

        Parameters are equivalent to :func:`bimets.stochastic_simulate`.
        """
        from bimets.mdl._stochastic import stochastic_simulate

        return stochastic_simulate(
            self,
            data,
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
        data: BimetsDataset | Mapping[str, BimetsSeries],
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
        """Compute multiplier matrices for this model.

        Parameters are equivalent to :func:`bimets.multiplier_matrix`.
        """
        from bimets.mdl._multipliers import multiplier_matrix

        return multiplier_matrix(
            self,
            data,
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
        data: BimetsDataset | Mapping[str, BimetsSeries],
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
        """Find instrument paths that achieve endogenous targets.

        Parameters are equivalent to :func:`bimets.renormalize`.
        """
        from bimets.mdl._renormalization import renormalize

        return renormalize(
            self,
            data,
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
        data: BimetsDataset | Mapping[str, BimetsSeries],
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
        """Perform Monte Carlo optimal control on this model.

        Parameters are equivalent to :func:`bimets.optimize_model`.
        """
        from bimets.mdl._optimization import optimize_model

        return optimize_model(
            self,
            data,
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

    @classmethod
    def from_text(cls, text: str, *, name: str = "<string>") -> BimetsModel:
        """Parse a model definition from a string.

        Parameters
        ----------
        text : str
            Complete MDL document, including ``MODEL`` and ``END``.
        name : str, default="<string>"
            Model identifier.

        Returns
        -------
        BimetsModel
            Parsed and validated model.
        """
        from bimets.mdl._parser import parse_mdl

        return parse_mdl(text, name=name)

    @classmethod
    def from_file(cls, path: str | Path) -> BimetsModel:
        """Parse a model definition from a UTF-8 text file.

        Parameters
        ----------
        path : str or pathlib.Path
            MDL source file.

        Returns
        -------
        BimetsModel
            Parsed and validated model named after ``path``.
        """
        from bimets.mdl._parser import load_model

        return load_model(model_file=path)

    def __repr__(self) -> str:
        return (
            f"BimetsModel(name={self.name!r}, behaviorals={len(self.behaviorals)}, "
            f"identities={len(self.identities)}, coefficients={self.coefficient_count}, "
            f"forward_looking={self.forward_looking})"
        )
