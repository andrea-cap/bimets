"""BIMETS Model Description Language parsing and public model structures."""

from bimets.mdl._binding import BoundModel, bind_model_data
from bimets.mdl._estimation import (
    ChowTestResult,
    EquationEstimationResult,
    ModelEstimationResult,
    estimate,
)
from bimets.mdl._evaluation import MdlValue, evaluate_expression
from bimets.mdl._expression import (
    BinaryExpression,
    FunctionCall,
    MdlError,
    MdlExpression,
    MdlSyntaxError,
    Number,
    UnaryExpression,
    Variable,
    parse_expression,
    temporal_offsets,
    variable_names,
    variable_offsets,
)
from bimets.mdl._model import (
    AutoregressiveError,
    BehavioralEquation,
    BimetsModel,
    CoefficientRestriction,
    IdentityAlternative,
    IdentityEquation,
    MdlEquation,
    MdlTimeRange,
    PdlDefinition,
    RestrictionTerm,
)
from bimets.mdl._multipliers import (
    MultiplierMatrixError,
    MultiplierMatrixResult,
    multiplier_matrix,
)
from bimets.mdl._optimization import (
    OptimizationBound,
    OptimizationError,
    OptimizationFunction,
    OptimizationRestriction,
    OptimizationResult,
    optimize_model,
)
from bimets.mdl._parser import load_model, parse_mdl
from bimets.mdl._renormalization import (
    RenormalizationError,
    RenormalizationResult,
    renormalize,
)
from bimets.mdl._simulation import (
    AdjustmentValue,
    CoefficientInput,
    ExogenizationValue,
    SimulationBlock,
    SimulationConvergenceError,
    SimulationResult,
    simulate,
)
from bimets.mdl._stochastic import (
    DisturbanceParameters,
    StochasticDisturbance,
    StochasticSeriesResult,
    StochasticSimulationError,
    StochasticSimulationResult,
    stochastic_simulate,
)

# BIMETS R compatibility names. Methods and canonical lowercase functions keep
# their native Python signatures and behavior.
ESTIMATE = estimate
LOAD_MODEL = load_model
LOAD_MODEL_DATA = bind_model_data
MULTMATRIX = multiplier_matrix
OPTIMIZE = optimize_model
RENORM = renormalize
SIMULATE = simulate
STOCHSIMULATE = stochastic_simulate

__all__ = [
    "ESTIMATE",
    "LOAD_MODEL",
    "LOAD_MODEL_DATA",
    "MULTMATRIX",
    "OPTIMIZE",
    "RENORM",
    "SIMULATE",
    "STOCHSIMULATE",
    "AdjustmentValue",
    "AutoregressiveError",
    "BehavioralEquation",
    "BimetsModel",
    "BinaryExpression",
    "BoundModel",
    "ChowTestResult",
    "CoefficientInput",
    "CoefficientRestriction",
    "DisturbanceParameters",
    "EquationEstimationResult",
    "ExogenizationValue",
    "FunctionCall",
    "IdentityAlternative",
    "IdentityEquation",
    "MdlEquation",
    "MdlError",
    "MdlExpression",
    "MdlSyntaxError",
    "MdlTimeRange",
    "MdlValue",
    "ModelEstimationResult",
    "MultiplierMatrixError",
    "MultiplierMatrixResult",
    "Number",
    "OptimizationBound",
    "OptimizationError",
    "OptimizationFunction",
    "OptimizationRestriction",
    "OptimizationResult",
    "PdlDefinition",
    "RenormalizationError",
    "RenormalizationResult",
    "RestrictionTerm",
    "SimulationBlock",
    "SimulationConvergenceError",
    "SimulationResult",
    "StochasticDisturbance",
    "StochasticSeriesResult",
    "StochasticSimulationError",
    "StochasticSimulationResult",
    "UnaryExpression",
    "Variable",
    "bind_model_data",
    "estimate",
    "evaluate_expression",
    "load_model",
    "multiplier_matrix",
    "optimize_model",
    "parse_expression",
    "parse_mdl",
    "renormalize",
    "simulate",
    "stochastic_simulate",
    "temporal_offsets",
    "variable_names",
    "variable_offsets",
]
