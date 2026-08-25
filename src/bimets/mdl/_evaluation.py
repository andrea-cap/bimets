"""Numerical evaluation of parsed MDL expressions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeGuard

import numpy as np

from bimets.mdl._expression import (
    BinaryExpression,
    FunctionCall,
    MdlExpression,
    Number,
    UnaryExpression,
    Variable,
    numeric_value,
)
from bimets.timeseries import BimetsMask, BimetsSeries

type MdlValue = float | bool | BimetsSeries | BimetsMask
type _NumericValue = float | BimetsSeries
type _LogicalValue = bool | BimetsMask


def evaluate_expression(
    expression: MdlExpression,
    data: Mapping[str, BimetsSeries],
) -> MdlValue:
    """Evaluate a parsed MDL expression against named time series.

    Parameters
    ----------
    expression : MdlExpression
        Expression returned by :func:`parse_expression`.
    data : mapping of str to BimetsSeries
        Values used to resolve variable references.

    Returns
    -------
    float, bool, BimetsSeries, or BimetsMask
        The scalar or indexed result of the expression.

    Raises
    ------
    KeyError
        If a referenced variable is absent from ``data``.
    TypeError
        If an operator or function receives an incompatible value.

    Notes
    -----
    Evaluation walks the typed expression tree directly. Source text is never
    passed to Python's :func:`eval`.

    Examples
    --------
    >>> from bimets import evaluate_expression, parse_expression, timeseries
    >>> data = {"x": timeseries([1, 2, 4], start=(2020, 1))}
    >>> result = evaluate_expression(parse_expression("TSDELTA(x) + 1"), data)
    >>> result.values.tolist()
    [2.0, 3.0]
    """
    if isinstance(expression, Number):
        return expression.value
    if isinstance(expression, Variable):
        if expression.name.lower() == "pi":
            return math.pi
        return data[expression.name]
    if isinstance(expression, UnaryExpression):
        operand = evaluate_expression(expression.operand, data)
        if not _is_numeric(operand):
            raise TypeError(f"unary {expression.operator} requires numeric operands")
        return operand if expression.operator == "+" else -operand
    if isinstance(expression, BinaryExpression):
        left = evaluate_expression(expression.left, data)
        right = evaluate_expression(expression.right, data)
        return _evaluate_binary(left, expression.operator, right)
    return _evaluate_function(expression, data)


def _evaluate_binary(left: MdlValue, operator: str, right: MdlValue) -> MdlValue:
    """Evaluate binary for internal processing."""
    if operator in {"&", "|"}:
        if not _is_logical(left) or not _is_logical(right):
            raise TypeError(f"{operator} requires logical operands")
        return left & right if operator == "&" else left | right

    if operator in {"==", "!=", "<", "<=", ">", ">="}:
        if not _is_numeric(left) or not _is_numeric(right):
            raise TypeError(f"{operator} requires numeric operands")
        if operator == "==":
            return left == right
        if operator == "!=":
            return left != right
        if operator == "<":
            return left < right
        if operator == "<=":
            return left <= right
        if operator == ">":
            return left > right
        return left >= right

    if not _is_numeric(left) or not _is_numeric(right):
        raise TypeError(f"{operator} requires numeric operands")
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        return left / right
    return left**right


def _evaluate_function(
    call: FunctionCall, data: Mapping[str, BimetsSeries]
) -> MdlValue:
    """Evaluate function for internal processing."""
    value = evaluate_expression(call.arguments[0], data)
    if call.name in {"ABS", "EXP", "LOG"}:
        if not _is_numeric(value):
            raise TypeError(f"{call.name} requires numeric operands")
        if isinstance(value, BimetsSeries):
            with np.errstate(all="ignore"):
                values = {
                    "ABS": np.abs,
                    "EXP": np.exp,
                    "LOG": np.log,
                }[call.name](value.values)
            return BimetsSeries(
                values,
                start=value.start,
                freq=value.freq,
                metadata=value.metadata,
            )
        scalar = float(value)
        with np.errstate(all="ignore"):
            return float(
                {"ABS": np.abs, "EXP": np.exp, "LOG": np.log}[call.name](scalar)
            )

    if not isinstance(value, BimetsSeries):
        raise TypeError(f"{call.name} requires a BimetsSeries argument")
    periods = 1
    if len(call.arguments) == 2:
        parsed_periods = numeric_value(call.arguments[1])
        assert parsed_periods is not None
        periods = int(parsed_periods)
    if call.name == "TSLAG":
        return value.lag(periods)
    if call.name == "TSLEAD":
        return value.lead(periods)
    if call.name == "TSDELTA":
        return value.delta(lag=periods)
    if call.name == "TSDELTALOG":
        return value.delta_log(lag=periods)
    if call.name == "TSDELTAP":
        return value.delta_percent(lag=periods)
    if call.name == "MOVAVG":
        return value.moving_average(periods)
    if call.name == "MOVSUM":
        return value.moving_sum(periods)
    raise AssertionError(f"unexpected MDL function: {call.name}")


def _is_numeric(value: MdlValue) -> TypeGuard[_NumericValue]:
    """Return whether numeric."""
    return not isinstance(value, (BimetsMask, bool))


def _is_logical(value: MdlValue) -> TypeGuard[_LogicalValue]:
    """Return whether logical."""
    return isinstance(value, (BimetsMask, bool))
