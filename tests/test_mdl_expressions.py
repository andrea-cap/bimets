from __future__ import annotations

import math

import pytest

from bimets import (
    BinaryExpression,
    FunctionCall,
    MdlSyntaxError,
    Number,
    UnaryExpression,
    Variable,
    parse_expression,
)
from bimets.mdl import temporal_offsets, variable_names, variable_offsets


def test_expression_parser_builds_a_precedence_aware_tree() -> None:
    expression = parse_expression("-2^2 + a*b / (c-1)")

    assert expression == BinaryExpression(
        UnaryExpression(
            "-",
            BinaryExpression(Number(2), "^", Number(2)),
        ),
        "+",
        BinaryExpression(
            BinaryExpression(Variable("a"), "*", Variable("b")),
            "/",
            BinaryExpression(Variable("c"), "-", Number(1)),
        ),
    )
    assert variable_names(expression) == frozenset({"a", "b", "c"})


def test_expression_parser_normalizes_functions_aliases_and_comparisons() -> None:
    expression = parse_expression("lag(x,2) + LEAD(y) + DEL(z) + MAVE(w,3) + MTOT(q,2)")
    assert variable_names(expression) == frozenset({"x", "y", "z", "w", "q"})
    names: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, FunctionCall):
            names.append(node.name)
            for argument in node.arguments:
                visit(argument)
        elif isinstance(node, BinaryExpression):
            visit(node.left)
            visit(node.right)

    visit(expression)
    assert names == ["TSLAG", "TSLEAD", "TSDELTA", "MOVAVG", "MOVSUM"]
    assert parse_expression("x.GE.0 & x.LT.10 | y != 2") == parse_expression(
        "x >= 0 & x < 10 | y != 2"
    )


def test_constants_scientific_notation_and_right_associative_power() -> None:
    expression = parse_expression("pi + .5e2 + 2^3^2")

    assert variable_names(expression) == frozenset()
    assert isinstance(expression, BinaryExpression)
    power = expression.right
    assert power == BinaryExpression(
        Number(2), "^", BinaryExpression(Number(3), "^", Number(2))
    )
    assert math.isclose(math.pi, 3.141592653589793)


def test_temporal_analysis_accounts_for_nested_transformations() -> None:
    expression = parse_expression(
        "TSLAG(TSDELTA(x,2),3)+TSLEAD(y,4)+MOVAVG(z,3)+LOG(w)"
    )

    assert sorted(temporal_offsets(expression)) == [-5, -3, -2, -1, 0, 0, 4]
    assert variable_offsets(expression) == {
        "x": frozenset({-3, -5}),
        "y": frozenset({4}),
        "z": frozenset({0, -1, -2}),
        "w": frozenset({0}),
    }


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "empty expression"),
        ("x @ y", "unexpected character"),
        ("x y", "unexpected token"),
        ("(x+1", "closing parenthesis"),
        ("LOG(x", "closing parenthesis"),
        ("UNKNOWN(x)", "unsupported MDL function"),
        ("LOG()", "exactly one"),
        ("ABS(x,2)", "exactly one"),
        ("TSLAG()", "one or two"),
        ("TSLAG(x,1,2)", "one or two"),
        ("TSLAG(x,-1)", "non-negative integer"),
        ("TSLEAD(x,1.5)", "non-negative integer"),
        ("TSDELTA(x,0)", "positive integer"),
        ("MOVSUM(x,y)", "positive integer"),
        ("x +", "expected an expression"),
    ],
)
def test_expression_validation(source: str, message: str) -> None:
    with pytest.raises(MdlSyntaxError, match=message):
        parse_expression(source, line=7)


def test_all_documented_scalar_functions_are_accepted() -> None:
    expression = parse_expression(
        "ABS(x)+EXP(x)+LOG(x)+TSDELTAP(x)+TSDELTALOG(x)+MOVAVG(x)+MOVSUM(x)"
    )

    assert variable_names(expression) == frozenset({"x"})
