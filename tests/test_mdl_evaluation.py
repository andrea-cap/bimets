from __future__ import annotations

import math

import numpy as np
import pytest

from bimets import (
    BimetsMask,
    BimetsSeries,
    evaluate_expression,
    parse_expression,
    timeseries,
)


def evaluate(source: str) -> float | bool | BimetsSeries | BimetsMask:
    data = {
        "x": timeseries([1, 2, 4, 8], start=(2020, 1), freq="Q"),
        "y": timeseries([2, 4, 8, 16], start=(2020, 1), freq="Q"),
    }
    return evaluate_expression(parse_expression(source), data)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("-2 + 3^2", 7.0),
        ("ABS(-2)", 2.0),
        ("LOG(EXP(2))", 2.0),
        ("pi > 3", True),
    ],
)
def test_scalar_expression_evaluation(source: str, expected: float | bool) -> None:
    assert evaluate(source) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("source", "expected", "start"),
    [
        ("x + y/2", [2, 4, 8, 16], (2020, 1)),
        ("TSLAG(x,2)", [1, 2, 4, 8], (2020, 3)),
        ("TSLEAD(x)", [1, 2, 4, 8], (2019, 4)),
        ("TSDELTA(x)", [1, 2, 4], (2020, 2)),
        ("TSDELTALOG(x)", [math.log(2)] * 3, (2020, 2)),
        ("TSDELTAP(x)", [100, 100, 100], (2020, 2)),
        ("MOVAVG(x,2)", [1.5, 3, 6], (2020, 2)),
        ("MOVSUM(x,2)", [3, 6, 12], (2020, 2)),
        ("LOG(EXP(x))", [1, 2, 4, 8], (2020, 1)),
    ],
)
def test_series_expression_evaluation(
    source: str, expected: list[float], start: tuple[int, int]
) -> None:
    result = evaluate(source)

    assert isinstance(result, BimetsSeries)
    np.testing.assert_allclose(result.values, expected)
    assert (result.start.year, result.start.period) == start


def test_comparison_and_logical_expression_returns_indexed_mask() -> None:
    result = evaluate("(x > 1) & (y <= 8)")

    assert isinstance(result, BimetsMask)
    assert list(result) == [False, True, True, False]


def test_evaluation_reports_missing_variables_and_invalid_operand_types() -> None:
    with pytest.raises(KeyError, match="missing"):
        evaluate_expression(parse_expression("missing + 1"), {})
    with pytest.raises(TypeError, match="logical operands"):
        evaluate("x & y")
    with pytest.raises(TypeError, match="numeric operands"):
        evaluate("-(x > 1)")
    with pytest.raises(TypeError, match="BimetsSeries"):
        evaluate("TSLAG(1)")
