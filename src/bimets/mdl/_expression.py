"""Safe expression tree and parser for BIMETS MDL."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, Never


class MdlError(ValueError):
    """Base exception for invalid MDL input."""


class MdlSyntaxError(MdlError):
    """An MDL syntax or semantic validation error.

    Parameters
    ----------
    message : str
        Human-readable error description.
    line : int, optional
        One-based source line included in the rendered message.

    Attributes
    ----------
    line : int or None
        Source line associated with the error.
    """

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        prefix = f"line {line}: " if line is not None else ""
        super().__init__(prefix + message)


@dataclass(frozen=True, slots=True)
class Number:
    """A numeric literal in an MDL expression.

    Attributes
    ----------
    value : float
        Parsed numeric value.
    """

    value: float


@dataclass(frozen=True, slots=True)
class Variable:
    """A variable reference in an MDL expression.

    Attributes
    ----------
    name : str
        Case-preserving variable name.
    """

    name: str


@dataclass(frozen=True, slots=True)
class UnaryExpression:
    """A unary arithmetic expression.

    Attributes
    ----------
    operator : str
        Unary ``+`` or ``-``.
    operand : MdlExpression
        Expression to which the operator applies.
    """

    operator: str
    operand: MdlExpression


@dataclass(frozen=True, slots=True)
class BinaryExpression:
    """A binary arithmetic, comparison, or logical expression.

    Attributes
    ----------
    left, right : MdlExpression
        Operands.
    operator : str
        Normalized MDL operator.
    """

    left: MdlExpression
    operator: str
    right: MdlExpression


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """A call to a supported MDL function.

    Attributes
    ----------
    name : str
        Canonical uppercase function name.
    arguments : tuple of MdlExpression
        Parsed positional arguments.
    """

    name: str
    arguments: tuple[MdlExpression, ...]


type MdlExpression = (
    Number | Variable | UnaryExpression | BinaryExpression | FunctionCall
)


FUNCTION_ALIASES: Final[dict[str, str]] = {
    "LAG": "TSLAG",
    "LEAD": "TSLEAD",
    "DEL": "TSDELTA",
    "MAVE": "MOVAVG",
    "MTOT": "MOVSUM",
}
SUPPORTED_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "ABS",
        "EXP",
        "LOG",
        "MOVAVG",
        "MOVSUM",
        "TSDELTA",
        "TSDELTALOG",
        "TSDELTAP",
        "TSLAG",
        "TSLEAD",
    }
)
LHS_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"EXP", "LOG", "TSDELTA", "TSDELTALOG", "TSDELTAP"}
)


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    value: str
    position: int


_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOT_OPERATORS: Final[dict[str, str]] = {
    ".EQ.": "==",
    ".NE.": "!=",
    ".GE.": ">=",
    ".LE.": "<=",
    ".GT.": ">",
    ".LT.": "<",
}
_BINDING_POWER: Final[dict[str, tuple[int, int]]] = {
    "|": (10, 11),
    "&": (20, 21),
    "==": (30, 31),
    "!=": (30, 31),
    "<": (30, 31),
    "<=": (30, 31),
    ">": (30, 31),
    ">=": (30, 31),
    "+": (40, 41),
    "-": (40, 41),
    "*": (50, 51),
    "/": (50, 51),
    "^": (70, 70),
}


def parse_expression(source: str, *, line: int | None = None) -> MdlExpression:
    """Parse an MDL expression without evaluating user input.

    Parameters
    ----------
    source : str
        Expression text.
    line : int, optional
        Source line attached to syntax errors.

    Returns
    -------
    MdlExpression
        Root of an immutable typed expression tree.

    Raises
    ------
    MdlSyntaxError
        If the expression contains invalid syntax, names, functions, or
        function arguments.

    Examples
    --------
    >>> from bimets import BinaryExpression, parse_expression
    >>> expression = parse_expression("income + TSLAG(tax, 1)")
    >>> isinstance(expression, BinaryExpression)
    True
    >>> expression.operator
    '+'
    """
    parser = _ExpressionParser(_tokenize(source, line=line), source, line)
    return parser.parse()


def variable_names(expression: MdlExpression) -> frozenset[str]:
    """Return variable names referenced by an expression.

    Parameters
    ----------
    expression : MdlExpression
        Parsed expression tree.

    Returns
    -------
    frozenset of str
        Referenced names, excluding the built-in constant ``pi``.

    Examples
    --------
    >>> from bimets.mdl import parse_expression, variable_names
    >>> variable_names(parse_expression("x + TSLAG(y, 1)")) == {"x", "y"}
    True
    """
    if isinstance(expression, Number):
        return frozenset()
    if isinstance(expression, Variable):
        return (
            frozenset()
            if expression.name.lower() == "pi"
            else frozenset({expression.name})
        )
    if isinstance(expression, UnaryExpression):
        return variable_names(expression.operand)
    if isinstance(expression, BinaryExpression):
        return variable_names(expression.left) | variable_names(expression.right)
    output: frozenset[str] = frozenset()
    for argument in expression.arguments:
        output |= variable_names(argument)
    return output


def temporal_offsets(expression: MdlExpression, offset: int = 0) -> tuple[int, ...]:
    """Return relative periods at which variables are referenced.

    Parameters
    ----------
    expression : MdlExpression
        Parsed expression tree.
    offset : int, default=0
        Base offset applied to every reference.

    Returns
    -------
    tuple of int
        Offsets in expression traversal order. Lags are negative and leads are
        positive.
    """
    if isinstance(expression, Number):
        return ()
    if isinstance(expression, Variable):
        return () if expression.name.lower() == "pi" else (offset,)
    if isinstance(expression, UnaryExpression):
        return temporal_offsets(expression.operand, offset)
    if isinstance(expression, BinaryExpression):
        return temporal_offsets(expression.left, offset) + temporal_offsets(
            expression.right, offset
        )
    first = expression.arguments[0]
    periods = _period_argument(expression)
    if expression.name == "TSLAG":
        return temporal_offsets(first, offset - periods)
    if expression.name == "TSLEAD":
        return temporal_offsets(first, offset + periods)
    if expression.name in {"TSDELTA", "TSDELTALOG", "TSDELTAP"}:
        return temporal_offsets(first, offset) + temporal_offsets(
            first, offset - periods
        )
    if expression.name in {"MOVAVG", "MOVSUM"}:
        return tuple(
            item
            for lag in range(periods)
            for item in temporal_offsets(first, offset - lag)
        )
    return tuple(
        item
        for argument in expression.arguments
        for item in temporal_offsets(argument, offset)
    )


def variable_offsets(
    expression: MdlExpression, offset: int = 0
) -> dict[str, frozenset[int]]:
    """Map each variable to all relative periods at which it is used.

    Parameters
    ----------
    expression : MdlExpression
        Parsed expression tree.
    offset : int, default=0
        Base offset applied to every reference.

    Returns
    -------
    dict of str to frozenset of int
        Per-variable temporal offsets.

    Examples
    --------
    >>> from bimets.mdl import parse_expression, variable_offsets
    >>> variable_offsets(parse_expression("x + TSLAG(x, 2) + TSLEAD(y, 1)"))
    {'x': frozenset({0, -2}), 'y': frozenset({1})}
    """
    output: dict[str, set[int]] = {}

    def visit(node: MdlExpression, current: int) -> None:
        """Collect variable offsets while traversing an expression tree."""
        if isinstance(node, Number):
            return
        if isinstance(node, Variable):
            if node.name.lower() != "pi":
                output.setdefault(node.name, set()).add(current)
            return
        if isinstance(node, UnaryExpression):
            visit(node.operand, current)
            return
        if isinstance(node, BinaryExpression):
            visit(node.left, current)
            visit(node.right, current)
            return
        periods = _period_argument(node)
        if node.name == "TSLAG":
            visit(node.arguments[0], current - periods)
        elif node.name == "TSLEAD":
            visit(node.arguments[0], current + periods)
        elif node.name in {"TSDELTA", "TSDELTALOG", "TSDELTAP"}:
            visit(node.arguments[0], current)
            visit(node.arguments[0], current - periods)
        elif node.name in {"MOVAVG", "MOVSUM"}:
            for lag in range(periods):
                visit(node.arguments[0], current - lag)
        else:
            for argument in node.arguments:
                visit(argument, current)

    visit(expression, offset)
    return {name: frozenset(offsets) for name, offsets in output.items()}


def numeric_value(expression: MdlExpression) -> float | None:
    """Return the value of a simple numeric constant expression.

    Parameters
    ----------
    expression : MdlExpression
        Parsed number, ``pi``, or unary signed constant.

    Returns
    -------
    float or None
        Constant value, or ``None`` when variables or compound operations are
        present.
    """
    if isinstance(expression, Number):
        return expression.value
    if isinstance(expression, Variable) and expression.name.lower() == "pi":
        return math.pi
    if isinstance(expression, UnaryExpression):
        value = numeric_value(expression.operand)
        if value is None:
            return None
        return value if expression.operator == "+" else -value
    return None


def _period_argument(call: FunctionCall) -> int:
    """Return and validate the integer period argument of a call."""
    if call.name not in {
        "MOVAVG",
        "MOVSUM",
        "TSDELTA",
        "TSDELTALOG",
        "TSDELTAP",
        "TSLAG",
        "TSLEAD",
    }:
        return 1
    if len(call.arguments) == 1:
        return 1
    value = numeric_value(call.arguments[1])
    assert value is not None
    return int(value)


def _tokenize(source: str, *, line: int | None) -> tuple[_Token, ...]:
    """Tokenize an MDL expression for recursive-descent parsing."""
    tokens: list[_Token] = []
    position = 0
    while position < len(source):
        if source[position].isspace():
            position += 1
            continue
        upper = source[position:].upper()
        dot_match = next(
            (item for item in _DOT_OPERATORS if upper.startswith(item)), None
        )
        if dot_match is not None:
            tokens.append(_Token("OP", _DOT_OPERATORS[dot_match], position))
            position += len(dot_match)
            continue
        number = _NUMBER.match(source, position)
        if number is not None:
            tokens.append(_Token("NUMBER", number.group(), position))
            position = number.end()
            continue
        name = _NAME.match(source, position)
        if name is not None:
            tokens.append(_Token("NAME", name.group(), position))
            position = name.end()
            continue
        two = source[position : position + 2]
        if two in {"<=", ">=", "==", "!="}:
            tokens.append(_Token("OP", two, position))
            position += 2
            continue
        character = source[position]
        if character in "+-*/^<>&|(),":
            kind = "OP" if character not in "()," else character
            tokens.append(_Token(kind, character, position))
            position += 1
            continue
        raise MdlSyntaxError(
            f"unexpected character {character!r} at column {position + 1}", line=line
        )
    tokens.append(_Token("EOF", "", len(source)))
    return tuple(tokens)


class _ExpressionParser:
    def __init__(
        self, tokens: tuple[_Token, ...], source: str, line: int | None
    ) -> None:
        """Initialize the parser state for a token sequence."""
        self.tokens = tokens
        self.source = source
        self.line = line
        self.position = 0

    @property
    def current(self) -> _Token:
        """Return the token at the current parser position."""
        return self.tokens[self.position]

    def parse(self) -> MdlExpression:
        """Parse the complete token sequence into an expression tree."""
        if self.current.kind == "EOF":
            raise MdlSyntaxError("empty expression", line=self.line)
        expression = self._expression(0)
        if self.current.kind != "EOF":
            self._error(f"unexpected token {self.current.value!r}")
        return expression

    def _expression(self, minimum_power: int) -> MdlExpression:
        """Parse an expression using precedence climbing."""
        token = self.current
        if token.kind == "OP" and token.value in {"+", "-"}:
            self.position += 1
            left: MdlExpression = UnaryExpression(token.value, self._expression(60))
        else:
            left = self._primary()

        while self.current.kind == "OP" and self.current.value in _BINDING_POWER:
            operator = self.current.value
            left_power, right_power = _BINDING_POWER[operator]
            if left_power < minimum_power:
                break
            self.position += 1
            right = self._expression(right_power)
            left = BinaryExpression(left, operator, right)
        return left

    def _primary(self) -> MdlExpression:
        """Parse a literal, variable, function call, or grouped expression."""
        token = self.current
        if token.kind == "NUMBER":
            self.position += 1
            return Number(float(token.value))
        if token.kind == "NAME":
            self.position += 1
            if self.current.kind != "(":
                return Variable(token.value)
            return self._call(token)
        if token.kind == "(":
            self.position += 1
            expression = self._expression(0)
            self._consume(")", "expected closing parenthesis")
            return expression
        self._error(f"expected an expression, found {token.value!r}")

    def _call(self, name_token: _Token) -> FunctionCall:
        """Parse and validate an MDL function call."""
        self.position += 1
        arguments: list[MdlExpression] = []
        if self.current.kind != ")":
            while True:
                arguments.append(self._expression(0))
                if self.current.kind != ",":
                    break
                self.position += 1
        self._consume(")", "expected closing parenthesis in function call")
        raw_name = name_token.value.upper()
        name = FUNCTION_ALIASES.get(raw_name, raw_name)
        if name not in SUPPORTED_FUNCTIONS:
            self._error(f"unsupported MDL function {name_token.value}()", name_token)
        call = FunctionCall(name, tuple(arguments))
        _validate_call(call, line=self.line)
        return call

    def _consume(self, kind: str, message: str) -> None:
        """Consume the expected token or raise a syntax error."""
        if self.current.kind != kind:
            self._error(message)
        self.position += 1

    def _error(self, message: str, token: _Token | None = None) -> Never:
        """Raise a syntax error at the selected token."""
        selected = self.current if token is None else token
        raise MdlSyntaxError(
            f"{message} at column {selected.position + 1}", line=self.line
        )


def _validate_call(call: FunctionCall, *, line: int | None) -> None:
    """Validate call for internal processing."""
    count = len(call.arguments)
    if call.name in {"ABS", "EXP", "LOG"}:
        if count != 1:
            raise MdlSyntaxError(
                f"{call.name}() requires exactly one argument", line=line
            )
        return
    if count not in {1, 2}:
        raise MdlSyntaxError(f"{call.name}() requires one or two arguments", line=line)
    if count == 1:
        return
    value = numeric_value(call.arguments[1])
    minimum = 0 if call.name in {"TSLAG", "TSLEAD"} else 1
    if value is None or not value.is_integer() or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise MdlSyntaxError(
            f"{call.name}() period must be a {qualifier} integer", line=line
        )
