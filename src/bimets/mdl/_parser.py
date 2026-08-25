"""Parser for BIMETS Model Description Language documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bimets.mdl._expression import (
    LHS_FUNCTIONS,
    SUPPORTED_FUNCTIONS,
    BinaryExpression,
    FunctionCall,
    MdlExpression,
    MdlSyntaxError,
    Number,
    UnaryExpression,
    Variable,
    numeric_value,
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

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DIRECTIVE = re.compile(
    r"^(BEHAVIORAL|EQUATION|IDENTITY|EQ|COEFF|ERROR|RESTRICT|PDL|IF|IV|STORE)\s*>(.*)$",
    re.IGNORECASE,
)
_TSRANGE = re.compile(r"^TSRANGE(?:\s+|$)(.*)$", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"(?<![<>=!])=(?!=)")


@dataclass(frozen=True, slots=True)
class _Statement:
    keyword: str
    content: str
    line: int
    source_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Group:
    kind: str
    header: _Statement
    statements: tuple[_Statement, ...]


def load_model(
    *,
    model_file: str | Path | None = None,
    model_text: str | None = None,
    name: str | None = None,
) -> BimetsModel:
    """Load an MDL model from text or a UTF-8 file.

    Parameters
    ----------
    model_file : str or pathlib.Path, optional
        UTF-8 MDL source file.
    model_text : str, optional
        Complete MDL document. It takes precedence over ``model_file`` when
        both are supplied.
    name : str, optional
        Model identifier. Defaults to ``<string>`` for text or the file path.

    Returns
    -------
    BimetsModel
        Parsed and semantically validated model.

    Raises
    ------
    ValueError
        If neither input is supplied.
    TypeError
        If ``model_text`` is not a string.
    MdlSyntaxError
        If the selected model definition is invalid.

    Examples
    --------
    >>> from bimets import load_model
    >>> text = "MODEL\\nIDENTITY> output\\nEQ> output = demand\\nEND"
    >>> model = load_model(model_text=text, name="market")
    >>> model.name, model.endogenous
    ('market', ('output',))
    """
    if model_text is not None:
        if not isinstance(model_text, str):
            raise TypeError("model_text must be a string")
        return parse_mdl(model_text, name=name or "<string>")
    if model_file is None:
        raise ValueError("model_file or model_text must be provided")
    path = Path(model_file)
    text = path.read_text(encoding="utf-8")
    return parse_mdl(text, name=name or str(path))


def parse_mdl(text: str, *, name: str = "<string>") -> BimetsModel:
    """Parse and semantically validate a BIMETS MDL definition.

    Parameters
    ----------
    text : str
        Complete model document beginning with ``MODEL`` and ending with
        ``END``.
    name : str, default="<string>"
        Identifier stored on the returned model.

    Returns
    -------
    BimetsModel
        Immutable model, equations, expression trees, discovered variables,
        temporal bounds, and dependencies.

    Raises
    ------
    TypeError
        If ``text`` is not a string.
    MdlSyntaxError
        If syntax or model-level semantic validation fails.

    Notes
    -----
    Parsing is evaluation-free: MDL text is never passed to Python ``eval``.
    Legacy ``STORE>`` declarations are syntax-checked for BIMETS compatibility
    and then discarded because they do not affect model execution.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise MdlSyntaxError("empty model definition")
    clean, numbered = _clean_lines(text)
    if not numbered or numbered[0][1] != "MODEL" or numbered[-1][1] != "END":
        raise MdlSyntaxError("model must begin with MODEL and end with END")
    body = numbered[1:-1]
    statements = _statements(body)
    groups = _groups(statements)
    if not groups:
        raise MdlSyntaxError("model contains no equations")

    behaviorals: list[BehavioralEquation] = []
    identity_order: list[str] = []
    identity_alternatives: dict[str, list[IdentityAlternative]] = {}
    endogenous_order: list[str] = []

    for group in groups:
        if group.kind == "BEHAVIORAL":
            behavioral = _parse_behavioral(group)
            if behavioral.name in endogenous_order:
                raise MdlSyntaxError(
                    f"duplicated endogenous name {behavioral.name!r}",
                    line=group.header.line,
                )
            behaviorals.append(behavioral)
            endogenous_order.append(behavioral.name)
        else:
            identity_name, alternative = _parse_identity(group)
            if identity_name in {item.name for item in behaviorals}:
                raise MdlSyntaxError(
                    f"duplicated endogenous name {identity_name!r}",
                    line=group.header.line,
                )
            if identity_name not in identity_alternatives:
                identity_order.append(identity_name)
                endogenous_order.append(identity_name)
                identity_alternatives[identity_name] = []
            existing = identity_alternatives[identity_name]
            if existing:
                previous = existing[0].equation
                current = alternative.equation
                if (
                    previous.lhs_function != current.lhs_function
                    or previous.lhs_periods != current.lhs_periods
                ):
                    raise MdlSyntaxError(
                        f"all definitions of identity {identity_name!r} must use "
                        "the same left-hand-side function",
                        line=current.line,
                    )
            existing.append(alternative)

    identities = tuple(
        IdentityEquation(name, tuple(identity_alternatives[name]))
        for name in identity_order
    )
    _validate_names_and_conditions(tuple(behaviorals), identities)
    endogenous = tuple(endogenous_order)
    exogenous = _exogenous(tuple(behaviorals), identities, endogenous)
    max_lag, max_lead = _model_lag_lead(tuple(behaviorals), identities)
    dependencies = _dependencies(tuple(behaviorals), identities, endogenous)
    return BimetsModel(
        name=name,
        raw_text=text,
        clean_lines=clean,
        behaviorals=tuple(behaviorals),
        identities=identities,
        endogenous=endogenous,
        exogenous=exogenous,
        max_lag=max_lag,
        max_lead=max_lead,
        dependencies=dependencies,
    )


def _clean_lines(text: str) -> tuple[tuple[str, ...], list[tuple[int, str]]]:
    """Remove comments and retain numbered, non-empty MDL lines."""
    clean: list[str] = []
    numbered: list[tuple[int, str]] = []
    for line_number, raw in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        value = raw.strip()
        if not value:
            continue
        if value.startswith("$") or re.match(r"^COMMENT\s*>", value, re.I):
            continue
        if "$" in value:
            raise MdlSyntaxError(
                "the dollar comment marker is allowed only at the start of a line",
                line=line_number,
            )
        clean.append(value)
        numbered.append((line_number, value))
    return tuple(clean[1:-1] if len(clean) >= 2 else clean), numbered


def _statements(lines: list[tuple[int, str]]) -> tuple[_Statement, ...]:
    """Convert cleaned lines into MDL statements."""
    statements: list[_Statement] = []
    keyword: str | None = None
    content: list[str] = []
    source_lines: list[str] = []
    start_line = 0

    def flush() -> None:
        """Emit the statement accumulated from the current source lines."""
        nonlocal keyword, content, source_lines
        if keyword is not None:
            separator = ";" if keyword == "RESTRICT" else " "
            statements.append(
                _Statement(
                    keyword,
                    separator.join(item for item in content if item).strip(),
                    start_line,
                    tuple(source_lines),
                )
            )
        keyword = None
        content = []
        source_lines = []

    for line_number, value in lines:
        directive = _DIRECTIVE.match(value)
        tsrange = _TSRANGE.match(value)
        if directive is not None or tsrange is not None:
            flush()
            if directive is not None:
                keyword = directive.group(1).upper()
                keyword = "BEHAVIORAL" if keyword == "EQUATION" else keyword
                current_content = directive.group(2).strip()
            else:
                keyword = "TSRANGE"
                assert tsrange is not None
                current_content = tsrange.group(1).strip()
            start_line = line_number
            content = [current_content]
            source_lines = [value]
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*>", value):
            raise MdlSyntaxError("unknown MDL directive", line=line_number)
        if keyword is None:
            raise MdlSyntaxError("content outside an equation group", line=line_number)
        content.append(value)
        source_lines.append(value)
    flush()
    return tuple(statements)


def _groups(statements: tuple[_Statement, ...]) -> tuple[_Group, ...]:
    """Group statements into model equation blocks."""
    groups: list[_Group] = []
    header: _Statement | None = None
    local: list[_Statement] = []
    for statement in statements:
        if statement.keyword in {"BEHAVIORAL", "IDENTITY"}:
            if header is not None:
                groups.append(_Group(header.keyword, header, tuple(local)))
            header = statement
            local = []
        elif header is None:
            raise MdlSyntaxError(
                f"{statement.keyword} appears outside an equation group",
                line=statement.line,
            )
        else:
            local.append(statement)
    if header is not None:
        groups.append(_Group(header.keyword, header, tuple(local)))
    return tuple(groups)


def _parse_behavioral(group: _Group) -> BehavioralEquation:
    """Parse behavioral for internal processing."""
    header_parts = re.split(r"\s+TSRANGE\s+", group.header.content, flags=re.I)
    name = header_parts[0].strip()
    _validate_name(name, "behavioral", group.header.line)
    inline_range = (
        _parse_range(header_parts[1], group.header.line)
        if len(header_parts) == 2
        else None
    )
    if len(header_parts) > 2:
        raise MdlSyntaxError("invalid inline TSRANGE", line=group.header.line)
    by_keyword = _by_keyword(group.statements)
    _reject_keywords(
        by_keyword,
        {"EQ", "COEFF", "TSRANGE", "ERROR", "RESTRICT", "PDL", "IV", "STORE"},
        "behavioral",
    )
    equation_statement = _exactly_one(by_keyword, "EQ", name)
    coefficient_statement = _exactly_one(by_keyword, "COEFF", name)
    ranges = by_keyword.get("TSRANGE", ())
    if inline_range is not None and ranges:
        raise MdlSyntaxError(
            f"multiple TSRANGE definitions in behavioral {name!r}",
            line=ranges[0].line,
        )
    if len(ranges) > 1:
        raise MdlSyntaxError(
            f"multiple TSRANGE definitions in behavioral {name!r}",
            line=ranges[1].line,
        )
    estimation_range = (
        inline_range
        if inline_range is not None
        else (_parse_range(ranges[0].content, ranges[0].line) if ranges else None)
    )
    equation = _parse_equation(equation_statement, name)
    coefficients = tuple(coefficient_statement.content.split())
    if not coefficients:
        raise MdlSyntaxError(
            f"empty COEFF definition in behavioral {name!r}",
            line=coefficient_statement.line,
        )
    if len(set(coefficients)) != len(coefficients):
        raise MdlSyntaxError(
            f"duplicated coefficient in behavioral {name!r}",
            line=coefficient_statement.line,
        )
    for coefficient in coefficients:
        _validate_name(coefficient, "coefficient", coefficient_statement.line)
        if coefficient.upper() in SUPPORTED_FUNCTIONS:
            raise MdlSyntaxError(
                f"reserved MDL function name {coefficient!r} cannot be a coefficient",
                line=coefficient_statement.line,
            )
    regressors = _extract_regressors(
        equation.rhs, coefficients, name, equation_statement.line
    )
    errors = by_keyword.get("ERROR", ())
    if len(errors) > 1:
        raise MdlSyntaxError(
            f"multiple ERROR definitions in behavioral {name!r}", line=errors[1].line
        )
    error = _parse_error(errors[0]) if errors else None
    pdls = tuple(
        _parse_pdl(item, coefficients, regressors) for item in by_keyword.get("PDL", ())
    )
    if len({item.coefficient for item in pdls}) != len(pdls):
        raise MdlSyntaxError(f"duplicated PDL definition in behavioral {name!r}")
    restrictions = tuple(
        restriction
        for statement in by_keyword.get("RESTRICT", ())
        for restriction in _parse_restrictions(statement, coefficients, pdls)
    )
    instruments = tuple(
        parse_expression(item.content, line=item.line)
        for item in by_keyword.get("IV", ())
    )
    stores = by_keyword.get("STORE", ())
    if len(stores) > 1:
        raise MdlSyntaxError(
            f"multiple STORE definitions in behavioral {name!r}",
            line=stores[1].line,
        )
    if stores:
        _validate_store(stores[0])
    return BehavioralEquation(
        name=name,
        equation=equation,
        coefficients=coefficients,
        regressors=regressors,
        estimation_range=estimation_range,
        error=error,
        restrictions=restrictions,
        pdls=pdls,
        instruments=instruments,
    )


def _parse_identity(group: _Group) -> tuple[str, IdentityAlternative]:
    """Parse identity for internal processing."""
    name = group.header.content.strip()
    _validate_name(name, "identity", group.header.line)
    by_keyword = _by_keyword(group.statements)
    _reject_keywords(by_keyword, {"EQ", "IF"}, "identity")
    equation = _parse_equation(_exactly_one(by_keyword, "EQ", name), name)
    conditions = by_keyword.get("IF", ())
    if len(conditions) > 1:
        raise MdlSyntaxError(
            f"multiple IF definitions in identity {name!r}", line=conditions[1].line
        )
    condition = (
        parse_expression(conditions[0].content, line=conditions[0].line)
        if conditions
        else None
    )
    if condition is not None and not _has_logical_operator(condition):
        raise MdlSyntaxError(
            f"IF condition in identity {name!r} has no logical operator",
            line=conditions[0].line,
        )
    return name, IdentityAlternative(
        equation,
        condition,
        conditions[0].content if conditions else None,
    )


def _parse_equation(statement: _Statement, dependent: str) -> MdlEquation:
    """Parse equation for internal processing."""
    assignments = list(_ASSIGNMENT.finditer(statement.content))
    if len(assignments) != 1:
        raise MdlSyntaxError(
            f"EQ for {dependent!r} must contain exactly one assignment",
            line=statement.line,
        )
    position = assignments[0].start()
    lhs_source = statement.content[:position].strip()
    rhs_source = statement.content[position + 1 :].strip()
    lhs = parse_expression(lhs_source, line=statement.line)
    rhs = parse_expression(rhs_source, line=statement.line)
    lhs_function, lhs_periods = _parse_lhs(lhs, dependent, statement.line)
    return MdlEquation(
        dependent=dependent,
        lhs_function=lhs_function,
        lhs_periods=lhs_periods,
        rhs=rhs,
        source=statement.content,
        line=statement.line,
    )


def _parse_lhs(expression: MdlExpression, dependent: str, line: int) -> tuple[str, int]:
    """Parse lhs for internal processing."""
    if isinstance(expression, Variable) and expression.name == dependent:
        return "IDENTITY", 0
    if not isinstance(expression, FunctionCall) or expression.name not in LHS_FUNCTIONS:
        raise MdlSyntaxError(
            f"left-hand side must be {dependent!r} or one supported transformation",
            line=line,
        )
    if (
        not isinstance(expression.arguments[0], Variable)
        or expression.arguments[0].name != dependent
    ):
        raise MdlSyntaxError(
            f"left-hand-side function must apply directly to {dependent!r}", line=line
        )
    periods = 0
    if expression.name in {"TSDELTA", "TSDELTALOG", "TSDELTAP"}:
        periods = (
            1
            if len(expression.arguments) == 1
            else int(numeric_value(expression.arguments[1]) or 0)
        )
    return expression.name, periods


def _extract_regressors(
    expression: MdlExpression,
    coefficients: tuple[str, ...],
    behavioral: str,
    line: int,
) -> tuple[MdlExpression, ...]:
    """Extract regressors for internal processing."""
    terms = _additive_terms(expression)
    found: list[str] = []
    regressors: list[MdlExpression] = []
    coefficient_set = set(coefficients)
    for sign, term in terms:
        coefficient: str | None = None
        regressor: MdlExpression
        if isinstance(term, Variable) and term.name in coefficient_set:
            coefficient = term.name
            regressor = Number(float(sign))
        elif (
            isinstance(term, UnaryExpression)
            and term.operator == "-"
            and isinstance(term.operand, Variable)
            and term.operand.name in coefficient_set
        ):
            coefficient = term.operand.name
            regressor = Number(float(-sign))
        else:
            product = _split_coefficient_product(term, coefficient_set)
            if product is None:
                names = variable_names(term) & coefficient_set
                if names:
                    raise MdlSyntaxError(
                        f"coefficient {sorted(names)[0]!r} in behavioral "
                        f"{behavioral!r} must be the left factor of one additive term",
                        line=line,
                    )
                raise MdlSyntaxError(
                    f"behavioral {behavioral!r} must be a linear combination "
                    "of coefficients",
                    line=line,
                )
            coefficient, regressor = product
            if sign == -1:
                regressor = UnaryExpression("-", regressor)
        if variable_names(regressor) & coefficient_set:
            raise MdlSyntaxError(
                f"nonlinear coefficient use in behavioral {behavioral!r}", line=line
            )
        found.append(coefficient)
        regressors.append(regressor)
    if tuple(found) != coefficients:
        raise MdlSyntaxError(
            f"coefficients in behavioral {behavioral!r} must appear once and in COEFF order",
            line=line,
        )
    numeric_regressors = sum(numeric_value(item) is not None for item in regressors)
    if numeric_regressors > 1:
        raise MdlSyntaxError(
            f"multiple intercepts in behavioral {behavioral!r}", line=line
        )
    return tuple(regressors)


def _split_coefficient_product(
    expression: MdlExpression, coefficients: set[str]
) -> tuple[str, MdlExpression] | None:
    """Split a leading coefficient from a multiplicative regressor chain."""
    if not isinstance(expression, BinaryExpression):
        return None
    if expression.operator == "*":
        if (
            isinstance(expression.left, Variable)
            and expression.left.name in coefficients
        ):
            return expression.left.name, expression.right
        if (
            isinstance(expression.left, UnaryExpression)
            and expression.left.operator == "-"
            and isinstance(expression.left.operand, Variable)
            and expression.left.operand.name in coefficients
        ):
            return expression.left.operand.name, UnaryExpression("-", expression.right)
    if expression.operator not in {"*", "/"}:
        return None
    nested = _split_coefficient_product(expression.left, coefficients)
    if nested is None:
        return None
    coefficient, regressor = nested
    return coefficient, BinaryExpression(
        regressor, expression.operator, expression.right
    )


def _additive_terms(expression: MdlExpression) -> list[tuple[int, MdlExpression]]:
    """Flatten an additive expression into signed terms."""
    if isinstance(expression, BinaryExpression) and expression.operator in {"+", "-"}:
        output = _additive_terms(expression.left)
        right = _additive_terms(expression.right)
        factor = 1 if expression.operator == "+" else -1
        return output + [(sign * factor, term) for sign, term in right]
    if isinstance(expression, UnaryExpression) and expression.operator == "-":
        return [(-sign, term) for sign, term in _additive_terms(expression.operand)]
    if isinstance(expression, UnaryExpression) and expression.operator == "+":
        return _additive_terms(expression.operand)
    return [(1, expression)]


def _parse_range(source: str, line: int) -> MdlTimeRange:
    """Parse range for internal processing."""
    parts = source.split()
    if len(parts) != 4:
        raise MdlSyntaxError("TSRANGE requires four positive integers", line=line)
    try:
        values = tuple(int(item) for item in parts)
    except ValueError as error:
        raise MdlSyntaxError(
            "TSRANGE requires four positive integers", line=line
        ) from error
    if any(value <= 0 for value in values):
        raise MdlSyntaxError("TSRANGE requires four positive integers", line=line)
    return MdlTimeRange(*values)


def _parse_error(statement: _Statement) -> AutoregressiveError:
    """Parse error for internal processing."""
    match = re.fullmatch(r"AUTO\(([1-9])\)", statement.content)
    if match is None:
        raise MdlSyntaxError(
            "ERROR must use AUTO(n), with n from 1 to 9", line=statement.line
        )
    return AutoregressiveError(int(match.group(1)))


def _validate_store(statement: _Statement) -> None:
    """Validate one legacy STORE declaration, whose value is then discarded."""
    source = re.sub(r"\s+", "", statement.content)
    match = re.fullmatch(r"([^()]*)\(([^()]*)\)", source)
    if match is None:
        raise MdlSyntaxError(
            "STORE usage is: STORE> variable(non-negative integer)",
            line=statement.line,
        )
    variable, position_source = match.groups()
    if not _NAME.fullmatch(variable):
        raise MdlSyntaxError(
            f"invalid STORE variable name {variable!r}", line=statement.line
        )
    if not re.fullmatch(r"[0-9]+", position_source):
        raise MdlSyntaxError(
            "STORE position must be a non-negative integer", line=statement.line
        )


def _parse_pdl(
    statement: _Statement,
    coefficients: tuple[str, ...],
    regressors: tuple[MdlExpression, ...],
) -> PdlDefinition:
    """Parse pdl for internal processing."""
    parts = statement.content.split()
    if len(parts) < 3 or len(parts) > 5:
        raise MdlSyntaxError(
            "PDL usage is: PDL> coefficient degree length [N] [F]",
            line=statement.line,
        )
    coefficient = parts[0]
    if coefficient not in coefficients:
        raise MdlSyntaxError(
            f"unknown coefficient {coefficient!r} in PDL", line=statement.line
        )
    try:
        degree, length = int(parts[1]), int(parts[2])
    except ValueError as error:
        raise MdlSyntaxError(
            "PDL degree and length must be integers", line=statement.line
        ) from error
    if degree < 0 or length <= 0 or length <= degree:
        raise MdlSyntaxError(
            "PDL degree must be non-negative and length must exceed degree",
            line=statement.line,
        )
    options = parts[3:]
    if any(item not in {"N", "F"} for item in options) or len(set(options)) != len(
        options
    ):
        raise MdlSyntaxError("PDL options must be N and/or F", line=statement.line)
    regressor = regressors[coefficients.index(coefficient)]
    if numeric_value(regressor) is not None:
        raise MdlSyntaxError(
            "PDL cannot be applied to an intercept", line=statement.line
        )
    if degree == 0 and length == 1:
        raise MdlSyntaxError(
            "a degree-0 length-1 PDL is redundant", line=statement.line
        )
    if length == 2 and set(options) == {"N", "F"}:
        raise MdlSyntaxError(
            "N and F are redundant for a length-2 PDL", line=statement.line
        )
    return PdlDefinition(
        coefficient,
        degree,
        length,
        zero_nearest="N" in options,
        zero_farthest="F" in options,
    )


def _parse_restrictions(
    statement: _Statement,
    coefficients: tuple[str, ...],
    pdls: tuple[PdlDefinition, ...],
) -> tuple[CoefficientRestriction, ...]:
    """Parse restrictions for internal processing."""
    definitions = {item.coefficient: item for item in pdls}
    sources = [item.strip() for item in statement.content.split(";") if item.strip()]
    output: list[CoefficientRestriction] = []
    for source in sources:
        assignments = list(_ASSIGNMENT.finditer(source))
        if len(assignments) != 1:
            raise MdlSyntaxError(
                "restriction must contain one assignment", line=statement.line
            )
        position = assignments[0].start()
        lhs = parse_expression(source[:position], line=statement.line)
        rhs = parse_expression(source[position + 1 :], line=statement.line)
        target = numeric_value(rhs)
        if target is None:
            raise MdlSyntaxError(
                "restriction target must be numeric", line=statement.line
            )
        terms: list[RestrictionTerm] = []
        for sign, term in _additive_terms(lhs):
            parsed = _restriction_term(term, sign, statement.line)
            if parsed.coefficient not in coefficients:
                raise MdlSyntaxError(
                    f"unknown coefficient {parsed.coefficient!r} in restriction",
                    line=statement.line,
                )
            if parsed.lag:
                pdl = definitions.get(parsed.coefficient)
                if pdl is None or parsed.lag >= pdl.length:
                    raise MdlSyntaxError(
                        f"LAG({parsed.coefficient},{parsed.lag}) has no matching PDL term",
                        line=statement.line,
                    )
            terms.append(parsed)
        if not terms or len({(item.coefficient, item.lag) for item in terms}) != len(
            terms
        ):
            raise MdlSyntaxError(
                "restriction must contain distinct coefficient terms",
                line=statement.line,
            )
        output.append(CoefficientRestriction(tuple(terms), target, source))
    return tuple(output)


def _restriction_term(
    expression: MdlExpression, sign: int, line: int
) -> RestrictionTerm:
    """Convert one restriction term into coefficients and a target."""
    multiplier = float(sign)
    target = expression
    if isinstance(expression, BinaryExpression) and expression.operator == "*":
        value = numeric_value(expression.left)
        if value is None:
            raise MdlSyntaxError(
                "restriction multiplier must precede its coefficient", line=line
            )
        multiplier *= value
        target = expression.right
    if isinstance(target, Variable):
        return RestrictionTerm(target.name, multiplier)
    if (
        isinstance(target, FunctionCall)
        and target.name == "TSLAG"
        and isinstance(target.arguments[0], Variable)
        and len(target.arguments) == 2
    ):
        lag_value = numeric_value(target.arguments[1])
        assert lag_value is not None
        return RestrictionTerm(target.arguments[0].name, multiplier, int(lag_value))
    raise MdlSyntaxError("restriction must be linear in coefficients", line=line)


def _by_keyword(
    statements: tuple[_Statement, ...],
) -> dict[str, tuple[_Statement, ...]]:
    """Group statements by their MDL keyword."""
    temporary: dict[str, list[_Statement]] = {}
    for statement in statements:
        temporary.setdefault(statement.keyword, []).append(statement)
    return {key: tuple(value) for key, value in temporary.items()}


def _reject_keywords(
    by_keyword: dict[str, tuple[_Statement, ...]],
    allowed: set[str],
    group_type: str,
) -> None:
    """Reject unsupported keywords found in an equation group."""
    unknown = set(by_keyword).difference(allowed)
    if unknown:
        selected = sorted(unknown)[0]
        raise MdlSyntaxError(
            f"{selected} is not allowed in an {group_type} group",
            line=by_keyword[selected][0].line,
        )


def _exactly_one(
    by_keyword: dict[str, tuple[_Statement, ...]], keyword: str, name: str
) -> _Statement:
    """Return the sole statement for a required keyword."""
    values = by_keyword.get(keyword, ())
    if len(values) != 1:
        line = values[1].line if len(values) > 1 else None
        raise MdlSyntaxError(
            f"equation {name!r} requires exactly one {keyword} definition", line=line
        )
    return values[0]


def _validate_name(name: str, kind: str, line: int) -> None:
    """Validate name for internal processing."""
    if not _NAME.fullmatch(name) or "__" in name:
        raise MdlSyntaxError(f"invalid {kind} name {name!r}", line=line)
    if name.upper() in SUPPORTED_FUNCTIONS:
        raise MdlSyntaxError(f"reserved MDL function name {name!r}", line=line)


def _validate_names_and_conditions(
    behaviorals: tuple[BehavioralEquation, ...],
    identities: tuple[IdentityEquation, ...],
) -> None:
    """Validate names and conditions for internal processing."""
    expressions: list[tuple[MdlExpression, int]] = []
    for behavioral in behaviorals:
        expressions.append((behavioral.equation.rhs, behavioral.equation.line))
        expressions.extend(
            (item, behavioral.equation.line) for item in behavioral.instruments
        )
    for identity in identities:
        for alternative in identity.alternatives:
            expressions.append((alternative.equation.rhs, alternative.equation.line))
            if alternative.condition is not None:
                if not variable_names(alternative.condition):
                    raise MdlSyntaxError(
                        f"IF condition in identity {identity.name!r} must reference a variable",
                        line=alternative.equation.line,
                    )
                expressions.append((alternative.condition, alternative.equation.line))
    for expression, line in expressions:
        for name in variable_names(expression):
            _validate_name(name, "variable", line)


def _has_logical_operator(expression: MdlExpression) -> bool:
    """Return whether has logical operator."""
    if isinstance(expression, BinaryExpression):
        if expression.operator in {"&", "|", "==", "!=", "<", "<=", ">", ">="}:
            return True
        return _has_logical_operator(expression.left) or _has_logical_operator(
            expression.right
        )
    if isinstance(expression, UnaryExpression):
        return _has_logical_operator(expression.operand)
    if isinstance(expression, FunctionCall):
        return any(_has_logical_operator(item) for item in expression.arguments)
    return False


def _exogenous(
    behaviorals: tuple[BehavioralEquation, ...],
    identities: tuple[IdentityEquation, ...],
    endogenous: tuple[str, ...],
) -> tuple[str, ...]:
    """Collect variables that are exogenous to the parsed model."""
    names: set[str] = set()
    coefficients = {
        coefficient
        for behavioral in behaviorals
        for coefficient in behavioral.coefficients
    }
    for behavioral in behaviorals:
        names.update(variable_names(behavioral.equation.rhs))
        for instrument in behavioral.instruments:
            names.update(variable_names(instrument))
    for identity in identities:
        for alternative in identity.alternatives:
            names.update(variable_names(alternative.equation.rhs))
            if alternative.condition is not None:
                names.update(variable_names(alternative.condition))
    names.difference_update(endogenous)
    names.difference_update(coefficients)
    return tuple(sorted(names))


def _model_lag_lead(
    behaviorals: tuple[BehavioralEquation, ...],
    identities: tuple[IdentityEquation, ...],
) -> tuple[int, int]:
    """Determine the largest lag and lead used by a model."""
    minimum = 0
    maximum = 0
    for behavioral in behaviorals:
        offsets = list(temporal_offsets(behavioral.equation.rhs))
        offsets.append(-behavioral.equation.lhs_periods)
        for instrument in behavioral.instruments:
            offsets.extend(temporal_offsets(instrument))
        pdl_lengths = {item.coefficient: item.length for item in behavioral.pdls}
        for coefficient, regressor in zip(
            behavioral.coefficients, behavioral.regressors, strict=True
        ):
            base_offsets = temporal_offsets(regressor)
            length = pdl_lengths.get(coefficient, 1)
            offsets.extend(
                offset - lag for offset in base_offsets for lag in range(length)
            )
        local_min = min(offsets, default=0)
        local_max = max(offsets, default=0)
        if behavioral.error is not None:
            local_min -= behavioral.error.order
        minimum = min(minimum, local_min)
        maximum = max(maximum, local_max)
    for identity in identities:
        for alternative in identity.alternatives:
            offsets = list(temporal_offsets(alternative.equation.rhs))
            offsets.append(-alternative.equation.lhs_periods)
            if alternative.condition is not None:
                offsets.extend(temporal_offsets(alternative.condition))
            minimum = min(minimum, min(offsets, default=0))
            maximum = max(maximum, max(offsets, default=0))
    return -minimum, maximum


def _dependencies(
    behaviorals: tuple[BehavioralEquation, ...],
    identities: tuple[IdentityEquation, ...],
    endogenous: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    """Collect contemporaneous dependencies from an expression."""
    endogenous_set = set(endogenous)
    output: dict[str, set[str]] = {name: set() for name in endogenous}
    for behavioral in behaviorals:
        offsets = variable_offsets(behavioral.equation.rhs)
        output[behavioral.name].update(
            name
            for name, values in offsets.items()
            if name in endogenous_set and 0 in values
        )
    for identity in identities:
        for alternative in identity.alternatives:
            expressions = [alternative.equation.rhs]
            if alternative.condition is not None:
                expressions.append(alternative.condition)
            for expression in expressions:
                offsets = variable_offsets(expression)
                output[identity.name].update(
                    name
                    for name, values in offsets.items()
                    if name in endogenous_set and 0 in values
                )
    return {name: frozenset(values) for name, values in output.items()}
