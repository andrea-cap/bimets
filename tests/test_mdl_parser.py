from __future__ import annotations

from pathlib import Path

import pytest

from bimets import (
    AutoregressiveError,
    BehavioralEquation,
    BimetsModel,
    BinaryExpression,
    FunctionCall,
    MdlSyntaxError,
    MdlTimeRange,
    Number,
    UnaryExpression,
    Variable,
    load_model,
    parse_expression,
    parse_mdl,
)

KLEIN_MODEL = """MODEL

COMMENT> Klein Model 1 of the U.S. Economy

COMMENT> Consumption
BEHAVIORAL> cn
TSRANGE 1925 1 1941 1
EQ> cn = a1 + a2*p + a3*TSLAG(p,1) + a4*(w1+w2)
COEFF> a1 a2 a3 a4
ERROR> AUTO(2)

COMMENT> Investment
BEHAVIORAL> i
TSRANGE 1923 1 1941 1
EQ> i = b1 + b2*p + b3*TSLAG(p,1) + b4*TSLAG(k,1)
COEFF> b1 b2 b3 b4
RESTRICT> b2 + b3 = 1

COMMENT> Demand for Labor
BEHAVIORAL> w1
TSRANGE 1925 1 1941 1
EQ> w1 = c1 + c2*(y+t-w2) + c3*TSLAG(y+t-w2,1) + c4*time
COEFF> c1 c2 c3 c4
PDL> c3 1 3

COMMENT> Gross National Product
IDENTITY> y
EQ> y = cn + i + g - t

COMMENT> Profits
IDENTITY> p
EQ> p = y - (w1+w2)

COMMENT> Capital Stock with switches
IDENTITY> k
EQ> k = TSLAG(k,1) + i
IF> i > 0
IDENTITY> k
EQ> k = TSLAG(k,1)
IF> i <= 0

END"""


@pytest.mark.source("bimets-R")
def test_klein_model_matches_documented_structure() -> None:
    model = parse_mdl(KLEIN_MODEL, name="klein")

    assert isinstance(model, BimetsModel)
    assert model.name == "klein"
    assert len(model.behaviorals) == 3
    assert len(model.identities) == 3
    assert model.coefficient_count == 12
    assert model.endogenous == ("cn", "i", "w1", "y", "p", "k")
    assert model.exogenous == ("g", "t", "time", "w2")
    assert model.max_lag == 3
    assert model.max_lead == 0
    assert model.forward_looking is False
    assert "behaviorals=3" in repr(model)
    assert model.dependencies == {
        "cn": frozenset({"p", "w1"}),
        "i": frozenset({"p"}),
        "w1": frozenset({"y"}),
        "y": frozenset({"cn", "i"}),
        "p": frozenset({"y", "w1"}),
        "k": frozenset({"i"}),
    }
    with pytest.raises(TypeError):
        model.dependencies["cn"] = frozenset()  # type: ignore[index]


@pytest.mark.source("bimets-R")
def test_behavioral_details_include_errors_restrictions_and_pdl() -> None:
    model = parse_mdl(KLEIN_MODEL)
    cn = model.behavioral("cn")
    investment = model.behavioral("i")
    labor = model.behavioral("w1")

    assert isinstance(cn, BehavioralEquation)
    assert cn.estimation_range == MdlTimeRange(1925, 1, 1941, 1)
    assert cn.error is not None and cn.error.order == 2
    assert cn.coefficients == ("a1", "a2", "a3", "a4")
    assert cn.regressors[0] == Number(1)
    assert investment.restrictions[0].target == 1
    assert [term.coefficient for term in investment.restrictions[0].terms] == [
        "b2",
        "b3",
    ]
    assert labor.pdls[0].coefficient == "c3"
    assert labor.pdls[0].degree == 1
    assert labor.pdls[0].length == 3
    assert labor.expanded_coefficients == (
        "c1",
        "c2",
        "c3",
        "c3__PDL__1",
        "c3__PDL__2",
        "c4",
    )
    with pytest.raises(KeyError):
        model.behavioral("missing")


def test_conditional_identity_alternatives_are_merged() -> None:
    identity = parse_mdl(KLEIN_MODEL).identity("k")

    assert identity.conditional
    assert len(identity.alternatives) == 2
    assert identity.alternatives[0].condition_source == "i > 0"
    assert identity.alternatives[1].condition_source == "i <= 0"
    assert identity.alternatives[0].equation.lhs_function == "IDENTITY"
    with pytest.raises(KeyError):
        parse_mdl(KLEIN_MODEL).identity("missing")


@pytest.mark.source("native")
def test_equation_lookup_rejects_the_wrong_definition_kind() -> None:
    model = parse_mdl(KLEIN_MODEL)

    with pytest.raises(KeyError):
        model.behavioral("k")
    with pytest.raises(KeyError):
        model.identity("cn")


def test_inline_range_lhs_functions_instruments_and_multiline_restrictions() -> None:
    model = parse_mdl(
        """MODEL
BEHAVIORAL> growth TSRANGE 2000 1 2010 4
EQ> TSDELTALOG(growth,2) = a + b*LOG(x) + c*TSLAG(z)
COEFF> a b c
PDL> b 1 3 N F
RESTRICT> b + LAG(b,2) = 1
c = -0.5
IV> 1
IV> TSLAG(instrument)*pi + 0.5
IDENTITY> level
EQ> EXP(level) = growth +
x
END"""
    )
    behavioral = model.behavioral("growth")

    assert behavioral.equation.lhs_function == "TSDELTALOG"
    assert behavioral.equation.lhs_periods == 2
    assert behavioral.estimation_range == MdlTimeRange(2000, 1, 2010, 4)
    assert behavioral.pdls[0].zero_nearest
    assert behavioral.pdls[0].zero_farthest
    assert len(behavioral.restrictions) == 2
    assert behavioral.restrictions[0].terms[1].lag == 2
    assert behavioral.restrictions[1].target == -0.5
    assert len(behavioral.instruments) == 2
    assert model.identity("level").alternatives[0].equation.lhs_function == "EXP"
    assert model.max_lag == 2


def test_forward_looking_model_and_dot_comparison_aliases() -> None:
    model = parse_mdl(
        """MODEL
IDENTITY> investment
EQ> investment=(MOVAVG(investment,2)+TSLEAD(investment,4))/2
IF> policy.GE.0 & policy.LT.10
END"""
    )

    assert model.forward_looking
    assert model.max_lag == 1
    assert model.max_lead == 4
    assert model.exogenous == ("policy",)
    assert model.dependencies["investment"] == frozenset({"investment"})


def test_load_model_from_text_file_and_class_constructors(tmp_path: Path) -> None:
    path = tmp_path / "klein.txt"
    path.write_text(KLEIN_MODEL, encoding="utf-8")

    from_file = load_model(model_file=path)
    preferred_text = load_model(
        model_file=tmp_path / "missing.txt", model_text=KLEIN_MODEL, name="preferred"
    )

    assert from_file.name == str(path)
    assert preferred_text.name == "preferred"
    assert BimetsModel.from_file(path).endogenous == from_file.endogenous
    assert BimetsModel.from_text(KLEIN_MODEL, name="text").name == "text"
    with pytest.raises(ValueError, match="must be provided"):
        load_model()
    with pytest.raises(TypeError, match="model_text"):
        load_model(model_text=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text"):
        parse_mdl(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "empty model definition"),
        ("IDENTITY> y\nEQ> y=x", "begin with MODEL"),
        ("MODEL\nEND", "no equations"),
        ("MODEL\n$ comment\nEND", "no equations"),
        ("MODEL\nFOO> x\nEND", "unknown MDL directive"),
        ("MODEL\nEQ> y=x\nEND", "outside an equation group"),
        ("MODEL\nIDENTITY> y\nSTORE> x(1)\nEQ> y=x\nEND", "not allowed"),
        ("MODEL\nIDENTITY> y\nEQ> y=x $ comment\nEND", "dollar comment marker"),
        ("MODEL\nIDENTITY> bad.name\nEQ> bad.name=x\nEND", "invalid identity name"),
        ("MODEL\nIDENTITY> LOG\nEQ> LOG=x\nEND", "reserved MDL function"),
        ("MODEL\nIDENTITY> y\nEND", "exactly one EQ"),
        ("MODEL\nIDENTITY> y\nEQ> y=x\nEQ> y=z\nEND", "exactly one EQ"),
        ("MODEL\nIDENTITY> y\nCOEFF> a\nEQ> y=x\nEND", "not allowed"),
        ("MODEL\nIDENTITY> y\nEQ> z=x\nEND", "left-hand side"),
        ("MODEL\nIDENTITY> y\nEQ> LOG(x)=x\nEND", "apply directly"),
        ("MODEL\nIDENTITY> y\nEQ> ABS(y)=x\nEND", "left-hand side"),
        ("MODEL\nIDENTITY> y\nEQ> y=x\nIF> x\nEND", "no logical operator"),
        ("MODEL\nIDENTITY> y\nEQ> y=x\nIF> 1>0\nEND", "must reference a variable"),
        ("MODEL\nIDENTITY> y\nEQ> y=x\nIF> x>0\nIF> x<1\nEND", "multiple IF"),
    ],
)
def test_document_validation(body: str, message: str) -> None:
    with pytest.raises(MdlSyntaxError, match=message):
        parse_mdl(body)


@pytest.mark.parametrize(
    ("declarations", "message"),
    [
        ("EQ> y=a*x", "exactly one COEFF"),
        ("COEFF> a", "exactly one EQ"),
        ("EQ> y=a*x\nCOEFF> a\nCOEFF> a", "exactly one COEFF"),
        ("EQ> y=a*x\nCOEFF> a a", "duplicated coefficient"),
        ("EQ> y=a*x\nCOEFF> bad.name", "invalid coefficient"),
        ("EQ> y=LOG*x\nCOEFF> LOG", "reserved MDL function"),
        ("EQ> y=x*a\nCOEFF> a", "left factor"),
        ("EQ> y=a*x+a*z\nCOEFF> a", "appear once"),
        ("EQ> y=a*x+b\nCOEFF> b a", "COEFF order"),
        ("EQ> y=a*x+x\nCOEFF> a", "linear combination"),
        ("EQ> y=a+b\nCOEFF> a b", "multiple intercepts"),
        ("EQ> y=a*(b*x)\nCOEFF> a b", "nonlinear coefficient"),
        ("TSRANGE 2000 1 2001\nEQ> y=a\nCOEFF> a", "four positive"),
        ("TSRANGE 2000 0 2001 1\nEQ> y=a\nCOEFF> a", "four positive"),
        ("TSRANGE 2000 x 2001 1\nEQ> y=a\nCOEFF> a", "four positive"),
        ("ERROR> AUTO(0)\nEQ> y=a\nCOEFF> a", "AUTO"),
        ("ERROR> AUTO(1)\nERROR> AUTO(2)\nEQ> y=a\nCOEFF> a", "multiple ERROR"),
        ("STORE> archive(1)\nSTORE> archive(2)\nEQ> y=a\nCOEFF> a", "multiple STORE"),
        ("STORE> archive\nEQ> y=a\nCOEFF> a", "STORE usage"),
        ("STORE> bad.name(1)\nEQ> y=a\nCOEFF> a", "invalid STORE variable"),
        ("STORE> archive(-1)\nEQ> y=a\nCOEFF> a", "non-negative integer"),
        ("STORE> archive(1.5)\nEQ> y=a\nCOEFF> a", "non-negative integer"),
    ],
)
def test_behavioral_validation(declarations: str, message: str) -> None:
    source = f"MODEL\nBEHAVIORAL> y\n{declarations}\nEND"
    with pytest.raises(MdlSyntaxError, match=message):
        parse_mdl(source)


@pytest.mark.source("bimets-R")
@pytest.mark.parametrize("behavioral_keyword", ["BEHAVIORAL", "EQUATION"])
def test_space_before_angle_bracket_is_accepted_for_all_r_directives(
    behavioral_keyword: str,
) -> None:
    source = f"""MODEL
COMMENT > synthetic directive-spacing model
{behavioral_keyword} > demand TSRANGE 2000 1 2005 1
EQ > demand=level+slope*driver
COEFF > level slope
STORE > archive_slot ( 2 )
PDL > slope 1 3
RESTRICT > slope=1
ERROR > AUTO(1)
IV > instrument
IDENTITY > balance
EQ > balance=demand+offset
IF > switch>0
END"""

    model = parse_mdl(source)

    behavioral = model.behavioral("demand")
    assert behavioral.pdls[0].coefficient == "slope"
    assert behavioral.restrictions[0].target == 1
    assert behavioral.error == AutoregressiveError(1)
    assert behavioral.instruments == (Variable("instrument"),)
    assert model.identity("balance").conditional


@pytest.mark.source("bimets-R")
def test_store_zero_position_is_accepted_and_discarded() -> None:
    model = parse_mdl(
        "MODEL\nBEHAVIORAL> output\nEQ> output=level\nCOEFF> level\n"
        "STORE > legacy_buffer(0)\nEND"
    )

    assert model.behavioral("output").name == "output"


@pytest.mark.parametrize(
    ("pdl", "message"),
    [
        ("PDL> b 1", "PDL usage"),
        ("PDL> missing 1 3", "unknown coefficient"),
        ("PDL> b x 3", "must be integers"),
        ("PDL> b -1 3", "degree must be"),
        ("PDL> b 3 3", "length must exceed"),
        ("PDL> b 1 3 X", "options must be"),
        ("PDL> b 1 3 N N", "options must be"),
        ("PDL> a 0 2", "intercept"),
        ("PDL> b 0 1", "redundant"),
        ("PDL> b 0 2 N F", "redundant"),
        ("PDL> b 1 3\nPDL> b 1 4", "duplicated PDL"),
    ],
)
def test_pdl_validation(pdl: str, message: str) -> None:
    source = f"""MODEL
BEHAVIORAL> y
EQ> y=a+b*x
COEFF> a b
{pdl}
END"""
    with pytest.raises(MdlSyntaxError, match=message):
        parse_mdl(source)


@pytest.mark.parametrize(
    ("restriction", "message"),
    [
        ("a", "one assignment"),
        ("a=x", "target must be numeric"),
        ("missing=0", "unknown coefficient"),
        ("a*a=0", "multiplier must precede"),
        ("a+LAG(a,1)=0", "no matching PDL"),
        ("LAG(b,3)=0", "no matching PDL"),
        ("a+a=0", "distinct coefficient"),
        ("LOG(a)=0", "linear in coefficients"),
    ],
)
def test_restriction_validation(restriction: str, message: str) -> None:
    source = f"""MODEL
BEHAVIORAL> y
EQ> y=a+b*x
COEFF> a b
PDL> b 1 3
RESTRICT> {restriction}
END"""
    with pytest.raises(MdlSyntaxError, match=message):
        parse_mdl(source)


def test_duplicate_names_and_inconsistent_identity_lhs_are_rejected() -> None:
    duplicate = """MODEL
IDENTITY> y
EQ> y=x
BEHAVIORAL> y
EQ> y=a*x
COEFF> a
END"""
    inconsistent = """MODEL
IDENTITY> y
EQ> y=x
IF> x>0
IDENTITY> y
EQ> LOG(y)=x
IF> x<=0
END"""

    with pytest.raises(MdlSyntaxError, match="duplicated endogenous"):
        parse_mdl(duplicate)
    with pytest.raises(MdlSyntaxError, match="same left-hand-side"):
        parse_mdl(inconsistent)


def test_unary_negative_regressor_is_preserved() -> None:
    model = parse_mdl("MODEL\nBEHAVIORAL> y\nEQ> y=-a*x*weight\nCOEFF> a\nEND")

    assert model.behavioral("y").regressors == (
        BinaryExpression(UnaryExpression("-", Variable("x")), "*", Variable("weight")),
    )


@pytest.mark.source("bimets-R")
@pytest.mark.parametrize(
    ("term", "regressor"),
    [
        ("slope*feature*weight", "feature*weight"),
        ("slope*feature/scale", "feature/scale"),
        ("slope*(feature*weight)", "feature*weight"),
        ("slope*TSLAG(feature,2)*(1-switch)", "TSLAG(feature,2)*(1-switch)"),
    ],
)
def test_leading_coefficient_supports_multiplicative_regressor_chains(
    term: str, regressor: str
) -> None:
    model = parse_mdl(
        f"MODEL\nBEHAVIORAL> result\nEQ> result={term}\nCOEFF> slope\nEND"
    )

    assert model.behavioral("result").regressors == (parse_expression(regressor),)


@pytest.mark.source("bimets-R")
@pytest.mark.parametrize(
    ("term", "message"),
    [
        ("feature*slope*weight", "left factor"),
        ("slope/scale", "left factor"),
        ("slope<feature", "left factor"),
        ("slope*feature*slope", "nonlinear coefficient"),
        ("slope*(other*feature)", "nonlinear coefficient"),
    ],
)
def test_multiplicative_regressor_chains_preserve_coefficient_rules(
    term: str, message: str
) -> None:
    coefficients = "slope other" if "other" in term else "slope"
    source = f"MODEL\nBEHAVIORAL> result\nEQ> result={term}\nCOEFF> {coefficients}\nEND"

    with pytest.raises(MdlSyntaxError, match=message):
        parse_mdl(source)


@pytest.mark.source("bimets-R")
def test_pdl_keeps_the_complete_multiplicative_regressor() -> None:
    model = parse_mdl(
        "MODEL\nBEHAVIORAL> result\n"
        "EQ> result=level+slope*feature*weight\n"
        "COEFF> level slope\nPDL> slope 1 3\nEND"
    )
    behavioral = model.behavioral("result")

    assert behavioral.regressors[1] == parse_expression("feature*weight")
    assert behavioral.pdls[0].coefficient == "slope"
    assert behavioral.expanded_coefficients == (
        "level",
        "slope",
        "slope__PDL__1",
        "slope__PDL__2",
    )


def test_lowercase_public_function_names_are_normalized() -> None:
    model = parse_mdl("MODEL\nIDENTITY> y\nEQ> log(y)=exp(x)\nEND")
    equation = model.identity("y").alternatives[0].equation

    assert equation.lhs_function == "LOG"
    assert equation.rhs == FunctionCall("EXP", (Variable("x"),))
