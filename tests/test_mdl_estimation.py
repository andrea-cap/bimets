from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from _paper_models import PAPER_ADVANCED_KLEIN, PAPER_DOI, PAPER_TRANSFORMED_KLEIN
from bimets import (
    BimetsDataset,
    BimetsModel,
    BimetsSeries,
    BoundModel,
    Frequency,
    ModelEstimationResult,
    YearPeriod,
    bind_model_data,
    cumsum,
    estimate,
    timeseries,
)
from bimets.mdl._estimation import _constrained_least_squares

KLEIN_CONSUMPTION = """MODEL
BEHAVIORAL> cn
TSRANGE 1921 1 1941 1
EQ> cn = a1 + a2*p + a3*TSLAG(p,1) + a4*(w1+w2)
COEFF> a1 a2 a3 a4
END"""

ADVANCED_KLEIN = """MODEL
BEHAVIORAL> cn
TSRANGE 1925 1 1941 1
EQ> cn = a1 + a2*p + a3*TSLAG(p,1) + a4*(w1+w2)
COEFF> a1 a2 a3 a4
ERROR> AUTO(2)
BEHAVIORAL> i
TSRANGE 1923 1 1941 1
EQ> i = b1 + b2*p + b3*TSLAG(p,1) + b4*TSLAG(k,1)
COEFF> b1 b2 b3 b4
RESTRICT> b2 + b3 = 1
BEHAVIORAL> w1
TSRANGE 1925 1 1941 1
EQ> w1 = c1 + c2*(y+t-w2) + c3*TSLAG(y+t-w2,1) + c4*time
COEFF> c1 c2 c3 c4
PDL> c3 1 3
END"""


def test_constrained_least_squares_validates_degenerate_systems() -> None:
    x = np.eye(2)
    y = np.asarray([1.0, 2.0])

    beta, covariance = _constrained_least_squares(x, y, np.eye(2), y, "fixed", 1e-12)
    np.testing.assert_allclose(beta, y)
    np.testing.assert_array_equal(covariance, np.zeros((2, 2)))

    with pytest.raises(ValueError, match="empty"):
        _constrained_least_squares(x, y, np.empty((0, 2)), np.empty(0), "empty", 1e-12)
    with pytest.raises(ValueError, match=r"restriction matrix.*singular"):
        _constrained_least_squares(
            x,
            y,
            np.asarray([[1.0, 0.0], [2.0, 0.0]]),
            np.asarray([1.0, 2.0]),
            "dependent",
            1e-12,
        )
    with pytest.raises(ValueError, match=r"regressor matrix.*singular"):
        _constrained_least_squares(
            np.zeros((2, 2)),
            y,
            np.asarray([[1.0, 0.0]]),
            np.asarray([1.0]),
            "unidentified",
            1e-12,
        )


def klein_data() -> BimetsDataset:
    values = {
        "cn": [
            39.8,
            41.9,
            45,
            49.2,
            50.6,
            52.6,
            55.1,
            56.2,
            57.3,
            57.8,
            55,
            50.9,
            45.6,
            46.5,
            48.7,
            51.3,
            57.7,
            58.7,
            57.5,
            61.6,
            65,
            69.7,
        ],
        "p": [
            12.7,
            12.4,
            16.9,
            18.4,
            19.4,
            20.1,
            19.6,
            19.8,
            21.1,
            21.7,
            15.6,
            11.4,
            7,
            11.2,
            12.3,
            14,
            17.6,
            17.3,
            15.3,
            19,
            21.1,
            23.5,
        ],
        "w1": [
            28.8,
            25.5,
            29.3,
            34.1,
            33.9,
            35.4,
            37.4,
            37.9,
            39.2,
            41.3,
            37.9,
            34.5,
            29,
            28.5,
            30.6,
            33.2,
            36.8,
            41,
            38.2,
            41.6,
            45,
            53.3,
        ],
        "w2": [
            2.2,
            2.7,
            2.9,
            2.9,
            3.1,
            3.2,
            3.3,
            3.6,
            3.7,
            4,
            4.2,
            4.8,
            5.3,
            5.6,
            6,
            6.1,
            7.4,
            6.7,
            7.7,
            7.8,
            8,
            8.5,
        ],
        "i": [
            2.7,
            -0.2,
            1.9,
            5.2,
            3,
            5.1,
            5.6,
            4.2,
            3,
            5.1,
            1,
            -3.4,
            -6.2,
            -5.1,
            -3,
            -1.3,
            2.1,
            2,
            -1.9,
            1.3,
            3.3,
            4.9,
        ],
        "k": [
            182.8,
            182.6,
            184.5,
            189.7,
            192.7,
            197.8,
            203.4,
            207.6,
            210.6,
            215.7,
            216.7,
            213.3,
            207.1,
            202,
            199,
            197.7,
            199.8,
            201.8,
            199.9,
            201.2,
            204.5,
            209.4,
        ],
        "y": [
            43.7,
            40.6,
            49.1,
            55.4,
            56.4,
            58.7,
            60.3,
            61.3,
            64,
            67,
            57.7,
            50.7,
            41.3,
            45.3,
            48.9,
            53.3,
            61.8,
            65,
            61.2,
            68.4,
            74.1,
            85.3,
        ],
        "t": [
            3.4,
            7.7,
            3.9,
            4.7,
            3.8,
            5.5,
            7,
            6.7,
            4.2,
            4,
            7.7,
            7.5,
            8.3,
            5.4,
            6.8,
            7.2,
            8.3,
            6.7,
            7.4,
            8.9,
            9.6,
            11.6,
        ],
        "time": [
            np.nan,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
        "g": [
            4.6,
            6.6,
            6.1,
            5.7,
            6.6,
            6.5,
            6.6,
            7.6,
            7.9,
            8.1,
            9.4,
            10.7,
            10.2,
            9.3,
            10,
            10.5,
            10.3,
            11,
            13,
            14.4,
            15.4,
            22.3,
        ],
    }
    return BimetsDataset(
        {name: timeseries(items, start=(1920, 1)) for name, items in values.items()}
    )


def linear_model(extra: str = "") -> BimetsModel:
    return BimetsModel.from_text(
        f"MODEL\nBEHAVIORAL> y\nEQ> y=a+b*x\nCOEFF> a b\n{extra}END",
        name="linear",
    )


def linear_data() -> Mapping[str, BimetsSeries]:
    return {
        "y": timeseries([1, 3, 5, 7]),
        "x": timeseries([0, 1, 2, 3]),
    }


def test_binding_validates_variables_and_frequency() -> None:
    model = linear_model()
    bound = bind_model_data(model, linear_data())

    assert isinstance(bound, BoundModel)
    assert bound.model is model
    assert bound.freq is Frequency.YEARLY
    with pytest.raises(KeyError, match="x"):
        model.bind({"y": timeseries([1, 2])})
    with pytest.raises(ValueError, match="same frequency"):
        model.bind(
            {
                "y": timeseries([1, 2], freq="Q"),
                "x": timeseries([1, 2], freq="M"),
            }
        )
    with pytest.raises(TypeError, match="BimetsModel"):
        bind_model_data(object(), {})  # type: ignore[arg-type]


def test_simple_ols_is_available_as_function_model_and_bound_method() -> None:
    model = linear_model()
    data = linear_data()

    direct = estimate(model, data)
    via_model = model.estimate(data)
    via_bound = model.bind(data).estimate(equations="y")

    for result in (direct, via_model, via_bound):
        assert isinstance(result, ModelEstimationResult)
        assert result.method == "OLS"
        assert dict(result["y"].coefficients) == pytest.approx(
            {"a": 1.0, "b": 2.0}, rel=0, abs=1e-12
        )
        np.testing.assert_allclose(result["y"].fitted_values.values, [1, 3, 5, 7])
        assert result["y"].observations == 4
        assert result["y"].degrees_of_freedom == 2
        assert "equations=('y',)" in repr(result)


@pytest.mark.source("bimets-R")
def test_legacy_store_does_not_change_estimation() -> None:
    baseline = estimate(linear_model(), linear_data())["y"]
    stored_model = linear_model("STORE > estimation_archive(3)\n")
    stored = estimate(stored_model, linear_data())["y"]

    assert stored.coefficients == pytest.approx(baseline.coefficients)
    np.testing.assert_allclose(
        stored.fitted_values.values, baseline.fitted_values.values
    )


@pytest.mark.source("bimets-R")
def test_klein_consumption_matches_original_bimets_numerical_example() -> None:
    model = BimetsModel.from_text(KLEIN_CONSUMPTION, name="klein-consumption")
    result = estimate(model, klein_data())["cn"]

    np.testing.assert_allclose(
        list(result.coefficients.values()),
        [16.2366003, 0.1929344, 0.0898849, 0.7962187],
        rtol=0,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        result.residuals.values,
        [
            -0.323893544,
            -1.250007790,
            -1.565741401,
            -0.493503129,
            0.007607907,
            0.869096295,
            1.338476868,
            1.054978943,
            -0.588557053,
            0.282311734,
            -0.229653489,
            -0.322131892,
            0.322281007,
            -0.058010257,
            -0.034662717,
            1.616497310,
            -0.435973632,
            0.210054350,
            0.989201310,
            0.785077489,
            -2.173448309,
        ],
        rtol=0,
        atol=5e-9,
    )
    assert (result.sample_start.year, result.sample_start.period) == (1921, 1)
    assert (result.sample_end.year, result.sample_end.period) == (1941, 1)
    assert result.observations == 21
    assert result.degrees_of_freedom == 17
    assert result.residual_sum_squares == pytest.approx(17.87945, abs=5e-6)
    assert result.standard_error == pytest.approx(1.02554, abs=5e-6)
    assert result.r_squared == pytest.approx(0.9810082, abs=5e-8)
    assert result.adjusted_r_squared == pytest.approx(0.9776567, abs=5e-8)
    assert result.log_likelihood == pytest.approx(-28.10857, abs=5e-6)
    assert result.covariance.flags.writeable is False
    with pytest.raises(TypeError):
        result.coefficients["a1"] = 0  # type: ignore[index]


@pytest.mark.source("bimets-R")
def test_advanced_klein_ols_matches_original_bimets_results() -> None:
    results = estimate(BimetsModel.from_text(ADVANCED_KLEIN), klein_data())

    consumption = results["cn"]
    np.testing.assert_allclose(
        list(consumption.coefficients.values()),
        [19.01352476, 0.34428157, 0.03443117, 0.69939052],
        atol=5e-8,
    )
    np.testing.assert_allclose(
        list(consumption.autoregressive_coefficients.values()),
        [0.057431312, 0.007785936],
        atol=5e-8,
    )
    np.testing.assert_allclose(
        list(consumption.autoregressive_standard_errors.values()),
        [0.3324101, 0.2647013],
        atol=5e-7,
    )
    assert consumption.autoregressive_iterations == 9
    assert consumption.residuals_without_error_correction is not None
    assert consumption.residual_sum_squares == pytest.approx(9.273455, abs=5e-6)
    assert consumption.standard_error == pytest.approx(0.9181728, abs=5e-7)
    assert consumption.durbin_watson == pytest.approx(1.966609, abs=5e-6)
    assert consumption.f_statistic == pytest.approx(147.0844, abs=5e-4)
    assert consumption.f_probability == pytest.approx(1.090551e-9, rel=5e-6)
    assert consumption.aic == pytest.approx(51.94093, abs=5e-5)
    assert consumption.bic == pytest.approx(57.77343, abs=5e-5)

    investment = results["i"]
    np.testing.assert_allclose(
        list(investment.coefficients.values()),
        [2.868104, 0.5787626, 0.4212374, -0.09160307],
        atol=5e-7,
    )
    assert investment.coefficients["b2"] + investment.coefficients["b3"] == (
        pytest.approx(1.0)
    )
    assert investment.restriction_f_statistic == pytest.approx(8.194478, abs=5e-6)
    assert investment.restriction_f_probability == pytest.approx(0.0118602, abs=5e-7)
    assert investment.degrees_of_freedom == 16
    assert investment.residual_sum_squares == pytest.approx(26.76483, abs=5e-5)

    labor = results["w1"]
    assert tuple(labor.coefficients) == (
        "c1",
        "c2",
        "c3",
        "c3__PDL__1",
        "c3__PDL__2",
        "c4",
    )
    np.testing.assert_allclose(
        list(labor.coefficients.values()),
        [1.12869024, 0.43987666, 0.10768118, 0.05074557, -0.00619005, 0.13682057],
        atol=5e-8,
    )
    assert labor.coefficients["c3"] - 2 * labor.coefficients["c3__PDL__1"] + (
        labor.coefficients["c3__PDL__2"]
    ) == pytest.approx(0.0, abs=1e-12)
    assert labor.restriction_f_statistic == pytest.approx(0.06920179, abs=5e-8)
    assert labor.restriction_f_probability == pytest.approx(0.7973647, abs=5e-7)
    assert labor.residual_sum_squares == pytest.approx(6.392707, abs=5e-6)


@pytest.mark.source("bimets-R")
def test_instrumental_variables_with_autoregressive_errors_matches_bimets() -> None:
    model = BimetsModel.from_text(ADVANCED_KLEIN)
    result = estimate(
        model,
        klein_data(),
        equations="cn",
        method="IV",
        instruments=("1", "TSLAG(y)", "TSLAG(w1)*pi+0.5", "EXP(w2)"),
    )["cn"]

    assert result.method == "IV"
    np.testing.assert_allclose(
        list(result.coefficients.values()),
        [18.07073, 0.2530483, 0.08631646, 0.7363227],
        atol=5e-6,
    )
    np.testing.assert_allclose(
        list(result.autoregressive_coefficients.values()),
        [0.01559806, -0.1196327],
        atol=5e-7,
    )
    assert result.autoregressive_iterations == 7
    assert result.residual_sum_squares == pytest.approx(9.867739, abs=5e-6)
    assert result.r_squared == pytest.approx(0.9843186, abs=5e-7)


def test_declared_instruments_are_used_only_for_iv_estimation() -> None:
    model = linear_model("IV> 1\nIV> x\n")

    assert estimate(model, linear_data()).method == "OLS"
    iv_result = model.bind(linear_data()).estimate(method="iv")
    assert iv_result.method == "IV"
    assert dict(iv_result["y"].coefficients) == pytest.approx({"a": 1, "b": 2})


@pytest.mark.source("bimets-R")
def test_estimation_uses_complete_multiplicative_regressor_chain() -> None:
    model = BimetsModel.from_text(
        "MODEL\nBEHAVIORAL> output\n"
        "EQ> output=level+sensitivity*driver*weight\n"
        "COEFF> level sensitivity\nEND"
    )
    driver = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    weight = np.array([2.0, 1.0, 3.0, 2.0, 4.0, 3.0])
    data = {
        "driver": timeseries(driver),
        "weight": timeseries(weight),
        "output": timeseries(3.0 + 2.0 * driver * weight),
    }

    result = estimate(model, data)["output"]

    assert dict(result.coefficients) == pytest.approx(
        {"level": 3.0, "sensitivity": 2.0}
    )


def test_pdl_endpoints_and_explicit_lag_restriction_are_combined() -> None:
    model = BimetsModel.from_text(
        """MODEL
BEHAVIORAL> y
TSRANGE 2004 1 2011 1
EQ> y=a+b*x
COEFF> a b
PDL> b 2 4 N F
RESTRICT> LAG(b,2)=0
END"""
    )
    data = {
        "x": timeseries([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5], start=(2000, 1)),
        "y": timeseries([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8], start=(2000, 1)),
    }

    result = estimate(model, data)["y"]

    assert tuple(result.coefficients) == (
        "a",
        "b",
        "b__PDL__1",
        "b__PDL__2",
        "b__PDL__3",
    )
    np.testing.assert_allclose(list(result.coefficients.values())[1:], 0, atol=1e-12)
    assert result.degrees_of_freedom == 7
    assert result.restriction_f_statistic is not None


@pytest.mark.source("bimets-R")
def test_chow_analysis_matches_original_klein_example() -> None:
    result = estimate(
        BimetsModel.from_text(ADVANCED_KLEIN),
        klein_data(),
        equations="w1",
        time_range=(1925, 1, 1935, 1),
        force_time_range=True,
        chow_test=True,
    )["w1"]

    np.testing.assert_allclose(
        list(result.coefficients.values()),
        [-4.48873, 0.545102, 0.0413985, 0.0493551, 0.0573116, 0.292018],
        atol=5e-6,
    )
    chow = result.chow_test
    assert chow is not None
    assert chow.f_statistic == pytest.approx(15.3457, abs=5e-4)
    assert chow.f_probability == pytest.approx(5.34447e-5, rel=5e-5)
    assert chow.numerator_degrees_of_freedom == 6
    assert chow.denominator_degrees_of_freedom == 12
    assert (chow.extended_end.year, chow.extended_end.period) == (1941, 1)
    np.testing.assert_allclose(chow.actual.values, [36.8, 41, 38.2, 41.6, 45, 53.3])
    np.testing.assert_allclose(
        chow.predicted.values,
        [38.439, 40.824, 39.6553, 45.0547, 49.0118, 56.6727],
        atol=5e-4,
    )
    np.testing.assert_allclose(
        chow.standard_errors.values,
        [0.547471, 0.630905, 0.672192, 0.834433, 0.966472, 1.23486],
        atol=5e-6,
    )
    assert list(chow.summary().columns) == [
        "actual",
        "predicted",
        "error",
        "standard_error",
        "t_statistic",
    ]


def test_call_level_time_range_precedence_and_result_summaries() -> None:
    model = BimetsModel.from_text(
        """MODEL
BEHAVIORAL> y
TSRANGE 2001 1 2003 1
EQ> y=a+b*x
COEFF> a b
END"""
    )
    data = {
        "y": timeseries([1, 2, 4, 8, 16, 32], start=(2000, 1)),
        "x": timeseries([0, 1, 2, 3, 4, 5], start=(2000, 1)),
    }

    local = estimate(model, data, time_range=(2000, 1, 2005, 1))
    forced = model.estimate(
        data,
        time_range=(2000, 1, 2005, 1),
        force_time_range=True,
    )

    assert local["y"].observations == 3
    assert forced["y"].observations == 6
    assert list(forced["y"].summary().columns) == [
        "coefficient",
        "standard_error",
        "t_statistic",
        "p_value",
    ]
    assert forced.summary().index.names == ["equation", "coefficient_name"]


def test_estimation_rejects_missing_nonfinite_underdetermined_and_singular_data() -> (
    None
):
    with pytest.raises(ValueError, match="not finite"):
        estimate(
            linear_model(),
            {"y": timeseries([1, 2, 3]), "x": timeseries([0, np.nan, 2])},
        )
    with pytest.raises(ValueError, match="positive residual"):
        estimate(linear_model(), {"y": timeseries([1, 2]), "x": timeseries([0, 1])})
    with pytest.raises(ValueError, match="singular"):
        estimate(
            linear_model(),
            {"y": timeseries([1, 2, 3]), "x": timeseries([1, 1, 1])},
        )
    with pytest.raises(TypeError, match="data are required"):
        estimate(linear_model())
    with pytest.raises(TypeError, match="must be omitted"):
        estimate(linear_model().bind(linear_data()), linear_data())
    with pytest.raises(ValueError, match="at least one"):
        estimate(linear_model(), linear_data(), equations=[])
    with pytest.raises(ValueError, match="duplicates"):
        estimate(linear_model(), linear_data(), equations=["y", "y"])
    with pytest.raises(ValueError, match=r"OLS.*IV"):
        estimate(linear_model(), linear_data(), method="GLS")
    with pytest.raises(ValueError, match="requires instruments"):
        estimate(linear_model(), linear_data(), method="IV")
    with pytest.raises(ValueError, match="must not be empty"):
        estimate(linear_model(), linear_data(), method="IV", instruments=[])
    with pytest.raises(TypeError, match="center_covariance"):
        estimate(linear_model(), linear_data(), center_covariance=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="instrument or regressor"):
        estimate(
            linear_model(),
            linear_data(),
            method="IV",
            instruments=("1", "1"),
        )
    with pytest.raises(ValueError, match="requires time_range"):
        estimate(linear_model(), linear_data(), force_time_range=True)
    with pytest.raises(ValueError, match="positive finite"):
        estimate(linear_model(), linear_data(), tol=0)
    with pytest.raises(ValueError, match="singular"):
        estimate(linear_model(), linear_data(), tol=0.5)
    with pytest.raises(ValueError, match="must follow"):
        estimate(
            linear_model(),
            linear_data(),
            chow_test=True,
            chow_end=(2002, 1),
        )


@pytest.mark.source(PAPER_DOI)
def test_klein_covariance_matrix_from_paper() -> None:
    """Validate the complete covariance matrix in paper section 3.2."""
    result = BimetsModel.from_text(KLEIN_CONSUMPTION).estimate(klein_data())["cn"]

    np.testing.assert_allclose(
        result.covariance,
        [
            [1.6970227814, 0.0005013886, -0.0177068887, -0.0329172192],
            [0.0005013886, 0.0083192948, -0.0052704304, -0.0013188865],
            [-0.0177068887, -0.0052704304, 0.0082170486, -0.0006710788],
            [-0.0329172192, -0.0013188865, -0.0006710788, 0.0015955167],
        ],
        rtol=0,
        atol=5e-11,
    )


@pytest.mark.source(PAPER_DOI)
def test_advanced_klein_estimation_from_paper() -> None:
    """Reproduce the AR, restriction, and PDL results in paper section 3.3."""
    results = BimetsModel.from_text(PAPER_ADVANCED_KLEIN).estimate(klein_data())

    consumption = results["cn"]
    np.testing.assert_allclose(
        list(consumption.coefficients.values()),
        [14.82685, 0.2589094, 0.01423821, 0.8390274],
        atol=5e-7,
    )
    np.testing.assert_allclose(
        list(consumption.autoregressive_coefficients.values()),
        [0.2542111, -0.05250591],
        atol=5e-8,
    )
    assert consumption.autoregressive_iterations == 6
    assert consumption.residual_sum_squares == pytest.approx(8.071633, abs=5e-6)

    investment = results["i"]
    np.testing.assert_allclose(
        list(investment.coefficients.values()),
        [0.5348561, 0.6267204, 0.3732796, -0.0796483],
        atol=5e-7,
    )
    assert investment.coefficients["b2"] + investment.coefficients["b3"] == (
        pytest.approx(1.0, abs=1e-12)
    )
    assert investment.restriction_f_statistic == pytest.approx(5.542962, abs=5e-6)

    labor = results["w1"]
    np.testing.assert_allclose(
        list(labor.coefficients.values()),
        [2.916775, 0.4229623, 0.1292072, 0.01035948, 0.1020647],
        atol=5e-7,
    )
    assert labor.residual_sum_squares == pytest.approx(6.59422, abs=5e-6)


@pytest.mark.source(PAPER_DOI)
def test_chow_consumption_example_from_paper() -> None:
    """Reproduce the structural-stability example in paper section 3.4."""
    result = BimetsModel.from_text(KLEIN_CONSUMPTION).estimate(
        klein_data(),
        equations="cn",
        time_range=(1921, 1, 1935, 1),
        force_time_range=True,
        chow_test=True,
    )["cn"]
    chow = result.chow_test
    assert chow is not None

    np.testing.assert_allclose(
        list(result.coefficients.values()),
        [13.12755, 0.1669801, 0.08856838, 0.887964],
        atol=5e-6,
    )
    assert chow.f_statistic == pytest.approx(4.488731, abs=5e-6)
    assert chow.f_probability == pytest.approx(0.006687229, abs=5e-9)
    np.testing.assert_allclose(
        chow.predicted.values,
        [56.55436, 59.93099, 57.97212, 61.52069, 65.39572, 73.79655],
        atol=5e-6,
    )
    np.testing.assert_allclose(
        chow.standard_errors.values,
        [1.01181, 1.020099, 0.9686377, 1.200479, 1.242267, 1.669299],
        atol=5e-7,
    )


@pytest.mark.source(PAPER_DOI)
def test_transformed_lhs_model_from_paper_estimates_and_simulates() -> None:
    """Exercise the combined transformed-LHS model in paper section 3.1."""
    adjusted = {}
    for name, series in klein_data().items():
        if name == "i":
            adjusted[name] = timeseries(
                np.exp(series.values),
                start=series.start,
                freq=series.freq,
            )
        elif name == "cn":
            adjusted[name] = timeseries(
                np.log(series.values),
                start=series.start,
                freq=series.freq,
            )
        elif name == "y":
            adjusted[name] = cumsum(series)
        else:
            adjusted[name] = series

    model = BimetsModel.from_text(PAPER_TRANSFORMED_KLEIN)
    data = BimetsDataset(adjusted)
    coefficients = model.estimate(data)
    result = model.simulate(
        data,
        coefficients=coefficients,
        time_range=(1925, 1, 1930, 1),
        convergence=1e-5,
    )

    assert set(coefficients) == {"cn", "i", "w1"}
    for series in result.values():
        assert np.all(np.isfinite(series.values))
    capital = result["k"]
    assert capital.start == YearPeriod(1925, 1)
    assert capital.values[2] == pytest.approx(capital.values[1])
    assert capital.values[3] == pytest.approx(capital.values[2])
