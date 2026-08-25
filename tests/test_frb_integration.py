"""Cross-component tests based on the BIMETS R FRB/US vignette.

Source: https://cran.r-project.org/web/packages/bimets/vignettes/frb2bimets.pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bimets import BimetsDataset, load_model

pytestmark = [pytest.mark.integration, pytest.mark.source("bimets-R")]

FIXTURES = Path(__file__).parent / "fixtures"


def test_frb_policy_excerpt_runs_from_dataframe_through_scenario_simulation() -> None:
    model = load_model(
        model_file=FIXTURES / "frb_policy_excerpt.mdl",
        name="FRB policy excerpt",
    )
    frame = pd.DataFrame(
        {
            "dmptmax": [0.0, 0.2, 0.4, 0.6, 0.8],
            "delrff": [0.0, 0.1, 0.1, 0.1, 0.1],
            "dmptlur": [0.1, 0.2, 0.3, 0.4, 0.5],
            "dmptpi": [0.4, 0.3, 0.2, 0.5, 0.1],
            "rff": [1.0, 1.1, 1.2, 1.3, 1.4],
        },
        index=pd.period_range("2019Q4", periods=5, freq="Q"),
    )
    data = BimetsDataset.from_frame(frame, metadata={"model": "FRB/US"})
    residual_check = model.bind(data).simulate(
        coefficients={},
        time_range=(2020, 1, 2020, 4),
        simulation_type="RESCHECK",
    )

    scenario = data.assign_range(
        {
            "dmptlur": [0.8, 0.1, 0.7, 0.2],
            "dmptpi": [0.2, 0.9, 0.3, 0.6],
            "rff": [1.5, 1.75, 1.25, 2.0],
        },
        start=(2020, 1),
        end=(2020, 4),
    )
    result = model.bind(scenario).simulate(
        coefficients={},
        time_range=(2020, 1, 2020, 4),
    )

    assert residual_check.constant_adjustments is not None
    assert set(residual_check.constant_adjustments) == {"dmptmax", "delrff"}
    np.testing.assert_allclose(result["dmptmax"].values, [0.8, 0.9, 0.7, 0.6])
    np.testing.assert_allclose(result["delrff"].values, [0.5, 0.25, -0.5, 0.75])
    result_frame = result.summary()
    assert list(result_frame.columns) == ["dmptmax", "delrff"]
    np.testing.assert_allclose(
        result_frame["delrff"].to_numpy(), result["delrff"].values
    )
    np.testing.assert_allclose(data["rff"].values, [1.0, 1.1, 1.2, 1.3, 1.4])
    assert scenario.metadata == {"model": "FRB/US"}


def test_frb_mce_lead_equation_uses_dataframe_terminal_condition() -> None:
    model = load_model(
        model_file=FIXTURES / "frb_mce_excerpt.mdl",
        name="FRB MCE excerpt",
    )
    frame = pd.DataFrame(
        {
            "zdivgr": [0.01, 0.01, 0.01, 0.02],
            "hgynid": [0.02, 0.03, 0.04, 0.05],
        },
        index=pd.period_range("2040Q1", periods=4, freq="Q"),
    )
    data = BimetsDataset.from_frame(frame)

    result = model.bind(data).simulate(
        coefficients={},
        time_range=(2040, 1, 2040, 3),
        algorithm="NEWTON",
        convergence=1e-10,
    )

    weight = 0.009757264257434617
    persistence = 0.9902427357425654
    expected_q3 = weight * 0.05 + persistence * 0.02
    expected_q2 = weight * 0.04 + persistence * expected_q3
    expected_q1 = weight * 0.03 + persistence * expected_q2
    np.testing.assert_allclose(
        result["zdivgr"].values,
        [expected_q1, expected_q2, expected_q3],
        rtol=1e-9,
        atol=1e-12,
    )
    assert model.forward_looking
    assert model.max_lead == 1
