import numpy as np
import pandas as pd
import pytest

from fuel_efficiency.inference import coefficient_table, fit_ols, vif


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": a,
            "b": b,
            "a_copy": a + rng.normal(scale=0.05, size=n),
            "y": 2.0 + 3.0 * a - 1.0 * b + rng.normal(scale=0.5, size=n),
        }
    )


def test_vif_is_near_one_for_independent_columns(synthetic):
    assert vif(synthetic, ["a", "b"]).max() < 1.1


def test_vif_flags_near_duplicate_columns(synthetic):
    values = vif(synthetic, ["a", "b", "a_copy"])
    assert values["a"] > 100 and values["a_copy"] > 100
    assert values["b"] < 1.1


def test_fit_ols_recovers_known_coefficients(synthetic):
    table = coefficient_table(fit_ols(synthetic, "y ~ a + b"))
    assert table.loc["a", "ci_low"] < 3.0 < table.loc["a", "ci_high"]
    assert table.loc["b", "ci_low"] < -1.0 < table.loc["b", "ci_high"]


def test_fit_ols_switches_to_robust_errors_when_variance_grows():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 10, size=500)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(scale=x, size=500)})
    result = fit_ols(df, "y ~ x")
    assert result.breusch_pagan_p < 0.05
    assert result.model.cov_type == "HC3"
