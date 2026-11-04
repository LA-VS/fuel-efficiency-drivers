"""Part A: explain what drives fuel use with ordinary least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Variance inflation factor for each column, with an intercept included."""
    x = df[columns].dropna().astype(float)
    x.insert(0, "const", 1.0)
    values = [variance_inflation_factor(x.values, i) for i in range(1, x.shape[1])]
    return pd.Series(values, index=columns, name="VIF").sort_values(ascending=False)


@dataclass
class OlsResult:
    model: object  # statsmodels results with robust (HC3) standard errors
    breusch_pagan_p: float
    cooks_distance: pd.Series
    n_influential: int


def fit_ols(df: pd.DataFrame, formula: str) -> OlsResult:
    """Fit OLS, test for heteroscedasticity, and refit with HC3 errors if needed."""
    data = df.dropna(subset=_formula_columns(formula, df))
    plain = smf.ols(formula, data=data).fit()
    _, bp_p, _, _ = het_breuschpagan(plain.resid, plain.model.exog)
    cooks = pd.Series(plain.get_influence().cooks_distance[0], index=data.index)
    threshold = 4 / len(data)
    model = plain.get_robustcov_results(cov_type="HC3") if bp_p < 0.05 else plain
    return OlsResult(
        model=model,
        breusch_pagan_p=float(bp_p),
        cooks_distance=cooks,
        n_influential=int((cooks > threshold).sum()),
    )


def coefficient_table(result: OlsResult) -> pd.DataFrame:
    """Coefficients with 95% confidence intervals as a tidy frame."""
    model = result.model
    names = model.model.exog_names
    ci = np.asarray(model.conf_int())
    return pd.DataFrame(
        {
            "coef": np.asarray(model.params),
            "ci_low": ci[:, 0],
            "ci_high": ci[:, 1],
            "p_value": np.asarray(model.pvalues),
        },
        index=names,
    )


def _formula_columns(formula: str, df: pd.DataFrame) -> list[str]:
    tokens = set(formula.replace("~", " ").replace("+", " ").replace("(", " ").replace(")", " ").split())
    return [c for c in df.columns if c in tokens]
