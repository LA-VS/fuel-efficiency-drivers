"""Part B: predict MPG and measure accuracy honestly.

Every learned step (imputation, scaling, encoding, tuning) sits inside a Pipeline
so it is fitted on training folds only. The original coursework scaled the full
dataset before splitting, which leaked test information into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = ["cylinders", "displacement", "horsepower", "weight_kg", "acceleration", "year"]
CATEGORICAL = ["origin"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "mpg"


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(drop="first", handle_unknown="ignore")
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def candidate_models() -> dict[str, object]:
    """Each model is a full pipeline. Tuned models wrap an inner grid search."""
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED)

    def pipe(estimator):
        return Pipeline([("prep", preprocessor()), ("model", estimator)])

    return {
        "Baseline (mean)": pipe(DummyRegressor(strategy="mean")),
        "Linear regression": pipe(LinearRegression()),
        "Ridge": GridSearchCV(
            pipe(Ridge()),
            {"model__alpha": np.logspace(-3, 3, 13)},
            cv=inner,
            scoring="neg_mean_absolute_error",
        ),
        "Lasso": GridSearchCV(
            pipe(Lasso(max_iter=50_000)),
            {"model__alpha": np.logspace(-3, 1, 9)},
            cv=inner,
            scoring="neg_mean_absolute_error",
        ),
        "Gradient boosting": GridSearchCV(
            pipe(HistGradientBoostingRegressor(random_state=SEED)),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3, None],
                "model__min_samples_leaf": [10, 20],
            },
            cv=inner,
            scoring="neg_mean_absolute_error",
        ),
    }


def evaluate(
    models: dict[str, object], X: pd.DataFrame, y: pd.Series, n_splits: int = 5, n_repeats: int = 3
) -> pd.DataFrame:
    """Nested repeated k fold: tuning happens inside each outer training fold."""
    outer = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=outer,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            n_jobs=-1,
        )
        mae, rmse = -scores["test_mae"], -scores["test_rmse"]
        rows.append(
            {
                "model": name,
                "mae_mean": mae.mean(),
                "mae_sd": mae.std(ddof=1),
                "rmse_mean": rmse.mean(),
                "rmse_sd": rmse.std(ddof=1),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("mae_mean")
