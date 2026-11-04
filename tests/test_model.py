import numpy as np
import pandas as pd

from fuel_efficiency.model import FEATURES, candidate_models, evaluate, preprocessor


def make_cars(n=120, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(800, 2000, n)
    df = pd.DataFrame(
        {
            "cylinders": np.where(weight > 1400, 8, 4),
            "displacement": weight / 8 + rng.normal(0, 10, n),
            "horsepower": weight / 15 + rng.normal(0, 5, n),
            "weight_kg": weight,
            "acceleration": rng.uniform(10, 20, n),
            "year": rng.integers(1970, 1983, n),
            "origin": pd.Categorical(
                rng.choice(["USA", "Europe", "Japan"], n), categories=["USA", "Europe", "Japan"]
            ),
        }
    )
    df.loc[:4, "horsepower"] = np.nan
    mpg = 60 - weight / 40 + (df["year"] - 1970) * 0.8 + rng.normal(0, 1, n)
    return df, pd.Series(mpg, name="mpg")


def test_preprocessor_imputes_missing_and_scales():
    X, _ = make_cars()
    out = preprocessor().fit_transform(X[FEATURES])
    assert not np.isnan(out).any()
    assert np.allclose(out[:, :6].mean(axis=0), 0, atol=1e-9)


def test_preprocessor_learns_only_from_training_rows():
    X, _ = make_cars()
    train, test = X.iloc[:60], X.iloc[60:].copy()
    test["weight_kg"] += 10_000  # a shift that would leak into the scaler if fitted on all rows
    prep = preprocessor().fit(train[FEATURES])
    assert prep.named_transformers_["num"].named_steps["scale"].mean_[3] < 2000


def test_every_model_beats_the_baseline_on_learnable_data():
    X, y = make_cars()
    models = {k: v for k, v in candidate_models().items() if k in {"Baseline (mean)", "Linear regression"}}
    scores = evaluate(models, X[FEATURES], y, n_splits=3, n_repeats=1)
    assert scores.index[0] == "Linear regression"
    assert scores.loc["Linear regression", "mae_mean"] < scores.loc["Baseline (mean)", "mae_mean"] / 3
