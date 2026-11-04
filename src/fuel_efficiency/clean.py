"""Cleaning and feature derivation for the Auto MPG data.

Nothing here is fitted to the data, so it is safe to apply before the train and
test split. Anything that learns from the data (imputation, scaling) lives in a
sklearn Pipeline in model.py instead.
"""

from __future__ import annotations

import pandas as pd

ORIGIN_LABELS = {1: "USA", 2: "Europe", 3: "Japan"}

# Misspellings and aliases found in the raw "name" column.
MAKE_FIXES = {
    "vw": "volkswagen",
    "vokswagen": "volkswagen",
    "chevy": "chevrolet",
    "chevroelt": "chevrolet",
    "maxda": "mazda",
    "toyouta": "toyota",
    "mercedes": "mercedes-benz",
    "hi": "international harvester",
    "capri": "mercury",
}

LB_TO_KG = 0.45359237
MPG_TO_L_PER_100KM = 235.214583  # US gallons


def extract_make(name: pd.Series) -> pd.Series:
    """Take the first word of the car name and fix known misspellings."""
    make = name.str.strip().str.lower().str.split().str[0]
    return make.replace(MAKE_FIXES)


def decode_origin(origin: pd.Series) -> pd.Series:
    """Turn the 1, 2, 3 origin codes into labelled categories (USA is the reference)."""
    unknown = set(origin.dropna().unique()) - set(ORIGIN_LABELS)
    if unknown:
        raise ValueError(f"Unknown origin codes: {sorted(unknown)}")
    return pd.Categorical(origin.map(ORIGIN_LABELS), categories=list(ORIGIN_LABELS.values()))


def mpg_to_l_per_100km(mpg: pd.Series) -> pd.Series:
    """Convert US miles per gallon to litres per 100 km."""
    return MPG_TO_L_PER_100KM / mpg


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy copy: decoded origin, fixed make, metric units and a full year."""
    df = raw.copy()
    df["origin"] = decode_origin(df["origin"])
    df["make"] = extract_make(df["name"])
    df["year"] = df["model_year"] + 1900
    df["weight_kg"] = df["weight"] * LB_TO_KG
    df["l_per_100km"] = mpg_to_l_per_100km(df["mpg"])
    return df.drop(columns=["model_year"])
