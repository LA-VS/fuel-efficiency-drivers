"""Fetch and load the UCI Auto MPG dataset.

Source: Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5859H (licence: CC BY 4.0).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "auto-mpg.data"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"

COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "name",
]


def fetch(force: bool = False) -> Path:
    """Download the raw data file into data/raw if it is not already there."""
    if RAW_FILE.exists() and not force:
        return RAW_FILE
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        RAW_FILE.write_bytes(archive.read("auto-mpg.data"))
    return RAW_FILE


def read_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Parse the raw whitespace separated file. Missing horsepower is stored as '?'."""
    path = Path(path) if path is not None else fetch()
    return pd.read_csv(
        path,
        sep=r"\s+",
        names=COLUMNS,
        na_values="?",
        quotechar='"',
    )
