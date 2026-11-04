import numpy as np
import pandas as pd
import pytest

from fuel_efficiency.clean import clean, decode_origin, extract_make, mpg_to_l_per_100km
from fuel_efficiency.data import read_raw

RAW_SAMPLE = """\
18.0   8   307.0      130.0      3504.      12.0   70  1\t"chevrolet chevelle malibu"
25.0   4   98.00      ?          2046.      19.0   71  1\t"ford pinto"
27.0   4   97.00      88.00      2130.      14.5   70  3\t"datsun pl510"
26.0   4   97.00      46.00      1835.      20.5   70  2\t"vw 1131 deluxe sedan"
"""


@pytest.fixture
def raw(tmp_path):
    path = tmp_path / "auto-mpg.data"
    path.write_text(RAW_SAMPLE)
    return read_raw(path)


def test_read_raw_turns_question_mark_into_missing(raw):
    assert raw.shape == (4, 9)
    assert raw["horsepower"].isna().sum() == 1
    assert raw.loc[0, "name"] == "chevrolet chevelle malibu"


@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("vw rabbit", "volkswagen"),
        ("vokswagen rabbit", "volkswagen"),
        ("chevy c20", "chevrolet"),
        ("chevroelt chevelle malibu", "chevrolet"),
        ("toyouta corona", "toyota"),
        ("maxda rx3", "mazda"),
        ("mercedes benz 300d", "mercedes-benz"),
        ("  Ford Pinto ", "ford"),
    ],
)
def test_extract_make_fixes_misspellings(name, make):
    assert extract_make(pd.Series([name])).iloc[0] == make


def test_decode_origin_labels_and_orders_categories():
    decoded = decode_origin(pd.Series([3, 1, 2]))
    assert list(decoded) == ["Japan", "USA", "Europe"]
    assert list(decoded.categories) == ["USA", "Europe", "Japan"]


def test_decode_origin_rejects_unknown_codes():
    with pytest.raises(ValueError, match="Unknown origin"):
        decode_origin(pd.Series([1, 4]))


def test_mpg_to_l_per_100km_known_value():
    # 23.5 US MPG is almost exactly 10 L/100 km
    assert mpg_to_l_per_100km(pd.Series([23.5214583])).iloc[0] == pytest.approx(10.0)


def test_clean_adds_derived_columns_without_touching_input(raw):
    before = raw.copy()
    df = clean(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert {"make", "year", "weight_kg", "l_per_100km"} <= set(df.columns)
    assert "model_year" not in df.columns
    assert df["year"].tolist() == [1970, 1971, 1970, 1970]
    assert df.loc[0, "weight_kg"] == pytest.approx(3504 * 0.45359237)
    assert np.isnan(df.loc[1, "horsepower"]), "imputation belongs in the model pipeline"
