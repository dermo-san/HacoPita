import pandas as pd
import pytest

from preprocessing import build_features
from schema import ALIASES, FEATURE_COLUMNS_24


def _base_dataframe(use_alias: bool = False) -> pd.DataFrame:
    alias_lookup = {canonical: alias for alias, canonical in ALIASES.items()}
    data = {}
    for idx, column in enumerate(FEATURE_COLUMNS_24):
        column_name = alias_lookup[column] if use_alias and column in alias_lookup else column
        data[column_name] = [idx + 1]
    data["box_id"] = [900]
    data["unrelated"] = [123]
    return pd.DataFrame(data)


def test_build_features_returns_only_feature_columns():
    df = _base_dataframe()
    features = build_features(df)

    assert list(features.columns) == FEATURE_COLUMNS_24
    assert features.shape[1] == len(FEATURE_COLUMNS_24)


def test_build_features_supports_aliases():
    df = _base_dataframe(use_alias=True)
    features = build_features(df)

    assert "other" in features.columns
    assert features.loc[0, "other"] == 3  # value assigned via alias


def test_build_features_requires_all_columns():
    df = _base_dataframe().drop(columns=[FEATURE_COLUMNS_24[0]])
    with pytest.raises(ValueError, match=FEATURE_COLUMNS_24[0]):
        build_features(df)


def test_build_features_allows_missing_box_id_column():
    df = _base_dataframe().drop(columns=["box_id"])
    features = build_features(df)

    assert features.shape[1] == len(FEATURE_COLUMNS_24)
    assert features.shape[0] == 1
