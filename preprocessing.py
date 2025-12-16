from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import pandas as pd

from schema import ALIASES, FEATURE_COLUMNS_24, load_feature_columns


def build_features(
    df: pd.DataFrame, feature_columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Project the incoming dataframe down to the 24 Azure ML features.

    Steps:
        1. Apply historical column aliases (others -> other, etc.).
        2. Validate that every required feature exists.
        3. Convert everything to numeric using pd.to_numeric(errors="coerce").
        4. Fill missing values with 0 to mirror training-time preprocessing.
        5. Reindex columns to FEATURE_COLUMNS_24 order.
        6. Guard against schema regressions via X.shape[1] == 24.
    """

    normalized = _apply_aliases(df.copy())

    features = list(feature_columns or load_feature_columns())

    missing = [col for col in features if col not in normalized.columns]
    if missing:
        raise ValueError(f"必須列が不足しています: {', '.join(sorted(missing))}")

    aligned = normalized.reindex(columns=features)
    numeric = aligned.apply(pd.to_numeric, errors="coerce").fillna(0)

    if numeric.shape[1] != len(features):
        raise AssertionError(
            f"Expected 24 features but found {numeric.shape[1]} columns: {numeric.columns.tolist()}"
        )

    return numeric


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for alias, canonical in ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical
        elif alias in df.columns and canonical in df.columns:
            logging.warning(
                "Both alias and canonical column detected (%s -> %s). Keeping canonical.",
                alias,
                canonical,
            )

    if rename_map:
        df = df.rename(columns=rename_map)
    return df
