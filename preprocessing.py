"""
前処理関数
入力CSVから必須特徴量を抽出・正規化する
"""
import logging
from typing import List, Set

import numpy as np
import pandas as pd

from schema import (
    DROP_COLUMNS,
    FEATURE_COLUMNS,
    FLOAT_COLUMNS,
    INTEGER_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)

LOGGER = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    入力DataFrameから必須特徴量を抽出・正規化する
    
    Args:
        df: 入力DataFrame（余分な列が含まれていてもOK）
        
    Returns:
        必須特徴量のみを含むDataFrame（FEATURE_COLUMNSの順序で固定）
        
    Raises:
        ValueError: 必須列が欠けている場合
    """
    # 必須列のチェック
    missing_required = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_required.append(col)
    
    if missing_required:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_required)}"
        )
    
    # 作業用コピー
    working = df.copy()
    
    # 除外列をログに記録（削除はしない、無視するだけ）
    extra_columns = [
        col
        for col in df.columns
        if col not in FEATURE_COLUMNS
        and col not in REQUIRED_COLUMNS
        and col != TARGET_COLUMN
    ]
    if extra_columns:
        LOGGER.info(
            "Ignoring extra columns (not used as features): %s", extra_columns
        )
    
    # 必須特徴量が欠けているかチェック
    missing_features = []
    for col in FEATURE_COLUMNS:
        if col not in working.columns:
            missing_features.append(col)
    
    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {', '.join(missing_features)}"
        )
    
    # 型変換と欠損処理
    normalized = _normalize_features(working)
    
    # FEATURE_COLUMNSの順序で抽出（列順を固定）
    features = normalized.reindex(columns=FEATURE_COLUMNS).copy()
    
    LOGGER.info(
        "Prepared features: shape=%s, columns=%s",
        features.shape,
        list(features.columns),
    )
    
    return features


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特徴量の型変換と欠損処理を行う
    
    Args:
        df: 入力DataFrame
        
    Returns:
        正規化されたDataFrame
    """
    normalized = df.copy()
    
    # 整数列の処理
    for column in INTEGER_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = (
            pd.to_numeric(normalized[column], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
    
    # 浮動小数点数列の処理
    for column in FLOAT_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = (
            pd.to_numeric(normalized[column], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
    
    return normalized


def get_training_box_id_set(training_csv_path: str) -> Set[str]:
    """
    学習データからbox_idの集合を取得する
    
    Args:
        training_csv_path: 学習データCSVのパス
        
    Returns:
        box_idの集合（文字列として）
    """
    try:
        df = pd.read_csv(training_csv_path, encoding="utf-8-sig")
        if TARGET_COLUMN not in df.columns:
            return set()
        box_ids = df[TARGET_COLUMN].dropna().unique()
        # 文字列に変換して返す
        return {str(bid) for bid in box_ids}
    except Exception as exc:
        LOGGER.warning(
            "Failed to load training box IDs from %s: %s",
            training_csv_path,
            exc,
        )
        return set()

