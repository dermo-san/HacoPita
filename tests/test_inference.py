"""
推論プログラムのテスト
要件を満たしていることを確認する
"""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# テスト用のインポート
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing import prepare_features
from label_decoder import decode_predictions
from schema import FEATURE_COLUMNS, DROP_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


def test_prepare_features_with_extra_columns():
    """テスト1: 入力CSVに除外列が含まれても推論できる"""
    # 必須特徴量 + 除外列を含むDataFrameを作成
    data = {
        "slip_number": [1, 2, 3],
        "total_items": [10, 20, 30],
        "bonsai": [1, 2, 3],
        "other": [0, 1, 0],
        "plastic_pots_trays": [1, 2, 3],
        "single_flower_vase": [0, 1, 0],
        "decorative_sand": [0, 0, 1],
        "saucers_mats": [1, 1, 1],
        "books": [0, 0, 0],
        "suiban": [0, 1, 0],
        "bonsai_seeds": [0, 0, 0],
        "for_bonsai_classes": [0, 0, 0],
        "bonsai_soil": [1, 1, 1],
        "bonsai_tools": [0, 0, 0],
        "bonsai_pots": [0, 0, 0],
        "bonsai_decorations": [0, 0, 0],
        "lucky_bag": [0, 0, 0],
        "moss": [0, 0, 0],
        "moss_bonsai": [0, 0, 0],
        "chemicals_fertilizer": [0, 0, 0],
        "wire": [0, 0, 0],
        "decorative_stones": [0, 0, 0],
        "max_item_long_cm": [10.0, 20.0, 30.0],
        "sum_item_volume_cm3": [100.0, 200.0, 300.0],
        # 除外列（無視される）
        "accessories": ["A", "B", "C"],
        "product_codes": ["P1", "P2", "P3"],
        "sizes_raw": ["S", "M", "L"],
        "max_item_mid_cm": [5.0, 10.0, 15.0],
        "max_item_short_cm": [3.0, 6.0, 9.0],
        "sum_item_area_cm2": [50.0, 100.0, 150.0],
        "avg_item_long_cm": [5.0, 10.0, 15.0],
        "unique_items": [1, 2, 3],
        "box_id": [7, 8, 9],  # ターゲット列（無視される）
    }
    df = pd.DataFrame(data)
    
    # prepare_featuresを実行
    features = prepare_features(df)
    
    # 検証
    assert features.shape[1] == len(FEATURE_COLUMNS)
    assert list(features.columns) == FEATURE_COLUMNS
    assert "accessories" not in features.columns
    assert "product_codes" not in features.columns
    assert "sizes_raw" not in features.columns
    assert "box_id" not in features.columns


def test_prepare_features_missing_required_column():
    """テスト2: 必須列が1つでも欠けたら分かりやすいエラーになる"""
    # slip_numberを欠いたDataFrame
    data = {
        "total_items": [10, 20, 30],
        "bonsai": [1, 2, 3],
        # ... 他の必須特徴量も欠けている
    }
    df = pd.DataFrame(data)
    
    # エラーが発生することを確認
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_features(df)


def test_prepare_features_missing_feature_column():
    """テスト2: 必須特徴量が欠けたらエラーになる"""
    # 必須特徴量の一部を欠いたDataFrame
    data = {
        "slip_number": [1, 2, 3],
        "total_items": [10, 20, 30],
        "bonsai": [1, 2, 3],
        # max_item_long_cmが欠けている
    }
    df = pd.DataFrame(data)
    
    # エラーが発生することを確認
    with pytest.raises(ValueError, match="Missing required feature columns"):
        prepare_features(df)


def test_decode_predictions_internal_class_ids():
    """テスト3: 内部クラスIDをbox_idに逆変換できる"""
    # ラベルクラス（学習時のbox_idのリスト）
    label_classes = ["7", "8", "9", "13", "18", "24"]
    
    # 学習データのbox_id集合
    train_box_id_set = {"7", "8", "9", "13", "18", "24"}
    
    # 内部クラスIDの予測値（0, 1, 2, ...）
    predictions = np.array([0, 1, 2, 3, 4, 5])
    
    # デコード（自動判定）
    decoded = decode_predictions(predictions, label_classes, train_box_id_set)
    
    # 検証
    assert decoded == ["7", "8", "9", "13", "18", "24"]
    assert all(bid in train_box_id_set for bid in decoded)
    
    # 明示的に内部クラスIDとして指定
    decoded_explicit = decode_predictions(
        predictions, label_classes, train_box_id_set, output_is_class_index=True
    )
    assert decoded_explicit == ["7", "8", "9", "13", "18", "24"]


def test_decode_predictions_already_box_ids():
    """テスト3: 予測値が既にbox_idの場合はそのまま返す"""
    # ラベルクラス
    label_classes = ["7", "8", "9", "13", "18", "24"]
    
    # 学習データのbox_id集合
    train_box_id_set = {"7", "8", "9", "13", "18", "24"}
    
    # 既にbox_idの予測値
    predictions = np.array([7, 8, 9, 13, 18, 24])
    
    # デコード（自動判定）
    decoded = decode_predictions(predictions, label_classes, train_box_id_set)
    
    # 検証（文字列に変換される）
    assert decoded == ["7", "8", "9", "13", "18", "24"]
    
    # 明示的にbox_idとして指定
    decoded_explicit = decode_predictions(
        predictions, label_classes, train_box_id_set, output_is_class_index=False
    )
    assert decoded_explicit == ["7", "8", "9", "13", "18", "24"]


def test_decode_predictions_invalid_box_id():
    """テスト3: 逆変換後のbox_idが学習データに存在しない場合はエラー"""
    # ラベルクラス
    label_classes = ["7", "8", "9"]
    
    # 学習データのbox_id集合（"10"は含まれていない）
    train_box_id_set = {"7", "8", "9"}
    
    # 内部クラスIDの予測値（範囲外）
    predictions = np.array([0, 1, 2, 3])  # 3は範囲外
    
    # エラーが発生することを確認
    with pytest.raises(ValueError, match="out of range"):
        decode_predictions(predictions, label_classes, train_box_id_set)
    
    # output_is_class_index=Trueで範囲外
    with pytest.raises(ValueError, match="out of range"):
        decode_predictions(
            predictions, label_classes, train_box_id_set, output_is_class_index=True
        )


def test_decode_predictions_output_is_class_index_false_with_invalid_box_ids():
    """テスト: output_is_class_index=Falseで内部IDっぽい入力を与えた場合、エラーになる"""
    label_classes = ["7", "8", "9", "13", "18", "24"]
    train_box_id_set = {"7", "8", "9", "13", "18", "24"}
    
    # 内部クラスIDっぽい予測値（0, 1, 2）をbox_idとして扱おうとする
    predictions = np.array([0, 1, 2])
    
    # train_box_id_setに"0", "1", "2"が含まれていない場合、エラーになる
    with pytest.raises(ValueError, match="not found in training data"):
        decode_predictions(
            predictions, label_classes, train_box_id_set, output_is_class_index=False
        )


def test_decode_predictions_output_is_class_index_true():
    """テスト: output_is_class_index=Trueで内部クラスIDを正しくデコード"""
    label_classes = ["7", "8", "9", "13", "18", "24"]
    train_box_id_set = {"7", "8", "9", "13", "18", "24"}
    
    # 内部クラスIDの予測値
    predictions = np.array([0, 1, 2])
    
    # デコード
    decoded = decode_predictions(
        predictions, label_classes, train_box_id_set, output_is_class_index=True
    )
    
    # 検証
    assert decoded == ["7", "8", "9"]


def test_decode_predictions_string_input():
    """テスト: 文字列形式の予測値でも安全に処理できる"""
    label_classes = ["7", "8", "9", "13", "18", "24"]
    train_box_id_set = {"7", "8", "9", "13", "18", "24"}
    
    # 文字列形式の予測値（有効な範囲内）
    predictions_valid = ["0", "1", "2"]
    decoded = decode_predictions(
        predictions_valid, label_classes, train_box_id_set, output_is_class_index=True
    )
    assert decoded == ["7", "8", "9"]
    
    # 浮動小数点数の文字列形式
    predictions_float_str = ["0.0", "1.0", "2.0"]
    decoded = decode_predictions(
        predictions_float_str, label_classes, train_box_id_set, output_is_class_index=True
    )
    assert decoded == ["7", "8", "9"]
    
    # 範囲外の文字列
    predictions_out_of_range = ["13.0", "14", "15.0"]
    with pytest.raises(ValueError, match="out of range"):
        decode_predictions(
            predictions_out_of_range, label_classes, train_box_id_set, output_is_class_index=True
        )


def test_decode_predictions_nan_raises_error():
    """テスト: NaNが含まれている場合は例外になる"""
    label_classes = ["7", "8", "9"]
    train_box_id_set = {"7", "8", "9"}
    
    # NaNを含む予測値
    predictions = np.array([0, np.nan, 2])
    
    # エラーが発生することを確認
    with pytest.raises(ValueError, match="NaN or invalid values"):
        decode_predictions(predictions, label_classes, train_box_id_set)
    
    # output_is_class_index=Trueでも同様
    with pytest.raises(ValueError, match="NaN or invalid values"):
        decode_predictions(
            predictions, label_classes, train_box_id_set, output_is_class_index=True
        )


def test_prepare_features_column_order():
    """テスト4: FEATURE_COLUMNSの順序で抽出される"""
    # ランダムな順序で列を持つDataFrame
    data = {
        col: [1] * 3 for col in FEATURE_COLUMNS
    }
    data["slip_number"] = [1, 2, 3]
    # 列順をランダムに
    cols = ["slip_number"] + list(np.random.permutation(FEATURE_COLUMNS))
    df = pd.DataFrame(data)[cols]
    
    # prepare_featuresを実行
    features = prepare_features(df)
    
    # 検証: 列順がFEATURE_COLUMNSと一致する
    assert list(features.columns) == FEATURE_COLUMNS


def test_prepare_features_type_conversion():
    """テスト4: 型変換が正しく行われる"""
    # 文字列として数値が入っているDataFrame
    data = {
        "slip_number": ["1", "2", "3"],
        "total_items": ["10", "20", "30"],  # 文字列
        "max_item_long_cm": ["10.5", "20.5", "30.5"],  # 文字列
        "sum_item_volume_cm3": ["100.5", "200.5", "300.5"],  # 文字列
    }
    # 他の必須特徴量も追加
    for col in FEATURE_COLUMNS:
        if col not in data:
            if col in ["max_item_long_cm", "sum_item_volume_cm3"]:
                data[col] = ["0.0"] * 3
            else:
                data[col] = ["0"] * 3
    
    df = pd.DataFrame(data)
    
    # prepare_featuresを実行
    features = prepare_features(df)
    
    # 検証: 型が正しく変換されている
    assert features["total_items"].dtype == "int64"
    assert features["max_item_long_cm"].dtype == "float64"
    assert features["sum_item_volume_cm3"].dtype == "float64"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

