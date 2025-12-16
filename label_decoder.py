"""
予測ラベルのデコード（内部クラスID → box_id）
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

LABEL_CLASSES_JSON_PATH = Path(__file__).resolve().parent / "label_classes.json"


def _to_int_safe(x) -> Optional[int]:
    """
    予測値を安全に整数に変換する
    
    Args:
        x: 予測値（"13", 13, 13.0, "13.0", np.int64(13), np.float64(13.0)等）
        
    Returns:
        整数値、またはNone（NaN/空文字/変換不能の場合）
    """
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return int(x)
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def load_label_classes(json_path: Path = None) -> List[str]:
    """
    ラベルクラス（box_idのリスト）をJSONから読み込む
    
    Args:
        json_path: JSONファイルのパス（Noneの場合はデフォルトパス）
        
    Returns:
        box_idのリスト（学習時のクラス順序）
    """
    if json_path is None:
        json_path = LABEL_CLASSES_JSON_PATH
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Label classes file not found: {json_path}. "
            "Please create it from training data or model."
        )
    
    with open(json_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    
    if not isinstance(classes, list):
        raise ValueError("label_classes.json must contain a list")
    
    # 文字列に変換
    return [str(c) for c in classes]


def decode_predictions(
    predictions: Union[np.ndarray, List, pd.Series],
    label_classes: List[str],
    train_box_id_set: Set[str],
    output_is_class_index: Optional[bool] = None,
) -> List[str]:
    """
    予測値をbox_idにデコードする
    
    予測値が既にbox_idの場合はそのまま返し、
    内部クラスIDの場合は逆変換してbox_idに戻す。
    
    Args:
        predictions: 予測値（内部クラスIDまたはbox_id）
        label_classes: 学習時のクラスリスト（box_idのリスト）
        train_box_id_set: 学習データに存在するbox_idの集合
        output_is_class_index: 予測値が内部クラスIDかどうかを明示的に指定
            - True: 常に内部クラスIDとして扱う（逆変換を実行）
            - False: 常にbox_idとして扱う（そのまま返す）
            - None: 自動判定（強化されたロジック）
        
    Returns:
        box_idのリスト（文字列）
        
    Raises:
        ValueError: 
            - 予測値にNaNが含まれている場合
            - 予測値を整数に変換できない場合
            - 内部クラスIDの範囲外の場合
            - デコード後のbox_idが学習データに存在しない場合
            - output_is_class_index=Falseで予測値が学習box_id集合に含まれない場合
    """
    # numpy配列に変換
    if isinstance(predictions, pd.Series):
        pred_array = predictions.values
    elif isinstance(predictions, list):
        pred_array = np.array(predictions)
    else:
        pred_array = np.asarray(predictions)
    
    # 1次元に変換
    pred_array = pred_array.reshape(-1)
    
    # 予測値を安全に整数化（NaN/不正値チェック）
    pred_ints = []
    nan_indices = []
    for i, pred in enumerate(pred_array):
        pred_int = _to_int_safe(pred)
        if pred_int is None:
            nan_indices.append(i)
        else:
            pred_ints.append(pred_int)
    
    # NaN/不正値が含まれている場合は例外
    if nan_indices:
        raise ValueError(
            f"Predictions contain NaN or invalid values at indices: {nan_indices[:10]}. "
            "Cannot decode predictions with missing values."
        )
    
    # ログ用にユニークな予測値を取得（常に定義）
    pred_unique_int = set(pred_ints)
    pred_unique_str = {str(p) for p in pred_unique_int}
    
    # output_is_class_indexが明示的に指定されている場合
    if output_is_class_index is True:
        # 常に内部クラスIDとして扱う
        mode = "class_index (explicit)"
        LOGGER.info(
            "Mode: %s | Unique predictions (int): %s (count=%d) | label_classes length: %d",
            mode,
            sorted(list(pred_unique_int))[:20],
            len(pred_unique_int),
            len(label_classes),
        )
        is_already_box_id = False
    elif output_is_class_index is False:
        # 常にbox_idとして扱う
        mode = "box_id (explicit)"
        LOGGER.info(
            "Mode: %s | Unique predictions (str): %s (count=%d) | train_box_id_set size: %d",
            mode,
            sorted(list(pred_unique_str))[:20],
            len(pred_unique_str),
            len(train_box_id_set),
        )
        is_already_box_id = True
    else:
        # 自動判定（強化されたロジック）
        mode = "auto"
        
        # 内部クラスIDの可能性を判定
        # 条件1: 予測値が0以上でlabel_classesの範囲内
        is_in_range = (
            len(pred_unique_int) > 0
            and min(pred_unique_int) >= 0
            and max(pred_unique_int) < len(label_classes)
        )
        
        # 条件2: 内部クラスIDっぽい特徴
        has_class_index_features = False
        if is_in_range:
            # 0が含まれる
            has_zero = 0 in pred_unique_int
            # 予測ユニーク数が小さい（<=10）
            is_small_count = len(pred_unique_int) <= 10
            # 値域が狭い（max-minが小さい、例：<=20）
            value_range = max(pred_unique_int) - min(pred_unique_int) if pred_unique_int else 0
            is_narrow_range = value_range <= 20
            
            has_class_index_features = has_zero or is_small_count or is_narrow_range
        
        # 内部クラスIDと判定
        if is_in_range and has_class_index_features:
            is_already_box_id = False
            mode = "auto->class_index"
        else:
            # box_idとして扱う（train_box_id_setで検証）
            if train_box_id_set:
                is_already_box_id = pred_unique_str.issubset(train_box_id_set)
                if is_already_box_id:
                    mode = "auto->box_id"
                else:
                    # train_box_id_setに含まれない場合は内部クラスIDとして試す
                    is_already_box_id = False
                    mode = "auto->class_index (fallback)"
            else:
                # train_box_id_setが空の場合は内部クラスIDとして扱う
                is_already_box_id = False
                mode = "auto->class_index (no train_box_id_set)"
        
        LOGGER.info(
            "Mode: %s | Unique predictions (int): %s (count=%d) | label_classes length: %d | train_box_id_set size: %d",
            mode,
            sorted(list(pred_unique_int))[:20],
            len(pred_unique_int),
            len(label_classes),
            len(train_box_id_set),
        )
    
    # デコード処理
    if is_already_box_id:
        # box_idとして扱う
        decoded = [str(p) for p in pred_ints]
        
        # output_is_class_index=Falseの場合でも、予測値が学習box_id集合に含まれない場合はエラー
        if output_is_class_index is False or train_box_id_set:
            decoded_unique = set(decoded)
            invalid_box_ids = decoded_unique - train_box_id_set
            if invalid_box_ids:
                raise ValueError(
                    f"Predictions contain box_ids not found in training data: {invalid_box_ids}. "
                    "This may indicate a configuration error (output_is_class_index=False but predictions are class indices)."
                )
    else:
        # 内部クラスIDとして扱い、逆変換する
        decoded = []
        for pred_idx in pred_ints:
            # インデックスの範囲チェック
            if pred_idx < 0 or pred_idx >= len(label_classes):
                raise ValueError(
                    f"Prediction index {pred_idx} is out of range "
                    f"(label_classes length: {len(label_classes)})"
                )
            
            box_id = str(label_classes[pred_idx])
            decoded.append(box_id)
        
        # 逆変換後のbox_idが学習データに存在するかチェック
        if train_box_id_set:
            decoded_unique = set(decoded)
            invalid_box_ids = decoded_unique - train_box_id_set
            if invalid_box_ids:
                raise ValueError(
                    f"Decoded box_ids not found in training data: {invalid_box_ids}. "
                    "This indicates a mapping corruption."
                )
    
    return decoded


def save_label_classes_from_model(model, output_path: Path = None) -> List[str]:
    """
    モデルからラベルクラスを取得してJSONに保存する
    
    Args:
        model: 学習済みモデル（classes_属性を持つ）
        output_path: 出力パス（Noneの場合はデフォルトパス）
        
    Returns:
        ラベルクラスのリスト
    """
    if output_path is None:
        output_path = LABEL_CLASSES_JSON_PATH
    
    if hasattr(model, "classes_"):
        classes = model.classes_
        if hasattr(classes, "tolist"):
            classes_list = classes.tolist()
        else:
            classes_list = list(classes)
    elif hasattr(model, "classes"):
        classes_list = list(model.classes)
    else:
        raise ValueError(
            "Model does not have 'classes_' or 'classes' attribute. "
            "Cannot extract label classes."
        )
    
    # 文字列に変換
    classes_list = [str(c) for c in classes_list]
    
    # JSONに保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, ensure_ascii=False, indent=2)
    
    LOGGER.info("Saved label classes to %s: %s", output_path, classes_list)
    
    return classes_list


def save_label_classes_from_training_data(
    training_csv_path: str, output_path: Path = None
) -> List[str]:
    """
    学習データからbox_idのユニーク値を取得してJSONに保存する
    
    Args:
        training_csv_path: 学習データCSVのパス
        output_path: 出力パス（Noneの場合はデフォルトパス）
        
    Returns:
        box_idのリスト（ソート済み）
    """
    if output_path is None:
        output_path = LABEL_CLASSES_JSON_PATH
    
    df = pd.read_csv(training_csv_path, encoding="utf-8-sig")
    
    if "box_id" not in df.columns:
        raise ValueError("Training CSV does not have 'box_id' column")
    
    box_ids = sorted(df["box_id"].dropna().unique())
    
    # 文字列に変換
    classes_list = [str(bid) for bid in box_ids]
    
    # JSONに保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, ensure_ascii=False, indent=2)
    
    LOGGER.info(
        "Saved label classes from training data to %s: %s",
        output_path,
        classes_list,
    )
    
    return classes_list

