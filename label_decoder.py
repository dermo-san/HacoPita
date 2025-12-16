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
            - None: 自動判定（既存のロジック）
        
    Returns:
        box_idのリスト（文字列）
        
    Raises:
        ValueError: 逆変換後のbox_idが学習データに存在しない場合
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
    
    # output_is_class_indexが明示的に指定されている場合
    if output_is_class_index is True:
        # 常に内部クラスIDとして扱う
        LOGGER.info(
            "output_is_class_index=True: Treating predictions as internal class IDs"
        )
        is_already_box_id = False
    elif output_is_class_index is False:
        # 常にbox_idとして扱う
        LOGGER.info(
            "output_is_class_index=False: Treating predictions as box_ids"
        )
        is_already_box_id = True
    else:
        # 自動判定（既存のロジック）
        # ユニークな予測値を取得（文字列に変換）
        pred_unique = set(str(int(p)) for p in pred_array if not pd.isna(p))
        
        # 予測値が既にbox_idかどうかを判定
        # 判定方法：
        # 1. train_box_id_setが空でない場合: pred_uniqueがtrain_box_id_setの部分集合なら「既にbox_id」
        # 2. train_box_id_setが空の場合: 予測値がlabel_classesのインデックス範囲内にあり、
        #    かつそのインデックスに対応するbox_idが予測値と一致しない場合は内部クラスIDと判定
        if train_box_id_set:
            is_already_box_id = pred_unique.issubset(train_box_id_set)
        else:
            # train_box_id_setが空の場合の判定
            # 予測値がlabel_classesのインデックス範囲内にあり、かつ
            # そのインデックスに対応するbox_idが予測値と一致しない場合は内部クラスID
            label_classes_set = set(label_classes)
            has_index_mismatch = any(
                int(p) >= 0
                and int(p) < len(label_classes)
                and str(label_classes[int(p)]) != str(int(p))
                for p in pred_array
                if not pd.isna(p)
            )
            # 予測値がlabel_classesに含まれていて、かつインデックス不一致がない場合は既にbox_id
            is_already_box_id = (
                pred_unique.issubset(label_classes_set) and not has_index_mismatch
            )
    
    if is_already_box_id:
        LOGGER.info(
            "Predictions are already box_ids: %s (subset of training box_ids)",
            pred_unique,
        )
        # そのまま文字列に変換して返す
        decoded = [str(int(p)) if not pd.isna(p) else "0" for p in pred_array]
    else:
        LOGGER.info(
            "Predictions appear to be internal class IDs. Decoding using label_classes..."
        )
        # 内部クラスIDとして扱い、逆変換する
        decoded = []
        for pred in pred_array:
            if pd.isna(pred):
                decoded.append("0")
                continue
            
            pred_idx = int(pred)
            
            # インデックスの範囲チェック
            if pred_idx < 0 or pred_idx >= len(label_classes):
                raise ValueError(
                    f"Prediction index {pred_idx} is out of range "
                    f"(label_classes length: {len(label_classes)})"
                )
            
            box_id = str(label_classes[pred_idx])
            decoded.append(box_id)
        
        # 逆変換後のbox_idが学習データに存在するかチェック
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

