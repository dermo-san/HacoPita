"""
特徴量スキーマ定義
学習時に使用した特徴量を固定し、推論時の前処理と一致させる
"""
from typing import List

# ターゲット列
TARGET_COLUMN = "box_id"

# 除外列（入力CSVにあっても無視する）
DROP_COLUMNS: List[str] = [
    "slip_number",  # 特徴量からは除外だが、突合キー・出力の識別子として保持
    "accessories",
    "product_codes",
    "sizes_raw",
    "max_item_mid_cm",
    "max_item_short_cm",
    "sum_item_area_cm2",
    "avg_item_long_cm",
    "unique_items",
]

# 必須特徴量（推論に使う列：この順序で固定）
# 学習データ列から除外列と目的変数を除いたもの
FEATURE_COLUMNS: List[str] = [
    "total_items",
    "bonsai",
    "other",
    "plastic_pots_trays",
    "single_flower_vase",
    "decorative_sand",
    "saucers_mats",
    "books",
    "suiban",
    "bonsai_seeds",
    "for_bonsai_classes",
    "bonsai_soil",
    "bonsai_tools",
    "bonsai_pots",
    "bonsai_decorations",
    "lucky_bag",
    "moss",
    "moss_bonsai",
    "chemicals_fertilizer",
    "wire",
    "decorative_stones",
    "max_item_long_cm",
    "sum_item_volume_cm3",
]

# 必須列（入力CSVに必ず存在する必要がある列）
# slip_numberは突合キーとして必要だが、特徴量には含めない
REQUIRED_COLUMNS: List[str] = ["slip_number"]

# 数値列（float型）
FLOAT_COLUMNS: set = {
    "max_item_long_cm",
    "sum_item_volume_cm3",
}

# 整数列（int型）- FEATURE_COLUMNSからFLOAT_COLUMNSを除いたもの
INTEGER_COLUMNS: List[str] = [
    col for col in FEATURE_COLUMNS if col not in FLOAT_COLUMNS
]

