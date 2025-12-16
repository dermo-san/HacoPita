"""簡易スモークテスト。サンプルCSVを使って前処理と推論入力整形を検証します。"""
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import (  # noqa: E402
    determine_expected_features,
    load_model,
    normalize_input_dataframe,
    prepare_model_input,
)

ALIAS_RENAMES = {
    "total_line_count": "total_items",
    "others": "other",
    "water_basins": "suiban",
    "bonsai_class_items": "for_bonsai_classes",
    "bonsai_decor": "bonsai_decorations",
    "chemicals_fertilizers": "chemicals_fertilizer",
}


def main() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "static" / "sample_input.csv"
    df = pd.read_csv(sample_path)

    model, model_error = load_model()
    if model_error:
        print(
            f"警告: モデル読み込みは失敗しました ({model_error})。feature list の自動取得のみ検証します。"
        )
        model = None

    features, feature_error = determine_expected_features(model)
    assert feature_error == "", f"特徴量取得でエラー: {feature_error}"

    variants = {
        "canonical": df,
        "alias": df.rename(columns=ALIAS_RENAMES),
    }

    for label, current_df in variants.items():
        normalized = normalize_input_dataframe(current_df)
        X = prepare_model_input(normalized, features)
        assert X.shape[1] == len(features), f"{label}: 列数が期待値と一致しません"
        print(f"{label} variant -> rows={len(current_df)} features={X.shape[1]}")

    print("Smoke test passed:")
    print(f"  feature count = {len(features)}")
    print(f"  head columns = {features[:5]}")


if __name__ == "__main__":
    main()
