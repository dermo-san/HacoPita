"""
予測結果の評価スクリプト
slip_numberでbox_idとpredicted_box_idを突合し、Accuracyを算出する
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from schema import TARGET_COLUMN


def evaluate_predictions(
    predictions_csv: str, ground_truth_csv: str = None
) -> float:
    """
    予測結果を評価する
    
    Args:
        predictions_csv: 予測結果を含むCSVファイル（predicted_box_id列を含む）
        ground_truth_csv: 正解データのCSVファイル（box_id列を含む）
                         指定しない場合、predictions_csvからbox_idを取得
        
    Returns:
        Accuracy（0.0-1.0）
    """
    # 予測結果を読み込む
    pred_df = pd.read_csv(predictions_csv, encoding="utf-8-sig")
    
    if "predicted_box_id" not in pred_df.columns:
        raise ValueError("predictions_csv must contain 'predicted_box_id' column")
    
    if "slip_number" not in pred_df.columns:
        raise ValueError("predictions_csv must contain 'slip_number' column")
    
    # 正解データを読み込む
    if ground_truth_csv:
        gt_df = pd.read_csv(ground_truth_csv, encoding="utf-8-sig")
        if "slip_number" not in gt_df.columns:
            raise ValueError("ground_truth_csv must contain 'slip_number' column")
        if TARGET_COLUMN not in gt_df.columns:
            raise ValueError(f"ground_truth_csv must contain '{TARGET_COLUMN}' column")
        
        # slip_numberでマージ
        merged = pred_df.merge(
            gt_df[["slip_number", TARGET_COLUMN]],
            on="slip_number",
            how="inner",
        )
    else:
        # predictions_csvからbox_idを取得
        if TARGET_COLUMN not in pred_df.columns:
            raise ValueError(
                f"predictions_csv must contain '{TARGET_COLUMN}' column "
                "or specify ground_truth_csv"
            )
        merged = pred_df.copy()
    
    # box_idとpredicted_box_idを文字列に変換して比較
    merged[TARGET_COLUMN] = merged[TARGET_COLUMN].astype(str)
    merged["predicted_box_id"] = merged["predicted_box_id"].astype(str)
    
    # 一致数をカウント
    correct = (merged[TARGET_COLUMN] == merged["predicted_box_id"]).sum()
    total = len(merged)
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 誤分類の詳細を表示
    incorrect = merged[merged[TARGET_COLUMN] != merged["predicted_box_id"]]
    if len(incorrect) > 0:
        print(f"\nIncorrect predictions ({len(incorrect)} samples):")
        print(
            incorrect[["slip_number", TARGET_COLUMN, "predicted_box_id"]].to_string(
                index=False
            )
        )
        
        # 予測値の分布を表示
        print("\nPredicted box_id distribution:")
        print(merged["predicted_box_id"].value_counts().to_string())
        
        # 正解box_idの分布を表示
        print("\nTrue box_id distribution:")
        print(merged[TARGET_COLUMN].value_counts().to_string())
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prediction results by comparing box_id and predicted_box_id"
    )
    parser.add_argument(
        "predictions_csv",
        type=str,
        help="CSV file with predictions (must contain 'predicted_box_id' column)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="CSV file with ground truth (must contain 'box_id' column). "
        "If not specified, uses 'box_id' from predictions_csv",
    )
    
    args = parser.parse_args()
    
    try:
        accuracy = evaluate_predictions(args.predictions_csv, args.ground_truth)
        sys.exit(0 if accuracy > 0 else 1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

