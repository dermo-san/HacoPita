"""
label_classes.jsonを生成するユーティリティスクリプト

学習成果物としてlabel_classes.jsonを生成するために使用します。
モデルから取得するか、学習データから取得します。

使用方法:
    # モデルから生成
    python generate_label_classes.py --from-model model.pkl
    
    # 学習データから生成
    python generate_label_classes.py --from-training data/学習データ.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pickle

from label_decoder import (
    LABEL_CLASSES_JSON_PATH,
    save_label_classes_from_model,
    save_label_classes_from_training_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate label_classes.json from model or training data"
    )
    parser.add_argument(
        "--from-model",
        type=str,
        help="Path to model.pkl file",
    )
    parser.add_argument(
        "--from-training",
        type=str,
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for label_classes.json (default: label_classes.json in project root)",
    )
    
    args = parser.parse_args()
    
    if not args.from_model and not args.from_training:
        parser.error("Either --from-model or --from-training must be specified")
    
    if args.from_model and args.from_training:
        parser.error("Cannot specify both --from-model and --from-training")
    
    output_path = Path(args.output) if args.output else LABEL_CLASSES_JSON_PATH
    
    try:
        if args.from_model:
            # モデルから生成
            model_path = Path(args.from_model)
            if not model_path.exists():
                print(f"Error: Model file not found: {model_path}", file=sys.stderr)
                sys.exit(1)
            
            print(f"Loading model from {model_path}...")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            print("Extracting label classes from model...")
            classes = save_label_classes_from_model(model, output_path)
            print(f"Successfully generated label_classes.json with {len(classes)} classes")
            print(f"Output: {output_path}")
            print(f"Classes: {classes}")
        
        elif args.from_training:
            # 学習データから生成
            training_path = Path(args.from_training)
            if not training_path.exists():
                print(f"Error: Training CSV file not found: {training_path}", file=sys.stderr)
                sys.exit(1)
            
            print(f"Loading training data from {training_path}...")
            classes = save_label_classes_from_training_data(str(training_path), output_path)
            print(f"Successfully generated label_classes.json with {len(classes)} classes")
            print(f"Output: {output_path}")
            print(f"Classes: {classes}")
    
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

