from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from schema import FEATURE_COLUMNS_24, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 24-feature model and export model_features.json."
    )
    parser.add_argument("--train-data", type=Path, required=True, help="CSV training data path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write model.pkl and model_features.json",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out ratio for evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training",
    )
    return parser.parse_args()


def _log_model_schema(model) -> None:
    logger = logging.getLogger("schema_check")
    logger.info("MODEL TYPE: %s", type(model))
    if hasattr(model, "n_features_in_"):
        logger.info("MODEL n_features_in_: %s", model.n_features_in_)
    if hasattr(model, "feature_names_in_"):
        logger.info("MODEL feature_names_in_: %s", list(model.feature_names_in_))
    if hasattr(model, "steps"):
        logger.info("PIPELINE STEPS: %s", [name for name, _ in model.steps])
        last = model.steps[-1][1]
        if hasattr(last, "n_features_in_"):
            logger.info("LAST n_features_in_: %s", last.n_features_in_)


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Training data is missing target column: {TARGET_COLUMN}")

    X = df.reindex(columns=FEATURE_COLUMNS_24)
    assert X.shape[1] == 24, f"TRAIN FEATURE COUNT MUST BE 24, got {X.shape[1]}"
    missing = [c for c in FEATURE_COLUMNS_24 if c not in df.columns]
    assert not missing, f"Missing required columns for training: {missing}"
    print("TRAIN FEATURES:", list(X.columns))

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    return X, y


def main() -> None:
    args = parse_args()
    logging.info("Loading training data from %s", args.train_data)
    df = pd.read_csv(args.train_data)

    X, y = _prepare_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=None
    )

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=4,
        n_jobs=-1,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)
    if hasattr(model, "n_features_in_"):
        print("MODEL n_features_in_:", model.n_features_in_)
    _log_model_schema(model)

    preds = model.predict(X_val)
    logging.info("Validation report:\n%s", classification_report(y_val, preds))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pkl"
    joblib.dump(model, model_path)
    logging.info("Saved model to %s", model_path)

    features_path = args.output_dir / "model_features.json"
    with features_path.open("w", encoding="utf-8") as handle:
        json.dump(FEATURE_COLUMNS_24, handle, ensure_ascii=False, indent=2)
    logging.info("Saved feature manifest to %s", features_path)


if __name__ == "__main__":
    main()
