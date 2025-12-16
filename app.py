import ast
import io
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

REQUIRED_COLUMNS: List[str] = ["slip_number"]

INTEGER_COLUMNS = [
    "slip_number",
    "subtotal_amount",
    "total_line_count",
    "bonsai",
    "others",
    "plastic_pots_trays",
    "single_flower_vase",
    "decorative_sand",
    "saucers_mats",
    "books",
    "water_basins",
    "bonsai_seeds",
    "bonsai_class_items",
    "bonsai_soil",
    "bonsai_tools",
    "bonsai_pots",
    "bonsai_decor",
    "lucky_bag",
    "moss",
    "moss_bonsai",
    "chemicals_fertilizers",
    "wire",
    "decorative_stones",
    "specification",
]

DATETIME_COLUMNS = ["shipment_confirmed_date"]
STRING_COLUMNS = ["dimensions"]
FALLBACK_DATE = pd.Timestamp("1970-01-01")
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
SCORING_FILE_PATH = Path(__file__).resolve().parent / "model" / "scoring_file_v_2_0_0.py"
EXPECTED_FEATURES_JSON_PATH = Path(__file__).resolve().parent / "expected_features.json"
LOGGER = logging.getLogger(__name__)
FEATURE_ALIASES: Dict[str, List[str]] = {
    "total_line_count": ["total_items"],
    "others": ["other"],
    "water_basins": ["suiban"],
    "bonsai_class_items": ["for_bonsai_classes"],
    "bonsai_decor": ["bonsai_decorations"],
    "chemicals_fertilizers": ["chemicals_fertilizer"],
}


def create_app() -> Flask:
    app = Flask(__name__)

    model, model_error = load_model()
    expected_features, feature_error = determine_expected_features(model)

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            model_error=model_error,
            feature_error=feature_error,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if model_error or feature_error:
            return (
                render_template(
                    "index.html",
                    error_message=model_error or feature_error,
                    model_error=model_error,
                    feature_error=feature_error,
                ),
                500,
            )

        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return (
                render_template(
                    "index.html",
                    error_message="CSVファイルを選択してください。",
                    model_error=model_error,
                    feature_error=feature_error,
                ),
                400,
            )

        try:
            result_df, predictions = process_file(
                uploaded_file, model, expected_features
            )
        except PredictionError as exc:
            return (
                render_template(
                    "index.html",
                    error_message=exc.message,
                    model_error=model_error,
                    feature_error=feature_error,
                ),
                exc.status_code,
            )

        csv_response = dataframe_to_csv_response(
            result_df, build_output_filename(uploaded_file.filename)
        )
        return csv_response

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        if model_error or feature_error:
            return jsonify({"error": model_error or feature_error}), 500

        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"error": "file field is required"}), 400

        try:
            result_df, predictions = process_file(
                uploaded_file, model, expected_features
            )
        except PredictionError as exc:
            return jsonify({"error": exc.message}), exc.status_code

        if request.args.get("format") == "json":
            payload = [
                {
                    "slip_number": int(result_df.iloc[i]["slip_number"]),
                    "predicted_box_id": int(pred),
                }
                for i, pred in enumerate(predictions)
            ]
            return jsonify({"predictions": payload})

        return dataframe_to_csv_response(
            result_df, build_output_filename(uploaded_file.filename)
        )

    return app


def load_model() -> Tuple[object, str]:
    if not MODEL_PATH.exists():
        message = f"モデルファイルが見つかりません: {MODEL_PATH}"
        logging.error(message)
        return None, message

    try:
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)
        return model, ""
    except Exception as exc:  # pragma: no cover - defensive logging
        message = f"モデルの読み込みに失敗しました: {exc}"
        logging.exception(message)
        return None, message


def determine_expected_features(model) -> Tuple[List[str], str]:
    try:
        from_model = infer_features_from_model(model)
        if from_model:
            log_detected_features(from_model, "model.pkl feature_names_in_")
            return from_model, ""

        from_scoring = infer_features_from_scoring_file()
        if from_scoring:
            log_detected_features(from_scoring, SCORING_FILE_PATH)
            return from_scoring, ""

        from_json = load_features_from_json()
        if from_json:
            log_detected_features(from_json, EXPECTED_FEATURES_JSON_PATH)
            return from_json, ""

        message = (
            "特徴量一覧を model.pkl / model/scoring_file_v_2_0_0.py / "
            "expected_features.json のいずれからも取得できません。"
        )
        LOGGER.error(message)
        return [], message
    except Exception as exc:
        message = f"特徴量一覧の取得に失敗しました: {exc}"
        LOGGER.exception(message)
        return [], message


def infer_features_from_model(model) -> List[str]:
    if model is None:
        return []
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return []
    return [str(name) for name in feature_names if str(name)]


def infer_features_from_scoring_file() -> List[str]:
    if not SCORING_FILE_PATH.exists():
        return []
    text = SCORING_FILE_PATH.read_text(encoding="utf-8")
    match = re.search(r"pd\\.DataFrame\\(\\{(.*?)\\}\\)", text, re.S)
    if not match:
        return []
    dict_body = "{" + match.group(1) + "}"
    dict_node = ast.parse(dict_body, mode="eval")
    if not isinstance(dict_node.body, ast.Dict):
        return []
    features: List[str] = []
    for key in dict_node.body.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            features.append(key.value)
    return features


def load_features_from_json() -> List[str]:
    if not EXPECTED_FEATURES_JSON_PATH.exists():
        return []
    raw = EXPECTED_FEATURES_JSON_PATH.read_text(encoding="utf-8")
    data = json.loads(raw or "[]")
    if not isinstance(data, list):
        raise ValueError("expected_features.json は配列形式である必要があります")
    return [str(item) for item in data if str(item)]


def log_detected_features(features: Sequence[str], source) -> None:
    count = len(features)
    head = list(features[:3])
    tail = list(features[-3:]) if count > 3 else []
    LOGGER.info(
        "Detected %s features from %s. head=%s%s",
        count,
        source,
        head,
        f", tail={tail}" if tail else "",
    )


class PredictionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def process_file(
    file_storage, model, expected_features: Sequence[str]
) -> Tuple[pd.DataFrame, List[int]]:
    if not expected_features:
        raise PredictionError("モデルの特徴量一覧が取得できませんでした。", status_code=500)

    input_df = read_csv_with_fallback(file_storage)
    missing = find_missing_required_columns(input_df)
    if missing:
        raise PredictionError(
            f"必須列が不足しています: {', '.join(missing)}", status_code=400
        )

    normalized_df = normalize_input_dataframe(input_df)
    features = prepare_model_input(normalized_df, expected_features)
    if features.shape[1] != len(expected_features):
        raise PredictionError(
            "モデル入力の列数が期待値と一致しません。", status_code=500
        )

    try:
        LOGGER.info(
            "Model input shape=%s columns=%s",
            features.shape,
            list(features.columns),
        )
        raw_predictions = model.predict(features)
    except Exception as exc:
        raise PredictionError(f"推論に失敗しました: {exc}", status_code=500) from exc

    flattened = np.asarray(raw_predictions).reshape(-1)
    predictions = pd.Series(flattened).fillna(0).astype("int64")
    result_df = input_df.copy()
    result_df["predicted_box_id"] = predictions.values
    return result_df, predictions.tolist()


def read_csv_with_fallback(file_storage) -> pd.DataFrame:
    file_storage.stream.seek(0)
    data = file_storage.read()
    last_error = None
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise PredictionError(f"CSVの読み込みに失敗しました: {exc}", 400) from exc

    raise PredictionError(
        "CSVの読み込みに失敗しました (文字コードを判定できませんでした)。", 400
    )


def normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in INTEGER_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = (
            pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype("int64")
        )

    for column in DATETIME_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce").fillna(
            FALLBACK_DATE
        )

    for column in STRING_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = normalized[column].fillna("").astype(str)

    return normalized


def prepare_model_input(
    df: pd.DataFrame, expected_features: Sequence[str]
) -> pd.DataFrame:
    working = df.copy()
    alias_usage: Dict[str, str] = {}
    for column in expected_features:
        if column in working.columns:
            continue
        alias = find_alias_column(df, column)
        if alias:
            working[column] = df[alias]
            alias_usage[column] = alias

    missing = [col for col in expected_features if col not in working.columns]
    extra_columns = [col for col in df.columns if col not in expected_features]
    LOGGER.info(
        "Incoming columns=%s | expected=%s | missing=%s | extra=%s | alias=%s",
        list(df.columns),
        list(expected_features),
        missing,
        extra_columns,
        alias_usage,
    )
    if missing:
        raise PredictionError(
            f"モデルに必要な列が不足しています: {', '.join(missing)}", status_code=400
        )

    features = working.reindex(columns=expected_features).copy()
    for column in expected_features:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)

    return features.astype("float64")


def find_missing_required_columns(df: pd.DataFrame) -> List[str]:
    missing: List[str] = []
    for column in REQUIRED_COLUMNS:
        if column in df.columns:
            continue
        if find_alias_column(df, column):
            continue
        missing.append(column)
    return missing


def find_alias_column(df: pd.DataFrame, column: str) -> Optional[str]:
    for alias in FEATURE_ALIASES.get(column, []):
        if alias in df.columns:
            return alias
    return None


def dataframe_to_csv_response(df: pd.DataFrame, filename: str) -> Response:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def build_output_filename(original_name: str) -> str:
    base = secure_filename(original_name) or "predictions.csv"
    if "." in base:
        name, _ = base.rsplit(".", 1)
        return f"{name}_with_predictions.csv"
    return f"{base}_with_predictions.csv"


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
