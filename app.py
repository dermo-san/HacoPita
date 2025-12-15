import io
import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

REQUIRED_COLUMNS: List[str] = [
    "slip_number",
    "shipment_confirmed_date",
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
    "dimensions",
]

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
EXPECTED_FEATURES: List[str] = [
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
]
FALLBACK_DATE = pd.Timestamp("1970-01-01")
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
LOGGER = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    model, model_error = load_model()

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", model_error=model_error)

    @app.route("/predict", methods=["POST"])
    def predict():
        if model_error:
            return (
                render_template("index.html", error_message=model_error, model_error=model_error),
                500,
            )

        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return (
                render_template(
                    "index.html",
                    error_message="CSVファイルを選択してください。",
                    model_error=model_error,
                ),
                400,
            )

        try:
            result_df, predictions = process_file(uploaded_file, model)
        except PredictionError as exc:
            return (
                render_template(
                    "index.html",
                    error_message=exc.message,
                    model_error=model_error,
                ),
                exc.status_code,
            )

        csv_response = dataframe_to_csv_response(
            result_df, build_output_filename(uploaded_file.filename)
        )
        return csv_response

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        if model_error:
            return jsonify({"error": model_error}), 500

        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"error": "file field is required"}), 400

        try:
            result_df, predictions = process_file(uploaded_file, model)
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


class PredictionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def process_file(file_storage, model) -> Tuple[pd.DataFrame, List[int]]:
    input_df = read_csv_with_fallback(file_storage)
    missing = [col for col in REQUIRED_COLUMNS if col not in input_df.columns]
    if missing:
        raise PredictionError(
            f"必須列が不足しています: {', '.join(missing)}", status_code=400
        )

    normalized_df = normalize_input_dataframe(input_df)
    features = prepare_model_input(normalized_df)

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
    result_df = normalized_df.copy()
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
        normalized[column] = (
            pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype("int64")
        )

    for column in DATETIME_COLUMNS:
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce").fillna(
            FALLBACK_DATE
        )

    for column in STRING_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str)

    return normalized


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing:
        raise PredictionError(
            f"モデルに必要な列が不足しています: {', '.join(missing)}", status_code=400
        )

    features = df.reindex(columns=EXPECTED_FEATURES).copy()
    for column in EXPECTED_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)

    return features.astype("float64")


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
