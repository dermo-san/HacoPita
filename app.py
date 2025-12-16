import io
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from preprocessing import build_features
from predictor import PredictorConfigurationError, build_predictor
from schema import FEATURE_COLUMNS_24, ID_COLUMN, load_feature_columns

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def create_app() -> Flask:
    app = Flask(__name__)

    feature_columns, schema_error = resolve_feature_columns()
    predictor, predictor_error = initialize_predictor()
    model_error = predictor_error or schema_error

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            model_error=model_error,
            error_message=None,
            feature_columns=feature_columns,
            feature_column_count=len(feature_columns),
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if model_error:
            return (
                render_template(
                    "index.html",
                    error_message=model_error,
                    model_error=model_error,
                    feature_columns=feature_columns,
                    feature_column_count=len(feature_columns),
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
                    feature_columns=feature_columns,
                    feature_column_count=len(feature_columns),
                ),
                400,
            )

        try:
            result_df, predictions = process_file(
                uploaded_file, predictor, feature_columns
            )
        except PredictionError as exc:
            return (
                render_template(
                    "index.html",
                    error_message=exc.message,
                    model_error=model_error,
                    feature_columns=feature_columns,
                    feature_column_count=len(feature_columns),
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
            result_df, predictions = process_file(
                uploaded_file, predictor, feature_columns
            )
        except PredictionError as exc:
            return jsonify({"error": exc.message}), exc.status_code

        if request.args.get("format") == "json":
            payload = build_api_payload(result_df, predictions)
            return jsonify({"predictions": payload})

        return dataframe_to_csv_response(
            result_df, build_output_filename(uploaded_file.filename)
        )

    return app


def resolve_feature_columns() -> Tuple[List[str], str]:
    try:
        columns = load_feature_columns()
        return columns, ""
    except ValueError as exc:
        logging.error("Feature column validation failed: %s", exc)
        return FEATURE_COLUMNS_24, str(exc)


def initialize_predictor():
    try:
        predictor = build_predictor(MODEL_PATH)
        return predictor, ""
    except PredictorConfigurationError as exc:
        logging.error("Predictor initialization failed: %s", exc)
        return None, str(exc)


def process_file(file_storage, predictor, feature_columns):
    input_df = read_csv_with_fallback(file_storage)

    try:
        features = build_features(input_df, feature_columns=feature_columns)
    except ValueError as exc:
        raise PredictionError(str(exc), status_code=400) from exc
    except AssertionError as exc:
        raise PredictionError(str(exc), status_code=500) from exc

    try:
        raw_predictions = predictor.predict(features)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Predictor failed to return results.")
        raise PredictionError(f"推論に失敗しました: {exc}", status_code=502) from exc

    flattened = np.asarray(raw_predictions).reshape(-1)
    predictions = pd.Series(flattened).fillna(0).astype("int64")

    if len(predictions) != len(input_df):
        raise PredictionError(
            "推論件数が入力件数と一致しません。Azure ML側のログを確認してください。",
            status_code=502,
        )

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
    ) from last_error


def build_api_payload(result_df: pd.DataFrame, predictions: List[int]) -> List[dict]:
    payload = []
    id_available = ID_COLUMN in result_df.columns
    id_series = (
        pd.to_numeric(result_df[ID_COLUMN], errors="coerce") if id_available else None
    )

    for i, pred in enumerate(predictions):
        record = {"predicted_box_id": int(pred)}
        if id_available and pd.notna(id_series.iloc[i]):
            record[ID_COLUMN] = int(id_series.iloc[i])
        else:
            record["row_index"] = i
        payload.append(record)
    return payload


class PredictionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


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
