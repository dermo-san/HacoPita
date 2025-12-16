import io
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from label_decoder import (
    decode_predictions,
    load_label_classes,
    save_label_classes_from_model,
    save_label_classes_from_training_data,
)
from preprocessing import get_training_box_id_set, prepare_features
from schema import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
TRAINING_CSV_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "テスト用BoxID空欄学習データ3_サイズ情報追加版.csv"
)
LOGGER = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    model, model_error = load_model()
    label_classes, label_error = load_label_classes_safe()
    train_box_id_set = get_training_box_id_set_safe()

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            model_error=model_error,
            feature_error=label_error,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if model_error or label_error:
            return (
                render_template(
                    "index.html",
                    error_message=model_error or label_error,
                    model_error=model_error,
                    feature_error=label_error,
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
                    feature_error=label_error,
                ),
                400,
            )

        try:
            result_df, predictions = process_file(
                uploaded_file, model, label_classes, train_box_id_set
            )
        except PredictionError as exc:
            return (
                render_template(
                    "index.html",
                    error_message=exc.message,
                    model_error=model_error,
                    feature_error=label_error,
                ),
                exc.status_code,
            )

        csv_response = dataframe_to_csv_response(
            result_df, build_output_filename(uploaded_file.filename)
        )
        return csv_response

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        if model_error or label_error:
            return jsonify({"error": model_error or label_error}), 500

        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"error": "file field is required"}), 400

        try:
            result_df, predictions = process_file(
                uploaded_file, model, label_classes, train_box_id_set
            )
        except PredictionError as exc:
            return jsonify({"error": exc.message}), exc.status_code

        if request.args.get("format") == "json":
            payload = [
                {
                    "slip_number": int(result_df.iloc[i]["slip_number"]),
                    "predicted_box_id": str(pred),  # box_idは文字列として扱う
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
        LOGGER.error(message)
        return None, message

    try:
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)
        
        # モデルからラベルクラスを取得して保存（初回のみ）
        try:
            from label_decoder import LABEL_CLASSES_JSON_PATH
            if not LABEL_CLASSES_JSON_PATH.exists():
                save_label_classes_from_model(model)
        except Exception as exc:
            LOGGER.warning(
                "Failed to save label classes from model: %s", exc
            )
        
        return model, ""
    except Exception as exc:  # pragma: no cover - defensive logging
        message = f"モデルの読み込みに失敗しました: {exc}"
        LOGGER.exception(message)
        return None, message


def load_label_classes_safe() -> Tuple[List[str], str]:
    """
    ラベルクラスを安全に読み込む
    モデルから取得できない場合は学習データから取得を試みる
    """
    try:
        label_classes = load_label_classes()
        return label_classes, ""
    except FileNotFoundError:
        # label_classes.jsonが存在しない場合、学習データから生成を試みる
        LOGGER.info(
            "label_classes.json not found. Attempting to generate from training data..."
        )
        try:
            if TRAINING_CSV_PATH.exists():
                label_classes = save_label_classes_from_training_data(
                    str(TRAINING_CSV_PATH)
                )
                return label_classes, ""
            else:
                return [], (
                    f"label_classes.json not found and training CSV not found: {TRAINING_CSV_PATH}"
                )
        except Exception as exc:
            return [], f"Failed to load label classes: {exc}"
    except Exception as exc:
        return [], f"Failed to load label classes: {exc}"


def get_training_box_id_set_safe() -> set:
    """
    学習データからbox_idの集合を安全に取得する
    """
    try:
        if TRAINING_CSV_PATH.exists():
            return get_training_box_id_set(str(TRAINING_CSV_PATH))
        else:
            LOGGER.warning(
                "Training CSV not found: %s. Cannot validate predicted box_ids.",
                TRAINING_CSV_PATH
            )
            return set()
    except Exception as exc:
        LOGGER.warning(
            "Failed to load training box_id set: %s", exc
        )
        return set()




class PredictionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def process_file(
    file_storage,
    model,
    label_classes: List[str],
    train_box_id_set: set,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    推論処理のメイン関数
    
    Args:
        file_storage: アップロードされたファイル
        model: 学習済みモデル
        label_classes: ラベルクラス（box_idのリスト）
        train_box_id_set: 学習データに存在するbox_idの集合
        
    Returns:
        (result_df, predictions): 結果DataFrameと予測box_idのリスト
    """
    if not label_classes:
        raise PredictionError(
            "ラベルクラスが取得できませんでした。", status_code=500
        )

    input_df = read_csv_with_fallback(file_storage)

    LOGGER.info(
        "Input CSV columns=%s (count=%s)",
        list(input_df.columns),
        len(input_df.columns),
    )

    # 必須特徴量の抽出・正規化
    try:
        features = prepare_features(input_df)
    except ValueError as exc:
        raise PredictionError(str(exc), status_code=400) from exc

    if features.shape[1] != len(FEATURE_COLUMNS):
        raise PredictionError(
            f"モデル入力の列数({features.shape[1]})が期待値({len(FEATURE_COLUMNS)})と一致しません。",
            status_code=500,
        )

    # 推論実行
    try:
        LOGGER.info(
            "Model input shape=%s columns=%s",
            features.shape,
            list(features.columns),
        )
        raw_predictions = model.predict(features)
    except Exception as exc:
        raise PredictionError(f"推論に失敗しました: {exc}", status_code=500) from exc

    # 予測値をbox_idにデコード
    try:
        decoded_predictions = decode_predictions(
            raw_predictions, label_classes, train_box_id_set
        )
    except ValueError as exc:
        raise PredictionError(
            f"予測ラベルのデコードに失敗しました: {exc}", status_code=500
        ) from exc

    # 結果DataFrameに予測値を追加
    result_df = input_df.copy()
    result_df["predicted_box_id"] = decoded_predictions

    return result_df, decoded_predictions


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
