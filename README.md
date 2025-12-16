# HacoPita

箱ID (`predicted_box_id`) を機械学習モデルで推論し、入力CSVに列を追加してダウンロードできるWebアプリです。Flask + gunicorn で動作し、Render Web Service を想定しています。

## リポジトリ構成

```
.
├── app.py                 # Flaskエントリポイント（PORT環境変数で待ち受け）
├── schema.py              # 特徴量スキーマ定義（FEATURE_COLUMNS, DROP_COLUMNS等）
├── preprocessing.py       # 前処理関数（prepare_features）
├── label_decoder.py       # ラベルデコード関数（内部クラスID→box_id逆変換）
├── evaluate_predictions.py # 評価スクリプト（Accuracy算出）
├── requirements.txt       # 既存のAzureML依存を含む推論環境
├── conda.yaml             # MLflow由来の依存定義
├── python_env.yaml        # MLflow由来の依存定義
├── MLmodel                # MLflowメタ情報
├── model.pkl              # 推論モデル本体（Renderへ同梱）
├── label_classes.json     # ラベルクラス（box_idのリスト、自動生成）
├── model/                 # Azure ML から取得した最新成果物（参照用）
│   ├── model.pkl
│   ├── conda_env_v_1_0_0.yml
│   └── scoring_file_v_2_0_0.py
├── expected_features.json # 特徴量リストのフォールバック（自動取得失敗時に参照）
├── data/
│   └── テスト用BoxID空欄学習データ3_サイズ情報追加版.csv # 特徴量推定用の学習データ抜粋
├── tests/
│   ├── smoke_test.py      # サンプルCSVを使った簡易スモークテスト
│   └── test_inference.py  # 推論プログラムのテスト
├── templates/
│   └── index.html         # アップロードフォーム
└── static/
    └── sample_input.csv   # 入力サンプル
```

## 実行方法

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py  # http://localhost:5000
```

Render での起動コマンド例：

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
```

- Python ランタイム：`runtime.txt` および `.python-version` で `3.10.19` を明示しているため、Render 側でも必ず 3.10 系を利用してください（3.12 などでビルドすると AzureML 依存が解決できません）。

## API/画面仕様

- `GET /`：CSVアップロードフォームと仕様説明を表示。サンプルCSV (`static/sample_input.csv`) をダウンロード可能。
- `POST /predict`：フォームからアップロードされたCSVを推論し、末尾に `predicted_box_id` 列を追加したCSVを添付ダウンロードとして返却。
- `POST /api/predict`：`multipart/form-data` の `file` フィールドを受け取り、
  - `?format=json` 付きでJSON（`slip_number`, `predicted_box_id` の配列）
  - それ以外はCSVを添付ダウンロード

## 特徴量スキーマ

**重要**: 特徴量スキーマは `schema.py` で固定されています。学習時に使用した特徴量と完全に一致させています。

### 必須特徴量（23列、この順序で固定）

`total_items`, `bonsai`, `other`, `plastic_pots_trays`, `single_flower_vase`, `decorative_sand`, `saucers_mats`, `books`, `suiban`, `bonsai_seeds`, `for_bonsai_classes`, `bonsai_soil`, `bonsai_tools`, `bonsai_pots`, `bonsai_decorations`, `lucky_bag`, `moss`, `moss_bonsai`, `chemicals_fertilizer`, `wire`, `decorative_stones`, `max_item_long_cm`, `sum_item_volume_cm3`

### 除外列（入力CSVにあっても無視される）

`slip_number`（突合キーとして保持）、`accessories`, `product_codes`, `sizes_raw`, `max_item_mid_cm`, `max_item_short_cm`, `sum_item_area_cm2`, `avg_item_long_cm`, `unique_items`

### 入力CSVの要件

- **必須列**: `slip_number`（突合キーとして必要）
- **必須特徴量**: 上記23列がすべて存在する必要があります
- **余分な列**: 除外列やその他の列が含まれていても問題ありません（無視されます）
- **型変換**: 数値列は自動的に数値型に変換され、NaNは0で埋められます
- **文字コード**: `utf-8-sig` を優先、失敗時は `cp932` で再読込

## ラベルエンコードの逆変換

モデルが内部クラスID（0, 1, 2, ...）を返す場合、自動的にbox_idに逆変換されます。

- `label_classes.json` にラベルクラス（box_idのリスト）が保存されます
- **重要**: `label_classes.json` は学習成果物として必ず同梱する必要があります。推論側では生成しません。
- 予測値が既にbox_idの場合はそのまま使用されます
- 逆変換後のbox_idが学習データに存在しない場合はエラーになります

### 出力形式の明示的指定

環境変数 `OUTPUT_IS_CLASS_INDEX` で予測値の形式を明示的に指定できます：

- `OUTPUT_IS_CLASS_INDEX=true` または `1`: 予測値は常に内部クラスIDとして扱う（逆変換を実行）
- `OUTPUT_IS_CLASS_INDEX=false` または `0`: 予測値は常にbox_idとして扱う（そのまま返す）
- 未設定: 自動判定（既存のロジック）

## 評価

予測結果を評価するには `evaluate_predictions.py` を使用します：

```bash
python evaluate_predictions.py predictions.csv [--ground-truth ground_truth.csv]
```

- `slip_number` で `box_id` と `predicted_box_id` を突合し、Accuracyを算出します
- `--ground-truth` を指定しない場合、`predictions.csv` から `box_id` を取得します

## モデルロード

- アプリ起動時に `model.pkl` を読み込み、リクエストごとに再ロードは行いません。
- 読み込み失敗や推論例外時は、ユーザー向けにエラーメッセージを返し、アプリを落としません。

## Render用メモ

- Build：`pip install -r requirements.txt`
- Runtime：Python 3.10.19（Renderのダッシュボードで指定）
- PORT は Render により注入されるため `app.py` では `PORT` 環境変数を参照して起動

## テスト

簡易スモークテスト（サンプルCSV＋新モデルで前処理～整形を確認）：

```bash
python tests/smoke_test.py
```

推論プログラムのテスト（pytest）：

```bash
pytest tests/test_inference.py -v
```

テスト内容：
- 入力CSVに除外列が含まれても推論できる
- 必須列が欠けたらエラーになる
- 内部クラスIDをbox_idに逆変換できる
- 予測値が既にbox_idの場合はそのまま使用される
- FEATURE_COLUMNSの順序で抽出される
- 型変換が正しく行われる

## ライセンス

社内利用を想定しているため別途指示に従ってください。
- `TRAINING_FEATURE_SAMPLE` 環境変数を設定すると学習データCSVの場所を上書きできます。未設定の場合は `data/テスト用BoxID空欄学習データ3_サイズ情報追加版.csv` とリポジトリ内の `*学習データ*.csv` を自動探索します。
