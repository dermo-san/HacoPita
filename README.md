# HacoPita

箱ID (`predicted_box_id`) を機械学習モデルで推論し、入力CSVに列を追加してダウンロードできるWebアプリです。Flask + gunicorn で動作し、Render Web Service を想定しています。

## リポジトリ構成

```
.
├── app.py                 # Flaskエントリポイント（PORT環境変数で待ち受け）
├── requirements.txt       # 既存のAzureML依存を含む推論環境
├── conda.yaml             # MLflow由来の依存定義
├── python_env.yaml        # MLflow由来の依存定義
├── MLmodel                # MLflowメタ情報
├── model.pkl              # 推論モデル本体（Renderへ同梱）
├── model/                 # Azure ML から取得した最新成果物（参照用）
│   ├── model.pkl
│   ├── conda_env_v_1_0_0.yml
│   └── scoring_file_v_2_0_0.py
├── expected_features.json # 特徴量リストのフォールバック（自動取得失敗時に参照）
├── data/
│   └── テスト用BoxID空欄学習データ3_サイズ情報追加版.csv # 特徴量推定用の学習データ抜粋
├── tests/
│   └── smoke_test.py      # サンプルCSVを使った簡易スモークテスト
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

- 特徴量26列の自動取得順
  1. ルート直下の `MLmodel` signature から列定義を取得（常にモデルと同期）
  2. 1で確定できなければ `model/scoring_file_v_2_0_0.py` の `data_sample` を解析
  3. それでも確定できなければ `data/テスト用BoxID空欄学習データ3_サイズ情報追加版.csv`（または `TRAINING_FEATURE_SAMPLE` で指定したCSV）のヘッダーから `box_id` を除いて推定
  4. 最後の手段として `expected_features.json`
- 文字コード：`utf-8-sig` を優先、失敗時は `cp932` で再読込。
- 入力CSVで必須なのは `slip_number` のみ（出力時に行を紐づけるため）。その他の列は可能な範囲で alias / デフォルト値補完を行いますが、学習CSVのヘッダに寄せるほど推論の再現性が上がります。
- モデルへ渡す26列（MLmodel signature 準拠）：
  `slip_number, shipment_confirmed_date, subtotal_amount, total_line_count, bonsai, others, plastic_pots_trays, single_flower_vase, decorative_sand, saucers_mats, books, water_basins, bonsai_seeds, bonsai_class_items, bonsai_soil, bonsai_tools, bonsai_pots, bonsai_decor, lucky_bag, moss, moss_bonsai, chemicals_fertilizers, wire, decorative_stones, specification, dimensions`
- `dimensions` は文字列列、`shipment_confirmed_date` は日時列として正規化し、その他の列は整数列として `pd.to_numeric(errors="coerce")` → NaN を 0 埋めします。CSVに存在しない列はログを出したうえでデフォルト値で補完します。
- `box_id`（ターゲット列）は入力CSVに含まれていても無視され、出力CSVにはそのまま残ります。

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

## ライセンス

社内利用を想定しているため別途指示に従ってください。
- `TRAINING_FEATURE_SAMPLE` 環境変数を設定すると学習データCSVの場所を上書きできます。未設定の場合は `data/テスト用BoxID空欄学習データ3_サイズ情報追加版.csv` とリポジトリ内の `*学習データ*.csv` を自動探索します。
