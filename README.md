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
├── model/                 # 元のAzureML成果物をそのまま配置
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

- Python ランタイム：`runtime.txt` および `.python-version` で `3.9.23` を明示しているため、Render 側でも必ず 3.9 系を利用してください（3.12 などでビルドすると AzureML 依存が解決できません）。

## API/画面仕様

- `GET /`：CSVアップロードフォームと仕様説明を表示。サンプルCSV (`static/sample_input.csv`) をダウンロード可能。
- `POST /predict`：フォームからアップロードされたCSVを推論し、末尾に `predicted_box_id` 列を追加したCSVを添付ダウンロードとして返却。
- `POST /api/predict`：`multipart/form-data` の `file` フィールドを受け取り、
  - `?format=json` 付きでJSON（`slip_number`, `predicted_box_id` の配列）
  - それ以外はCSVを添付ダウンロード

## CSV要件

- 文字コード：`utf-8-sig` を優先、失敗時は `cp932` で再読込。
- 必須26列（順序固定）：
  `slip_number, shipment_confirmed_date, subtotal_amount, total_line_count, bonsai, others, plastic_pots_trays, single_flower_vase, decorative_sand, saucers_mats, books, water_basins, bonsai_seeds, bonsai_class_items, bonsai_soil, bonsai_tools, bonsai_pots, bonsai_decor, lucky_bag, moss, moss_bonsai, chemicals_fertilizers, wire, decorative_stones, specification, dimensions`。
- 型変換：整数列は `int64`（欠損・不正値は0置換）、日時列は `pd.Timestamp("1970-01-01")` で補完、文字列列は空文字で補完。
- 追加列は推論には使用せず、出力CSVでは保持します。
- 不足列があれば 400 (Bad Request) で欠損列名を返します。

## モデルロード

- アプリ起動時に `model.pkl` を読み込み、リクエストごとに再ロードは行いません。
- 読み込み失敗や推論例外時は、ユーザー向けにエラーメッセージを返し、アプリを落としません。

## Render用メモ

- Build：`pip install -r requirements.txt`
- Runtime：Python 3.9.23（Renderのダッシュボードで指定）
- PORT は Render により注入されるため `app.py` では `PORT` 環境変数を参照して起動

## ライセンス

社内利用を想定しているため別途指示に従ってください。
