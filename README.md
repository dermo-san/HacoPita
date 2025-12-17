# HacoPita

箱ID (`predicted_box_id`) を機械学習モデルで推論し、入力CSVに列を追加してダウンロードできるWebアプリです。  
Azure ML オンラインエンドポイントを第一候補として呼び出し、設定されていない場合のみローカルの `model.pkl` にフォールバックします。Flask + gunicorn で動作し、Render Web Service を想定しています。

## リポジトリ構成

```
.
├── app.py                 # Flaskエントリポイント。Azure ML/ローカル推論を自動判別
├── schema.py              # 24特徴量やエイリアスの単一ソース
├── preprocessing.py       # build_features() で前処理を一元化
├── predictor.py           # Azure ML オンラインエンドポイント / ローカル推論のラッパー
├── train.py               # 24特徴量のみで学習し model_features.json を出力する学習スクリプト
├── model_features.json    # 学習時に保存する 24特徴量リスト (順序固定)
├── requirements.txt       # 既存のAzureML依存を含む推論環境
├── conda.yaml / python_env.yaml / MLmodel  # MLflow由来の依存メタ情報
├── model.pkl              # ローカル推論用モデル（Renderへ同梱する場合のみ使用）
├── model/                 # 元のAzureML成果物をそのまま配置
├── templates/
│   └── index.html         # アップロードフォーム
└── static/
    └── sample_input.csv   # 入力サンプル（24特徴量、box_idなし）
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

## Azure ML オンラインエンドポイント

- 環境変数 `AZUREML_ENDPOINT_URL` と `AZUREML_ENDPOINT_KEY`（または `AZUREML_ENDPOINT_TOKEN`）を設定すると Azure ML オンラインエンドポイントを呼び出します。`AZUREML_DEPLOYMENT_NAME` を指定すればデプロイメントを固定できます。
- いずれかの秘密情報が未設定の場合は自動的に `model.pkl` を読み込むローカル推論モードにフォールバックします。
- 送信payloadは `build_features()` が返す24列のみで構成され、送信前に `X.shape[1] == 24` を検証してログに列名を出力します。

## API/画面仕様

- `GET /`：CSVアップロードフォームと仕様説明を表示。サンプルCSV (`static/sample_input.csv`) をダウンロード可能。
- `POST /predict`：フォームからアップロードされたCSVを推論し、末尾に `predicted_box_id` 列を追加したCSVを添付ダウンロードとして返却。
- `POST /api/predict`：`multipart/form-data` の `file` フィールドを受け取り、
  - `?format=json` 付きでJSON（`slip_number`, `predicted_box_id` の配列）
  - それ以外はCSVを添付ダウンロード
- `slip_number` 列が存在しない/空欄でも推論結果は `row_index` とともに返却されます。

## CSV要件

- 文字コード：`utf-8-sig` を優先、失敗時は `cp932` で再読込。
- 推論に使用するのは以下24列のみ（順序固定）。入力CSVに余分な列があっても構いませんが、いずれかが欠けるとエラーになります。
  1. `total_items`
  2. `bonsai`
  3. `other`
  4. `plastic_pots_trays`
  5. `single_flower_vase`
  6. `decorative_sand`
  7. `saucers_mats`
  8. `books`
  9. `suiban`
  10. `bonsai_seeds`
  11. `for_bonsai_classes`
  12. `bonsai_soil`
  13. `bonsai_tools`
  14. `bonsai_pots`
  15. `bonsai_decorations`
  16. `lucky_bag`
  17. `moss`
  18. `moss_bonsai`
  19. `chemicals_fertilizer`
  20. `wire`
  21. `decorative_stones`
  22. `accessories`
  23. `max_item_long_cm`
  24. `sum_item_volume_cm3`
- `others` → `other`、`water_basins` → `suiban` など互換カラムは `schema.ALISES` で自動的に正規化されます。
- 24列すべてを `pd.to_numeric(errors="coerce")` → `fillna(0)` で数値化し、最終的に `X.shape[1] == 24` を `assert` しています。
- `box_id` 列は入力にあってもなくても構いません。推論に利用はされず、出力CSVではそのまま保持されます。
- 不足列があれば 400 (Bad Request) で欠損列名を返します。

## モデルとスキーマの同期

- 24列の順序を `schema.FEATURE_COLUMNS_24` に固定し、学習ジョブでは同じリストを `model_features.json` として成果物に出力します。
- 推論側はアプリ起動時に `model_features.json` を読み込み、内容が `FEATURE_COLUMNS_24` と異なる場合は起動エラーとして扱います（旧21列モデルを誤ってデプロイしている場合の早期検知）。
- `build_features()` も同じ `model_features.json` を参照し、推論payloadの真実を単一ソースに固定します。

## Azure ML モデル入れ替え手順（21 → 24列）

1. **学習スクリプトの更新**
   - `train.py` をそのまま使うか、同様のロジックを取り込んでください。内部では `X = df.reindex(columns=FEATURE_COLUMNS_24)` で入力列順を固定し、 `assert X.shape[1] == 24` / `Missing required columns` で 24 列以外なら学習を失敗させます。
   - `build_features()` を再利用すれば学習・推論で同じ前処理を使い回せます。
2. **成果物の書き出し**
   - `model.pkl`（または MLflow モデル）と同じディレクトリに `model_features.json` を保存します。内容は `FEATURE_COLUMNS_24` の順序付きリストそのものです。
   - モデルが内部クラスIDを返す場合は `label_classes.json` も同梱してください。
3. **モデル登録 & デプロイ**
   - Azure ML にモデルを登録し、`model_features.json` を含む形で新しいバージョンを Online endpoint の deployment に割り当てます（旧 21 列モデルへの参照を外す）。
4. **疎通確認**
   - `tests/test_inference.py` を通して前処理が 24 列のままになっていることを確認し、Render/本番アプリから Azure ML endpoint を呼び出して `fitted data (21)` エラーが消えていることをチェックします。
   - エンドポイントのログに送信列数 (`n_cols=24`) が出力されるため、列不一致が再発した場合も即座に検知できます。
5. **デプロイされたモデルの確認**
   - Azure ML Studio → Online endpoints → 該当 endpoint → Deployments → 使用中の deployment を開き、`Model name` / `Model version` が 24 列モデルの登録名になっているか確認します。
   - 併せて Azure ML の `scoring_file` ログ (`MODEL FILE PATH`, `MODEL NAME`, `MODEL VERSION`) にも出力されるため、旧26列モデルが誤って残っていないかログから追跡できます。

## Render用メモ

- Build：`pip install -r requirements.txt`
- Runtime：Python 3.9.23（Renderのダッシュボードで指定）
- PORT は Render により注入されるため `app.py` では `PORT` 環境変数を参照して起動

## テスト

`pytest` を実行すると `tests/test_inference.py` が以下を検証します。

- 余分な列があっても最終 payload が 24 列のみになること
- 必須24列のうち1列でも欠けると、不足列名を含む例外が発生すること
- box_id が欠けていても前処理が通ること
- 履歴互換の列名（others など）がエイリアスで補正されること
- `model_features.json`（またはその内容）で指定された順序どおりに `reindex` されること

## ライセンス

社内利用を想定しているため別途指示に従ってください。
