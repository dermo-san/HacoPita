# 変更内容サマリー

## 概要

推論プログラムの不具合を解消し、以下の要件を満たすようにリファクタリングしました：

1. 推論の前処理を学習時の特徴量定義と100%一致させる
2. 予測ラベルが内部クラスIDの場合は必ずbox_idに逆変換する
3. 入力CSVに余分な列があっても推論できるようにする

## 主な変更点

### 1. 新規ファイル

#### `schema.py`
- 特徴量スキーマを固定定義
- `FEATURE_COLUMNS`: 必須特徴量23列（順序固定）
- `DROP_COLUMNS`: 除外列の定義
- `TARGET_COLUMN`: ターゲット列（box_id）
- `REQUIRED_COLUMNS`: 必須列（slip_number）

#### `preprocessing.py`
- `prepare_features()`: 入力CSVから必須特徴量を抽出・正規化
  - 必須列チェック
  - 型変換（数値列は自動変換、NaNは0埋め）
  - FEATURE_COLUMNSの順序で抽出
- `get_training_box_id_set()`: 学習データからbox_idの集合を取得

#### `label_decoder.py`
- `decode_predictions()`: 予測値をbox_idにデコード
  - 予測値が既にbox_idか内部クラスIDかを自動判定
  - 内部クラスIDの場合は逆変換してbox_idに戻す
  - 逆変換後のbox_idが学習データに存在しない場合はエラー
- `load_label_classes()`: ラベルクラスをJSONから読み込む
- `save_label_classes_from_model()`: モデルからラベルクラスを取得して保存
- `save_label_classes_from_training_data()`: 学習データからラベルクラスを生成して保存

#### `evaluate_predictions.py`
- 予測結果の評価スクリプト
- `slip_number`で`box_id`と`predicted_box_id`を突合し、Accuracyを算出

#### `tests/test_inference.py`
- 推論プログラムのテストコード
- 要件を満たしていることを確認

### 2. 変更されたファイル

#### `app.py`
- 大幅にリファクタリング
- 古い特徴量定義（26列）を削除し、`schema.py`からインポート
- `process_file()`を修正：
  - `prepare_features()`を使用
  - `decode_predictions()`を使用して予測値をbox_idに変換
- `load_label_classes_safe()`: ラベルクラスを安全に読み込む
- `get_training_box_id_set_safe()`: 学習データからbox_idの集合を取得
- 古い前処理関数を削除（`prepare_model_input()`, `normalize_input_dataframe()`等）

#### `README.md`
- 変更内容を反映
- 特徴量スキーマの説明を追加
- ラベルエンコードの逆変換の説明を追加
- 評価方法の説明を追加

## 解決した問題

### 1. 特徴量スキーマの不一致
- **問題**: 古いコードでは26列の特徴量を使用していたが、実際の学習データとは異なっていた
- **解決**: `schema.py`で正しい23列の特徴量を固定定義し、学習時と完全に一致

### 2. ラベルエンコードの逆変換漏れ
- **問題**: 予測値が内部クラスID（0, 1, 2, ...）のまま出力され、box_idと一致していなかった
- **解決**: `label_decoder.py`で自動的にbox_idに逆変換

### 3. 入力CSVの柔軟性
- **問題**: 除外列が含まれているとエラーになる可能性があった
- **解決**: 除外列は無視し、必須特徴量のみを抽出

### 4. 型変換の問題
- **問題**: 型ズレで全列がobjectになる可能性があった
- **解決**: `prepare_features()`で明示的に型変換を実行

## 使用方法

### 推論の実行

```bash
# アプリを起動
python app.py

# または
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
```

### 評価の実行

```bash
# 予測結果を評価
python evaluate_predictions.py predictions.csv [--ground-truth ground_truth.csv]
```

### テストの実行

```bash
# 推論プログラムのテスト
pytest tests/test_inference.py -v
```

## 注意事項

1. **ラベルクラスの生成**: 初回起動時に`label_classes.json`が自動生成されます。モデルの`classes_`属性から取得できない場合は、学習データから生成されます。

2. **学習データのパス**: 学習データは `data/テスト用BoxID空欄学習データ3_サイズ情報追加版.csv` を想定しています。

3. **必須特徴量**: 入力CSVには23列の必須特徴量がすべて含まれている必要があります。欠けている場合はエラーになります。

4. **除外列**: 除外列（`slip_number`, `accessories`, `product_codes`等）は無視されますが、`slip_number`は突合キーとして必要です。

## 今後の改善点

- [ ] ラベルクラスの自動検出精度の向上
- [ ] エラーメッセージの改善
- [ ] ログ出力の最適化
- [ ] パフォーマンスの最適化

