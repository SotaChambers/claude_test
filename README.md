# データ分析プロジェクト

このプロジェクトはPythonを使用したデータ分析環境と2クラス分類の実装例を提供します。

## 構成

### データ分析環境
- `requirements.txt`: データ分析に必要なライブラリ
- `setup_guide.md`: 環境構築ガイド
- `data_analysis_sample.ipynb`: データ分析のサンプルNotebook
- `venv/`: Python仮想環境

### 2クラス分類
- `binary_classification_analysis.py`: LightGBMを使用した2クラス分類のメイン分析スクリプト
- `binary_classification_notebook.ipynb`: 2クラス分類のJupyter Notebook
- `generate_sample_data.py`: サンプルデータ生成スクリプト
- `binary_classification_data.csv`: 分析用サンプルデータ

## セットアップ

```bash
# 仮想環境の有効化
source venv/bin/activate

# ライブラリのインストール
pip install -r requirements.txt

# Jupyter Labの起動
jupyter lab
```

## 実行方法

### 2クラス分類分析
```bash
# Pythonスクリプトで実行
python binary_classification_analysis.py

# またはJupyter Notebookで実行
jupyter lab binary_classification_notebook.ipynb
```

### データ分析サンプル
```bash
jupyter lab data_analysis_sample.ipynb
```

## 主な機能

- **データ可視化**: matplotlib, seaborn, plotlyを使用
- **統計分析**: pandas, numpy, scipyを使用  
- **機械学習**: scikit-learn、LightGBMを使用した分類・回帰・クラスタリング
- **2クラス分類**: LightGBMによる高速で高精度な分類
- **評価指標**: Recall, Precision, F1-Score, Accuracyの算出

## 変更履歴

### 2025-08-11: LightGBMへの移行
- 2値分類モデルをRandom Forest、Logistic Regression、SVMからLightGBMに変更
- スケーラビリティと精度の向上を実現
- LightGBM 4.1.0をrequirements.txtに追加
