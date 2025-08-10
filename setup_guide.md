# データ分析環境セットアップガイド

このガイドでは、Pythonを使用したデータ分析環境の構築方法を説明します。

## 必要な環境

- Python 3.8以上
- pip（パッケージ管理ツール）

## セットアップ手順

### 1. 仮想環境の作成と有効化

```bash
# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化（macOS/Linux）
source venv/bin/activate

# 仮想環境の有効化（Windows）
# venv\Scripts\activate
```

### 2. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

### 3. Jupyter Labの起動

```bash
jupyter lab
```

または

```bash
jupyter notebook
```

## インストールされるライブラリ

### 基本的なデータ分析ライブラリ
- **pandas**: データフレーム操作とデータ分析
- **numpy**: 数値計算
- **scipy**: 統計分析と科学計算

### 可視化ライブラリ
- **matplotlib**: 基本的なグラフ作成
- **seaborn**: 統計的な可視化
- **plotly**: インタラクティブな可視化

### 機械学習ライブラリ
- **scikit-learn**: 機械学習アルゴリズム

### ファイル処理
- **openpyxl**: Excelファイルの読み書き

### 開発環境
- **jupyter**: Jupyter Notebook
- **jupyterlab**: Jupyter Lab（より高機能なNotebook環境）

## サンプルNotebookの使用方法

1. `data_analysis_sample.ipynb`をJupyter Lab/Notebookで開く
2. セルを上から順番に実行（Shift+Enterまたは実行ボタン）
3. 各セクションの説明を読みながら分析手法を学習

## 実際のデータでの分析

サンプルNotebookでは自動生成されたデータを使用していますが、実際のCSVファイルやExcelファイルを使用する場合は以下のようにデータを読み込みます：

```python
# CSVファイルの読み込み
df = pd.read_csv('your_data.csv')

# Excelファイルの読み込み
df = pd.read_excel('your_data.xlsx')
```

## トラブルシューティング

### インストールエラーが発生する場合

```bash
# pipのアップグレード
pip install --upgrade pip

# 個別にライブラリをインストール
pip install pandas numpy matplotlib seaborn
```

### Jupyter Labが起動しない場合

```bash
# Jupyter Labの再インストール
pip uninstall jupyterlab
pip install jupyterlab
```

### 仮想環境の無効化

```bash
deactivate
```

## 参考資料

- [Pandas公式ドキュメント](https://pandas.pydata.org/docs/)
- [Matplotlib公式ドキュメント](https://matplotlib.org/stable/contents.html)
- [Seaborn公式ドキュメント](https://seaborn.pydata.org/)
- [Scikit-learn公式ドキュメント](https://scikit-learn.org/stable/)