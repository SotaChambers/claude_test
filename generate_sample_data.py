import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# シード値を設定して再現可能にする
np.random.seed(42)

# 2クラス分類用のサンプルデータを作成
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42
)

# 特徴量の名前を設定
feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]

# データフレームを作成
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# CSVファイルとして保存
df.to_csv('binary_classification_data.csv', index=False)

print(f"データセットを作成しました:")
print(f"形状: {df.shape}")
print(f"クラス分布:")
print(df['target'].value_counts())
print(f"\nCSVファイル 'binary_classification_data.csv' を保存しました。")