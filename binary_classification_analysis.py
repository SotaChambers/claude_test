import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, recall_score,
    precision_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

def load_data(csv_path):
    """CSVデータを読み込む"""
    df = pd.read_csv(csv_path)
    print(f"データ形状: {df.shape}")
    print(f"クラス分布:")
    print(df['target'].value_counts())
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """学習データと評価データに分割"""
    # 特徴量とターゲットを分離
    X = df.drop('target', axis=1)
    y = df['target']
    # 学習データと評価データに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"学習データ形状: {X_train.shape}")
    print(f"評価データ形状: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """LightGBMモデルを学習"""
    models = {}

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        objective='binary',
        metric='binary_logloss',
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model

    print("LightGBMモデル学習完了")
    return models

def evaluate_models(models, X_test, y_test):
    """LightGBMモデルの評価とRecallの算出"""
    results = {}

    for model_name, model in models.items():
        print(f"\n=== {model_name} の評価結果 ===")

        # 予測
        y_pred = model.predict(X_test)

        # 評価指標の計算
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 結果を保存
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # 詳細なレポート
        print("\n詳細な分類レポート:")
        print(classification_report(y_test, y_pred))

        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - 混同行列')
        plt.ylabel('実際のクラス')
        plt.xlabel('予測クラス')
        plt.show()

    return results

def compare_models(results):
    """LightGBMモデルの結果表示"""
    comparison_df = pd.DataFrame(results).T

    print("\n=== LightGBMモデル結果 ===")
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1']].round(4))

    # 結果をグラフで表示
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [comparison_df.iloc[0][metric] for metric in metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'orange'])
    plt.ylim(0, 1)
    plt.title('LightGBM性能指標')
    plt.ylabel('スコア')
    plt.xlabel('評価指標')

    # 各バーの上に値を表示
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    plt.show()

    return comparison_df

def main():
    """メイン実行関数"""
    print("=== LightGBMを用いた2クラス分類分析開始 ===")

    # 1. データ読み込み
    df = load_data('binary_classification_data.csv')

    # 2. データ分割
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 3. LightGBMモデル学習
    models = train_models(X_train, y_train)

    # 4. モデル評価とRecall算出
    results = evaluate_models(models, X_test, y_test)

    # 5. 結果表示
    comparison_df = compare_models(results)

    print("\n=== LightGBM分析完了 ===")
    return results, comparison_df

if __name__ == "__main__":
    results, comparison = main()