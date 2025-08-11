import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    """複数のモデルを学習"""
    models = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # Logistic Regression（データを標準化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = (lr, scaler)
    
    # SVM
    svm = SVC(random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)
    models['SVM'] = (svm, scaler)
    
    print("モデル学習完了")
    return models

def evaluate_models(models, X_test, y_test):
    """モデルの評価とRecallの算出"""
    results = {}
    
    for model_name, model in models.items():
        print(f"\n=== {model_name} の評価結果 ===")
        
        # モデルとスケーラーの処理
        if isinstance(model, tuple):
            classifier, scaler = model
            X_test_processed = scaler.transform(X_test)
        else:
            classifier = model
            X_test_processed = X_test
        
        # 予測
        y_pred = classifier.predict(X_test_processed)
        
        # 評価指標の計算
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)  # 重要: Recallの算出
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
        print(f"Recall: {recall:.4f}")  # Recallを表示
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
    """モデルの比較"""
    comparison_df = pd.DataFrame(results).T
    
    print("\n=== モデル比較 ===")
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1']].round(4))
    
    # Recallでソートして最良モデルを特定
    best_model = comparison_df['recall'].idxmax()
    best_recall = comparison_df.loc[best_model, 'recall']
    
    print(f"\n最高Recall: {best_model} (Recall: {best_recall:.4f})")
    
    # 視覚的比較
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(comparison_df.index))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, comparison_df[metric], width, label=metric)
    
    plt.xlabel('モデル')
    plt.ylabel('スコア')
    plt.title('モデル性能比較')
    plt.xticks(x + width*1.5, comparison_df.index)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
    
    return comparison_df

def main():
    """メイン実行関数"""
    print("=== 2クラス分類分析開始 ===")
    
    # 1. データ読み込み
    df = load_data('binary_classification_data.csv')
    
    # 2. データ分割
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # 3. モデル学習
    models = train_models(X_train, y_train)
    
    # 4. モデル評価とRecall算出
    results = evaluate_models(models, X_test, y_test)
    
    # 5. モデル比較
    comparison_df = compare_models(results)
    
    print("\n=== 分析完了 ===")
    return results, comparison_df

if __name__ == "__main__":
    results, comparison = main()