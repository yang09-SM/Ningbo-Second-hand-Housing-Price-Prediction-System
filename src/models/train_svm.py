import pandas as pd
import numpy as np
import os
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def load_data(train_path, val_path, target_col='总价'):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    return X_train, y_train, X_val, y_val

def train_model(X_train, y_train):
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, model_path)

def main():
    train_path = os.path.join('data', 'train.csv')
    val_path = os.path.join('data', 'val.csv')
    model_path = os.path.join('models', 'svm.joblib')
    
    print("=" * 80)
    print("宁波二手房价格预测 - SVM 模型训练")
    print("=" * 80)
    
    print("\n1. 正在加载训练数据...")
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)
    print(f"   训练集: {len(X_train)} 条")
    print(f"   验证集: {len(X_val)} 条")
    print(f"   特征数: {X_train.shape[1]} 个")
    
    print("\n2. 正在标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("\n3. 正在训练 SVM 模型...")
    model = train_model(X_train_scaled, y_train)
    
    print("\n4. 正在评估模型...")
    train_score = model.score(X_train_scaled, y_train)
    val_score = model.score(X_val_scaled, y_val)
    print(f"   训练集 R²: {train_score:.4f}")
    print(f"   验证集 R²: {val_score:.4f}")
    
    print("\n5. 正在保存模型...")
    save_model(model, scaler, model_path)
    
    print(f"\n" + "=" * 80)
    print(f"SVM 模型训练完成!")
    print(f"  - 模型已保存至: {os.path.abspath(model_path)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
