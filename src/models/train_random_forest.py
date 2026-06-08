import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
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
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, model_path)

def main():
    train_path = os.path.join('data', 'train.csv')
    val_path = os.path.join('data', 'val.csv')
    model_path = os.path.join('models', 'random_forest.joblib')
    
    print("=" * 80)
    print("宁波二手房价格预测 - 随机森林模型训练")
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
    
    print("\n3. 正在训练随机森林模型...")
    model = train_model(X_train_scaled, y_train)
    
    print("\n4. 正在评估模型...")
    train_score = model.score(X_train_scaled, y_train)
    val_score = model.score(X_val_scaled, y_val)
    print(f"   训练集 R²: {train_score:.4f}")
    print(f"   验证集 R²: {val_score:.4f}")
    
    print("\n5. 正在保存模型...")
    save_model(model, scaler, model_path)
    
    print(f"\n" + "=" * 80)
    print(f"随机森林模型训练完成!")
    print(f"  - 模型已保存至: {os.path.abspath(model_path)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
