import pandas as pd
import numpy as np
import os
import joblib
import optuna
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_data(train_path, val_path, target_col='总价'):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    return X_train, y_train, X_val, y_val


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    return r2


def save_model(model, scaler, best_params, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'best_params': best_params}, model_path)


def main():
    train_path = os.path.join('data', 'train.csv')
    val_path = os.path.join('data', 'val.csv')
    model_path = os.path.join('models', 'lightgbm.joblib')

    print("=" * 80)
    print("宁波二手房价格预测 - LightGBM 模型训练（Optuna 超参数搜索）")
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

    print("\n3. 正在使用 Optuna 进行超参数搜索（50次试验，目标：最大化验证集 R²）...")
    study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val),
                   n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value
    print(f"\n   最优 R²: {best_value:.4f}")
    print(f"   最优参数: {best_params}")

    print("\n4. 使用最优参数重新训练最终模型...")
    final_params = best_params.copy()
    final_params['random_state'] = 42
    final_params['n_jobs'] = -1
    final_params['verbose'] = -1
    model = LGBMRegressor(**final_params)
    model.fit(X_train_scaled, y_train)

    print("\n5. 正在评估最终模型...")
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)

    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"   训练集 R²:     {train_r2:.4f}")
    print(f"   验证集 R²:     {val_r2:.4f}")
    print(f"   验证集 MAE:    {val_mae:.4f}")
    print(f"   验证集 RMSE:   {val_rmse:.4f}")

    print("\n6. 正在保存模型...")
    save_model(model, scaler, best_params, model_path)

    print(f"\n" + "=" * 80)
    print(f"LightGBM 模型训练完成!")
    print(f"  - 模型已保存至: {os.path.abspath(model_path)}")
    print(f"  - 验证集 R²: {val_r2:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
