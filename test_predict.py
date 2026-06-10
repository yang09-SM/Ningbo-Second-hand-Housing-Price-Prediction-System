import joblib
import numpy as np
import os

base_dir = r'f:\Git Hub Project\Ningbo Second-hand Housing Price Prediction System'

# 加载模型和scaler
model_data = joblib.load(os.path.join(base_dir, 'models', 'lightgbm.joblib'))
model = model_data['model']
scaler = model_data['scaler']

print(f"Model type: {type(model).__name__}")
print(f"Scaler n_features: {scaler.n_features_in_}")
print(f"Model n_features_in_: {model.n_features_in_}")
print(f"Scaler mean (first 5): {scaler.mean_[:5]}")
print(f"Scaler scale (first 5): {scaler.scale_[:5]}")

# 构造两条不同的测试数据（17特征：6数值 + 11编码）
# 特征顺序: 单价,建筑面积,总楼层数,室数,厅数,卫数,区_encoded,板块_encoded,楼层位置_encoded,房屋朝向_encoded,装修情况_encoded,建筑类型_encoded,建筑结构_encoded,交易权属_encoded,房屋用途_encoded,配备电梯_encoded,产权所属_encoded

test1 = np.array([[25000, 89.5, 18, 3, 1, 1, 8, 60, 1, 17, 3, 5, 5, 0, 4, 2, 1]], dtype=np.float64)
test2 = np.array([[15000, 65.0, 6, 2, 1, 1, 6, 55, 1, 17, 2, 2, 0, 0, 4, 1, 0]], dtype=np.float64)

print(f"\nTest1 raw: {test1[0][:6]}")
print(f"Test2 raw: {test2[0][:6]}")

test1_scaled = scaler.transform(test1)
test2_scaled = scaler.transform(test2)

print(f"\nTest1 scaled (first 5): {test1_scaled[0][:5]}")
print(f"Test2 scaled (first 5): {test2_scaled[0][:5]}")
print(f"Scaled values differ: {not np.allclose(test1_scaled, test2_scaled)}")

pred1 = model.predict(test1_scaled)
pred2 = model.predict(test2_scaled)

print(f"\nPrediction 1: {pred1[0]:.2f} ({pred1[0]/10000:.2f}万)")
print(f"Prediction 2: {pred2[0]:.2f} ({pred2[0]/10000:.2f}万)")
print(f"Predictions differ: {pred1[0] != pred2[0]}")

# 也测试CatBoost
print("\n=== Testing CatBoost ===")
cb_data = joblib.load(os.path.join(base_dir, 'models', 'catboost.joblib'))
cb_model = cb_data['model']
cb_scaler = cb_data['scaler']

cb_pred1 = cb_model.predict(cb_scaler.transform(test1))
cb_pred2 = cb_model.predict(cb_scaler.transform(test2))
print(f"CatBoost Pred1: {cb_pred1[0]:.2f} ({cb_pred1[0]/10000:.2f}万)")
print(f"CatBoost Pred2: {cb_pred2[0]:.2f} ({cb_pred2[0]/10000:.2f}万)")
print(f"CatBoost predictions differ: {cb_pred1[0] != cb_pred2[0]}")

# 测试XGBoost
print("\n=== Testing XGBoost ===")
xgb_data = joblib.load(os.path.join(base_dir, 'models', 'xgboost.joblib'))
xgb_model = xgb_data['model']
xgb_scaler = xgb_data['scaler']

xgb_pred1 = xgb_model.predict(xgb_scaler.transform(test1))
xgb_pred2 = xgb_model.predict(xgb_scaler.transform(test2))
print(f"XGBoost Pred1: {xgb_pred1[0]:.2f} ({xgb_pred1[0]/10000:.2f}万)")
print(f"XGBoost Pred2: {xgb_pred2[0]:.2f} ({xgb_pred2[0]/10000:.2f}万)")
print(f"XGBoost predictions differ: {xgb_pred1[0] != xgb_pred2[0]}")
