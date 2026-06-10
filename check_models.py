import joblib, os

base_dir = r'f:\Git Hub Project\Ningbo Second-hand Housing Price Prediction System'
info = joblib.load(os.path.join(base_dir, 'models', 'best_model_info.joblib'))
print('=== Best Model Info ===')
for k, v in info.items():
    print(f'{k}: {v}')

print()
# 检查各模型的scaler维度
for name in ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'stacking_ensemble']:
    path = os.path.join(base_dir, 'models', f'{name}.joblib')
    if os.path.exists(path):
        data = joblib.load(path)
        scaler = data.get('scaler')
        if scaler is not None:
            print(f'{name}: scaler n_features={scaler.n_features_in_}, scale_.shape={scaler.scale_.shape}')
        else:
            print(f'{name}: no scaler')
        if 'model' in data:
            m = data['model']
            print(f'  model type: {type(m).__name__}')
            if hasattr(m, 'n_features_in_'):
                print(f'  model n_features_in_: {m.n_features_in_}')
        elif 'meta_model' in data:
            m = data['meta_model']
            print(f'  meta_model type: {type(m).__name__}')
            if hasattr(m, 'n_features_in_'):
                print(f'  meta_model n_features_in_: {m.n_features_in_}')

print()
print('feature_order length:', len([
    '单价', '建筑面积', '总楼层数', '室数', '厅数', '卫数',
    '人均面积', '楼层密度', '厅室比', '卫室比', '总房间数',
    '单价等级', '面积等级',
    '区域均价', '区域中位价', '区域价格标准差', '区域房源数', '相对价格指数',
    '区_encoded', '板块_encoded', '楼层位置_encoded', '房屋朝向_encoded',
    '装修情况_encoded', '建筑类型_encoded', '建筑结构_encoded',
    '交易权属_encoded', '房屋用途_encoded', '配备电梯_encoded', '产权所属_encoded'
]))
