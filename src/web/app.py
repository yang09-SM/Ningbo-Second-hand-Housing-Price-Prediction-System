from flask import Flask, render_template, request, jsonify, session
import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import sys
import csv
from datetime import datetime

# 设置正确的工作目录和模板路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'src'))

# 导入可视化大屏数据处理模块
from visualization.dashboard import (get_dashboard_stats, get_heatmap_data,
                                     get_price_distribution, get_district_price_trend,
                                     get_model_comparison_data, get_prediction_scatter_data)

# 导入 SHAP 可解释性分析模块（只导入需要的类和函数）
from explainability.shap_analyzer import SHAPAnalyzer

app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'ningbo-house-prediction-secret-key-2024'

DB_PATH = os.path.join(base_dir, 'data', 'houses.db')
BEST_MODEL_INFO_PATH = os.path.join(base_dir, 'models', 'best_model_info.joblib')
LABEL_ENCODERS_PATH = os.path.join(base_dir, 'models', 'label_encoders.joblib')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

from functools import wraps

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = session.get('user')
        if not user or user.get('role') != 'admin':
            return jsonify({'error': '需要管理员权限'}), 403
        return f(*args, **kwargs)
    return decorated_function

def load_models():
    best_model_info = joblib.load(BEST_MODEL_INFO_PATH)
    model_path = best_model_info['best_model_path']
    model_data = joblib.load(model_path)

    # 兼容单模型格式 {'model': ...} 和 Stacking 格式 {'meta_model': ...}
    if 'model' in model_data:
        model = model_data['model']
        scaler = model_data.get('scaler')
    elif 'meta_model' in model_data:
        # Stacking 集成：对于在线预测，改用 LightGBM（最佳单模型，特征匹配可靠）
        lgbm_path = os.path.join(base_dir, 'models', 'lightgbm.joblib')
        if os.path.exists(lgbm_path):
            lgbm_data = joblib.load(lgbm_path)
            model = lgbm_data.get('model', lgbm_data.get('meta_model'))
            scaler = lgbm_data.get('scaler')  # 使用LightGBM自身的scaler
            best_model_info['_actual_model_name'] = 'LightGBM (Stacking替代)'
        else:
            catboost_path = os.path.join(base_dir, 'models', 'catboost.joblib')
            if os.path.exists(catboost_path):
                catboost_data = joblib.load(catboost_path)
                model = catboost_data.get('model', catboost_data.get('meta_model'))
                scaler = catboost_data.get('scaler')
                best_model_info['_actual_model_name'] = 'CatBoost (Stacking替代)'
            else:
                model = model_data['meta_model']
                scaler = model_data.get('scaler')
    else:
        raise KeyError("模型文件中未找到 'model' 或 'meta_model' 键")

    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    return model, scaler, label_encoders, best_model_info


def compute_advanced_features(input_data):
    """
    从基础特征计算高级工程特征
    确保与 preprocess_data.py 中的 create_cross_features() 和 create_region_features() 保持一致
    """
    # 提取基础数值特征
    unit_price = float(input_data.get('单价', 0))
    area = float(input_data.get('建筑面积', 0))
    total_floors = float(input_data.get('总楼层数', 0))
    rooms = float(input_data.get('室数', 0))
    halls = float(input_data.get('厅数', 0))
    baths = float(input_data.get('卫数', 0))

    # 计算交叉特征（与 preprocess_data.create_cross_features 一致）
    input_data['人均面积'] = area / (rooms + 1) if rooms > 0 else 0
    input_data['楼层密度'] = area / (total_floors + 1) if total_floors > 0 else 0
    input_data['厅室比'] = halls / (rooms + 1) if rooms > 0 else 0
    input_data['卫室比'] = baths / (rooms + 1) if rooms > 0 else 0
    input_data['总房间数'] = rooms + halls + baths

    # 分桶特征（简化处理，实际应基于训练数据的分箱边界）
    # 这里使用固定值作为默认值，生产环境应从训练数据中获取分箱边界
    input_data['单价等级'] = 2  # 默认中等水平
    input_data['面积等级'] = 2  # 默认中等水平

    # 区域统计特征（无法从单条样本计算，使用默认值或区域均值）
    # 这些特征通常需要预计算的区域统计信息，这里使用0作为占位符
    # 实际应用中应该维护一张区域统计表
    input_data['区域均价'] = 0
    input_data['区域中位价'] = 0
    input_data['区域价格标准差'] = 0
    input_data['区域房源数'] = 0
    input_data['相对价格指数'] = 1.0  # 默认值为1表示与区域均价持平

    return input_data

model, scaler, label_encoders, best_model_info = load_models()

# 初始化 SHAP 分析器（延迟加载）
_shap_analyzer = None

# 模块级别加载 best_model_info，供 get_shap_analyzer() 使用
_best_model_info_global = joblib.load(BEST_MODEL_INFO_PATH)

def get_shap_analyzer():
    """获取 SHAP 分析器单例"""
    global _shap_analyzer
    if _shap_analyzer is None:
        _shap_analyzer = SHAPAnalyzer(_best_model_info_global['best_model_path'])
    return _shap_analyzer

# 特征顺序（与 train.csv 保持一致：6基础数值 + 11编码分类 = 17特征）
# 注意：此列表需要与 train.csv 的列顺序一致
feature_order = ['单价', '建筑面积', '总楼层数', '室数', '厅数', '卫数',
                 '区_encoded', '板块_encoded', '楼层位置_encoded', '房屋朝向_encoded',
                 '装修情况_encoded', '建筑类型_encoded', '建筑结构_encoded',
                 '交易权属_encoded', '房屋用途_encoded', '配备电梯_encoded', '产权所属_encoded']

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '')
    password = data.get('password', '')

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    if user and user['password'] == password:
        session['user'] = {'id': user['id'], 'username': user['username'], 'role': user['role']}
        return jsonify({'id': user['id'], 'username': user['username'], 'role': user['role']})
    else:
        return jsonify({'error': '用户名或密码错误'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'message': '已登出'})

@app.route('/api/current-user', methods=['GET'])
def current_user():
    user = session.get('user')
    if user:
        return jsonify(user)
    return jsonify(None), 200

@app.route('/test')
def test():
    return "Flask is working! 🚀"

@app.route('/api/houses', methods=['GET'])
def get_houses():
    conn = get_db_connection()
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = request.args.get('search', '')
    district = request.args.get('district', '')
    min_price = request.args.get('min_price', '')
    max_price = request.args.get('max_price', '')
    
    query = 'SELECT * FROM houses WHERE 1=1'
    params = []
    
    if search:
        query += ' AND (小区名称 LIKE ? OR 所在区域 LIKE ?)'
        params.extend([f'%{search}%', f'%{search}%'])
    
    if district:
        query += ' AND 区 = ?'
        params.append(district)
    
    count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
    
    query += ' ORDER BY id DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])
    
    houses = conn.execute(query, params).fetchall()
    total = conn.execute(count_query, params[:-2] if len(params) >= 2 else params).fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'houses': [dict(house) for house in houses],
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page
    })

@app.route('/api/houses/<int:id>', methods=['GET'])
def get_house(id):
    conn = get_db_connection()
    house = conn.execute('SELECT * FROM houses WHERE id = ?', (id,)).fetchone()
    conn.close()
    
    if house is None:
        return jsonify({'error': 'House not found'}), 404
    
    return jsonify(dict(house))

@app.route('/api/houses', methods=['POST'])
@admin_required
def create_house():
    data = request.get_json()
    
    required_fields = ['总价', '单价', '小区名称', '所在区域', '房屋户型', '建筑面积']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    conn = get_db_connection()
    
    columns = list(data.keys())
    placeholders = ', '.join(['?' for _ in columns])
    values = [data[col] for col in columns]
    
    query = f'INSERT INTO houses ({", ".join(columns)}) VALUES ({placeholders})'
    cursor = conn.execute(query, values)
    conn.commit()
    
    house_id = cursor.lastrowid
    house = conn.execute('SELECT * FROM houses WHERE id = ?', (house_id,)).fetchone()
    conn.close()
    
    return jsonify(dict(house)), 201

@app.route('/api/houses/<int:id>', methods=['PUT'])
@admin_required
def update_house(id):
    data = request.get_json()
    
    conn = get_db_connection()
    house = conn.execute('SELECT * FROM houses WHERE id = ?', (id,)).fetchone()
    
    if house is None:
        conn.close()
        return jsonify({'error': 'House not found'}), 404
    
    columns = list(data.keys())
    set_clause = ', '.join([f'{col} = ?' for col in columns])
    values = [data[col] for col in columns] + [id]
    
    query = f'UPDATE houses SET {set_clause} WHERE id = ?'
    conn.execute(query, values)
    conn.commit()
    
    house = conn.execute('SELECT * FROM houses WHERE id = ?', (id,)).fetchone()
    conn.close()
    
    return jsonify(dict(house))

@app.route('/api/houses/<int:id>', methods=['DELETE'])
@admin_required
def delete_house(id):
    conn = get_db_connection()
    house = conn.execute('SELECT * FROM houses WHERE id = ?', (id,)).fetchone()
    
    if house is None:
        conn.close()
        return jsonify({'error': 'House not found'}), 404
    
    conn.execute('DELETE FROM houses WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'House deleted successfully'})

@app.route('/api/districts', methods=['GET'])
def get_districts():
    conn = get_db_connection()
    districts = conn.execute('SELECT DISTINCT 区 FROM houses WHERE 区 IS NOT NULL ORDER BY 区').fetchall()
    conn.close()
    return jsonify([d['区'] for d in districts])

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_data = {}

        numeric_features = ['单价', '建筑面积', '总楼层数', '室数', '厅数', '卫数']
        for feature in numeric_features:
            input_data[feature] = float(data.get(feature, 0))

        categorical_features = ['区', '板块', '楼层位置', '房屋朝向', '装修情况', '建筑类型',
                               '建筑结构', '交易权属', '房屋用途', '配备电梯', '产权所属']
        for feature in categorical_features:
            value = data.get(feature, '未知')
            le = label_encoders[feature]
            if value in le.classes_:
                input_data[feature + '_encoded'] = le.transform([value])[0]
            else:
                input_data[feature + '_encoded'] = 0

        # 计算高级工程特征（确保特征向量完整）
        # 注：当前模型训练数据(train.csv)仅含17个基础+编码特征，无需计算高级特征
        # input_data = compute_advanced_features(input_data)

        features = np.array([input_data[feature] for feature in feature_order], dtype=np.float64).reshape(1, -1)
        features_df = pd.DataFrame(features, columns=feature_order)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            'predicted_price': float(prediction),
            'predicted_price_wan': f'{prediction/10000:.2f}万',
            'model_used': best_model_info['best_model_name']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-options', methods=['GET'])
def get_feature_options():
    options = {}
    for feature, le in label_encoders.items():
        # 确保转换为原生Python类型，避免numpy类型导致JSON序列化失败
        options[feature] = [str(x) for x in le.classes_]
    return jsonify(options)

# ==================== 可视化大屏数据API路由 ====================

@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    """获取仪表盘关键指标卡片数据"""
    try:
        data = get_dashboard_stats()
        return jsonify({'code': 0, 'data': data, 'message': 'success'})
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

@app.route('/api/dashboard/heatmap', methods=['GET'])
def dashboard_heatmap():
    """获取区域房价热力图数据"""
    try:
        data = get_heatmap_data()
        return jsonify({'code': 0, 'data': data, 'message': 'success'})
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

@app.route('/api/dashboard/distribution', methods=['GET'])
def dashboard_distribution():
    """获取价格分布直方图数据"""
    try:
        data = get_price_distribution()
        return jsonify({'code': 0, 'data': data, 'message': 'success'})
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

@app.route('/api/dashboard/districts', methods=['GET'])
def dashboard_districts():
    """获取各区域价格概况数据"""
    try:
        data = get_district_price_trend()
        return jsonify({'code': 0, 'data': data, 'message': 'success'})
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

@app.route('/api/model/comparison', methods=['GET'])
def model_comparison():
    """获取模型性能对比数据"""
    try:
        data = get_model_comparison_data()
        return jsonify({'code': 0, 'data': data, 'message': 'success'})
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

@app.route('/api/model/scatter', methods=['GET'])
def model_scatter():
    """获取预测值vs实际值的散点图数据"""
    try:
        sample_size = request.args.get('sample_size', 200, type=int)
        data = get_prediction_scatter_data(sample_size)
        return jsonify({'code': 0, 'data': data, 'message': 'success'})
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

# ==================== SHAP 可解释性分析API路由 ====================

@app.route('/api/model/shap/global', methods=['GET'])
def shap_global_importance():
    """获取全局特征重要性（SHAP分析）"""
    try:
        analyzer = get_shap_analyzer()
        max_features = request.args.get('max_features', 20, type=int)
        result = analyzer.get_global_feature_importance(max_features=max_features)

        return jsonify({
            'code': 0,
            'data': {
                'features': result['feature_names'],
                'importance': result['importance_mean'],
                'summary_data': result['summary_data'],
                'model_used': analyzer.model_name
            },
            'message': 'success'
        })
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

@app.route('/api/predict/explain', methods=['POST'])
def predict_with_explanation():
    """带 SHAP 解释的单次预测"""
    try:
        analyzer = get_shap_analyzer()
        data = request.get_json()

        # 处理输入数据（与 /api/predict 相同的逻辑）
        input_data = {}

        numeric_features = ['单价', '建筑面积', '总楼层数', '室数', '厅数', '卫数']
        for feature in numeric_features:
            input_data[feature] = float(data.get(feature, 0))

        categorical_features = ['区', '板块', '楼层位置', '房屋朝向', '装修情况', '建筑类型',
                               '建筑结构', '交易权属', '房屋用途', '配备电梯', '产权所属']
        for feature in categorical_features:
            value = data.get(feature, '未知')
            le = label_encoders.get(feature)
            if le is not None and value in le.classes_:
                input_data[feature + '_encoded'] = int(le.transform([value])[0])
            else:
                input_data[feature + '_encoded'] = 0

        # 计算高级工程特征（当前模型不需要，train.csv仅含17特征）
        # input_data = compute_advanced_features(input_data)

        # 使用与模型训练一致的特征顺序
        features = np.array([input_data.get(feat, 0) for feat in feature_order], dtype=np.float64).reshape(1, -1)
        features_df = pd.DataFrame(features, columns=feature_order)
        features_scaled = scaler.transform(features_df)

        # 获取 SHAP 解释
        explanation = analyzer.explain_single_prediction(
            features_scaled.flatten(), feature_order
        )

        return jsonify({
            'code': 0,
            'data': {
                'predicted_price': round(explanation['prediction'], 2),
                'predicted_price_wan': f'{explanation["prediction"]/10000:.2f}万',
                'model_used': analyzer.model_name,
                'shap_explanation': {
                    'base_value': round(explanation['base_value'], 2),
                    'feature_contributions': explanation['feature_contributions'],
                    'top_positive': explanation['top_positive'],
                    'top_negative': explanation['top_negative']
                }
            },
            'message': 'success'
        })
    except Exception as e:
        return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500

# ==================== 统一响应格式装饰器 ====================

def api_response(func):
    """统一 API 响应格式装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # 如果返回的是 tuple (response, status_code)
            if isinstance(result, tuple):
                return result
            # 如果已经是标准格式
            if isinstance(result, dict) and 'code' in result:
                return jsonify(result)
            # 包装为标准格式
            return jsonify({'code': 0, 'data': result, 'message': 'success'})
        except Exception as e:
            return jsonify({'code': -1, 'data': None, 'message': str(e)}), 500
    return wrapper

# ==================== 批量预测接口 ====================

@app.route('/api/predict/batch', methods=['POST'])
@api_response
def batch_predict():
    """
    批量房价预测
    接收: {'houses': [{特征字段...}, {特征字段...}, ...]} 最多100条
    返回: {
        'code': 0,
        'data': {
            'predictions': [
                {'predicted_price': float, 'predicted_price_wan': str},
                ...
            ],
            'total': int,
            'model_used': str
        },
        'message': 'success'
    }
    """
    data = request.get_json()
    houses = data.get('houses', [])

    if not houses:
        return {'code': -1, 'data': None, 'message': '请提供至少一条房屋数据'}

    if len(houses) > 100:
        return {'code': -1, 'data': None, 'message': '批量预测最多支持100条数据'}

    predictions = []
    numeric_features = ['单价', '建筑面积', '总楼层数', '室数', '厅数', '卫数']
    categorical_features = ['区', '板块', '楼层位置', '房屋朝向', '装修情况', '建筑类型',
                           '建筑结构', '交易权属', '房屋用途', '配备电梯', '产权所属']

    for idx, house_data in enumerate(houses):
        try:
            # DEBUG: 打印原始接收数据（仅第1条）
            if idx == 0:
                print(f'[RAW-DATA] house_data={house_data}', flush=True)

            input_data = {}

            for feature in numeric_features:
                val = house_data.get(feature, 0)
                input_data[feature] = float(val)

            for feature in categorical_features:
                value = house_data.get(feature, '未知')
                le = label_encoders.get(feature)
                if le is not None and value in le.classes_:
                    input_data[feature + '_encoded'] = int(le.transform([value])[0])
                else:
                    input_data[feature + '_encoded'] = 0

            # 计算高级工程特征（当前模型不需要，train.csv仅含17特征）
            # input_data = compute_advanced_features(input_data)

            # DEBUG: 打印完整的input_data（仅前2条）
            if idx < 2:
                print(f'[DEBUG-{idx}] input_data keys={list(input_data.keys())} 单价={input_data.get("单价")} 区_enc={input_data.get("区_encoded")}', flush=True)

            features = np.array([input_data[feat] for feat in feature_order], dtype=np.float64).reshape(1, -1)
            # 使用DataFrame保持与训练时一致的列名（scaler和model都是用DataFrame拟合的）
            features_df = pd.DataFrame(features, columns=feature_order)
            features_scaled = scaler.transform(features_df)
            prediction = model.predict(features_scaled)[0]

            # DEBUG: 输出前两条预测的调试信息（使用print确保可见）
            if idx < 2:
                print(f'[DEBUG-PREDICT] idx={idx} raw[:6]={features[0][:6]} scaled[:6]={features_scaled[0][:6]} pred={prediction} model={type(model).__name__}', flush=True)

            predictions.append({
                'predicted_price': float(prediction),
                'predicted_price_wan': f'{prediction/10000:.2f}万',
                'index': idx
            })
        except Exception as e:
            predictions.append({
                'predicted_price': None,
                'predicted_price_wan': None,
                'index': idx,
                'error': str(e)
            })

    return {
        'code': 0,
        'data': {
            'predictions': predictions,
            'total': len(predictions),
            'model_used': best_model_info['best_model_name']
        },
        'message': 'success'
    }

# ==================== 预测报告导出接口 ====================

@app.route('/api/export/report', methods=['GET'])
@api_response
def export_report():
    """
    导出模型评估报告为 JSON 格式
    包含: 所有模型指标对比、最佳模型信息、SHAP 全局重要性摘要
    查询参数: format=json (默认)
    """
    report_format = request.args.get('format', 'json')

    if report_format != 'json':
        return {'code': -1, 'data': None, 'message': f'不支持的导出格式: {report_format}'}

    model_comparison_path = os.path.join(base_dir, 'reports', 'model_comparison.csv')
    train_csv_path = os.path.join(base_dir, 'data', 'train.csv')

    # 读取模型对比数据
    model_metrics = []
    if os.path.exists(model_comparison_path):
        with open(model_comparison_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_metrics.append(row)

    # 获取最佳模型信息
    best_model = {
        'name': best_model_info.get('best_model_name', '未知'),
        'metrics': best_model_info.get('best_model_metrics', {})
    }

    # 获取 SHAP 全局重要性摘要（如果可用）
    shap_summary = None
    try:
        analyzer = get_shap_analyzer()
        shap_result = analyzer.get_global_feature_importance(max_features=10)
        shap_summary = {
            'features': shap_result['feature_names'],
            'importance': shap_result['importance_mean']
        }
    except Exception:
        shap_summary = None

    # 训练数据大小
    training_data_size = None
    if os.path.exists(train_csv_path):
        with open(train_csv_path, 'r', encoding='utf-8') as f:
            training_data_size = sum(1 for _ in f) - 1  # 减去表头

    return {
        'code': 0,
        'data': {
            'generated_at': datetime.now().isoformat(),
            'model_comparison': model_metrics,
            'best_model': best_model,
            'shap_summary': shap_summary,
            'training_data_size': training_data_size,
            'features_count': len(feature_order)
        },
        'message': 'success'
    }

# ==================== 系统信息接口 ====================

@app.route('/api/system/info', methods=['GET'])
@api_response
def system_info():
    """返回系统版本信息和可用模型列表"""
    train_csv_path = os.path.join(base_dir, 'data', 'train.csv')
    training_data_size = None
    if os.path.exists(train_csv_path):
        with open(train_csv_path, 'r', encoding='utf-8') as f:
            training_data_size = sum(1 for _ in f) - 1

    return {
        'code': 0,
        'data': {
            'version': '2.0.0',
            'available_models': ['线性回归', 'SVM', '随机森林', '神经网络', 'XGBoost', 'LightGBM', 'CatBoost', 'Stacking'],
            'features_count': len(feature_order),
            'base_features': 6,  # 基础数值特征
            'advanced_features': len(feature_order) - 6 - 11,  # 高级工程特征（减去基础特征和编码特征）
            'encoded_features': 11,  # 编码后的分类特征
            'training_data_size': training_data_size,
            'last_updated': datetime.now().isoformat()
        }
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
