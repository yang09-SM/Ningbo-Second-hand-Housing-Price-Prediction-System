from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import sys

# 设置正确的工作目录和模板路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
sys.path.append(base_dir)

app = Flask(__name__, template_folder=template_dir)

DB_PATH = os.path.join(base_dir, 'data', 'houses.db')
BEST_MODEL_INFO_PATH = os.path.join(base_dir, 'models', 'best_model_info.joblib')
LABEL_ENCODERS_PATH = os.path.join(base_dir, 'models', 'label_encoders.joblib')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_models():
    best_model_info = joblib.load(BEST_MODEL_INFO_PATH)
    model_data = joblib.load(best_model_info['best_model_path'])
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    return model, scaler, label_encoders, best_model_info

model, scaler, label_encoders, best_model_info = load_models()

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
        
        features = np.array([input_data[feature] for feature in feature_order]).reshape(1, -1)
        features_scaled = scaler.transform(features)
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
        options[feature] = list(le.classes_)
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
