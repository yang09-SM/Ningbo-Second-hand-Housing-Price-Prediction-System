"""
SHAP 可解释性分析模块 - 宁波二手房房价预测系统
提供全局特征重要性分析、单样本预测解释、可视化图表生成等功能
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
from flask import Blueprint, request, jsonify

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入各种树模型类型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


class SHAPAnalyzer:
    """SHAP 可解释性分析核心类"""

    # 基础特征顺序（17个）
    BASE_FEATURE_ORDER = ['单价', '建筑面积', '总楼层数', '室数', '厅数', '卫数',
                          '区_encoded', '板块_encoded', '楼层位置_encoded', '房屋朝向_encoded',
                          '装修情况_encoded', '建筑类型_encoded', '建筑结构_encoded',
                          '交易权属_encoded', '房屋用途_encoded', '配备电梯_encoded', '产权所属_encoded']

    def __init__(self, model_path, test_data_path='data/test.csv', target_col='总价'):
        """
        初始化 SHAP 分析器

        Args:
            model_path: 模型文件路径 (joblib格式，包含 {'model': model, 'scaler': scaler})
            test_data_path: 测试数据 CSV 路径
            target_col: 目标列名
        """
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.model_path = model_path if os.path.isabs(model_path) else os.path.join(self.base_dir, model_path)
        self.test_data_path = test_data_path if os.path.isabs(test_data_path) else os.path.join(self.base_dir, test_data_path)
        self.target_col = target_col

        # 加载模型和数据
        self._load_model()
        self._load_test_data()
        self._create_explainer()

    def _load_model(self):
        """加载模型、标准化器和标签编码器"""
        print(f"正在加载模型: {self.model_path}")
        model_data = joblib.load(self.model_path)

        # 兼容单模型格式 {'model': ..., 'scaler': ...} 和 Stacking 格式 {'meta_model': ..., 'scaler': ...}
        if 'model' in model_data:
            self.model = model_data['model']
            self.is_stacking = False
        elif 'meta_model' in model_data:
            # Stacking 集成模型的 Ridge 元模型无法做有意义的特征级 SHAP 分析（只有4个基模型输入）
            # 自动切换到 CatBoost（最佳单模型）进行 TreeExplainer 分析
            catboost_path = os.path.join(self.base_dir, 'models', 'catboost.joblib')
            if os.path.exists(catboost_path):
                print(f"  检测到 Stacking 集成模型，自动切换至 CatBoost 进行 SHAP 分析")
                catboost_data = joblib.load(catboost_path)
                self.model = catboost_data.get('model', catboost_data.get('meta_model'))
                self.scaler = catboost_data.get('scaler', model_data.get('scaler'))
                self.is_stacking = False  # 切换后当作普通单模型处理
                self._switched_from_stacking = True
            else:
                self.model = model_data['meta_model']
                self.is_stacking = True
        else:
            raise KeyError("模型文件中未找到 'model' 或 'meta_model' 键")

        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = model_data.get('scaler')
        self.is_stacking = getattr(self, 'is_stacking', False) or model_data.get('type') == 'stacking_ensemble'

        # 加载标签编码器
        label_encoders_path = os.path.join(self.base_dir, 'models', 'label_encoders.joblib')
        if os.path.exists(label_encoders_path):
            self.label_encoders = joblib.load(label_encoders_path)
        else:
            self.label_encoders = {}

        # 加载最佳模型信息
        best_model_info_path = os.path.join(self.base_dir, 'models', 'best_model_info.joblib')
        if os.path.exists(best_model_info_path):
            self.best_model_info = joblib.load(best_model_info_path)
            self.model_name = self.best_model_info.get('best_model_name', 'Unknown')
            # 如果从Stacking切换到CatBoost，更新显示名称
            if getattr(self, '_switched_from_stacking', False):
                self.model_name = f"CatBoost ({self.model_name} 替代)"
        else:
            self.model_name = type(self.model).__name__

        print(f"  模型类型: {type(self.model).__name__}")

    def _load_test_data(self):
        """加载测试数据并动态获取特征列"""
        print(f"正在加载测试数据: {self.test_data_path}")
        self.test_df = pd.read_csv(self.test_data_path)

        # 动态获取特征列（排除目标列）
        self.feature_names = [col for col in self.test_df.columns if col != self.target_col]
        print(f"  特征数量: {len(self.feature_names)}")
        print(f"  特征列表: {self.feature_names}")

        # 准备特征矩阵
        self.X_test = self.test_df[self.feature_names].values
        self.y_test = self.test_df[self.target_col].values if self.target_col in self.test_df.columns else None

        # 标准化特征
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def _create_explainer(self):
        """根据模型类型创建合适的 SHAP explainer"""
        print("正在创建 SHAP Explainer...")

        TREE_MODELS = (RandomForestRegressor, GradientBoostingRegressor)
        if XGBRegressor is not None:
            TREE_MODELS = TREE_MODELS + (XGBRegressor,)
        if LGBMRegressor is not None:
            TREE_MODELS = TREE_MODELS + (LGBMRegressor,)
        if CatBoostRegressor is not None:
            TREE_MODELS = TREE_MODELS + (CatBoostRegressor,)

        if isinstance(self.model, TREE_MODELS):
            print("  使用 TreeExplainer (树模型)")
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = 'tree'
        else:
            print("  使用 KernelExplainer (非树模型)")
            # 使用部分样本作为背景数据
            background_size = min(50, len(self.X_test_scaled))
            background_data = shap.sample(self.X_test_scaled, background_size)
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
            self.explainer_type = 'kernel'

    def compute_shap_values(self, sample_size=100):
        """
        计算 SHAP 值

        Args:
            sample_size: 用于计算 SHAP 值的样本数量

        Returns:
            shap_values: SHAP 值对象
        """
        print(f"正在计算 SHAP 值 (样本数: {sample_size})...")

        # 限制样本数量以加快计算速度
        actual_sample_size = min(sample_size, len(self.X_test_scaled))
        sample_indices = np.random.choice(len(self.X_test_scaled), actual_sample_size, replace=False)
        X_sample = self.X_test_scaled[sample_indices]

        self.shap_values = self.explainer.shap_values(X_sample)
        self.shap_X = X_sample

        # 如果是列表形式（某些模型的输出），取第一个元素
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]

        print(f"  SHAP 值计算完成, shape: {np.array(self.shap_values).shape}")
        return self.shap_values

    def get_global_feature_importance(self, max_features=20):
        """
        全局特征重要性分析

        Args:
            max_features: 返回的最大特征数

        Returns:
            dict: 包含特征重要性数据的字典
        """
        if not hasattr(self, 'shap_values') or self.shap_values is None:
            self.compute_shap_values()

        shap_vals = np.array(self.shap_values)

        # 计算每个特征的 |SHAP value| 的均值作为重要性
        importance_mean = np.abs(shap_vals).mean(axis=0)

        # 创建特征重要性 DataFrame
        importance_df = pd.DataFrame({
            'feature_names': self.feature_names,
            'importance_mean': importance_mean
        }).sort_values('importance_mean', ascending=False)

        # 取 Top N 特征
        top_features = importance_df.head(max_features)

        # 构建详细摘要数据
        summary_data = []
        for _, row in top_features.iterrows():
            feature_idx = list(importance_df['feature_names']).index(row['feature_names'])
            mean_contribution = shap_vals[:, feature_idx].mean()
            direction = 'positive' if mean_contribution >= 0 else 'negative'
            summary_data.append({
                'feature': row['feature_names'],
                'importance': round(float(row['importance_mean']), 6),
                'direction': direction,
                'mean_contribution': round(float(mean_contribution), 2)
            })

        return {
            'feature_names': top_features['feature_names'].tolist(),
            'importance_mean': top_features['importance_mean'].tolist(),
            'feature_values': self.shap_X.T.tolist() if hasattr(self, 'shap_X') else None,
            'summary_data': summary_data,
            'all_importances': {
                'features': importance_df['feature_names'].tolist(),
                'values': importance_df['importance_mean'].tolist()
            }
        }

    def explain_single_prediction(self, feature_vector, feature_names=None):
        """
        单样本预测解释

        Args:
            feature_vector: 1D numpy array (已标准化的特征向量)
            feature_names: 特征名称列表（可选）

        Returns:
            dict: 单样本解释结果
        """
        if feature_names is None:
            feature_names = self.feature_names

        # 确保是二维数组
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # 预测值
        prediction = float(self.model.predict(feature_vector)[0])

        # 计算 SHAP 值
        shap_vals = self.explainer.shap_values(feature_vector)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.array(shap_vals).flatten()

        # 获取基准值
        if hasattr(self.explainer, 'expected_value'):
            base_value = float(self.explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
        else:
            base_value = float(np.mean(self.y_test)) if self.y_test is not None else 0.0

        # 构建特征贡献列表
        feature_contributions = []
        for i, (feat_name, feat_val, shap_val) in enumerate(zip(feature_names, feature_vector.flatten(), shap_vals)):
            direction = 'positive' if shap_val >= 0 else 'negative'
            feature_contributions.append({
                'feature': feat_name,
                'value': round(float(feat_val), 4),
                'contribution': round(float(shap_val), 4),
                'direction': direction
            })

        # 按 |贡献| 排序获取 Top 正向和负向贡献
        sorted_contributions = sorted(feature_contributions, key=lambda x: abs(x['contribution']), reverse=True)
        top_positive = sorted([c for c in sorted_contributions if c['direction'] == 'positive'],
                             key=lambda x: x['contribution'], reverse=True)[:5]
        top_negative = sorted([c for c in sorted_contributions if c['direction'] == 'negative'],
                             key=lambda x: x['contribution'])[:5]

        return {
            'base_value': round(base_value, 2),
            'prediction': round(prediction, 2),
            'feature_contributions': feature_contributions,
            'top_positive': top_positive,
            'top_negative': top_negative
        }

    def generate_summary_plot(self, output_dir='reports'):
        """
        生成 Summary Plot（条形图 + 蜂群图）并保存为 PNG

        Args:
            output_dir: 输出目录
        """
        if not hasattr(self, 'shap_values') or self.shap_values is None:
            self.compute_shap_values()

        # 确保输出目录存在
        shap_output_dir = os.path.join(self.base_dir, output_dir, 'shap')
        os.makedirs(shap_output_dir, exist_ok=True)

        shap_vals = np.array(self.shap_values)

        # 1. 条形图 - 全局特征重要性
        plt.figure(figsize=(12, max(8, len(self.feature_names) * 0.4)))
        shap.summary_plot(shap_vals, self.shap_X, feature_names=self.feature_names,
                         plot_type='bar', show=False, max_display=min(20, len(self.feature_names)))
        plt.title('SHAP 全局特征重要性 (条形图)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        bar_plot_path = os.path.join(shap_output_dir, 'shap_summary_bar.png')
        plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  条形图已保存: {bar_plot_path}")

        # 2. 蜂群图 - 特征分布与影响
        plt.figure(figsize=(12, max(8, len(self.feature_names) * 0.4)))
        shap.summary_plot(shap_vals, self.shap_X, feature_names=self.feature_names,
                         show=False, max_display=min(20, len(self.feature_names)))
        plt.title('SHAP 特征影响蜂群图', fontsize=14, fontweight='bold')
        plt.tight_layout()
        swarm_plot_path = os.path.join(shap_output_dir, 'shap_summary_swarm.png')
        plt.savefig(swarm_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  蜂群图已保存: {swarm_plot_path}")

        return {
            'bar_plot': bar_plot_path,
            'swarm_plot': swarm_plot_path
        }

    def generate_force_plot_data(self, feature_vector, feature_names=None):
        """
        生成 Force Plot 所需的 JSON 数据结构（用于前端渲染）

        Args:
            feature_vector: 1D numpy array (已标准化的特征向量)
            feature_names: 特征名称列表

        Returns:
            dict: Force Plot 数据结构
        """
        explanation = self.explain_single_prediction(feature_vector, feature_names)

        base_value = explanation['base_value']
        prediction = explanation['prediction']

        # 构建 force plot 数据
        force_data = {
            'baseValue': base_value,
            'outValue': prediction,
            'features': [],
            'featureNames': explanation['feature_contributions'],
            'linkColor': '#b40000' if prediction > base_value else '#006b00',
            'linkLabel': f'{prediction:.0f}'
        }

        # 添加每个特征的数据点
        current_value = base_value
        for contrib in explanation['feature_contributions']:
            force_data['features'].append({
                'name': contrib['feature'],
                'value': contrib['value'],
                'contribution': contrib['contribution'],
                'direction': contrib['direction'],
                'effect': current_value + contrib['contribution']
            })
            current_value += contrib['contribution']

        return force_data

    def generate_waterfall_plot_data(self, feature_vector, feature_names=None):
        """
        生成 Waterfall Plot 数据

        Args:
            feature_vector: 1D numpy array (已标准化的特征向量)
            feature_names: 特征名称列表

        Returns:
            dict: Waterfall Plot 数据结构
        """
        explanation = self.explain_single_prediction(feature_vector, feature_names)

        # 按 |贡献| 排序
        sorted_contribs = sorted(explanation['feature_contributions'],
                                key=lambda x: abs(x['contribution']), reverse=True)

        waterfall_data = {
            'base_value': explanation['base_value'],
            'prediction': explanation['prediction'],
            'steps': [{'label': '基准值', 'value': explanation['base_value'], 'type': 'base'}]
        }

        running_total = explanation['base_value']
        for contrib in sorted_contribs[:15]:  # 取前15个最重要的特征
            running_total += contrib['contribution']
            waterfall_data['steps'].append({
                'label': contrib['feature'],
                'value': round(contrib['contribution'], 2),
                'type': contrib['direction'],
                'cumulative': round(running_total, 2),
                'feature_value': contrib['value']
            })

        # 最终值
        waterfall_data['steps'].append({
            'label': '预测值',
            'value': explanation['prediction'],
            'type': 'total'
        })

        return waterfall_data


def create_shap_api_blueprint():
    """
    创建 Flask Blueprint，注册 SHAP 相关路由

    Returns:
        Blueprint: Flask Blueprint 对象
    """
    shap_bp = Blueprint('shap', __name__, url_prefix='/api')

    # 全局变量缓存 SHAP 分析器实例
    _analyzer_instance = None

    def get_analyzer():
        """获取或创建 SHAP 分析器单例"""
        nonlocal _analyzer_instance
        if _analyzer_instance is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            best_model_info = joblib.load(os.path.join(base_dir, 'models', 'best_model_info.joblib'))
            model_path = best_model_info['best_model_path']
            _analyzer_instance = SHAPAnalyzer(model_path)
        return _analyzer_instance

    @shap_bp.route('/model/shap/global', methods=['GET'])
    def get_global_feature_importance():
        """获取全局特征重要性数据"""
        try:
            analyzer = get_analyzer()
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

    @shap_bp.route('/predict/explain', methods=['POST'])
    def predict_with_explanation():
        """带 SHAP 解释的单次预测"""
        try:
            analyzer = get_analyzer()
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
                le = analyzer.label_encoders.get(feature)
                if le is not None and value in le.classes_:
                    input_data[feature + '_encoded'] = int(le.transform([value])[0])
                else:
                    input_data[feature + '_encoded'] = 0

            # 使用实际的特征列顺序（可能包含高级特征）
            features = np.array([input_data.get(feat, 0) for feat in analyzer.feature_names]).reshape(1, -1)
            features_scaled = analyzer.scaler.transform(features)

            # 获取 SHAP 解释
            explanation = analyzer.explain_single_prediction(
                features_scaled.flatten(), analyzer.feature_names
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

    return shap_bp


def main():
    """命令行入口：执行完整的 SHAP 分析并生成报告"""
    print("=" * 80)
    print("宁波二手房房价预测系统 - SHAP 可解释性分析")
    print("=" * 80)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # 加载最佳模型信息
    best_model_info_path = os.path.join(base_dir, 'models', 'best_model_info.joblib')
    if not os.path.exists(best_model_info_path):
        print("错误: 找不到最佳模型信息文件!")
        return

    best_model_info = joblib.load(best_model_info_path)
    model_path = best_model_info['best_model_path']
    model_name = best_model_info.get('best_model_name', 'Unknown')

    # 对于 Stacking 集成模型，改用 CatBoost 做 SHAP 分析（Ridge 元模型特征归因意义不大）
    if 'stacking' in model_name.lower() or '集成' in model_name:
        catboost_path = os.path.join(base_dir, 'models', 'catboost.joblib')
        if os.path.exists(catboost_path):
            model_path = catboost_path
            model_name = 'CatBoost (用于SHAP分析)'
            print(f"  注意: 检测到 Stacking 集成，切换至 CatBoost 进行特征级SHAP分析")

    print(f"\n使用模型: {model_name}")
    print(f"模型路径: {model_path}")

    # 初始化分析器
    print("\n" + "-" * 40)
    analyzer = SHAPAnalyzer(model_path)

    # 1. 计算 SHAP 值
    print("\n" + "-" * 40)
    print("\n1. 计算 SHAP 值...")
    shap_values = analyzer.compute_shap_values(sample_size=100)

    # 2. 输出全局 Top-10 重要特征
    print("\n" + "-" * 40)
    print("\n2. 全局特征重要性 Top-10:")
    importance_result = analyzer.get_global_feature_importance(max_features=10)
    print("-" * 60)
    print(f"{'排名':<4} {'特征名称':<20} {'重要性均值':<15} {'方向':<10}")
    print("-" * 60)
    for i, (feat, imp, info) in enumerate(zip(
        importance_result['feature_names'],
        importance_result['importance_mean'],
        importance_result['summary_data']
    ), 1):
        direction_symbol = "↑" if info['direction'] == 'positive' else "↓"
        print(f"{i:<4} {feat:<20} {imp:<15.6f} {info['direction']:<10} {direction_symbol}")
    print("-" * 60)

    # 3. 生成 Summary Plot
    print("\n" + "-" * 40)
    print("\n3. 生成 Summary Plot...")
    plot_paths = analyzer.generate_summary_plot(output_dir='reports')
    print(f"   条形图: {plot_paths['bar_plot']}")
    print(f"   蜂群图: {plot_paths['swarm_plot']}")

    # 4. 对随机样本生成单样本解释
    print("\n" + "-" * 40)
    print("\n4. 单样本预测解释 (随机选取一个测试样本):")
    random_idx = np.random.randint(len(analyzer.X_test_scaled))
    random_sample = analyzer.X_test_scaled[random_idx]
    single_explanation = analyzer.explain_single_prediction(random_sample, analyzer.feature_names)

    print(f"\n   样本索引: {random_idx}")
    print(f"   基准值: {single_explanation['base_value']:,.2f}")
    print(f"   预测值: {single_explanation['prediction']:,.2f} ({single_explanation['prediction']/10000:.2f}万)")
    print(f"\n   Top-5 正向贡献因素:")
    for i, contrib in enumerate(single_explanation['top_positive'], 1):
        print(f"     {i}. {contrib['feature']}: +{contrib['contribution']:,.2f}")
    print(f"\n   Top-5 负向贡献因素:")
    for i, contrib in enumerate(single_explanation['top_negative'], 1):
        print(f"     {i}. {contrib['feature']}: {contrib['contribution']:,.2f}")

    # 5. 生成 Force Plot 和 Waterfall Plot 数据示例
    print("\n" + "-" * 40)
    print("\n5. 生成 Force Plot / Waterfall Plot 数据:")

    force_data = analyzer.generate_force_plot_data(random_sample, analyzer.feature_names)
    force_json_path = os.path.join(base_dir, 'reports', 'shap', 'force_plot_data.json')
    os.makedirs(os.path.dirname(force_json_path), exist_ok=True)
    with open(force_json_path, 'w', encoding='utf-8') as f:
        json.dump(force_data, f, ensure_ascii=False, indent=2)
    print(f"   Force Plot 数据已保存: {force_json_path}")

    waterfall_data = analyzer.generate_waterfall_plot_data(random_sample, analyzer.feature_names)
    waterfall_json_path = os.path.join(base_dir, 'reports', 'shap', 'waterfall_plot_data.json')
    with open(waterfall_json_path, 'w', encoding='utf-8') as f:
        json.dump(waterfall_data, f, ensure_ascii=False, indent=2)
    print(f"   Waterfall Plot 数据已保存: {waterfall_json_path}")

    print("\n" + "=" * 80)
    print("SHAP 分析完成! 所有报告已保存至 reports/shap/ 目录")
    print("=" * 80)


if __name__ == '__main__':
    main()
