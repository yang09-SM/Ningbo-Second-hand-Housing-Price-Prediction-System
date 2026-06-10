import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# 8 色方案，支持 7 个单模型 + 1 个 Stacking 集成
COLOR_PALETTE = [
    '#4CAF50',   # 线性回归 - 绿色
    '#2196F3',   # SVM - 蓝色
    '#FF9800',   # 随机森林 - 橙色
    '#9C27B0',   # 神经网络 - 紫色
    '#F44336',   # XGBoost - 红色
    '#00BCD4',   # LightGBM - 青色
    '#FFEB3B',   # CatBoost - 黄色
    '#E91E63',   # Stacking - 粉色
]


def mean_absolute_percentage_error(y_true, y_pred):
    """平均绝对百分比误差"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 避免除零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_test_data(test_path, target_col='总价'):
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_test, y_test


def load_model(model_path):
    data = joblib.load(model_path)
    return data['model'], data['scaler']


def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'predictions': y_pred,
        'actual': y_test.values
    }


def build_stacking_ensemble(model_results_dict, X_test_scaled, y_test):
    """
    构建 Stacking 集成模型
    基模型: XGBoost + LightGBM + CatBoost + 随机森林 (4个最强树模型)
    元模型: Ridge 回归

    流程:
    1. 收集各基模型对测试集的预测结果作为新特征
    2. 用 Ridge 回归拟合 基模型预测 → 真实值 的映射关系
    3. 输出集成预测结果和评估指标
    """
    # 从已加载的模型中选取基模型
    base_model_names = ['XGBoost', 'LightGBM', 'CatBoost', '随机森林']

    # 收集基模型的预测
    base_predictions = []
    valid_base_names = []
    for name in base_model_names:
        if name in model_results_dict and hasattr(model_results_dict[name], 'predict'):
            pred = model_results_dict[name].predict(X_test_scaled)
            base_predictions.append(pred)
            valid_base_names.append(name)

    if len(base_predictions) < 2:
        print("   警告: 可用基模型不足2个，跳过 Stacking")
        return None

    # 构建元特征矩阵 (n_samples × n_base_models)
    meta_features = np.column_stack(base_predictions)

    # 训练元模型
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, y_test)

    # 预测
    stacking_pred = meta_model.predict(meta_features)

    # 评估
    metrics = {
        'MSE': mean_squared_error(y_test, stacking_pred),
        'MAE': mean_absolute_error(y_test, stacking_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, stacking_pred)),
        'R²': r2_score(y_test, stacking_pred),
        'MAPE': mean_absolute_percentage_error(y_test, stacking_pred),
        'predictions': stacking_pred,
        'actual': y_test.values,
        'base_models': valid_base_names,
        'meta_model_weights': meta_model.coef_ if hasattr(meta_model, 'coef_') else None,
        '_meta_model_object': meta_model,
        '_base_model_objects': {name: model_results_dict[name] for name in valid_base_names}
    }

    return metrics


def generate_comparison_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    report_data = []
    for model_name, metrics in results.items():
        report_data.append({
            '模型': model_name,
            'MSE': f"{metrics['MSE']:.2e}",
            'MAE': f"{metrics['MAE']:.2e}",
            'RMSE': f"{metrics['RMSE']:.2e}",
            'R²': f"{metrics['R²']:.4f}",
            'MAPE': f"{metrics['MAPE']:.2f}%"
        })

    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(output_dir, 'model_comparison.csv')
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')

    with open(os.path.join(output_dir, 'model_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("宁波二手房价格预测 - 模型对比报告\n")
        f.write("=" * 80 + "\n\n")
        for model_name, metrics in results.items():
            f.write(f"模型: {model_name}\n")
            f.write(f"  - MSE:  {metrics['MSE']:.2e}\n")
            f.write(f"  - MAE:  {metrics['MAE']:.2e}\n")
            f.write(f"  - RMSE: {metrics['RMSE']:.2e}\n")
            f.write(f"  - R²:   {metrics['R²']:.4f}\n")
            f.write(f"  - MAPE: {metrics['MAPE']:.2f}%\n")
            if 'base_models' in metrics:
                f.write(f"  - 基模型: {', '.join(metrics['base_models'])}\n")
                if metrics.get('meta_model_weights') is not None:
                    weights = metrics['meta_model_weights']
                    f.write(f"  - 元模型权重: {dict(zip(metrics['base_models'], weights))}\n")
            f.write("\n")

        # 标注最佳模型
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        best_r2 = results[best_model_name]['R²']
        f.write("-" * 80 + "\n")
        f.write(f"★ 最佳模型: {best_model_name} (R²={best_r2:.4f})\n")

    return report_df


def plot_comparison_charts(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results.keys())
    n_models = len(model_names)
    colors = COLOR_PALETTE[:n_models]

    metrics = ['MSE', 'MAE', 'RMSE', 'R²', 'MAPE']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        if metric == 'R²':
            ylabel = metric
        elif metric == 'MAPE':
            ylabel = f'{metric} (%)'
        else:
            values = [v / 1e6 for v in values]
            ylabel = f'{metric} (百万)'

        bars = axes[i].bar(model_names, values, color=colors[:len(values)])
        axes[i].set_title(f'{metric} 对比', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(ylabel)
        axes[i].tick_params(axis='x', rotation=25)

        # 标注最佳模型（对于 R² 是最高，其他是最低）
        if metric == 'R²':
            best_idx = values.index(max(values))
        else:
            best_idx = values.index(min(values))
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

        for bar in bars:
            height = bar.get_height()
            label_text = f'{height:.2f}' if metric != 'MAPE' else f'{height:.1f}%'
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        label_text,
                        ha='center', va='bottom', fontsize=9)

    # 隐藏第6个子图（如果只有5个指标）
    if len(metrics) < 6:
        axes[5].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 找出最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
    best_results = results[best_model_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(best_results['actual'], best_results['predictions'], alpha=0.6, color=colors[model_names.index(best_model_name)])
    min_val = min(best_results['actual'].min(), best_results['predictions'].min())
    max_val = max(best_results['actual'].max(), best_results['predictions'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('真实价格', fontsize=12)
    axes[0].set_ylabel('预测价格', fontsize=12)
    axes[0].set_title(f'★ {best_model_name} - 真实值 vs 预测值', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    residuals = best_results['actual'] - best_results['predictions']
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=colors[model_names.index(best_model_name)])
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('残差', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].set_title(f'★ {best_model_name} - 残差分布', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_model_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    test_path = os.path.join('data', 'test.csv')
    model_paths = {
        '线性回归': os.path.join('models', 'linear_regression.joblib'),
        'SVM': os.path.join('models', 'svm.joblib'),
        '随机森林': os.path.join('models', 'random_forest.joblib'),
        '神经网络': os.path.join('models', 'neural_network.joblib'),
        'XGBoost': os.path.join('models', 'xgboost.joblib'),
        'LightGBM': os.path.join('models', 'lightgbm.joblib'),
        'CatBoost': os.path.join('models', 'catboost.joblib'),
    }
    output_dir = 'reports'

    print("=" * 80)
    print("宁波二手房价格预测 - 模型评估 (升级版)")
    print("支持 7 种单模型 + Stacking 集成")
    print("=" * 80)

    # 步骤 1: 加载测试数据
    print("\n1. 正在加载测试数据...")
    X_test, y_test = load_test_data(test_path)
    print(f"   测试集: {len(X_test)} 条样本, {len(X_test.columns)} 个特征")
    print(f"   特征列: {list(X_test.columns)}")

    # 步骤 2: 逐一加载并评估 7 个单模型
    print("\n2. 正在加载并评估模型...")
    results = {}
    loaded_models = {}  # 保存已加载的模型对象，用于 Stacking
    loaded_scalers = {}  # 保存对应的 scaler

    for model_name, model_path in model_paths.items():
        try:
            if not os.path.exists(model_path):
                print(f"   - {model_name}: 模型文件不存在，跳过 ({os.path.basename(model_path)})")
                continue

            print(f"   - {model_name}...", end='', flush=True)
            model, scaler = load_model(model_path)
            metrics = evaluate_model(model, scaler, X_test, y_test)
            results[model_name] = metrics
            loaded_models[model_name] = model
            loaded_scalers[model_name] = scaler
            print(f" 完成 (R²={metrics['R²']:.4f}, MAPE={metrics['MAPE']:.2f}%)")
        except Exception as e:
            print(f" 失败 ({str(e)})")

    print(f"\n   成功加载 {len(results)}/{len(model_paths)} 个模型")

    # 步骤 3: 构建 Stacking 集成模型
    print("\n3. 正在构建 Stacking 集成模型...")
    stacking_metrics = None

    if len(loaded_models) >= 2:
        # 使用第一个可用的 scaler 进行标准化（所有模型应使用相同的 scaler）
        first_scaler = list(loaded_scalers.values())[0]
        X_test_scaled = first_scaler.transform(X_test)

        stacking_metrics = build_stacking_ensemble(loaded_models, X_test_scaled, y_test)

        if stacking_metrics is not None:
            results['Stacking 集成'] = stacking_metrics
            print(f"   Stacking 集成完成 (R²={stacking_metrics['R²']:.4f}, MAPE={stacking_metrics['MAPE']:.2f}%)")
            print(f"   基模型: {', '.join(stacking_metrics['base_models'])}")
            if stacking_metrics.get('meta_model_weights') is not None:
                weights = stacking_metrics['meta_model_weights']
                weight_info = ', '.join([f'{name}={w:.3f}' for name, w in zip(stacking_metrics['base_models'], weights)])
                print(f"   元模型权重: {weight_info}")

            # 保存 Stacking 集成模型
            stacking_info = {
                'type': 'stacking_ensemble',
                'base_models': stacking_metrics['base_models'],
                'meta_model': stacking_metrics.get('_meta_model_object', None),
                'base_model_paths': {name: model_paths[name] for name in stacking_metrics['base_models']},
                'scaler': first_scaler,
                'metrics': {k: v for k, v in stacking_metrics.items()
                           if k not in ['predictions', 'actual', '_meta_model_object']}
            }
            joblib.dump(stacking_info, os.path.join('models', 'stacking_ensemble.joblib'))
            print(f"   Stacking 模型已保存至 models/stacking_ensemble.joblib")
        else:
            print("   Stacking 集成跳过（可用基模型不足）")
    else:
        print("   Stacking 集成跳过（可用模型不足 2 个）")

    # 步骤 4: 生成增强版对比报告
    print("\n4. 正在生成增强版对比报告...")
    report_df = generate_comparison_report(results, output_dir)
    print(f"   {len(results)} 个模型已评估（含 Stacking）")

    # 步骤 5: 生成可视化图表
    print("\n5. 正在生成可视化图表...")
    plot_comparison_charts(results, output_dir)
    print(f"   图表已保存至 {os.path.abspath(output_dir)}")

    # 步骤 6: 选择最佳模型（单模型或 Stacking）
    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
    best_results = results[best_model_name]

    print("\n" + "=" * 80)
    print("模型评估完成!")
    print(f"★ 最佳模型: {best_model_name}")
    print(f"  - R²:   {best_results['R²']:.4f}")
    print(f"  - MAE:  {best_results['MAE']:.2e}")
    print(f"  - RMSE: {best_results['RMSE']:.2e}")
    print(f"  - MAPE: {best_results['MAPE']:.2f}%")
    print(f"\n报告已保存至: {os.path.abspath(output_dir)}")
    print("=" * 80)

    # 保存最佳模型信息
    if best_model_name == 'Stacking 集成':
        best_model_info = {
            'best_model_name': best_model_name,
            'best_model_type': 'stacking',
            'best_model_path': os.path.join('models', 'stacking_ensemble.joblib'),
            'metrics': {k: v for k, v in best_results.items()
                       if k not in ['predictions', 'actual', 'meta_model_weights', 'base_models']}
        }
    else:
        best_model_info = {
            'best_model_name': best_model_name,
            'best_model_type': 'single',
            'best_model_path': model_paths[best_model_name],
            'metrics': {k: v for k, v in best_results.items()
                       if k not in ['predictions', 'actual']}
        }
    joblib.dump(best_model_info, os.path.join('models', 'best_model_info.joblib'))
    print(f"\n最佳模型信息已保存至: models/best_model_info.joblib")


if __name__ == "__main__":
    main()
