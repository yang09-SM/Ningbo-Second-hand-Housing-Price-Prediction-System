import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'predictions': y_pred,
        'actual': y_test.values
    }

def generate_comparison_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    report_data = []
    for model_name, metrics in results.items():
        report_data.append({
            '模型': model_name,
            'MSE': f"{metrics['MSE']:.2e}",
            'MAE': f"{metrics['MAE']:.2e}",
            'RMSE': f"{metrics['RMSE']:.2e}",
            'R²': f"{metrics['R²']:.4f}"
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
            f.write("\n")
    
    return report_df

def plot_comparison_charts(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(results.keys())
    metrics = ['MSE', 'MAE', 'RMSE', 'R²']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        if metric != 'R²':
            values = [v / 1e6 for v in values]
            ylabel = f'{metric} (百万)'
        else:
            ylabel = metric
        
        bars = axes[i].bar(model_names, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
        axes[i].set_title(f'{metric} 对比', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(ylabel)
        axes[i].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
    best_results = results[best_model_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(best_results['actual'], best_results['predictions'], alpha=0.6)
    min_val = min(best_results['actual'].min(), best_results['predictions'].min())
    max_val = max(best_results['actual'].max(), best_results['predictions'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('真实价格', fontsize=12)
    axes[0].set_ylabel('预测价格', fontsize=12)
    axes[0].set_title(f'{best_model_name} - 真实值 vs 预测值', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    residuals = best_results['actual'] - best_results['predictions']
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('残差', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].set_title(f'{best_model_name} - 残差分布', fontsize=14, fontweight='bold')
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
        '神经网络': os.path.join('models', 'neural_network.joblib')
    }
    output_dir = 'reports'
    
    print("=" * 80)
    print("宁波二手房价格预测 - 模型评估")
    print("=" * 80)
    
    print("\n1. 正在加载测试数据...")
    X_test, y_test = load_test_data(test_path)
    print(f"   测试集: {len(X_test)} 条")
    
    print("\n2. 正在加载并评估模型...")
    results = {}
    for model_name, model_path in model_paths.items():
        print(f"   - {model_name}...", end='', flush=True)
        model, scaler = load_model(model_path)
        metrics = evaluate_model(model, scaler, X_test, y_test)
        results[model_name] = metrics
        print(f" 完成 (R²={metrics['R²']:.4f})")
    
    print("\n3. 正在生成对比报告...")
    report_df = generate_comparison_report(results, output_dir)
    print(f"   {len(results)} 个模型已评估")
    
    print("\n4. 正在生成可视化图表...")
    plot_comparison_charts(results, output_dir)
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
    print(f"\n" + "=" * 80)
    print(f"模型评估完成!")
    print(f"最佳模型: {best_model_name}")
    print(f"  - R²: {results[best_model_name]['R²']:.4f}")
    print(f"  - MAE: {results[best_model_name]['MAE']:.2e}")
    print(f"报告已保存至: {os.path.abspath(output_dir)}")
    print("=" * 80)
    
    best_model_info = {
        'best_model_name': best_model_name,
        'best_model_path': model_paths[best_model_name]
    }
    joblib.dump(best_model_info, os.path.join('models', 'best_model_info.joblib'))

if __name__ == "__main__":
    main()
