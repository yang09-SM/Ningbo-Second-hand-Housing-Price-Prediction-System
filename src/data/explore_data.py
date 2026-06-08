import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    value = str(value)
    if '万' in value:
        return float(value.replace('万', '')) * 10000
    if '元/平米' in value:
        return float(value.replace('元/平米', ''))
    if '㎡' in value:
        return float(value.replace('㎡', ''))
    if value == '暂无数据' or value == 'None':
        return np.nan
    try:
        return float(value)
    except:
        return np.nan

def preprocess_data(df):
    df_clean = df.copy()
    
    numeric_columns = ['总价', '单价', '建筑面积', '套内面积']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric_value)
    
    if '所在楼层' in df_clean.columns:
        df_clean['楼层位置'] = df_clean['所在楼层'].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else x)
        df_clean['总楼层数'] = df_clean['所在楼层'].apply(lambda x: int(str(x).split('共')[1].split('层')[0]) if pd.notna(x) and '共' in str(x) else np.nan)
    
    if '所在区域' in df_clean.columns:
        df_clean[['区', '板块']] = df_clean['所在区域'].str.split(' ', expand=True, n=1)
    
    return df_clean

def generate_basic_info(df, df_clean):
    info = []
    info.append("=" * 80)
    info.append("数据探索报告")
    info.append("=" * 80)
    info.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info.append("")
    info.append("1. 数据集基本信息")
    info.append("-" * 80)
    info.append(f"原始数据集行数: {len(df)}")
    info.append(f"原始数据集列数: {len(df.columns)}")
    info.append("")
    info.append("列名:")
    for i, col in enumerate(df.columns, 1):
        info.append(f"  {i}. {col}")
    info.append("")
    info.append("原始数据类型:")
    for col in df.columns:
        info.append(f"  {col}: {df[col].dtype}")
    return info

def generate_missing_values_info(df, df_clean):
    info = []
    info.append("2. 缺失值情况")
    info.append("-" * 80)
    missing = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_percent.round(2)})
    missing_df = missing_df.sort_values('缺失数量', ascending=False)
    info.append(missing_df.to_string())
    return info

def generate_statistical_analysis(df, df_clean):
    info = []
    info.append("")
    info.append("3. 基本统计分析")
    info.append("-" * 80)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info.append("数值型变量统计:")
        info.append(df_clean[numeric_cols].describe().to_string())
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df_clean[col].nunique()
        if unique_count <= 20:
            info.append("")
            info.append(f"{col} 的分布:")
            value_counts = df_clean[col].value_counts()
            info.append(value_counts.to_string())
    return info

def generate_outlier_analysis(df, df_clean):
    info = []
    info.append("")
    info.append("4. 异常值识别")
    info.append("-" * 80)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].notna().sum() > 0:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            info.append("")
            info.append(f"{col} 的异常值分析:")
            info.append(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            info.append(f"  异常值范围: < {lower_bound:.2f} 或 > {upper_bound:.2f}")
            info.append(f"  异常值数量: {len(outliers)} ({len(outliers)/len(df_clean)*100:.2f}%)")
    return info

def save_visualizations(df_clean, output_dir):
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 1:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols[:6], 1):
            plt.subplot(2, 3, i)
            df_clean[col].hist(bins=30)
            plt.title(f'{col} 分布直方图')
            plt.xlabel(col)
            plt.ylabel('频数')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'numeric_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    if '总价' in df_clean.columns and '单价' in df_clean.columns and df_clean['总价'].notna().sum() > 0 and df_clean['单价'].notna().sum() > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_clean['单价'], df_clean['总价'], alpha=0.6)
        plt.xlabel('单价 (元/平米)')
        plt.ylabel('总价 (万元)')
        plt.title('单价 vs 总价散点图')
        plt.savefig(os.path.join(output_dir, 'price_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(12, 10))
        corr = df_clean[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('数值变量相关性热力图')
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    if '区' in df_clean.columns and '总价' in df_clean.columns:
        plt.figure(figsize=(12, 6))
        region_price = df_clean.groupby('区')['总价'].median().sort_values(ascending=False)
        region_price.plot(kind='bar')
        plt.title('各区域房价中位数')
        plt.xlabel('区域')
        plt.ylabel('房价中位数 (万元)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'region_price.png'), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    data_path = os.path.join('data', 'lianjia.csv')
    report_dir = 'reports'
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    print("正在加载数据...")
    df = load_data(data_path)
    
    print("正在预处理数据...")
    df_clean = preprocess_data(df)
    
    print("正在生成报告内容...")
    report_content = []
    report_content.extend(generate_basic_info(df, df_clean))
    report_content.extend(generate_missing_values_info(df, df_clean))
    report_content.extend(generate_statistical_analysis(df, df_clean))
    report_content.extend(generate_outlier_analysis(df, df_clean))
    
    report_text = '\n'.join(report_content)
    report_file = os.path.join(report_dir, 'data_exploration_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"正在保存可视化图表到 {report_dir}...")
    save_visualizations(df_clean, report_dir)
    
    print("=" * 80)
    print(report_text)
    print("=" * 80)
    print(f"\n报告已保存到: {os.path.abspath(report_file)}")
    print(f"可视化图表已保存到: {os.path.abspath(report_dir)}")

if __name__ == "__main__":
    main()
