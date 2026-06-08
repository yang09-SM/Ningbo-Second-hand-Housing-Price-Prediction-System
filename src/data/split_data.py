import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_dataset(df, target_col='总价', random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, val_df, test_df

def save_datasets(train_df, val_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    return train_path, val_path, test_path

def main():
    processed_path = os.path.join('data', 'processed_house_data.csv')
    output_dir = 'data'
    
    print("=" * 80)
    print("宁波二手房价格预测 - 数据集划分")
    print("=" * 80)
    
    print("\n1. 正在加载处理后的数据...")
    df = load_processed_data(processed_path)
    print(f"   数据总量: {len(df)} 条记录")
    print(f"   特征数量: {len(df.columns) - 1} 个")
    
    print("\n2. 正在划分数据集...")
    train_df, val_df, test_df = split_dataset(df)
    print(f"   训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    
    print("\n3. 正在保存划分后的数据...")
    train_path, val_path, test_path = save_datasets(train_df, val_df, test_df, output_dir)
    
    print(f"\n" + "=" * 80)
    print(f"数据集划分完成!")
    print(f"  - 训练集: {os.path.abspath(train_path)}")
    print(f"  - 验证集: {os.path.abspath(val_path)}")
    print(f"  - 测试集: {os.path.abspath(test_path)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
