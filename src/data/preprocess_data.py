import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

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
    if value == '暂无数据' or value == 'None' or value == '':
        return np.nan
    try:
        return float(value)
    except:
        return np.nan

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df_clean = df.copy()
    
    numeric_columns = ['总价', '单价', '建筑面积', '套内面积']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric_value)
    
    df_clean = df_clean.dropna(subset=['总价'])
    
    for col in ['单价', '建筑面积']:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('未知')
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['总价', '单价', '建筑面积']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def feature_engineering(df):
    df_feat = df.copy()
    
    if '所在区域' in df_feat.columns:
        df_feat['所在区域'] = df_feat['所在区域'].str.strip()
        region_split = df_feat['所在区域'].str.split(r'\s+', expand=True, n=1)
        df_feat['区'] = region_split[0]
        df_feat['板块'] = region_split[1] if region_split.shape[1] > 1 else '未知'
    
    if '所在楼层' in df_feat.columns:
        def extract_floor_info(x):
            if pd.isna(x) or x == '未知':
                return pd.Series(['未知', np.nan])
            x = str(x)
            floor_pos = '未知'
            total_floor = np.nan
            if '低楼层' in x:
                floor_pos = '低楼层'
            elif '中楼层' in x:
                floor_pos = '中楼层'
            elif '高楼层' in x:
                floor_pos = '高楼层'
            if '共' in x and '层' in x:
                try:
                    total_floor = int(x.split('共')[1].split('层')[0])
                except:
                    pass
            return pd.Series([floor_pos, total_floor])
        
        df_feat[['楼层位置', '总楼层数']] = df_feat['所在楼层'].apply(extract_floor_info)
        df_feat['总楼层数'] = df_feat['总楼层数'].fillna(df_feat['总楼层数'].median())
    
    if '房屋户型' in df_feat.columns:
        def extract_rooms(x):
            if pd.isna(x) or x == '未知':
                return pd.Series([np.nan, np.nan, np.nan, np.nan])
            x = str(x)
            rooms = np.nan
            halls = np.nan
            kitchens = np.nan
            bathrooms = np.nan
            if '室' in x:
                try:
                    rooms = int(x.split('室')[0])
                except:
                    pass
            if '厅' in x:
                try:
                    halls = int(x.split('厅')[0].split('室')[-1])
                except:
                    pass
            if '厨' in x:
                try:
                    kitchens = int(x.split('厨')[0].split('厅')[-1])
                except:
                    pass
            if '卫' in x:
                try:
                    bathrooms = int(x.split('卫')[0].split('厨')[-1])
                except:
                    pass
            return pd.Series([rooms, halls, kitchens, bathrooms])
        
        df_feat[['室数', '厅数', '厨数', '卫数']] = df_feat['房屋户型'].apply(extract_rooms)
        for col in ['室数', '厅数', '卫数']:
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())
    
    return df_feat

def encode_features(df, target_col='总价'):
    df_encode = df.copy()
    
    categorical_cols = ['区', '板块', '楼层位置', '房屋朝向', '装修情况', '建筑类型', 
                       '建筑结构', '交易权属', '房屋用途', '配备电梯', '产权所属']
    
    categorical_cols = [col for col in categorical_cols if col in df_encode.columns]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encode[col + '_encoded'] = le.fit_transform(df_encode[col].astype(str))
        label_encoders[col] = le
    
    return df_encode, label_encoders

def select_features(df, target_col='总价'):
    feature_cols = []
    
    numeric_features = ['单价', '建筑面积', '总楼层数', '室数', '厅数', '卫数']
    for col in numeric_features:
        if col in df.columns:
            feature_cols.append(col)
    
    encoded_features = [col for col in df.columns if col.endswith('_encoded')]
    feature_cols.extend(encoded_features)
    
    feature_cols = [col for col in feature_cols if col in df.columns and col != target_col]
    
    df_selected = df[feature_cols + [target_col]].copy()
    
    df_selected = df_selected.dropna()
    
    return df_selected, feature_cols

def prepare_db_data(df):
    df_db = df.copy()
    
    if '总价' in df_db.columns:
        df_db['总价'] = df_db['总价'].apply(lambda x: f'{x/10000:.0f}万' if pd.notna(x) else x)
    if '单价' in df_db.columns:
        df_db['单价'] = df_db['单价'].apply(lambda x: f'{x:.0f}元/平米' if pd.notna(x) else x)
    if '建筑面积' in df_db.columns:
        df_db['建筑面积'] = df_db['建筑面积'].apply(lambda x: f'{x:.2f}㎡' if pd.notna(x) else x)
    if '套内面积' in df_db.columns:
        df_db['套内面积'] = df_db['套内面积'].apply(lambda x: f'{x:.2f}㎡' if pd.notna(x) else x)
    
    return df_db

def main():
    data_path = os.path.join('data', 'lianjia.csv')
    processed_path = os.path.join('data', 'processed_house_data.csv')
    db_path = os.path.join('data', 'house_data_for_db.csv')
    
    print("=" * 80)
    print("宁波二手房价格预测 - 数据预处理")
    print("=" * 80)
    
    print("\n1. 正在加载原始数据...")
    df = load_data(data_path)
    print(f"   原始数据: {len(df)} 条记录, {len(df.columns)} 个特征")
    
    print("\n2. 正在清洗数据...")
    df_clean = clean_data(df)
    print(f"   清洗后: {len(df_clean)} 条记录")
    
    print("\n3. 正在进行特征工程...")
    df_feat = feature_engineering(df_clean)
    print("   特征工程完成")
    
    print("\n4. 正在编码分类变量...")
    df_encoded, label_encoders = encode_features(df_feat)
    print(f"   编码完成, 生成了 {len(label_encoders)} 个标签编码器")
    
    print("\n5. 正在选择重要特征...")
    df_selected, feature_cols = select_features(df_encoded)
    print(f"   选择了 {len(feature_cols)} 个特征用于预测")
    print(f"   特征列表: {', '.join(feature_cols)}")
    
    print("\n6. 正在准备数据库数据...")
    df_db = prepare_db_data(df_feat)
    
    print("\n7. 正在保存处理后的数据...")
    df_selected.to_csv(processed_path, index=False, encoding='utf-8-sig')
    df_db.to_csv(db_path, index=False, encoding='utf-8-sig')
    
    print("\n8. 正在保存标签编码器...")
    joblib.dump(label_encoders, os.path.join('models', 'label_encoders.joblib'))
    
    print(f"\n" + "=" * 80)
    print(f"数据预处理完成!")
    print(f"  - 预测用数据已保存至: {os.path.abspath(processed_path)}")
    print(f"  - 管理系统数据已保存至: {os.path.abspath(db_path)}")
    print(f"  - 预测特征: {len(feature_cols)} 个")
    print(f"  - 最终数据量: {len(df_selected)} 条")
    print("=" * 80)

if __name__ == "__main__":
    main()
