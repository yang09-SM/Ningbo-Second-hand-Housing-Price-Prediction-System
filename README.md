# 宁波二手房价格预测系统

一个基于机器学习的宁波二手房价格预测系统，包含数据处理、模型训练、评估和 Web 应用功能。

## 功能特性

- 数据预处理和探索性分析
- 多种机器学习模型训练（线性回归、SVM、随机森林、神经网络）
- 模型性能评估和对比
- SQLite 数据库存储
- Flask 后端 API
- 现代化 Web 界面
- 二手房价格预测功能
- 房源列表和搜索功能

## 项目结构

```
Ningbo Second-hand Housing Price Prediction System/
├── data/                          # 数据目录
│   ├── lianjia.csv                # 原始数据
│   ├── processed_house_data.csv   # 处理后的数据
│   ├── house_data_for_db.csv      # 数据库导入数据
│   ├── train.csv                  # 训练集
│   ├── val.csv                    # 验证集
│   ├── test.csv                   # 测试集
│   └── houses.db                  # SQLite 数据库
├── models/                        # 模型目录
│   ├── linear_regression.joblib   # 线性回归模型
│   ├── svm.joblib                 # SVM 模型
│   ├── random_forest.joblib       # 随机森林模型
│   ├── neural_network.joblib      # 神经网络模型
│   ├── label_encoders.joblib      # 标签编码器
│   └── best_model_info.joblib     # 最佳模型信息
├── reports/                       # 报告目录
├── src/                           # 源代码目录
│   ├── data/                      # 数据处理模块
│   │   ├── explore_data.py        # 数据探索
│   │   ├── preprocess_data.py     # 数据预处理
│   │   └── split_data.py          # 数据集划分
│   ├── models/                    # 模型模块
│   │   ├── train_linear_regression.py
│   │   ├── train_svm.py
│   │   ├── train_random_forest.py
│   │   ├── train_neural_network.py
│   │   └── evaluate_models.py     # 模型评估
│   ├── database/                  # 数据库模块
│   │   └── init_db.py             # 数据库初始化
│   └── web/                       # Web 应用模块
│       ├── app.py                 # Flask 应用
│       ├── templates/
│       │   └── index.html         # 前端页面
│       └── static/
├── requirements.txt               # 依赖包
└── README.md                      # 项目说明
```

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
python src/data/preprocess_data.py
```

### 3. 数据集划分

```bash
python src/data/split_data.py
```

### 4. 训练模型

依次运行四个模型训练脚本：

```bash
python src/models/train_linear_regression.py
python src/models/train_svm.py
python src/models/train_random_forest.py
python src/models/train_neural_network.py
```

### 5. 评估模型

```bash
python src/models/evaluate_models.py
```

### 6. 初始化数据库

```bash
python src/database/init_db.py
```

### 7. 启动 Web 应用

```bash
python src/web/app.py
```

然后在浏览器中访问：http://localhost:5000

## API 说明

### 房源相关

- `GET /api/houses` - 获取房源列表（支持分页、搜索、筛选）
- `GET /api/houses/<id>` - 获取单个房源详情
- `POST /api/houses` - 创建新房源
- `PUT /api/houses/<id>` - 更新房源信息
- `DELETE /api/houses/<id>` - 删除房源
- `GET /api/districts` - 获取区域列表

### 预测相关

- `POST /api/predict` - 预测二手房价格
- `GET /api/feature-options` - 获取特征选项

## 使用说明

### 房源列表

1. 在首页可以浏览所有房源
2. 使用搜索框搜索小区名称或区域
3. 使用区域筛选器按区域筛选
4. 点击分页按钮翻页

### 价格预测

1. 切换到"价格预测"标签
2. 填写房源信息（单价、建筑面积、楼层等）
3. 点击"预测价格"按钮
4. 查看预测结果和使用的模型

## 技术栈

- **后端**: Python, Flask
- **数据库**: SQLite
- **机器学习**: scikit-learn, pandas, numpy
- **前端**: HTML, Tailwind CSS, JavaScript
- **可视化**: matplotlib, seaborn

## 模型说明

系统训练了四种机器学习模型：

1. **线性回归** - 简单的线性模型
2. **支持向量机 (SVM)** - 基于核函数的非线性模型
3. **随机森林** - 集成学习模型，通常表现最佳
4. **神经网络** - 使用 MLPRegressor 的深度学习模型

系统会自动选择表现最好的模型用于预测。

## 许可证

MIT License
