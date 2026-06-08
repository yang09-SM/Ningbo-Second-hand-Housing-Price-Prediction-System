# 宁波二手房房价预测系统 - Product Requirement Document

## Overview
- **Summary**: 本项目旨在构建一个完整的宁波二手房房价预测与管理系统，包含数据预处理、多模型预测、数据库管理和Web应用集成等功能模块。
- **Purpose**: 利用宁波二手房数据集，通过机器学习算法预测房价，并提供二手房信息的管理和展示功能。
- **Target Users**: 房产经纪人、购房者、数据分析人员。

## Goals
- 完成数据集的清洗、预处理和划分
- 构建四种不同的机器学习模型（集成学习、线性回归、支持向量机、神经网络）
- 构建数据库并开发二手房管理系统（增删改查、数据展示）
- 整合预测模型与管理系统，形成完整的智能管理系统

## Non-Goals (Out of Scope)
- 不包括实时房产数据爬虫功能
- 不包括移动端应用
- 不涉及房产交易功能

## Background & Context
- 已有数据集：`lianjia (1).csv`，包含宁波二手房详细信息
- 需要使用 Python 机器学习库（scikit-learn、TensorFlow/PyTorch等）
- 需要构建 Web 应用作为管理系统界面

## Functional Requirements
- **FR-1**: 数据集清洗与预处理功能
- **FR-2**: 数据集划分功能（训练集/验证集/测试集）
- **FR-3**: 线性回归模型构建与训练
- **FR-4**: 支持向量机（SVM）模型构建与训练
- **FR-5**: 集成学习模型（如随机森林、XGBoost等）构建与训练
- **FR-6**: 神经网络模型构建与训练
- **FR-7**: 模型性能评估与对比
- **FR-8**: 数据库设计与创建
- **FR-9**: 二手房信息增删改查功能
- **FR-10**: 二手房数据展示功能
- **FR-11**: 房价预测功能集成
- **FR-12**: Web管理界面

## Non-Functional Requirements
- **NFR-1**: 模型训练时间合理（单模型训练时间不超过10分钟）
- **NFR-2**: Web界面响应时间小于2秒
- **NFR-3**: 代码结构清晰，模块化设计
- **NFR-4**: 数据库设计合理，索引优化

## Constraints
- **Technical**: 使用 Python 作为主要开发语言，推荐使用 Flask 或 Django 作为 Web 框架，SQLite 或 MySQL 作为数据库
- **Business**: 项目应具有良好的可维护性和可扩展性
- **Dependencies**: scikit-learn, pandas, numpy, matplotlib, flask/django, sqlite3/mysql-connector

## Assumptions
- 数据集 `lianjia (1).csv` 是完整且有效的
- Python 3.x 环境可用
- 必要的机器学习和Web开发库可以安装

## Acceptance Criteria

### AC-1: 数据预处理完成
- **Given**: 原始数据集 `lianjia (1).csv`
- **When**: 执行数据清洗和预处理脚本
- **Then**: 生成清洗后的数据集，处理缺失值、异常值，并进行特征工程
- **Verification**: `programmatic`

### AC-2: 数据集划分完成
- **Given**: 清洗后的数据集
- **When**: 执行数据集划分脚本
- **Then**: 数据集被划分为训练集（70%）、验证集（15%）和测试集（15%）
- **Verification**: `programmatic`

### AC-3: 四种模型构建完成
- **Given**: 训练数据集
- **When**: 执行模型训练脚本
- **Then**: 成功构建并训练线性回归、SVM、集成学习和神经网络四种模型
- **Verification**: `programmatic`

### AC-4: 模型性能评价完成
- **Given**: 四种训练好的模型和测试数据集
- **When**: 执行模型评估脚本
- **Then**: 输出各模型的性能指标（MSE、MAE、R²等）并进行对比
- **Verification**: `programmatic`

### AC-5: 数据库构建完成
- **Given**: 清洗后的数据集
- **When**: 执行数据库初始化脚本
- **Then**: 成功创建数据库和表结构，并导入数据
- **Verification**: `programmatic`

### AC-6: 二手房管理系统功能完成
- **Given**: 已构建的数据库
- **When**: 启动Web应用并进行操作
- **Then**: 可以成功进行二手房信息的增删改查和数据展示
- **Verification**: `human-judgment`

### AC-7: 系统集成完成
- **Given**: 训练好的模型和管理系统
- **When**: 在Web界面输入房屋信息进行预测
- **Then**: 系统返回预测房价，并展示相关信息
- **Verification**: `human-judgment`

## Open Questions
- [ ] 集成学习模型具体使用哪种算法？（随机森林、XGBoost、LightGBM等）
- [ ] 神经网络使用哪种框架？（TensorFlow/Keras、PyTorch等）
- [ ] Web框架使用 Flask 还是 Django？
- [ ] 数据库使用 SQLite 还是 MySQL？
