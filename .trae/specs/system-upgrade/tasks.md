# 宁波二手房房价预测系统 - 迭代升级任务清单

## [x] Task 1: 梯度提升模型集成（XGBoost / LightGBM / CatBoost）
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 安装 xgboost、lightgbm、catboost、optuna 依赖
  - 创建 `src/models/train_xgboost.py` — XGBoost 回归模型训练脚本
  - 创建 `src/models/train_lightgbm.py` — LightGBM 回归模型训练脚本
  - 创建 `src/models/train_catboost.py` — CatBoost 回归模型训练脚本
  - 使用 Optuna 实现超参数自动搜索（目标：最大化 R²）
  - 保存训练好的模型和最优超参数配置
- **Acceptance Criteria Addressed**: 高精度梯度提升模型集成
- **Test Requirements**:
  - `programmatic` TR-1.1: 三种模型均成功训练并保存
  - `programmatic` TR-1.2: Optuna 超参搜索完成，输出最优参数
  - `programmatic` TR-1.3: 模型 R² > 0.85（在测试集上）

## [x] Task 2: SHAP 可解释性模块开发
- **Priority**: P0
- **Depends On**: [Task 1]
- **Description**:
  - 安装 shap 依赖
  - 创建 `src/explainability/shap_analyzer.py` — SHAP 分析核心模块
  - 实现 SHAP 值计算（支持 TreeExplainer + KernelExplainer）
  - 实现全局特征重要性分析（Summary Plot — bar chart + bee swarm）
  - 实现单样本预测解释（Force Plot / Waterfall Plot / Dependence Plot）
  - 生成 SHAP 可视化图表并保存为 HTML/PNG
  - 提供 Flask API 接口供前端调用 SHAP 结果
- **Acceptance Criteria Addressed**: SHAP 可解释性分析
- **Test Requirements**:
  - `programmatic` TR-2.1: SHAP 值计算正确，各特征贡献值之和等于预测偏差
  - `programmatic` TR-2.2: 全局/局部可视化图表生成成功
  - `programmatic` TR-2.3: SHAP API 接口可正常返回数据

## [x] Task 3: 模型评估体系升级与 Stacking 集成
- **Priority**: P0
- **Depends On**: [Task 1, Task 2]
- **Description**:
  - 升级 `src/models/evaluate_models.py` — 纳入全部 7 种模型的对比评估
  - 新增 RMSE、MAPE 评估指标
  - 实现 Stacking 集成策略（以 XGBoost/LightGBM/CatBoost 为基模型，线性回归为元模型）
  - 生成增强版模型对比报告（含 SHAP 一致性分析）
  - 更新最佳模型选择逻辑，支持 Stacking 模型作为默认预测模型
- **Acceptance Criteria Addressed**: 模型评估升级 + Stacking 集成
- **Test Requirements**:
  - `programmatic` TR-3.1: 全部 7+Stacking 模型评估完成
  - `programmatic` TR-3.2: Stacking 集成模型 R² 高于任一单模型
  - `programmatic` TR-3.3: 增强版对比报告生成成功

## [x] Task 4: 交互式数据可视化大屏
- **Priority**: P1
- **Depends On**: None
- **Description**:
  - 安装 plotly / echarts-python（或前端直接使用 ECharts CDN）
  - 创建 `src/visualization/dashboard.py` — 可视化数据处理 API
  - 实现首页仪表盘数据接口：
    - 区域房价热力图数据（按宁波各区聚合）
    - 价格分布直方图数据
    - 关键指标卡片（均价、中位数、房源量、模型 R²）
    - 价格趋势折线图数据（按月份）
  - 实现模型性能看板数据接口：
    - 多模型指标雷达图数据
    - 预测 vs 实际散点图数据
    - 残差分布图数据
    - SHAP 特征重要性排行数据
- **Acceptance Criteria Addressed**: 交互式数据可视化大屏
- **Test Requirements**:
  - `programmatic` TR-4.1: 所有可视化 API 接口返回正确格式数据
  - `human-judgement` TR-4.2: 前端图表渲染正常且支持交互

## [x] Task 5: 前端界面现代化改造
- **Priority**: P1
- **Depends On**: [Task 4]
- **Description**:
  - 重构 `src/web/templates/index.html`，引入现代 UI 设计
  - 引入 Element Plus CDN（或本地静态资源）作为 UI 组件库
  - 引入 ECharts CDN 用于绑定交互式图表
  - 重构页面布局为多标签页结构：
    - Tab 1: 数据概览仪表盘（首页）
    - Tab 2: 房源数据管理
    - Tab 3: 智能房价预测（增强版表单 + 结果展示）
    - Tab 4: 模型分析中心（性能对比 + SHAP 解释）
  - 优化预测表单：联动下拉选择、实时校验、智能默认值
  - 增强预测结果页：置信区间、Top-N 影响因素、"查看解释"按钮
- **Acceptance Criteria Addressed**: 前端用户体验升级
- **Test Requirements**:
  - `human-judgement` TR-5.1: 页面布局清晰美观，组件样式统一
  - `human-judgement` TR-5.2: 表单联动和校验功能正常
  - `human-judgement` TR-5.3: 预测结果展示完整且包含解释入口

## [x] Task 6: 后端 API 扩展
- **Priority**: P1
- **Depends On**: [Task 2, Task 3, Task 4]
- **Description**:
  - 扩展 `src/web/app.py`，新增以下 API 端点：
    - `GET /api/dashboard/stats` — 仪表盘关键指标
    - `GET /api/dashboard/heatmap` — 区域房价热力图数据
    - `GET /api/dashboard/trends` — 价格趋势时序数据
    - `GET /api/model/comparison` — 全部模型性能对比数据
    - `GET /api/model/shap/global` — SHAP 全局特征重要性
    - `POST /api/predict/explain` — 带解释的预测接口
    - `POST /api/predict/batch` — 批量预测接口
    - `GET /api/export/report` — 预测报告导出（Excel/PDF）
  - 统一 API 响应格式（JSON 标准 envelope）
  - 添加请求参数校验和错误处理
- **Acceptance Criteria Addressed**:系统工程化增强 + 前后端对接
- **Test Requirements**:
  - `programmatic` TR-6.1: 所有新增 API 端点正常工作
  - `programmatic` TR-6.2: API 响应格式统一，错误处理完善

## [x] Task 7: 高级特征工程管道
- **Priority**: P2
- **Depends On**: None
- **Description**:
  - 扩展 `src/data/preprocess_data.py`，新增高级特征构造逻辑：
    - 地理距离特征：基于经纬度计算到市中心/地铁/学校的距离（若数据中有坐标信息）或基于区域名称的代理变量
    - POI 密度特征：统计各区域周边配套设施数量（若有外部数据源）
    - 时序市场特征：基于历史成交数据计算区域均价滚动均值、价格环比变化率等
    - 交叉特征：面积×单价等级、楼层×建筑类型等有意义的特征组合
  - 更新特征工程流程文档
- **Acceptance Criteria Addressed**: 高级特征工程管道
- **Test Requirements**:
  - `programmatic` TR-7.1: 新增特征成功构造并写入清洗后数据集
  - `programmatic` TR-7.2: 特征相关性分析显示新特征与目标变量相关

## [x] Task 8: 系统集成测试与优化
- **Priority**: P2
- **Depends On**: [Task 3, Task 5, Task 6]
- **Description**:
  - 进行全流程集成测试（数据→模型→API→前端）
  - 验证预测功能端到端正常（输入房屋信息 → 返回带解释的预测结果）
  - 验证 SHAP 解释功能前后端打通
  - 验证可视化大屏数据正确渲染
  - 性能优化：预测响应时间 < 500ms，页面加载时间 < 3s
  - 修复测试中发现的问题
- **Acceptance Criteria Addressed**: 全系统集成验证
- **Test Requirements**:
  - `human-judgement` TR-8.1: 系统全功能正常运行
  - `programmatic` TR-8.2: 预测响应时间满足要求
  - `human-judgement` TR-8.3: 用户体验流畅无明显缺陷

# Task Dependencies
- [Task 2] depends on [Task 1]（SHAP 需要已训练好的树模型）
- [Task 3] depends on [Task 1, Task 2]（评估和集成需要所有模型和 SHAP）
- [Task 4] 无强依赖，可与 Task 1 并行
- [Task 5] depends on [Task 4]（前端需要先有可视化数据 API）
- [Task 6] depends on [Task 2, Task 3, Task 4]（API 需要整合 SHAP、评估、可视化的后端逻辑）
- [Task 7] 无强依赖，可与 Task 1 并行
- [Task 8] depends on [Task 3, Task 5, Task 6]（最终集成测试需要核心模块就绪）
