# HR员工流失预测机器学习项目

## 项目简介

本项目是一个完整的机器学习解决方案，用于预测HR领域的员工流失问题。项目实现了从数据预处理、特征工程、模型训练到预测评估的完整流程，采用了多种高级集成学习算法和超参数优化技术。

## 项目结构

```
ml_project/
├── data/                          # 数据文件夹
│   ├── train.csv                  # 训练数据集
│   ├── test.csv                   # 测试数据集
│   ├── training_results.png       # 训练结果可视化图
│   ├── prediction_results.png     # 预测结果可视化图
│   └── predictions.csv            # 预测结果CSV文件
├── src/                           # 源代码文件夹
│   ├── train.py                   # 模型训练脚本
│   └── predict.py                 # 模型预测脚本
├── log/                           # 日志文件夹
│   ├── train_*.log                # 训练日志
│   └── predict_*.log              # 预测日志
├── model/                         # 模型文件夹
│   ├── best_model.pkl             # 最佳模型
│   ├── voting_model.pkl           # Voting集成模型
│   ├── ultra_ensemble_model.pkl   # Ultra Ensemble模型
│   ├── catboost_model.pkl         # CatBoost模型
│   ├── deep_stacking_model.pkl    # Deep Stacking模型
│   ├── xgb_optimized_model.pkl    # 优化后的XGBoost模型
│   ├── scaler.pkl                 # 标准化器
│   ├── label_encoders.pkl         # 标签编码器
│   └── feature_names.pkl          # 特征名称列表
└── util/                          # 工具类文件夹
    ├── __init__.py                # 包初始化文件
    ├── logUtil.py                 # 日志工具类
    └── commonUtil.py              # 通用工具类
```

## 核心功能

### 1. 数据处理

- **重复样本移除**: 自动检测并移除训练集中与测试集相同的样本，确保模型评估的准确性
- **缺失值处理**: 智能处理数值型和类别型特征的缺失值
- **数据标准化**: 使用StandardScaler对数值特征进行标准化

### 2. 特征工程（阶段一）

基于HR领域知识创建了多个衍生特征：

**工作经验相关特征**
- ExperienceBeforeCompany: 入职前工作经验
- CompanyTenureRatio: 公司工作年限占比

**职业发展特征**
- PromotionRate: 晋升频率
- YearsWithoutPromotion: 未晋升年数

**工作稳定性特征**
- RoleStability: 角色稳定性
- ManagerStability: 管理者稳定性

**薪资相关特征**
- IncomePerAge: 年龄收入比
- IncomePerExperience: 经验收入比
- IncomePerJobLevel: 职级收入比

**满意度综合指标**
- OverallSatisfaction: 总体满意度
- SatisfactionStd: 满意度标准差

**分组特征**
- AgeGroup: 年龄分组
- IncomeGroup: 收入分组
- ExperienceGroup: 经验分组

**其他特征**
- JobHoppingRate: 跳槽频率
- InvolvementSatisfactionProduct: 投入与满意度交互项

### 3. 高级集成学习（阶段二）

#### Voting分类器
集成了多个基础模型进行软投票：
- RandomForest
- GradientBoosting
- XGBoost
- LightGBM

#### Ultra Ensemble
多层次集成学习架构：
- 第一层：7个多样化的基础模型（RF, GB, XGBoost, LightGBM, LogisticRegression, SVC, KNN）
- 第二层：使用LogisticRegression作为元学习器

#### CatBoost优化
专门针对类别特征优化的梯度提升算法

#### Deep Stacking
深度堆叠集成：
- 第一层：RF, XGBoost, LightGBM
- 元学习器：GradientBoosting
- 使用5折交叉验证

### 4. 深度超参数优化（阶段三）

使用RandomizedSearchCV对XGBoost进行超参数优化：
- n_estimators: 树的数量
- max_depth: 树的最大深度
- learning_rate: 学习率
- subsample: 样本采样比例
- colsample_bytree: 特征采样比例
- min_child_weight: 最小叶子节点权重

## 模型性能

### 训练集性能

| 模型 | 验证集准确率 |
|------|-------------|
| Voting分类器 | 0.8682 |
| Ultra Ensemble | 0.8545 |
| CatBoost | 0.8545 |
| Deep Stacking | 0.8409 |
| **XGBoost优化** | **0.8727** |

**最佳模型**: XGBoost优化版，验证集准确率达到87.27%

### 测试集性能

- **准确率 (Accuracy)**: 0.8743
- **精确率 (Precision)**: 0.6667
- **召回率 (Recall)**: 0.3396
- **F1分数**: 0.4500
- **AUC**: 0.8351

### 混淆矩阵

|  | 预测未流失 | 预测流失 |
|---|-----------|---------|
| **真实未流失** | 288 (TN) | 9 (FP) |
| **真实流失** | 35 (FN) | 18 (TP) |

## 使用方法

### 环境要求

```bash
Python 3.11+
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
scipy
```

### 安装依赖

```bash
cd ml_project
python3.11 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn scipy
```

### 训练模型

```bash
cd src
python train.py
```

训练脚本会：
1. 加载训练和测试数据
2. 移除训练集中与测试集重复的样本
3. 执行特征工程
4. 训练多个集成学习模型
5. 进行超参数优化
6. 选择最佳模型
7. 保存所有模型和预处理器
8. 生成训练结果可视化图

### 预测

```bash
cd src
python predict.py
```

预测脚本会：
1. 加载最佳模型和预处理器
2. 加载测试数据
3. 应用相同的特征工程
4. 进行预测
5. 评估预测结果（如果有真实标签）
6. 生成预测结果可视化图
7. 保存预测结果到CSV文件

## 日志系统

项目实现了完整的日志记录系统：

- **文件日志**: 所有运行日志保存在`log/`文件夹下
- **控制台日志**: 实时显示运行进度
- **日志级别**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **时间戳**: 每条日志都包含精确的时间戳

## 输出文件

### 训练阶段

1. **模型文件** (model/)
   - 所有训练的模型（pkl格式）
   - 预处理器（scaler, label_encoders）
   - 特征名称列表

2. **可视化图** (data/training_results.png)
   - 模型性能对比
   - 特征重要性
   - 混淆矩阵
   - 目标分布

3. **日志文件** (log/train_*.log)
   - 详细的训练过程记录

### 预测阶段

1. **预测结果** (data/predictions.csv)
   - 样本索引
   - 预测标签
   - 预测概率
   - 真实标签（如果有）
   - 预测是否正确（如果有真实标签）

2. **可视化图** (data/prediction_results.png)
   - 真实值vs预测值对比
   - 混淆矩阵
   - 分布对比
   - ROC曲线或性能指标

3. **日志文件** (log/predict_*.log)
   - 详细的预测过程记录
   - 性能指标
   - 分类报告

## 技术亮点

1. **数据清洗**: 自动移除训练集与测试集的重复样本
2. **领域特征工程**: 基于HR领域知识创建高质量特征
3. **多模型集成**: 实现了4种不同的集成学习策略
4. **自动化超参数优化**: 使用随机搜索优化模型性能
5. **完整的日志系统**: 便于调试和追踪
6. **可视化分析**: 自动生成多种可视化图表
7. **模块化设计**: 代码结构清晰，易于维护和扩展

## 项目特色

- ✅ 完整的机器学习流程
- ✅ 高级集成学习算法
- ✅ 自动化特征工程
- ✅ 超参数优化
- ✅ 详细的日志记录
- ✅ 丰富的可视化输出
- ✅ 模块化代码设计
- ✅ 生产级代码质量

## 性能分析

### 优势

1. **高准确率**: 测试集准确率达到87.43%
2. **良好的泛化能力**: AUC达到0.8351
3. **高精确率**: 对于流失预测，精确率达到66.67%

### 改进方向

1. **召回率提升**: 当前召回率为33.96%，可以通过调整决策阈值或使用SMOTE等方法处理类别不平衡问题
2. **特征选择**: 可以进一步进行特征选择，移除冗余特征
3. **模型融合**: 可以尝试更复杂的模型融合策略

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过项目日志系统查看详细运行信息。
