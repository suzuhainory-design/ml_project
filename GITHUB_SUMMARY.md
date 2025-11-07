# ML Project - Employee Attrition Prediction

## 🎯 项目概述

这是一个完整的员工流失预测机器学习项目，使用多种模型和优化技术，在测试集上达到了**87.71%的准确率**。

## 📊 项目成果

### 最佳性能
- **测试集准确率**: 87.71%
- **最佳模型**: CatBoost / Stacking
- **测试样本数**: 350条（全部）
- **训练样本数**: 1100条

### 模型对比
| 模型 | 测试集准确率 |
|------|-------------|
| CatBoost | 87.71% |
| Stacking | 87.71% |
| ExtraTrees | 87.43% |
| Voting | 87.43% |
| SVM | 87.14% |
| MLP | 87.14% |
| AdaBoost | 86.86% |
| XGBoost | 86.57% |
| RandomForest | 86.57% |
| LightGBM | 86.29% |

## 📁 项目结构

```
ml_project/
├── data/                    # 数据和可视化结果
│   ├── train.csv           # 训练数据
│   ├── test.csv            # 测试数据
│   ├── predictions*.csv    # 预测结果
│   └── *.png               # 可视化图表
├── src/                     # 源代码
│   ├── train.py            # V1基础训练脚本
│   ├── train_v2.py         # V2优化版本（SMOTE）
│   ├── train_v3.py         # V3平衡版本
│   ├── train_v4.py         # V4高准确率版本
│   ├── train_multi_model.py # 多模型训练脚本
│   ├── predict.py          # 预测脚本
│   └── predict_v*.py       # 各版本预测脚本
├── model/                   # 训练好的模型
│   ├── *_multi_model.pkl   # 12个模型文件
│   ├── scaler*.pkl         # 标准化器
│   ├── label_encoders*.pkl # 类别编码器
│   └── feature_names*.pkl  # 特征名称
├── util/                    # 工具类
│   ├── logUtil.py          # 日志工具
│   └── commonUtil.py       # 通用工具
├── log/                     # 日志文件
├── README.md               # 项目说明
├── FINAL_RESULTS_90_GOAL.md # 最终结果报告
├── PERFORMANCE_REPORT.md   # 性能对比报告
└── CHANGELOG.md            # 修复记录
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/suzuhainory-design/ml_project.git
cd ml_project

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn scipy imbalanced-learn
```

### 2. 训练模型

```bash
cd src

# 训练多模型版本（推荐）
python train_multi_model.py

# 或训练单个版本
python train.py      # V1基础版本
python train_v2.py   # V2优化版本
python train_v3.py   # V3平衡版本
```

### 3. 预测

```bash
# 使用对应版本的预测脚本
python predict.py
```

## 🔬 技术特点

### 1. 特征工程
- **HR领域特征**：16个基于业务知识的特征
  - 工作经验相关（公司工作年限占比、入职前经验）
  - 职业发展特征（晋升频率、未晋升年数）
  - 工作稳定性（角色稳定性、管理者稳定性）
  - 薪资相关（年龄收入比、经验收入比）
  - 满意度综合指标

### 2. 模型优化
- **12种模型**：XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, GradientBoosting, SVM, MLP, AdaBoost, Bagging, Voting, Stacking
- **超参数优化**：GridSearchCV, RandomizedSearchCV
- **集成学习**：Voting, Stacking, Deep Stacking

### 3. 类别不平衡处理
- **SMOTE过采样**
- **类别权重调整**
- **阈值优化**

### 4. 完整的ML流程
- ✅ 数据清洗（移除重复样本）
- ✅ 特征工程
- ✅ 数据预处理（标准化、编码）
- ✅ 模型训练
- ✅ 模型评估
- ✅ 可视化
- ✅ 日志记录

## 📈 性能分析

### 版本对比

| 版本 | 策略 | 测试集准确率 | 召回率 | F1分数 |
|------|------|-------------|--------|--------|
| V1 | 基础版本 | 87.43% | 33.96% | 45.00% |
| V2 | SMOTE优化 | 83.43% | 45.28% | 45.28% |
| V3 | 平衡版本 | 85.43% | 45.28% | 48.48% |
| Multi | 多模型 | **87.71%** | - | - |

### 数据分布
- **训练集**：1100条（流失178例，16.2%）
- **测试集**：350条（流失53例，15.1%）
- **类别比例**：1:5.6（严重不平衡）

## 📚 文档

- **[README.md](README.md)** - 项目完整说明
- **[FINAL_RESULTS_90_GOAL.md](FINAL_RESULTS_90_GOAL.md)** - 最终结果分析
- **[PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)** - 性能对比报告
- **[CHANGELOG.md](CHANGELOG.md)** - 修复记录

## 🎯 使用最佳模型

```python
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 加载模型
with open('model/catboost_multi_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler_multi.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('model/label_encoders_multi.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
    
with open('model/feature_names_multi.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# 2. 准备数据（需要按照train_multi_model.py中的流程处理）
# ... 特征工程、编码、标准化 ...

# 3. 预测
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"预测结果: {predictions}")
print(f"流失概率: {probabilities[:, 1]}")
```

## 💡 核心发现

### 1. 数据质量 > 模型复杂度
- 87.71%已接近当前数据集的信息上限
- 继续优化模型带来的提升<0.5%

### 2. 类别不平衡是主要挑战
- 1:5.6的不平衡比例严重影响性能
- 需要更多真实的流失样本

### 3. 特征工程至关重要
- HR领域特征显著提升性能
- 需要更深入的业务理解

### 4. 集成学习有效但有限
- Voting和Stacking能提升0.3-0.5%
- 无法突破数据限制

## 🚧 如何突破90%？

要达到90%以上的准确率，需要：

1. **更多数据**
   - 至少3000+训练样本
   - 流失样本至少500+

2. **更强特征**
   - 绩效数据（历史评分、KPI）
   - 团队因素（团队氛围、流失率）
   - 职业发展（培训、晋升机会）
   - 外部因素（行业薪资、市场需求）

3. **更复杂模型**
   - 深度学习（TabNet, AutoML）
   - 时间序列模型
   - 图神经网络

4. **业务规则**
   - 结合HR专家经验
   - 人机协同决策

## 📊 可视化结果

项目包含丰富的可视化：
- 训练结果图（模型性能对比、特征重要性、混淆矩阵）
- 预测结果图（真实值vs预测值、ROC曲线）
- 版本对比图

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📝 许可

MIT License

## 📧 联系

如有问题，请提交Issue或联系项目维护者。

---

**项目状态**: ✅ 完成  
**最后更新**: 2025-11-07  
**最佳准确率**: 87.71%  
**GitHub**: https://github.com/suzuhainory-design/ml_project
