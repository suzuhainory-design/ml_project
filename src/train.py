"""
模型训练脚本
实现数据预处理、特征工程、高级集成学习和超参数优化
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.logUtil import LogUtil
from util.commonUtil import CommonUtil

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform


class HRAttritionTrainer:
    """HR员工流失预测模型训练器"""
    
    def __init__(self, log_dir='../log', data_dir='../data', model_dir='../model'):
        """初始化训练器"""
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # 创建日志记录器
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=log_dir, log_name=f'train_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("HR员工流失预测模型训练开始")
        self.logger.info("=" * 80)
        
        # 确保目录存在
        CommonUtil.ensure_dir(model_dir)
        CommonUtil.ensure_dir(data_dir)
        
        # 数据存储
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = None
        
        # 模型存储
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def load_data(self):
        """加载数据"""
        self.logger.info("开始加载数据...")
        
        # 读取训练集
        train_path = os.path.join(self.data_dir, 'train.csv')
        self.train_df = pd.read_csv(train_path)
        
        # 读取测试集
        test_path = os.path.join(self.data_dir, 'test.csv')
        self.test_df = pd.read_csv(test_path)
        
        self.logger.info(f"训练集形状: {self.train_df.shape}")
        self.logger.info(f"测试集形状: {self.test_df.shape}")
        self.logger.info(f"训练集列: {list(self.train_df.columns)}")
        self.logger.info(f"测试集列: {list(self.test_df.columns)}")
        
        return self
    
    def remove_duplicates(self):
        """移除训练集中与测试集相同的样本"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始移除训练集中与测试集重复的样本...")
        self.logger.info("=" * 80)
        
        # 获取特征列（排除目标列）
        feature_cols = [col for col in self.train_df.columns if col != 'Attrition']
        
        # 确保测试集也有这些特征列
        test_feature_cols = [col for col in self.test_df.columns if col != 'Attrition']
        
        # 使用共同的特征列
        common_features = list(set(feature_cols) & set(test_feature_cols))
        
        self.logger.info(f"用于比较的特征列数量: {len(common_features)}")
        
        # 移除重复样本
        self.train_df = CommonUtil.remove_duplicates_from_train(
            self.train_df, 
            self.test_df, 
            common_features, 
            self.logger
        )
        
        return self
    
    def feature_engineering(self):
        """阶段一: 特征工程 - 基于HR领域知识创建新特征"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段一: 特征工程")
        self.logger.info("=" * 80)
        
        def create_features(df):
            """创建新特征"""
            df = df.copy()
            
            # 转换数值列
            numeric_cols = ['Age', 'DistanceFromHome', 'Education', 'EmployeeNumber', 
                          'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                          'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                          'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                          'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
                          'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                          'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 1. 工作经验相关特征
            if 'TotalWorkingYears' in df.columns and 'YearsAtCompany' in df.columns:
                df['ExperienceBeforeCompany'] = df['TotalWorkingYears'] - df['YearsAtCompany']
                df['CompanyTenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
            
            # 2. 职业发展特征
            if 'YearsAtCompany' in df.columns and 'YearsSinceLastPromotion' in df.columns:
                df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
                df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
            
            # 3. 工作稳定性特征
            if 'YearsAtCompany' in df.columns and 'YearsInCurrentRole' in df.columns:
                df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
            
            if 'YearsAtCompany' in df.columns and 'YearsWithCurrManager' in df.columns:
                df['ManagerStability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
            
            # 4. 薪资相关特征
            if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
                df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
            
            if 'MonthlyIncome' in df.columns and 'TotalWorkingYears' in df.columns:
                df['IncomePerExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
            
            if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
                df['IncomePerJobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
            
            # 5. 工作满意度综合指标
            satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                               'RelationshipSatisfaction', 'WorkLifeBalance']
            available_satisfaction = [col for col in satisfaction_cols if col in df.columns]
            if available_satisfaction:
                df['OverallSatisfaction'] = df[available_satisfaction].mean(axis=1)
                df['SatisfactionStd'] = df[available_satisfaction].std(axis=1)
            
            # 6. 年龄分组
            if 'Age' in df.columns:
                df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                                       labels=['Young', 'Middle', 'Senior', 'Veteran'])
            
            # 7. 收入分组
            if 'MonthlyIncome' in df.columns:
                df['IncomeGroup'] = pd.qcut(df['MonthlyIncome'], q=4, 
                                           labels=['Low', 'Medium', 'High', 'VeryHigh'],
                                           duplicates='drop')
            
            # 8. 工作年限分组
            if 'TotalWorkingYears' in df.columns:
                df['ExperienceGroup'] = pd.cut(df['TotalWorkingYears'], 
                                              bins=[-1, 5, 10, 20, 100],
                                              labels=['Entry', 'Mid', 'Senior', 'Expert'])
            
            # 9. 跳槽频率
            if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
                df['JobHoppingRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
            
            # 10. 工作投入与满意度交互
            if 'JobInvolvement' in df.columns and 'JobSatisfaction' in df.columns:
                df['InvolvementSatisfactionProduct'] = df['JobInvolvement'] * df['JobSatisfaction']
            
            return df
        
        self.logger.info("创建HR领域特征...")
        self.train_df = create_features(self.train_df)
        self.test_df = create_features(self.test_df)
        
        self.logger.info(f"特征工程后训练集形状: {self.train_df.shape}")
        self.logger.info(f"特征工程后测试集形状: {self.test_df.shape}")
        
        return self
    
    def preprocess_data(self):
        """数据预处理"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始数据预处理...")
        self.logger.info("=" * 80)
        
        # 处理目标变量
        if 'Attrition' in self.train_df.columns:
            # 先转换为数值，再确保为0或1
            self.train_df['Attrition'] = pd.to_numeric(self.train_df['Attrition'], errors='coerce')
            # 处理可能的字符串值
            self.train_df.loc[self.train_df['Attrition'].isna(), 'Attrition'] = self.train_df.loc[self.train_df['Attrition'].isna(), 'Attrition'].map({'Yes': 1, 'No': 0})
            self.train_df['Attrition'] = self.train_df['Attrition'].fillna(0).astype(int)
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours']
        for col in cols_to_drop:
            if col in self.train_df.columns:
                self.train_df = self.train_df.drop(columns=[col])
            if col in self.test_df.columns:
                self.test_df = self.test_df.drop(columns=[col])
        
        # 分离特征和目标
        X = self.train_df.drop(columns=['Attrition'])
        y = self.train_df['Attrition'].astype(int)
        
        # 先填充缺失值（在编码之前）
        numeric_cols_before = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols_before:
            X[col] = X[col].fillna(X[col].median())
        
        # 处理类别特征（包括特征工程创建的category类型）
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.logger.info(f"类别特征: {categorical_cols}")
        
        for col in categorical_cols:
            # 先转换为字符串，再填充缺失值
            X[col] = X[col].astype(str)
            X[col] = X[col].replace('nan', 'Unknown').replace('None', 'Unknown')
            
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # 划分训练集和验证集
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )
        
        self.feature_names = self.X_train.columns.tolist()
        
        self.logger.info(f"训练集大小: {self.X_train.shape}")
        self.logger.info(f"验证集大小: {self.X_val.shape}")
        self.logger.info(f"特征数量: {len(self.feature_names)}")
        self.logger.info(f"目标分布 - 流失: {(self.y_train == 1).sum()}, 未流失: {(self.y_train == 0).sum()}")
        
        return self
    
    def train_voting_classifier(self):
        """阶段二: 实现Voting分类器"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段二: 高级集成学习 - Voting分类器")
        self.logger.info("=" * 80)
        
        # 定义基础模型
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ]
        
        # 创建Voting分类器
        voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        self.logger.info("训练Voting分类器...")
        voting_clf.fit(self.X_train, self.y_train)
        
        # 评估
        train_score = voting_clf.score(self.X_train, self.y_train)
        val_score = voting_clf.score(self.X_val, self.y_val)
        
        self.logger.info(f"Voting分类器 - 训练集准确率: {train_score:.4f}")
        self.logger.info(f"Voting分类器 - 验证集准确率: {val_score:.4f}")
        
        self.models['voting'] = voting_clf
        
        return self
    
    def train_ultra_ensemble(self):
        """实现Ultra Ensemble（多层次集成）"""
        self.logger.info("\n实现Ultra Ensemble...")
        
        # 第一层：多样化的基础模型
        base_models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
            'xgb': XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='logloss'),
            'lgbm': LGBMClassifier(n_estimators=200, max_depth=8, random_state=42, verbose=-1),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'svc': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        
        self.logger.info("训练Ultra Ensemble基础模型...")
        for name, model in base_models.items():
            model.fit(self.X_train, self.y_train)
            val_score = model.score(self.X_val, self.y_val)
            self.logger.info(f"  {name}: 验证集准确率 = {val_score:.4f}")
        
        # 第二层：使用基础模型的预测作为元特征
        meta_features_train = np.column_stack([
            model.predict_proba(self.X_train)[:, 1] for model in base_models.values()
        ])
        meta_features_val = np.column_stack([
            model.predict_proba(self.X_val)[:, 1] for model in base_models.values()
        ])
        
        # 元学习器
        meta_learner = LogisticRegression(random_state=42)
        meta_learner.fit(meta_features_train, self.y_train)
        
        val_score = meta_learner.score(meta_features_val, self.y_val)
        self.logger.info(f"Ultra Ensemble - 验证集准确率: {val_score:.4f}")
        
        self.models['ultra_ensemble'] = {
            'base_models': base_models,
            'meta_learner': meta_learner
        }
        
        return self
    
    def train_catboost(self):
        """实现CatBoost优化"""
        self.logger.info("\n实现CatBoost优化...")
        
        if CatBoostClassifier is None:
            self.logger.warning("CatBoost未安装，跳过此模型")
            return self
        
        catboost_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            random_state=42,
            verbose=False
        )
        
        catboost_model.fit(self.X_train, self.y_train)
        
        train_score = catboost_model.score(self.X_train, self.y_train)
        val_score = catboost_model.score(self.X_val, self.y_val)
        
        self.logger.info(f"CatBoost - 训练集准确率: {train_score:.4f}")
        self.logger.info(f"CatBoost - 验证集准确率: {val_score:.4f}")
        
        self.models['catboost'] = catboost_model
        
        return self
    
    def train_deep_stacking(self):
        """实现Deep Stacking"""
        self.logger.info("\n实现Deep Stacking...")
        
        # 第一层模型
        level1_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ]
        
        # 第二层模型（元学习器）
        meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # 创建Stacking分类器
        stacking_clf = StackingClassifier(
            estimators=level1_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        self.logger.info("训练Deep Stacking模型...")
        stacking_clf.fit(self.X_train, self.y_train)
        
        train_score = stacking_clf.score(self.X_train, self.y_train)
        val_score = stacking_clf.score(self.X_val, self.y_val)
        
        self.logger.info(f"Deep Stacking - 训练集准确率: {train_score:.4f}")
        self.logger.info(f"Deep Stacking - 验证集准确率: {val_score:.4f}")
        
        self.models['deep_stacking'] = stacking_clf
        
        return self
    
    def hyperparameter_optimization(self):
        """阶段三: 深度超参数优化"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段三: 深度超参数优化")
        self.logger.info("=" * 80)
        
        # 对最佳模型进行超参数优化
        self.logger.info("对XGBoost进行超参数优化...")
        
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 10)
        }
        
        xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
        
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_xgb = random_search.best_estimator_
        train_score = best_xgb.score(self.X_train, self.y_train)
        val_score = best_xgb.score(self.X_val, self.y_val)
        
        self.logger.info(f"最佳参数: {random_search.best_params_}")
        self.logger.info(f"优化后XGBoost - 训练集准确率: {train_score:.4f}")
        self.logger.info(f"优化后XGBoost - 验证集准确率: {val_score:.4f}")
        
        self.models['xgb_optimized'] = best_xgb
        
        return self
    
    def select_best_model(self):
        """选择最佳模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("选择最佳模型...")
        self.logger.info("=" * 80)
        
        best_score = 0
        best_name = None
        
        for name, model in self.models.items():
            if name == 'ultra_ensemble':
                # Ultra Ensemble需要特殊处理
                base_models = model['base_models']
                meta_learner = model['meta_learner']
                meta_features = np.column_stack([
                    m.predict_proba(self.X_val)[:, 1] for m in base_models.values()
                ])
                score = meta_learner.score(meta_features, self.y_val)
            else:
                score = model.score(self.X_val, self.y_val)
            
            self.logger.info(f"{name}: 验证集准确率 = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        self.logger.info(f"\n最佳模型: {best_name}, 验证集准确率: {best_score:.4f}")
        
        return self
    
    def save_models(self):
        """保存模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("保存模型...")
        self.logger.info("=" * 80)
        
        # 保存所有模型
        for name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{name}_model.pkl')
            CommonUtil.save_model(model, model_path)
            self.logger.info(f"已保存模型: {model_path}")
        
        # 保存预处理器
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        CommonUtil.save_model(self.scaler, scaler_path)
        self.logger.info(f"已保存标准化器: {scaler_path}")
        
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        CommonUtil.save_model(self.label_encoders, encoders_path)
        self.logger.info(f"已保存标签编码器: {encoders_path}")
        
        # 保存特征名称
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        CommonUtil.save_model(self.feature_names, features_path)
        self.logger.info(f"已保存特征名称: {features_path}")
        
        # 保存最佳模型信息
        best_model_info = {
            'name': self.best_model_name,
            'model': self.best_model
        }
        best_model_path = os.path.join(self.model_dir, 'best_model.pkl')
        CommonUtil.save_model(best_model_info, best_model_path)
        self.logger.info(f"已保存最佳模型信息: {best_model_path}")
        
        return self
    
    def plot_results(self):
        """绘制训练结果"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("绘制训练结果...")
        self.logger.info("=" * 80)
        
        # 1. 模型性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准备数据
        model_names = []
        train_scores = []
        val_scores = []
        
        for name, model in self.models.items():
            if name == 'ultra_ensemble':
                base_models = model['base_models']
                meta_learner = model['meta_learner']
                
                meta_features_train = np.column_stack([
                    m.predict_proba(self.X_train)[:, 1] for m in base_models.values()
                ])
                meta_features_val = np.column_stack([
                    m.predict_proba(self.X_val)[:, 1] for m in base_models.values()
                ])
                
                train_score = meta_learner.score(meta_features_train, self.y_train)
                val_score = meta_learner.score(meta_features_val, self.y_val)
            else:
                train_score = model.score(self.X_train, self.y_train)
                val_score = model.score(self.X_val, self.y_val)
            
            model_names.append(name)
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        # 绘制性能对比
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_scores, width, label='训练集', alpha=0.8)
        axes[0, 0].bar(x + width/2, val_scores, width, label='验证集', alpha=0.8)
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_title('模型性能对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 特征重要性（使用最佳模型）
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-20:]  # 前20个重要特征
            
            axes[0, 1].barh(range(len(indices)), importances[indices], alpha=0.8)
            axes[0, 1].set_yticks(range(len(indices)))
            axes[0, 1].set_yticklabels([self.feature_names[i] for i in indices], fontsize=8)
            axes[0, 1].set_xlabel('重要性')
            axes[0, 1].set_title(f'特征重要性 ({self.best_model_name})')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '该模型不支持特征重要性', 
                          ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('特征重要性')
        
        # 3. 混淆矩阵
        y_pred = self.best_model.predict(self.X_val)
        cm = confusion_matrix(self.y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('预测标签')
        axes[1, 0].set_ylabel('真实标签')
        axes[1, 0].set_title(f'混淆矩阵 ({self.best_model_name})')
        
        # 4. 目标分布
        target_counts = pd.Series(self.y_train).value_counts()
        axes[1, 1].bar(['未流失', '流失'], target_counts.values, alpha=0.8, color=['green', 'red'])
        axes[1, 1].set_ylabel('样本数量')
        axes[1, 1].set_title('训练集目标分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        for i, v in enumerate(target_counts.values):
            axes[1, 1].text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图形
        plot_path = os.path.join(self.data_dir, 'training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"已保存训练结果图: {plot_path}")
        
        plt.close()
        
        return self
    
    def run(self):
        """运行完整的训练流程"""
        try:
            self.load_data()
            self.remove_duplicates()
            self.feature_engineering()
            self.preprocess_data()
            self.train_voting_classifier()
            self.train_ultra_ensemble()
            self.train_catboost()
            self.train_deep_stacking()
            self.hyperparameter_optimization()
            self.select_best_model()
            self.save_models()
            self.plot_results()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("训练完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = HRAttritionTrainer()
    trainer.run()
