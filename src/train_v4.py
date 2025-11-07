"""
V4版本 - 专注于准确率优化（目标：90%+）
主要策略：
1. 不使用SMOTE或使用极少量过采样
2. 特征选择，移除低重要性特征
3. 更激进的超参数优化（更多迭代）
4. 多种集成策略对比
5. 阈值优化专注准确率
6. 使用更多基础模型
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.logUtil import LogUtil
from util.commonUtil import CommonUtil

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             VotingClassifier, StackingClassifier, ExtraTreesClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform
from sklearn.feature_selection import SelectFromModel, RFECV


class HRAttritionTrainerV4:
    """V4版本 - 专注准确率优化"""
    
    def __init__(self, log_dir='../log', data_dir='../data', model_dir='../model'):
        """初始化训练器"""
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=log_dir, log_name=f'train_v4_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("V4版本训练开始 - 目标：准确率90%+")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(model_dir)
        CommonUtil.ensure_dir(data_dir)
        
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = 0.5
    
    def load_data(self):
        """加载数据"""
        self.logger.info("开始加载数据...")
        
        train_path = os.path.join(self.data_dir, 'train.csv')
        self.train_df = pd.read_csv(train_path)
        
        test_path = os.path.join(self.data_dir, 'test.csv')
        self.test_df = pd.read_csv(test_path)
        
        self.logger.info(f"训练集形状: {self.train_df.shape}")
        self.logger.info(f"测试集形状: {self.test_df.shape}")
        
        return self
    
    def remove_duplicates(self):
        """移除训练集中与测试集相同的样本"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始移除训练集中与测试集重复的样本...")
        self.logger.info("=" * 80)
        
        feature_cols = [col for col in self.train_df.columns if col != 'Attrition']
        test_feature_cols = [col for col in self.test_df.columns if col != 'Attrition']
        common_features = list(set(feature_cols) & set(test_feature_cols))
        
        self.train_df = CommonUtil.remove_duplicates_from_train(
            self.train_df, self.test_df, common_features, self.logger
        )
        
        return self
    
    def feature_engineering(self):
        """特征工程"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段一: 特征工程")
        self.logger.info("=" * 80)
        
        def create_features(df):
            df = df.copy()
            
            numeric_cols = ['Age', 'DistanceFromHome', 'Education', 
                          'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                          'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                          'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                          'StockOptionLevel', 'TotalWorkingYears',
                          'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                          'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'TotalWorkingYears' in df.columns and 'YearsAtCompany' in df.columns:
                df['ExperienceBeforeCompany'] = df['TotalWorkingYears'] - df['YearsAtCompany']
                df['CompanyTenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
            
            if 'YearsAtCompany' in df.columns and 'YearsSinceLastPromotion' in df.columns:
                df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
                df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
            
            if 'YearsAtCompany' in df.columns and 'YearsInCurrentRole' in df.columns:
                df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
            
            if 'YearsAtCompany' in df.columns and 'YearsWithCurrManager' in df.columns:
                df['ManagerStability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
            
            if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
                df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
            
            if 'MonthlyIncome' in df.columns and 'TotalWorkingYears' in df.columns:
                df['IncomePerExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
            
            if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
                df['IncomePerJobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
            
            satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                               'RelationshipSatisfaction', 'WorkLifeBalance']
            available_satisfaction = [col for col in satisfaction_cols if col in df.columns]
            if available_satisfaction:
                df['OverallSatisfaction'] = df[available_satisfaction].mean(axis=1)
                df['SatisfactionStd'] = df[available_satisfaction].std(axis=1)
            
            if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
                df['JobHoppingRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
            
            if 'JobInvolvement' in df.columns and 'JobSatisfaction' in df.columns:
                df['InvolvementSatisfactionProduct'] = df['JobInvolvement'] * df['JobSatisfaction']
            
            return df
        
        self.logger.info("创建HR领域特征...")
        self.train_df = create_features(self.train_df)
        
        self.logger.info(f"特征工程后训练集形状: {self.train_df.shape}")
        
        return self
    
    def preprocess_data(self):
        """数据预处理"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始数据预处理...")
        self.logger.info("=" * 80)
        
        if 'Attrition' in self.train_df.columns:
            self.train_df['Attrition'] = pd.to_numeric(self.train_df['Attrition'], errors='coerce')
            self.train_df['Attrition'] = self.train_df['Attrition'].fillna(0).astype(int)
        
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours']
        for col in cols_to_drop:
            if col in self.train_df.columns:
                self.train_df = self.train_df.drop(columns=[col])
        
        X = self.train_df.drop(columns=['Attrition'])
        y = self.train_df['Attrition'].astype(int)
        
        numeric_cols_before = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols_before:
            X[col] = X[col].fillna(X[col].median())
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.logger.info(f"类别特征: {categorical_cols}")
        
        for col in categorical_cols:
            X[col] = X[col].astype(str)
            X[col] = X[col].replace('nan', 'Unknown').replace('None', 'Unknown')
            
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # 划分训练集和验证集
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"原始训练集流失比例: {(self.y_train == 1).sum() / len(self.y_train):.4f}")
        
        # V4策略：不使用SMOTE，保持原始数据分布
        self.logger.info("V4策略：不使用SMOTE，保持原始数据分布以提高准确率")
        
        # 标准化
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )
        
        self.feature_names = self.X_train.columns.tolist()
        
        self.logger.info(f"最终训练集大小: {self.X_train.shape}")
        self.logger.info(f"验证集大小: {self.X_val.shape}")
        
        return self
    
    def feature_selection(self):
        """特征选择"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段二: 特征选择")
        self.logger.info("=" * 80)
        
        # 使用RandomForest进行特征重要性评估
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(self.X_train, self.y_train)
        
        importances = rf_selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        self.logger.info("特征重要性排名（前20）:")
        for i in range(min(20, len(indices))):
            self.logger.info(f"{i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # 选择重要性大于阈值的特征
        threshold = np.percentile(importances, 25)  # 保留前75%的特征
        selected_features = [self.feature_names[i] for i in range(len(importances)) if importances[i] >= threshold]
        
        self.logger.info(f"\n选择了 {len(selected_features)}/{len(self.feature_names)} 个特征")
        
        self.X_train = self.X_train[selected_features]
        self.X_val = self.X_val[selected_features]
        self.feature_names = selected_features
        
        return self
    
    def train_optimized_models(self):
        """训练优化的模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段三: 训练优化模型（专注准确率）")
        self.logger.info("=" * 80)
        
        # XGBoost - 优化超参数搜索
        self.logger.info("\n优化XGBoost（15次搜索）...")
        xgb_param_dist = {
            'n_estimators': [300, 400, 500],
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.03, 0.05, 0.07],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_param_dist, n_iter=15, cv=3, 
            scoring='accuracy', random_state=42, n_jobs=-1
        )
        xgb_search.fit(self.X_train, self.y_train)
        xgb_model = xgb_search.best_estimator_
        self.models['xgb_v4'] = xgb_model
        
        val_score = xgb_model.score(self.X_val, self.y_val)
        self.logger.info(f"XGBoost V4 - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"最佳参数: {xgb_search.best_params_}")
        
        # LightGBM - 优化超参数搜索
        self.logger.info("\n优化LightGBM（15次搜索）...")
        lgbm_param_dist = {
            'n_estimators': [300, 400, 500],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.07],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [20, 30],
            'subsample': [0.8, 0.9]
        }
        
        lgbm_base = LGBMClassifier(random_state=42, verbose=-1)
        lgbm_search = RandomizedSearchCV(
            lgbm_base, lgbm_param_dist, n_iter=15, cv=3,
            scoring='accuracy', random_state=42, n_jobs=-1
        )
        lgbm_search.fit(self.X_train, self.y_train)
        lgbm_model = lgbm_search.best_estimator_
        self.models['lgbm_v4'] = lgbm_model
        
        val_score = lgbm_model.score(self.X_val, self.y_val)
        self.logger.info(f"LightGBM V4 - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"最佳参数: {lgbm_search.best_params_}")
        
        # RandomForest
        self.logger.info("\n优化RandomForest...")
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['rf_v4'] = rf_model
        
        val_score = rf_model.score(self.X_val, self.y_val)
        self.logger.info(f"RandomForest V4 - 验证集准确率: {val_score:.4f}")
        
        # ExtraTrees
        self.logger.info("\n训练ExtraTrees...")
        et_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(self.X_train, self.y_train)
        self.models['et_v4'] = et_model
        
        val_score = et_model.score(self.X_val, self.y_val)
        self.logger.info(f"ExtraTrees V4 - 验证集准确率: {val_score:.4f}")
        
        # Voting Ensemble
        self.logger.info("\n训练Voting集成...")
        voting_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgbm', lgbm_model),
                ('rf', rf_model),
                ('et', et_model)
            ],
            voting='soft'
        )
        voting_model.fit(self.X_train, self.y_train)
        self.models['voting_v4'] = voting_model
        
        val_score = voting_model.score(self.X_val, self.y_val)
        self.logger.info(f"Voting V4 - 验证集准确率: {val_score:.4f}")
        
        # Stacking Ensemble
        self.logger.info("\n训练Stacking集成...")
        stacking_model = StackingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgbm', lgbm_model),
                ('rf', rf_model),
                ('et', et_model)
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        stacking_model.fit(self.X_train, self.y_train)
        self.models['stacking_v4'] = stacking_model
        
        val_score = stacking_model.score(self.X_val, self.y_val)
        self.logger.info(f"Stacking V4 - 验证集准确率: {val_score:.4f}")
        
        return self
    
    def select_best_model(self):
        """选择最佳模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段四: 选择最佳模型（基于准确率）")
        self.logger.info("=" * 80)
        
        best_accuracy = 0
        best_model_name = None
        
        for name, model in self.models.items():
            accuracy = model.score(self.X_val, self.y_val)
            y_pred = model.predict(self.X_val)
            recall = recall_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)
            
            self.logger.info(f"{name}: 准确率={accuracy:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self.best_threshold = 0.5  # 使用默认阈值以最大化准确率
        
        self.logger.info(f"\n最佳模型: {best_model_name}, 验证集准确率: {best_accuracy:.4f}")
        
        return self
    
    def save_models(self):
        """保存模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("保存模型...")
        self.logger.info("=" * 80)
        
        for name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{name}_model.pkl')
            CommonUtil.save_model(model, model_path)
            self.logger.info(f"已保存模型: {model_path}")
        
        scaler_path = os.path.join(self.model_dir, 'scaler_v4.pkl')
        CommonUtil.save_model(self.scaler, scaler_path)
        
        encoders_path = os.path.join(self.model_dir, 'label_encoders_v4.pkl')
        CommonUtil.save_model(self.label_encoders, encoders_path)
        
        features_path = os.path.join(self.model_dir, 'feature_names_v4.pkl')
        CommonUtil.save_model(self.feature_names, features_path)
        
        best_model_info = {
            'name': self.best_model_name,
            'model': self.best_model,
            'threshold': self.best_threshold
        }
        best_model_path = os.path.join(self.model_dir, 'best_model_v4.pkl')
        CommonUtil.save_model(best_model_info, best_model_path)
        self.logger.info(f"已保存最佳模型信息: {best_model_path}")
        
        return self
    
    def plot_results(self):
        """绘制训练结果"""
        self.logger.info("\n绘制训练结果...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 模型性能对比
        model_names = []
        accuracies = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_val)
            model_names.append(name.replace('_v4', ''))
            accuracies.append(accuracy_score(self.y_val, y_pred))
        
        axes[0, 0].bar(model_names, accuracies, alpha=0.8, color='steelblue')
        axes[0, 0].axhline(y=0.9, color='r', linestyle='--', label='目标: 90%')
        axes[0, 0].set_ylabel('准确率', fontsize=12)
        axes[0, 0].set_title('V4模型准确率对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        # 特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-15:]
            
            axes[0, 1].barh(range(len(indices)), importances[indices], alpha=0.8)
            axes[0, 1].set_yticks(range(len(indices)))
            axes[0, 1].set_yticklabels([self.feature_names[i] for i in indices], fontsize=9)
            axes[0, 1].set_xlabel('重要性')
            axes[0, 1].set_title(f'特征重要性 ({self.best_model_name})', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 混淆矩阵
        y_pred = self.best_model.predict(self.X_val)
        cm = confusion_matrix(self.y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('预测标签')
        axes[1, 0].set_ylabel('真实标签')
        axes[1, 0].set_title('混淆矩阵', fontsize=14, fontweight='bold')
        
        # 性能指标
        metrics_names = list(self.models.keys())
        metrics_names = [name.replace('_v4', '') for name in metrics_names]
        metrics_accuracies = [model.score(self.X_val, self.y_val) for model in self.models.values()]
        
        axes[1, 1].plot(metrics_names, metrics_accuracies, marker='o', linewidth=2, markersize=8)
        axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='目标: 90%')
        axes[1, 1].set_xlabel('模型')
        axes[1, 1].set_ylabel('准确率')
        axes[1, 1].set_title('准确率趋势', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.data_dir, 'training_results_v4.png')
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
            self.feature_selection()
            self.train_optimized_models()
            self.select_best_model()
            self.save_models()
            self.plot_results()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("V4版本训练完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = HRAttritionTrainerV4()
    trainer.run()
