"""
V1_optimized - 基于V1的全面超参数优化版本
目标：最大化测试集准确率

优化策略：
1. GridSearchCV - 精细网格搜索
2. RandomizedSearchCV - 大范围随机搜索
3. 多模型对比（XGBoost, LightGBM, CatBoost, RF, ET, GB）
4. 集成策略优化（Voting权重优化）
5. 阈值优化
6. 交叉验证优化
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

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             VotingClassifier, ExtraTreesClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer
from scipy.stats import randint, uniform


class HRAttritionTrainerV1Optimized:
    """V1优化版 - 全面超参数调优"""
    
    def __init__(self, log_dir='../log', data_dir='../data', model_dir='../model'):
        """初始化训练器"""
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=log_dir, log_name=f'train_v1_optimized_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("V1优化版训练开始 - 全面超参数调优")
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
        """特征工程（与V1相同）"""
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
        
        # 使用stratified split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"训练集流失比例: {(self.y_train == 1).sum() / len(self.y_train):.4f}")
        self.logger.info(f"验证集流失比例: {(self.y_val == 1).sum() / len(self.y_val):.4f}")
        
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
    
    def optimize_models(self):
        """全面优化模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段二: 全面超参数优化")
        self.logger.info("=" * 80)
        
        # 使用accuracy作为评分标准
        accuracy_scorer = make_scorer(accuracy_score)
        
        # XGBoost - GridSearchCV精细搜索
        self.logger.info("\n1. XGBoost - GridSearchCV精细搜索...")
        xgb_param_grid = {
            'n_estimators': [400, 500, 600],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.03, 0.05, 0.07],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1]
        }
        
        xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_grid = GridSearchCV(
            xgb_base, xgb_param_grid, cv=5, 
            scoring=accuracy_scorer, n_jobs=-1, verbose=0
        )
        xgb_grid.fit(self.X_train, self.y_train)
        xgb_model = xgb_grid.best_estimator_
        self.models['xgb_grid'] = xgb_model
        
        val_score = xgb_model.score(self.X_val, self.y_val)
        self.logger.info(f"XGBoost Grid - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"最佳参数: {xgb_grid.best_params_}")
        
        # LightGBM - GridSearchCV
        self.logger.info("\n2. LightGBM - GridSearchCV精细搜索...")
        lgbm_param_grid = {
            'n_estimators': [400, 500, 600],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.07],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [20, 30]
        }
        
        lgbm_base = LGBMClassifier(random_state=42, verbose=-1)
        lgbm_grid = GridSearchCV(
            lgbm_base, lgbm_param_grid, cv=5,
            scoring=accuracy_scorer, n_jobs=-1, verbose=0
        )
        lgbm_grid.fit(self.X_train, self.y_train)
        lgbm_model = lgbm_grid.best_estimator_
        self.models['lgbm_grid'] = lgbm_model
        
        val_score = lgbm_model.score(self.X_val, self.y_val)
        self.logger.info(f"LightGBM Grid - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"最佳参数: {lgbm_grid.best_params_}")
        
        # RandomForest - GridSearchCV
        self.logger.info("\n3. RandomForest - GridSearchCV...")
        rf_param_grid = {
            'n_estimators': [400, 500, 600],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(
            rf_base, rf_param_grid, cv=5,
            scoring=accuracy_scorer, n_jobs=-1, verbose=0
        )
        rf_grid.fit(self.X_train, self.y_train)
        rf_model = rf_grid.best_estimator_
        self.models['rf_grid'] = rf_model
        
        val_score = rf_model.score(self.X_val, self.y_val)
        self.logger.info(f"RandomForest Grid - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"最佳参数: {rf_grid.best_params_}")
        
        # GradientBoosting - GridSearchCV
        self.logger.info("\n4. GradientBoosting - GridSearchCV...")
        gb_param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'min_samples_split': [2, 5]
        }
        
        gb_base = GradientBoostingClassifier(random_state=42)
        gb_grid = GridSearchCV(
            gb_base, gb_param_grid, cv=5,
            scoring=accuracy_scorer, n_jobs=-1, verbose=0
        )
        gb_grid.fit(self.X_train, self.y_train)
        gb_model = gb_grid.best_estimator_
        self.models['gb_grid'] = gb_model
        
        val_score = gb_model.score(self.X_val, self.y_val)
        self.logger.info(f"GradientBoosting Grid - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"最佳参数: {gb_grid.best_params_}")
        
        # ExtraTrees
        self.logger.info("\n5. ExtraTrees - GridSearchCV...")
        et_param_grid = {
            'n_estimators': [400, 500, 600],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        et_base = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        et_grid = GridSearchCV(
            et_base, et_param_grid, cv=5,
            scoring=accuracy_scorer, n_jobs=-1, verbose=0
        )
        et_grid.fit(self.X_train, self.y_train)
        et_model = et_grid.best_estimator_
        self.models['et_grid'] = et_model
        
        val_score = et_model.score(self.X_val, self.y_val)
        self.logger.info(f"ExtraTrees Grid - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"最佳参数: {et_grid.best_params_}")
        
        # CatBoost（如果可用）
        if CatBoostClassifier is not None:
            self.logger.info("\n6. CatBoost - 优化训练...")
            cat_model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=0
            )
            cat_model.fit(self.X_train, self.y_train)
            self.models['catboost'] = cat_model
            
            val_score = cat_model.score(self.X_val, self.y_val)
            self.logger.info(f"CatBoost - 验证集准确率: {val_score:.4f}")
        
        return self
    
    def create_ensemble(self):
        """创建优化的集成模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段三: 创建优化集成模型")
        self.logger.info("=" * 80)
        
        # 软投票集成
        self.logger.info("\n创建Voting集成（软投票）...")
        estimators = [(name, model) for name, model in self.models.items()]
        
        voting_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        voting_model.fit(self.X_train, self.y_train)
        self.models['voting_soft'] = voting_model
        
        val_score = voting_model.score(self.X_val, self.y_val)
        self.logger.info(f"Voting Soft - 验证集准确率: {val_score:.4f}")
        
        # 加权投票（根据验证集性能）
        self.logger.info("\n创建加权Voting集成...")
        weights = []
        for name, model in self.models.items():
            if name != 'voting_soft':
                weights.append(model.score(self.X_val, self.y_val))
        
        weighted_voting = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items() if name != 'voting_soft'],
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        weighted_voting.fit(self.X_train, self.y_train)
        self.models['voting_weighted'] = weighted_voting
        
        val_score = weighted_voting.score(self.X_val, self.y_val)
        self.logger.info(f"Voting Weighted - 验证集准确率: {val_score:.4f}")
        self.logger.info(f"权重: {[f'{w:.4f}' for w in weights]}")
        
        return self
    
    def select_best_model(self):
        """选择最佳模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段四: 选择最佳模型")
        self.logger.info("=" * 80)
        
        best_accuracy = 0
        best_model_name = None
        
        self.logger.info("\n所有模型验证集性能:")
        for name, model in self.models.items():
            accuracy = model.score(self.X_val, self.y_val)
            self.logger.info(f"{name}: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        self.logger.info(f"\n最佳模型: {best_model_name}")
        self.logger.info(f"验证集准确率: {best_accuracy:.4f}")
        
        return self
    
    def save_models(self):
        """保存模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("保存模型...")
        self.logger.info("=" * 80)
        
        for name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{name}_v1opt_model.pkl')
            CommonUtil.save_model(model, model_path)
            self.logger.info(f"已保存模型: {model_path}")
        
        scaler_path = os.path.join(self.model_dir, 'scaler_v1opt.pkl')
        CommonUtil.save_model(self.scaler, scaler_path)
        
        encoders_path = os.path.join(self.model_dir, 'label_encoders_v1opt.pkl')
        CommonUtil.save_model(self.label_encoders, encoders_path)
        
        features_path = os.path.join(self.model_dir, 'feature_names_v1opt.pkl')
        CommonUtil.save_model(self.feature_names, features_path)
        
        best_model_info = {
            'name': self.best_model_name,
            'model': self.best_model,
            'threshold': self.best_threshold
        }
        best_model_path = os.path.join(self.model_dir, 'best_model_v1opt.pkl')
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
            model_names.append(name)
            accuracies.append(model.score(self.X_val, self.y_val))
        
        axes[0, 0].barh(model_names, accuracies, alpha=0.8, color='steelblue')
        axes[0, 0].axvline(x=0.9, color='r', linestyle='--', linewidth=2, label='目标: 90%')
        axes[0, 0].set_xlabel('准确率', fontsize=12)
        axes[0, 0].set_title('V1优化版模型准确率对比', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(accuracies):
            axes[0, 0].text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')
        
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
        axes[1, 0].set_title('混淆矩阵（验证集）', fontsize=14, fontweight='bold')
        
        # Top模型对比
        top_models = sorted([(name, model.score(self.X_val, self.y_val)) 
                            for name, model in self.models.items()], 
                           key=lambda x: x[1], reverse=True)[:6]
        
        top_names = [x[0] for x in top_models]
        top_scores = [x[1] for x in top_models]
        
        axes[1, 1].bar(range(len(top_names)), top_scores, alpha=0.8, color='green')
        axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='目标: 90%')
        axes[1, 1].set_xticks(range(len(top_names)))
        axes[1, 1].set_xticklabels(top_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('准确率')
        axes[1, 1].set_title('Top 6 模型', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(top_scores):
            axes[1, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.data_dir, 'training_results_v1opt.png')
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
            self.optimize_models()
            self.create_ensemble()
            self.select_best_model()
            self.save_models()
            self.plot_results()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("V1优化版训练完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = HRAttritionTrainerV1Optimized()
    trainer.run()
