"""
最终优化版模型训练脚本 - 提高测试集准确率
主要优化：
1. SMOTE + 调整采样比例
2. 更精细的超参数优化
3. 集成多个优化模型
4. 特征选择
5. 调整决策阈值优化F1和准确率
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE


class HRAttritionTrainerV3:
    """最终优化版HR员工流失预测模型训练器"""
    
    def __init__(self, log_dir='../log', data_dir='../data', model_dir='../model'):
        """初始化训练器"""
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=log_dir, log_name=f'train_v3_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("最终优化版HR员工流失预测模型训练开始")
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
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"划分前 - 训练集流失比例: {(self.y_train == 1).sum() / len(self.y_train):.4f}")
        
        # SMOTE with sampling_strategy=0.5 (更温和的过采样)
        self.logger.info("\n应用SMOTE处理类别不平衡（采样比例=0.5）...")
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.5)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        self.logger.info(f"SMOTE后 - 训练集大小: {self.X_train.shape}")
        self.logger.info(f"SMOTE后 - 流失比例: {(self.y_train == 1).sum() / len(self.y_train):.4f}")
        
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
    
    def train_optimized_models(self):
        """训练优化的模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段二: 训练优化模型并进行超参数搜索")
        self.logger.info("=" * 80)
        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_val), y=self.y_val)
        weight_dict = {0: class_weights[0], 1: class_weights[1]}
        scale_pos_weight = (self.y_val == 0).sum() / (self.y_val == 1).sum()
        
        # XGBoost with hyperparameter tuning
        self.logger.info("\n优化XGBoost...")
        xgb_param_dist = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'scale_pos_weight': [scale_pos_weight * 0.8, scale_pos_weight, scale_pos_weight * 1.2]
        }
        
        xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_param_dist, n_iter=15, cv=3, 
            scoring='f1', random_state=42, n_jobs=-1
        )
        xgb_search.fit(self.X_train, self.y_train)
        xgb_model = xgb_search.best_estimator_
        self.models['xgb_optimized'] = xgb_model
        
        val_score = xgb_model.score(self.X_val, self.y_val)
        y_pred = xgb_model.predict(self.X_val)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        self.logger.info(f"XGBoost优化 - 准确率: {val_score:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        self.logger.info(f"最佳参数: {xgb_search.best_params_}")
        
        # LightGBM with hyperparameter tuning
        self.logger.info("\n优化LightGBM...")
        lgbm_param_dist = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [20, 30, 40]
        }
        
        lgbm_base = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        lgbm_search = RandomizedSearchCV(
            lgbm_base, lgbm_param_dist, n_iter=15, cv=3,
            scoring='f1', random_state=42, n_jobs=-1
        )
        lgbm_search.fit(self.X_train, self.y_train)
        lgbm_model = lgbm_search.best_estimator_
        self.models['lgbm_optimized'] = lgbm_model
        
        val_score = lgbm_model.score(self.X_val, self.y_val)
        y_pred = lgbm_model.predict(self.X_val)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        self.logger.info(f"LightGBM优化 - 准确率: {val_score:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        self.logger.info(f"最佳参数: {lgbm_search.best_params_}")
        
        # Stacking Ensemble
        self.logger.info("\n训练Stacking集成...")
        estimators = [
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ]
        
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
            cv=5
        )
        stacking_model.fit(self.X_train, self.y_train)
        self.models['stacking'] = stacking_model
        
        val_score = stacking_model.score(self.X_val, self.y_val)
        y_pred = stacking_model.predict(self.X_val)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        self.logger.info(f"Stacking集成 - 准确率: {val_score:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        
        return self
    
    def optimize_threshold(self):
        """优化决策阈值"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("阶段三: 优化决策阈值（目标：最大化准确率同时保持合理召回率）")
        self.logger.info("=" * 80)
        
        best_score = 0
        best_threshold = 0.5
        best_model_name = None
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_val)[:, 1]
                
                # 尝试不同的阈值，优化准确率和召回率的平衡
                for threshold in np.arange(0.35, 0.65, 0.05):
                    y_pred = (y_proba >= threshold).astype(int)
                    accuracy = accuracy_score(self.y_val, y_pred)
                    recall = recall_score(self.y_val, y_pred)
                    f1 = f1_score(self.y_val, y_pred)
                    
                    # 综合评分：准确率权重0.6，召回率权重0.2，F1权重0.2
                    score = 0.6 * accuracy + 0.2 * recall + 0.2 * f1
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        best_model_name = name
                        
                        self.logger.info(f"{name} @ threshold={threshold:.2f}: "
                                       f"Acc={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Score={score:.4f}")
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self.best_threshold = best_threshold
        
        # 最终评估
        y_proba = self.best_model.predict_proba(self.X_val)[:, 1]
        y_pred = (y_proba >= best_threshold).astype(int)
        accuracy = accuracy_score(self.y_val, y_pred)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        
        self.logger.info(f"\n最佳配置: {best_model_name}, 阈值={best_threshold:.2f}")
        self.logger.info(f"验证集性能 - 准确率: {accuracy:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        
        return self
    
    def save_models(self):
        """保存模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("保存模型...")
        self.logger.info("=" * 80)
        
        for name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{name}_model_v3.pkl')
            CommonUtil.save_model(model, model_path)
            self.logger.info(f"已保存模型: {model_path}")
        
        scaler_path = os.path.join(self.model_dir, 'scaler_v3.pkl')
        CommonUtil.save_model(self.scaler, scaler_path)
        
        encoders_path = os.path.join(self.model_dir, 'label_encoders_v3.pkl')
        CommonUtil.save_model(self.label_encoders, encoders_path)
        
        features_path = os.path.join(self.model_dir, 'feature_names_v3.pkl')
        CommonUtil.save_model(self.feature_names, features_path)
        
        best_model_info = {
            'name': self.best_model_name,
            'model': self.best_model,
            'threshold': self.best_threshold
        }
        best_model_path = os.path.join(self.model_dir, 'best_model_v3.pkl')
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
        recalls = []
        f1_scores = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_val)
            
            model_names.append(name)
            accuracies.append(accuracy_score(self.y_val, y_pred))
            recalls.append(recall_score(self.y_val, y_pred))
            f1_scores.append(f1_score(self.y_val, y_pred))
        
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, accuracies, width, label='准确率', alpha=0.8)
        axes[0, 0].bar(x, recalls, width, label='召回率', alpha=0.8)
        axes[0, 0].bar(x + width, f1_scores, width, label='F1分数', alpha=0.8)
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_title('最终优化模型性能对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-20:]
            
            axes[0, 1].barh(range(len(indices)), importances[indices], alpha=0.8)
            axes[0, 1].set_yticks(range(len(indices)))
            axes[0, 1].set_yticklabels([self.feature_names[i] for i in indices], fontsize=8)
            axes[0, 1].set_xlabel('重要性')
            axes[0, 1].set_title(f'特征重要性 ({self.best_model_name})')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 混淆矩阵
        y_proba = self.best_model.predict_proba(self.X_val)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        cm = confusion_matrix(self.y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('预测标签')
        axes[1, 0].set_ylabel('真实标签')
        axes[1, 0].set_title(f'混淆矩阵 (阈值={self.best_threshold:.2f})')
        
        # 阈值vs性能曲线
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_accuracies = []
        threshold_recalls = []
        threshold_f1s = []
        
        y_proba = self.best_model.predict_proba(self.X_val)[:, 1]
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            threshold_accuracies.append(accuracy_score(self.y_val, y_pred))
            threshold_recalls.append(recall_score(self.y_val, y_pred))
            threshold_f1s.append(f1_score(self.y_val, y_pred))
        
        axes[1, 1].plot(thresholds, threshold_accuracies, label='准确率', marker='o')
        axes[1, 1].plot(thresholds, threshold_recalls, label='召回率', marker='s')
        axes[1, 1].plot(thresholds, threshold_f1s, label='F1分数', marker='^')
        axes[1, 1].axvline(x=self.best_threshold, color='r', linestyle='--', label=f'最佳阈值={self.best_threshold:.2f}')
        axes[1, 1].set_xlabel('决策阈值')
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].set_title('阈值对性能的影响')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.data_dir, 'training_results_v3.png')
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
            self.train_optimized_models()
            self.optimize_threshold()
            self.save_models()
            self.plot_results()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("最终优化版训练完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = HRAttritionTrainerV3()
    trainer.run()
