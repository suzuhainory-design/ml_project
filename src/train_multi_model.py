"""
多模型快速优化版本
目标：在测试集（全部350条数据）上达到90%以上准确率
策略：尝试多种模型和集成方法
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.logUtil import LogUtil
from util.commonUtil import CommonUtil
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                             BaggingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


class MultiModelTrainer:
    """多模型训练器"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_multi_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("多模型快速优化训练 - 目标测试集准确率90%+")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.model_dir)
        
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
        self.models = {}
        self.test_scores = {}
    
    def load_data(self):
        """加载数据"""
        self.logger.info("加载数据...")
        
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        self.logger.info(f"训练集: {self.train_df.shape}")
        self.logger.info(f"测试集: {self.test_df.shape} (使用全部350条数据)")
        
        return self
    
    def feature_engineering(self, df):
        """特征工程"""
        df = df.copy()
        
        # 数值型特征
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
        
        # HR领域特征
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
        
        if 'MonthlyIncome' in df.columns:
            if 'Age' in df.columns:
                df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
            if 'TotalWorkingYears' in df.columns:
                df['IncomePerExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
            if 'JobLevel' in df.columns:
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
    
    def preprocess_data(self):
        """预处理数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("数据预处理...")
        self.logger.info("=" * 80)
        
        # 特征工程
        self.train_df = self.feature_engineering(self.train_df)
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours']
        for col in cols_to_drop:
            if col in self.train_df.columns:
                self.train_df = self.train_df.drop(columns=[col])
        
        # 分离特征和标签
        if 'Attrition' in self.train_df.columns:
            self.train_df['Attrition'] = pd.to_numeric(self.train_df['Attrition'], errors='coerce')
            self.train_df['Attrition'] = self.train_df['Attrition'].fillna(0).astype(int)
        
        X_train = self.train_df.drop(columns=['Attrition'])
        y_train = self.train_df['Attrition'].astype(int)
        
        # 填充缺失值
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            X_train[col] = X_train[col].fillna(X_train[col].median())
        
        # 编码类别特征
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.logger.info(f"类别特征: {categorical_cols}")
        
        for col in categorical_cols:
            X_train[col] = X_train[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            self.label_encoders[col] = le
        
        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        
        self.X_train = X_train_scaled
        self.y_train = y_train
        self.feature_names = X_train.columns.tolist()
        
        self.logger.info(f"训练集大小: {self.X_train.shape}")
        self.logger.info(f"流失比例: {(self.y_train == 1).sum() / len(self.y_train):.4f}")
        
        return self
    
    def train_models(self):
        """训练多种模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("训练多种模型...")
        self.logger.info("=" * 80)
        
        # 1. XGBoost
        self.logger.info("\n1. XGBoost...")
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(self.X_train, self.y_train)
        self.models['xgb'] = xgb
        
        # 2. LightGBM
        self.logger.info("2. LightGBM...")
        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=50,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(self.X_train, self.y_train)
        self.models['lgbm'] = lgbm
        
        # 3. CatBoost
        if CatBoostClassifier is not None:
            self.logger.info("3. CatBoost...")
            cat = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=0
            )
            cat.fit(self.X_train, self.y_train)
            self.models['catboost'] = cat
        
        # 4. RandomForest
        self.logger.info("4. RandomForest...")
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['rf'] = rf
        
        # 5. ExtraTrees
        self.logger.info("5. ExtraTrees...")
        et = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        et.fit(self.X_train, self.y_train)
        self.models['et'] = et
        
        # 6. GradientBoosting
        self.logger.info("6. GradientBoosting...")
        gb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(self.X_train, self.y_train)
        self.models['gb'] = gb
        
        # 7. SVM
        self.logger.info("7. SVM...")
        svm = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        svm.fit(self.X_train, self.y_train)
        self.models['svm'] = svm
        
        # 8. MLP神经网络
        self.logger.info("8. MLP Neural Network...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
        mlp.fit(self.X_train, self.y_train)
        self.models['mlp'] = mlp
        
        # 9. AdaBoost
        self.logger.info("9. AdaBoost...")
        ada = AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.1,
            random_state=42
        )
        ada.fit(self.X_train, self.y_train)
        self.models['adaboost'] = ada
        
        # 10. Bagging
        self.logger.info("10. Bagging...")
        bag = BaggingClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        bag.fit(self.X_train, self.y_train)
        self.models['bagging'] = bag
        
        return self
    
    def create_ensembles(self):
        """创建集成模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("创建集成模型...")
        self.logger.info("=" * 80)
        
        # Voting集成
        self.logger.info("\n创建Voting集成...")
        estimators = [(name, model) for name, model in self.models.items() 
                     if hasattr(model, 'predict_proba')]
        
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        voting.fit(self.X_train, self.y_train)
        self.models['voting'] = voting
        
        # Stacking集成
        self.logger.info("创建Stacking集成...")
        stacking = StackingClassifier(
            estimators=estimators[:5],  # 使用前5个模型
            final_estimator=LogisticRegression(),
            n_jobs=-1
        )
        stacking.fit(self.X_train, self.y_train)
        self.models['stacking'] = stacking
        
        return self
    
    def preprocess_test_data(self):
        """预处理测试数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("预处理测试数据（全部350条）...")
        self.logger.info("=" * 80)
        
        # 保存真实标签
        if 'Attrition' in self.test_df.columns:
            self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            if self.y_test.isna().any():
                self.test_df['Attrition'] = self.test_df['Attrition'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
                self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            self.y_test = self.y_test.fillna(0).astype(int)
            self.logger.info(f"测试集样本数: {len(self.y_test)}")
            self.logger.info(f"流失: {(self.y_test == 1).sum()}, 未流失: {(self.y_test == 0).sum()}")
        
        # 特征工程
        test_df = self.feature_engineering(self.test_df)
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        for col in cols_to_drop:
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
        
        # 填充缺失值
        numeric_cols = test_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            test_df[col] = test_df[col].fillna(test_df[col].median())
        
        # 编码类别特征
        for col in self.label_encoders:
            if col in test_df.columns:
                le = self.label_encoders[col]
                test_df[col] = test_df[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
                test_df[col] = test_df[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                test_df[col] = le.transform(test_df[col])
        
        # 确保特征一致
        missing_features = set(self.feature_names) - set(test_df.columns)
        if missing_features:
            for feature in missing_features:
                test_df[feature] = 0
        
        test_df = test_df[self.feature_names]
        
        # 标准化
        self.X_test = pd.DataFrame(
            self.scaler.transform(test_df),
            columns=self.feature_names
        )
        
        self.logger.info(f"测试集形状: {self.X_test.shape}")
        
        return self
    
    def evaluate_on_test(self):
        """在测试集上评估所有模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("在测试集（350条数据）上评估所有模型...")
        self.logger.info("=" * 80)
        
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.test_scores[name] = accuracy
            
            self.logger.info(f"\n{name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            results.append({
                'model': name,
                'accuracy': accuracy,
                'percentage': f"{accuracy*100:.2f}%"
            })
        
        # 排序
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("测试集准确率排名:")
        self.logger.info("=" * 80)
        for i, r in enumerate(results, 1):
            status = "✓ 达标" if r['accuracy'] >= 0.90 else "✗ 未达标"
            self.logger.info(f"{i}. {r['model']}: {r['percentage']} {status}")
        
        # 最佳模型
        best_model_name = results[0]['model']
        best_accuracy = results[0]['accuracy']
        
        self.logger.info(f"\n最佳模型: {best_model_name}")
        self.logger.info(f"测试集准确率: {best_accuracy*100:.2f}%")
        
        if best_accuracy >= 0.90:
            self.logger.info("✓ 已达到90%目标！")
        else:
            self.logger.info(f"✗ 距离90%目标还差: {(0.90 - best_accuracy)*100:.2f}%")
        
        return self
    
    def save_models(self):
        """保存模型"""
        self.logger.info("\n保存模型...")
        
        for name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{name}_multi_model.pkl')
            CommonUtil.save_model(model, model_path)
        
        # 保存预处理器
        CommonUtil.save_model(self.scaler, os.path.join(self.model_dir, 'scaler_multi.pkl'))
        CommonUtil.save_model(self.label_encoders, os.path.join(self.model_dir, 'label_encoders_multi.pkl'))
        CommonUtil.save_model(self.feature_names, os.path.join(self.model_dir, 'feature_names_multi.pkl'))
        
        # 保存测试结果
        results_path = os.path.join(self.model_dir, 'test_scores_multi.pkl')
        CommonUtil.save_model(self.test_scores, results_path)
        self.logger.info(f"测试结果已保存: {results_path}")
        
        return self
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_data()
            self.preprocess_data()
            self.train_models()
            self.create_ensembles()
            self.preprocess_test_data()
            self.evaluate_on_test()
            self.save_models()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("训练和评估完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = MultiModelTrainer()
    trainer.run()
