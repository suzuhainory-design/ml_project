"""
使用已找到的最佳参数进行评估
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

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from imblearn.over_sampling import ADASYN
import pickle

from xgboost import XGBClassifier


# 最佳参数
BEST_ADASYN_PARAMS = {'sampling_strategy': 0.5, 'n_neighbors': 5}
BEST_XGB_PARAMS = {
    'subsample': 0.8,
    'reg_lambda': 1.5,
    'reg_alpha': 0.5,
    'n_estimators': 400,
    'min_child_weight': 1,
    'max_depth': 7,
    'learning_rate': 0.03,
    'gamma': 0.1,
    'colsample_bytree': 0.75,
    'random_state': 42,
    'eval_metric': 'logloss'
}


class BestParamsEvaluator:
    """使用最佳参数进行评估"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'eval_best_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("使用最佳参数进行评估")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.model_dir)
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        self.logger.info("\n加载和预处理数据...")
        
        # 加载数据
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        self.logger.info(f"训练集: {self.train_df.shape}, 测试集: {self.test_df.shape}")
        
        # 特征工程
        self.train_df = self.feature_engineering(self.train_df, fit=True)
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours']
        for col in cols_to_drop:
            if col in self.train_df.columns:
                self.train_df = self.train_df.drop(columns=[col])
        
        # 分离特征和标签
        if 'Attrition' in self.train_df.columns:
            self.train_df['Attrition'] = pd.to_numeric(self.train_df['Attrition'], errors='coerce')
            self.train_df['Attrition'] = self.train_df['Attrition'].fillna(0).astype(int)
        
        X = self.train_df.drop(columns=['Attrition'])
        y = self.train_df['Attrition'].astype(int)
        
        self.logger.info(f"流失比例: {y.mean():.4f}")
        
        # KNN填充
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            self.knn_imputer = KNNImputer(n_neighbors=5)
            X[numeric_cols] = self.knn_imputer.fit_transform(X[numeric_cols])
        
        # 编码类别特征
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders = {col: le}
        
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        self.X_train = X_scaled
        self.y_train = y.values
        self.feature_names = X.columns.tolist()
        
        self.logger.info(f"特征数量: {len(self.feature_names)}")
        
        return self
    
    def feature_engineering(self, df, fit=True):
        """特征工程"""
        df = df.copy()
        
        numeric_cols = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
                       'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                       'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 聚类特征
        if fit and 'Age' in df.columns and 'MonthlyIncome' in df.columns:
            from sklearn.cluster import KMeans
            cluster_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']
            available = [f for f in cluster_features if f in df.columns]
            if len(available) >= 2:
                cluster_data = df[available].fillna(0)
                self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                df['Employee_Cluster'] = self.kmeans.fit_predict(cluster_data)
                distances = self.kmeans.transform(cluster_data)
                df['Cluster_Distance'] = distances.min(axis=1)
        elif hasattr(self, 'kmeans'):
            cluster_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']
            available = [f for f in cluster_features if f in df.columns]
            if len(available) >= 2:
                cluster_data = df[available].fillna(0)
                df['Employee_Cluster'] = self.kmeans.predict(cluster_data)
                distances = self.kmeans.transform(cluster_data)
                df['Cluster_Distance'] = distances.min(axis=1)
        
        # 交互特征
        if 'Age' in df.columns and 'MonthlyIncome' in df.columns:
            df['Age_Income_Interaction'] = df['Age'] * df['MonthlyIncome'] / 10000
        
        if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
            df['CompanyTenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
        
        # 比率特征
        if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
            df['Income_Age_Ratio'] = df['MonthlyIncome'] / (df['Age'] + 1)
        
        if 'YearsInCurrentRole' in df.columns and 'YearsAtCompany' in df.columns:
            df['Role_Tenure_Ratio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
        
        return df
    
    def evaluate(self):
        """评估最佳参数"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("使用最佳参数进行评估")
        self.logger.info("=" * 80)
        
        self.logger.info(f"\nADASYN参数: {BEST_ADASYN_PARAMS}")
        self.logger.info(f"XGBoost参数: {BEST_XGB_PARAMS}")
        
        # 应用ADASYN
        adasyn = ADASYN(**BEST_ADASYN_PARAMS, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(self.X_train, self.y_train)
        
        self.logger.info(f"\n过采样后训练集: {X_resampled.shape}")
        
        # 10折CV评估
        model = XGBClassifier(**BEST_XGB_PARAMS)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='accuracy', n_jobs=-1)
        
        self.logger.info(f"\n10折CV准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        self.logger.info(f"各折准确率: {[f'{s:.4f}' for s in cv_scores]}")
        
        # 在全部数据上训练最终模型
        final_model = XGBClassifier(**BEST_XGB_PARAMS)
        final_model.fit(X_resampled, y_resampled)
        
        # 测试集评估
        test_acc = self.evaluate_on_test(final_model)
        
        # 保存模型
        self.logger.info("\n保存最佳模型...")
        CommonUtil.save_model(final_model, os.path.join(self.model_dir, 'best_finetuned_model.pkl'))
        CommonUtil.save_model({
            'adasyn': BEST_ADASYN_PARAMS,
            'xgboost': BEST_XGB_PARAMS,
            'cv_score': cv_scores.mean(),
            'test_score': test_acc
        }, os.path.join(self.model_dir, 'best_finetuned_params.pkl'))
        
        return test_acc
    
    def evaluate_on_test(self, model):
        """在测试集上评估"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("测试集评估（350条数据）")
        self.logger.info("=" * 80)
        
        # 预处理测试集
        if 'Attrition' in self.test_df.columns:
            y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            if y_test.isna().any():
                self.test_df['Attrition'] = self.test_df['Attrition'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
                y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            y_test = y_test.fillna(0).astype(int).values
        
        test_df = self.feature_engineering(self.test_df, fit=False)
        
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        for col in cols_to_drop:
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
        
        # KNN填充
        numeric_cols = test_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0 and hasattr(self, 'knn_imputer'):
            test_df[numeric_cols] = self.knn_imputer.transform(test_df[numeric_cols])
        
        # 编码
        for col in self.label_encoders:
            if col in test_df.columns:
                le = self.label_encoders[col]
                test_df[col] = test_df[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
                test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                test_df[col] = le.transform(test_df[col])
        
        # 确保特征一致
        missing_features = set(self.feature_names) - set(test_df.columns)
        if missing_features:
            for feature in missing_features:
                test_df[feature] = 0
        
        test_df = test_df[self.feature_names]
        
        # 标准化
        X_test = self.scaler.transform(test_df)
        
        # 预测
        preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        
        self.logger.info(f"\n测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        if test_acc >= 0.90:
            self.logger.info("\n🎉 已达到90%目标！")
        else:
            self.logger.info(f"\n距离90%目标还差: {(0.90 - test_acc)*100:.2f}%")
        
        # 详细报告
        self.logger.info(f"\n分类报告:")
        self.logger.info("\n" + classification_report(y_test, preds))
        
        self.logger.info(f"\n混淆矩阵:")
        cm = confusion_matrix(y_test, preds)
        self.logger.info(f"\n{cm}")
        self.logger.info(f"\nTN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        return test_acc
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_and_preprocess_data()
            test_acc = self.evaluate()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("评估完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    evaluator = BestParamsEvaluator()
    evaluator.run()
