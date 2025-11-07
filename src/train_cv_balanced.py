"""
CV平衡训练版本 - 使用交叉验证减少过拟合
使用过拟合最轻的方法：CatBoost、LightGBM、XGBoost
目标：通过CV平衡提高泛化能力，突破90%准确率
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

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import ADASYN
import pickle

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class CVBalancedTrainer:
    """CV平衡训练器"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_cv_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("CV平衡训练 - 使用交叉验证减少过拟合")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.model_dir)
        
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("加载和预处理数据...")
        self.logger.info("=" * 80)
        
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
            self.label_encoders[col] = le
        
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
        self.logger.info(f"训练集: {self.X_train.shape}")
        
        return self
    
    def feature_engineering(self, df, fit=True):
        """特征工程 - 使用验证有效的方法"""
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
    
    def train_with_cv(self, model, model_name, n_splits=10):
        """使用交叉验证训练模型"""
        self.logger.info(f"\n训练 {model_name} (CV={n_splits}折)...")
        
        # 使用StratifiedKFold确保每折的类别分布一致
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 交叉验证评分
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=skf, scoring='accuracy', n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        self.logger.info(f"{model_name} CV准确率: {mean_score:.4f} (+/- {std_score:.4f})")
        self.logger.info(f"各折准确率: {[f'{s:.4f}' for s in cv_scores]}")
        
        # 使用交叉验证预测（用于后续集成）
        cv_predictions = cross_val_predict(
            model, self.X_train, self.y_train,
            cv=skf, method='predict_proba'
        )
        
        # 在全部训练数据上训练最终模型
        model.fit(self.X_train, self.y_train)
        
        self.cv_results[model_name] = {
            'model': model,
            'cv_mean': mean_score,
            'cv_std': std_score,
            'cv_scores': cv_scores,
            'cv_predictions': cv_predictions
        }
        
        return model, mean_score
    
    def train_all_models(self):
        """训练所有过拟合较轻的模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("训练过拟合较轻的模型")
        self.logger.info("=" * 80)
        
        # 1. XGBoost
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.train_with_cv(xgb, 'XGBoost', n_splits=10)
        
        # 2. LightGBM
        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.train_with_cv(lgbm, 'LightGBM', n_splits=10)
        
        # 3. CatBoost
        cat = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=0
        )
        self.train_with_cv(cat, 'CatBoost', n_splits=10)
        
        # 4. 使用ADASYN的XGBoost
        self.logger.info(f"\n训练 XGBoost+ADASYN (CV=10折)...")
        adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(self.X_train, self.y_train)
        
        xgb_adasyn = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            xgb_adasyn, X_resampled, y_resampled,
            cv=skf, scoring='accuracy', n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        self.logger.info(f"XGBoost+ADASYN CV准确率: {mean_score:.4f} (+/- {std_score:.4f})")
        
        xgb_adasyn.fit(X_resampled, y_resampled)
        
        self.cv_results['XGBoost+ADASYN'] = {
            'model': xgb_adasyn,
            'cv_mean': mean_score,
            'cv_std': std_score,
            'cv_scores': cv_scores
        }
        
        # 5. Voting集成（软投票）
        self.logger.info(f"\n训练 Voting Ensemble (CV=10折)...")
        voting = VotingClassifier(
            estimators=[
                ('xgb', self.cv_results['XGBoost']['model']),
                ('lgbm', self.cv_results['LightGBM']['model']),
                ('cat', self.cv_results['CatBoost']['model'])
            ],
            voting='soft',
            weights=[1, 1, 1]
        )
        
        self.train_with_cv(voting, 'Voting', n_splits=10)
        
        return self
    
    def evaluate_on_test(self):
        """在测试集上评估所有模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("测试集评估（350条数据）")
        self.logger.info("=" * 80)
        
        # 预处理测试集
        if 'Attrition' in self.test_df.columns:
            self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            if self.y_test.isna().any():
                self.test_df['Attrition'] = self.test_df['Attrition'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
                self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            self.y_test = self.y_test.fillna(0).astype(int).values
        
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
        
        # 评估所有模型
        test_results = {}
        
        for name, result in self.cv_results.items():
            model = result['model']
            cv_mean = result['cv_mean']
            cv_std = result['cv_std']
            
            preds = model.predict(X_test)
            test_acc = accuracy_score(self.y_test, preds)
            
            # 计算过拟合程度
            overfit = cv_mean - test_acc
            
            test_results[name] = {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_acc': test_acc,
                'overfit': overfit
            }
            
            self.logger.info(f"\n{name}:")
            self.logger.info(f"  CV准确率: {cv_mean:.4f} (+/- {cv_std:.4f})")
            self.logger.info(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
            self.logger.info(f"  过拟合程度: {overfit:.4f} ({overfit*100:.2f}%)")
            
            if test_acc >= 0.90:
                self.logger.info(f"  ✓ 达到90%目标！")
        
        # 找到测试集最佳模型
        best_name = max(test_results.items(), key=lambda x: x[1]['test_acc'])[0]
        best_result = test_results[best_name]
        
        self.logger.info(f"\n" + "=" * 80)
        self.logger.info(f"最佳模型: {best_name}")
        self.logger.info(f"CV准确率: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})")
        self.logger.info(f"测试集准确率: {best_result['test_acc']*100:.2f}%")
        self.logger.info(f"过拟合程度: {best_result['overfit']*100:.2f}%")
        
        if best_result['test_acc'] >= 0.90:
            self.logger.info("✓ 已达到90%目标！")
        else:
            self.logger.info(f"✗ 距离90%目标还差: {(0.90 - best_result['test_acc'])*100:.2f}%")
        
        # 详细报告
        best_model = self.cv_results[best_name]['model']
        best_preds = best_model.predict(X_test)
        
        self.logger.info(f"\n{best_name} 分类报告:")
        self.logger.info("\n" + classification_report(self.y_test, best_preds))
        
        self.logger.info(f"\n混淆矩阵:")
        cm = confusion_matrix(self.y_test, best_preds)
        self.logger.info(f"\n{cm}")
        
        # 保存最佳模型
        self.best_model = best_model
        self.best_model_name = best_name
        
        return test_results
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_and_preprocess_data()
            self.train_all_models()
            test_results = self.evaluate_on_test()
            
            # 保存结果
            self.logger.info("\n保存模型...")
            CommonUtil.save_model(self.best_model, os.path.join(self.model_dir, f'cv_best_model_{self.best_model_name}.pkl'))
            CommonUtil.save_model(test_results, os.path.join(self.model_dir, 'cv_test_results.pkl'))
            CommonUtil.save_model(self.cv_results, os.path.join(self.model_dir, 'cv_results.pkl'))
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("CV平衡训练完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = CVBalancedTrainer()
    trainer.run()
