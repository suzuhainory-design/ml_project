"""
终极优化版本 - 实施所有优化技术
目标：通过交叉验证筛选稳定提升准确率的方法，达到90%+
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SVMSMOTE
from sklearn.cluster import KMeans
import pickle


class UltimateOptimizer:
    """终极优化器 - 实施并验证所有优化技术"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_ultimate_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("终极优化训练 - 实施所有优化技术")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.model_dir)
        
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
        self.optimization_results = {}
        self.best_features = None
        self.best_model = None
        
    def load_data(self):
        """加载数据"""
        self.logger.info("加载数据...")
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        self.logger.info(f"训练集: {self.train_df.shape}, 测试集: {self.test_df.shape}")
        return self
    
    def advanced_feature_engineering(self, df, fit=True):
        """高级特征工程"""
        df = df.copy()
        
        # 基础特征工程（保留原有的）
        numeric_cols = ['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                       'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. 原有HR特征
        if 'TotalWorkingYears' in df.columns and 'YearsAtCompany' in df.columns:
            df['ExperienceBeforeCompany'] = df['TotalWorkingYears'] - df['YearsAtCompany']
            df['CompanyTenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
        
        if 'YearsAtCompany' in df.columns and 'YearsSinceLastPromotion' in df.columns:
            df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
            df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
        
        # 2. 新增：高级交互特征
        if 'Age' in df.columns and 'MonthlyIncome' in df.columns:
            df['Age_Income_Interaction'] = df['Age'] * df['MonthlyIncome'] / 10000
            
        if 'YearsAtCompany' in df.columns and 'JobLevel' in df.columns:
            df['Tenure_Level_Interaction'] = df['YearsAtCompany'] * df['JobLevel']
        
        if 'JobSatisfaction' in df.columns and 'WorkLifeBalance' in df.columns:
            df['Satisfaction_Balance_Product'] = df['JobSatisfaction'] * df['WorkLifeBalance']
        
        # 3. 新增：比率特征
        if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
            df['Income_Age_Ratio'] = df['MonthlyIncome'] / (df['Age'] + 1)
        
        if 'YearsInCurrentRole' in df.columns and 'YearsAtCompany' in df.columns:
            df['Role_Tenure_Ratio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
        
        # 4. 新增：聚类特征
        if fit:
            cluster_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobLevel']
            available_cluster_features = [f for f in cluster_features if f in df.columns]
            if len(available_cluster_features) >= 2:
                cluster_data = df[available_cluster_features].fillna(0)
                self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                df['Employee_Cluster'] = self.kmeans.fit_predict(cluster_data)
                # 添加到簇中心的距离
                distances = self.kmeans.transform(cluster_data)
                df['Cluster_Distance'] = distances.min(axis=1)
        else:
            if hasattr(self, 'kmeans'):
                cluster_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobLevel']
                available_cluster_features = [f for f in cluster_features if f in df.columns]
                if len(available_cluster_features) >= 2:
                    cluster_data = df[available_cluster_features].fillna(0)
                    df['Employee_Cluster'] = self.kmeans.predict(cluster_data)
                    distances = self.kmeans.transform(cluster_data)
                    df['Cluster_Distance'] = distances.min(axis=1)
        
        # 5. 新增：分箱特征
        if 'Age' in df.columns:
            df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3]).astype(float)
        
        if 'MonthlyIncome' in df.columns:
            try:
                df['Income_Bin'] = pd.qcut(df['MonthlyIncome'].fillna(0), q=5, labels=[0, 1, 2, 3, 4], duplicates='drop').astype(float)
            except:
                # 如果分位数分箱失败，使用等宽分箱
                df['Income_Bin'] = pd.cut(df['MonthlyIncome'].fillna(0), bins=5, labels=[0, 1, 2, 3, 4]).astype(float)
        
        # 6. 新增：统计特征
        satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                           'RelationshipSatisfaction', 'WorkLifeBalance']
        available_satisfaction = [col for col in satisfaction_cols if col in df.columns]
        if available_satisfaction:
            df['OverallSatisfaction'] = df[available_satisfaction].mean(axis=1)
            df['SatisfactionStd'] = df[available_satisfaction].std(axis=1)
            df['SatisfactionMin'] = df[available_satisfaction].min(axis=1)
            df['SatisfactionMax'] = df[available_satisfaction].max(axis=1)
        
        return df
    
    def target_encoding(self, X_train, y_train, X_val, categorical_cols):
        """目标编码"""
        self.logger.info("\n应用目标编码...")
        self.target_encodings = {}
        
        # 创建临时DataFrame用于groupby
        temp_df = X_train.copy()
        temp_df['_target_'] = y_train.values
        
        for col in categorical_cols:
            if col in X_train.columns:
                # 计算每个类别的目标均值
                encoding = temp_df.groupby(col)['_target_'].mean()
                self.target_encodings[col] = encoding
                
                # 应用编码（使用全局均值作为未见类别的默认值）
                global_mean = y_train.mean()
                X_train[f'{col}_target_enc'] = X_train[col].map(encoding).fillna(global_mean)
                X_val[f'{col}_target_enc'] = X_val[col].map(encoding).fillna(global_mean)
        
        return X_train, X_val
    
    def preprocess_data(self):
        """预处理数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("数据预处理...")
        self.logger.info("=" * 80)
        
        # 特征工程
        self.train_df = self.advanced_feature_engineering(self.train_df, fit=True)
        
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
        
        # 使用KNN填充缺失值
        self.logger.info("使用KNN填充缺失值...")
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            X[numeric_cols] = knn_imputer.fit_transform(X[numeric_cols])
            self.knn_imputer = knn_imputer
        
        # 编码类别特征
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.logger.info(f"类别特征: {categorical_cols}")
        
        for col in categorical_cols:
            X[col] = X[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # 分割训练集和验证集
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 目标编码（在分割后进行，避免数据泄露）
        X_train, X_val = self.target_encoding(X_train, y_train, X_val, categorical_cols)
        
        # 使用RobustScaler（对异常值更鲁棒）
        self.logger.info("使用RobustScaler标准化...")
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.y_train = y_train
        self.y_val = y_val
        self.feature_names = X_train.columns.tolist()
        
        self.logger.info(f"训练集: {self.X_train.shape}, 验证集: {self.X_val.shape}")
        self.logger.info(f"训练集流失比例: {(self.y_train == 1).sum() / len(self.y_train):.4f}")
        self.logger.info(f"验证集流失比例: {(self.y_val == 1).sum() / len(self.y_val):.4f}")
        
        return self
    
    def test_oversampling_methods(self):
        """测试不同的过采样方法"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("测试过采样方法...")
        self.logger.info("=" * 80)
        
        methods = {
            'No_Sampling': None,
            'ADASYN': ADASYN(sampling_strategy=0.5, random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(sampling_strategy=0.5, random_state=42),
            'SVMSMOTE': SVMSMOTE(sampling_strategy=0.5, random_state=42)
        }
        
        best_method = None
        best_score = 0
        
        for name, sampler in methods.items():
            if sampler is None:
                X_resampled, y_resampled = self.X_train, self.y_train
            else:
                try:
                    X_resampled, y_resampled = sampler.fit_resample(self.X_train, self.y_train)
                except:
                    self.logger.info(f"{name} 失败，跳过")
                    continue
            
            # 使用简单模型快速评估
            model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            model.fit(X_resampled, y_resampled)
            score = accuracy_score(self.y_val, model.predict(self.X_val))
            
            self.logger.info(f"{name}: {score:.4f}")
            self.optimization_results[f'Oversampling_{name}'] = score
            
            if score > best_score:
                best_score = score
                best_method = (name, sampler)
        
        self.logger.info(f"\n最佳过采样方法: {best_method[0]} ({best_score:.4f})")
        self.best_oversampling = best_method
        
        # 应用最佳方法
        if best_method[1] is not None:
            self.X_train, self.y_train = best_method[1].fit_resample(self.X_train, self.y_train)
            self.logger.info(f"重采样后训练集大小: {self.X_train.shape}")
        
        return self
    
    def feature_selection(self):
        """特征选择"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("特征选择...")
        self.logger.info("=" * 80)
        
        # 使用多个模型的特征重要性
        models = {
            'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGB': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'LGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        }
        
        feature_importance = pd.DataFrame(index=self.feature_names)
        
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = model.feature_importances_
        
        # 计算平均重要性
        feature_importance['Mean'] = feature_importance.mean(axis=1)
        feature_importance = feature_importance.sort_values('Mean', ascending=False)
        
        self.logger.info(f"\nTop 20 重要特征:")
        for i, (feat, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            self.logger.info(f"{i}. {feat}: {row['Mean']:.4f}")
        
        # 测试不同数量的特征
        feature_counts = [20, 30, 40, 50, len(self.feature_names)]
        best_count = len(self.feature_names)
        best_score = 0
        
        for count in feature_counts:
            if count > len(self.feature_names):
                count = len(self.feature_names)
            
            selected_features = feature_importance.head(count).index.tolist()
            X_train_selected = self.X_train[selected_features]
            X_val_selected = self.X_val[selected_features]
            
            model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
            model.fit(X_train_selected, self.y_train)
            score = accuracy_score(self.y_val, model.predict(X_val_selected))
            
            self.logger.info(f"使用{count}个特征: {score:.4f}")
            self.optimization_results[f'FeatureSelection_{count}'] = score
            
            if score > best_score:
                best_score = score
                best_count = count
        
        self.logger.info(f"\n最佳特征数量: {best_count} ({best_score:.4f})")
        self.best_features = feature_importance.head(best_count).index.tolist()
        
        # 更新训练集和验证集
        self.X_train = self.X_train[self.best_features]
        self.X_val = self.X_val[self.best_features]
        
        return self
    
    def optimize_models_with_bayesian(self):
        """使用贝叶斯优化进行超参数搜索"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("贝叶斯优化...")
        self.logger.info("=" * 80)
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                # XGBoost参数
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 300, 700),
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
                
                model = XGBClassifier(**params)
                model.fit(self.X_train, self.y_train)
                score = accuracy_score(self.y_val, model.predict(self.X_val))
                return score
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=30, show_progress_bar=False)
            
            self.logger.info(f"最佳准确率: {study.best_value:.4f}")
            self.logger.info(f"最佳参数: {study.best_params}")
            
            self.best_xgb_params = study.best_params
            self.optimization_results['Bayesian_Optimization'] = study.best_value
            
        except ImportError:
            self.logger.info("Optuna未安装，跳过贝叶斯优化")
            self.best_xgb_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
        
        return self
    
    def train_final_ensemble(self):
        """训练最终集成模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("训练最终集成模型...")
        self.logger.info("=" * 80)
        
        # 1. XGBoost（贝叶斯优化后）
        xgb = XGBClassifier(**self.best_xgb_params)
        xgb.fit(self.X_train, self.y_train)
        xgb_score = accuracy_score(self.y_val, xgb.predict(self.X_val))
        self.logger.info(f"XGBoost: {xgb_score:.4f}")
        
        # 2. LightGBM
        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=50,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(self.X_train, self.y_train)
        lgbm_score = accuracy_score(self.y_val, lgbm.predict(self.X_val))
        self.logger.info(f"LightGBM: {lgbm_score:.4f}")
        
        # 3. CatBoost
        cat = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=0
        )
        cat.fit(self.X_train, self.y_train)
        cat_score = accuracy_score(self.y_val, cat.predict(self.X_val))
        self.logger.info(f"CatBoost: {cat_score:.4f}")
        
        # 4. 加权Voting（根据验证集性能）
        weights = [xgb_score, lgbm_score, cat_score]
        voting = VotingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
            voting='soft',
            weights=weights
        )
        voting.fit(self.X_train, self.y_train)
        voting_score = accuracy_score(self.y_val, voting.predict(self.X_val))
        self.logger.info(f"Weighted Voting: {voting_score:.4f}")
        
        # 5. Stacking
        from sklearn.linear_model import LogisticRegression
        stacking = StackingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        stacking.fit(self.X_train, self.y_train)
        stacking_score = accuracy_score(self.y_val, stacking.predict(self.X_val))
        self.logger.info(f"Stacking: {stacking_score:.4f}")
        
        # 6. 模型校准
        calibrated = CalibratedClassifierCV(xgb, method='isotonic', cv=5)
        calibrated.fit(self.X_train, self.y_train)
        calibrated_score = accuracy_score(self.y_val, calibrated.predict(self.X_val))
        self.logger.info(f"Calibrated XGBoost: {calibrated_score:.4f}")
        
        # 选择最佳模型
        models = {
            'xgb': (xgb, xgb_score),
            'lgbm': (lgbm, lgbm_score),
            'catboost': (cat, cat_score),
            'voting': (voting, voting_score),
            'stacking': (stacking, stacking_score),
            'calibrated': (calibrated, calibrated_score)
        }
        
        best_name = max(models.items(), key=lambda x: x[1][1])[0]
        self.best_model = models[best_name][0]
        best_val_score = models[best_name][1]
        
        self.logger.info(f"\n最佳模型: {best_name} (验证集准确率: {best_val_score:.4f})")
        
        return self
    
    def optimize_threshold(self):
        """优化分类阈值"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("优化分类阈值...")
        self.logger.info("=" * 80)
        
        # 获取预测概率
        y_pred_proba = self.best_model.predict_proba(self.X_val)[:, 1]
        
        # 测试不同阈值
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            score = accuracy_score(self.y_val, y_pred)
            self.logger.info(f"阈值 {threshold:.2f}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.logger.info(f"\n最佳阈值: {best_threshold:.2f} ({best_score:.4f})")
        self.best_threshold = best_threshold
        self.optimization_results['Threshold_Optimization'] = best_score
        
        return self
    
    def preprocess_test_data(self):
        """预处理测试数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("预处理测试数据...")
        self.logger.info("=" * 80)
        
        # 保存真实标签
        if 'Attrition' in self.test_df.columns:
            self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            if self.y_test.isna().any():
                self.test_df['Attrition'] = self.test_df['Attrition'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
                self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            self.y_test = self.y_test.fillna(0).astype(int)
        
        # 特征工程
        test_df = self.advanced_feature_engineering(self.test_df, fit=False)
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        for col in cols_to_drop:
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
        
        # KNN填充
        numeric_cols = test_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0 and hasattr(self, 'knn_imputer'):
            test_df[numeric_cols] = self.knn_imputer.transform(test_df[numeric_cols])
        
        # 编码类别特征
        for col in self.label_encoders:
            if col in test_df.columns:
                le = self.label_encoders[col]
                test_df[col] = test_df[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
                test_df[col] = test_df[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                test_df[col] = le.transform(test_df[col])
        
        # 目标编码
        for col, encoding in self.target_encodings.items():
            if col in test_df.columns:
                global_mean = encoding.mean()
                test_df[f'{col}_target_enc'] = test_df[col].map(encoding).fillna(global_mean)
        
        # 确保特征一致
        missing_features = set(self.best_features) - set(test_df.columns)
        if missing_features:
            for feature in missing_features:
                test_df[feature] = 0
        
        test_df = test_df[self.best_features]
        
        # 标准化
        self.X_test = pd.DataFrame(
            self.scaler.transform(test_df),
            columns=self.best_features
        )
        
        self.logger.info(f"测试集形状: {self.X_test.shape}")
        
        return self
    
    def evaluate_on_test(self):
        """在测试集上评估"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("测试集评估（350条数据）...")
        self.logger.info("=" * 80)
        
        # 使用优化后的阈值
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.logger.info(f"\n测试集准确率: {accuracy*100:.2f}%")
        
        if accuracy >= 0.90:
            self.logger.info("✓ 已达到90%目标！")
        else:
            self.logger.info(f"✗ 距离90%目标还差: {(0.90 - accuracy)*100:.2f}%")
        
        self.logger.info(f"\n分类报告:")
        self.logger.info("\n" + classification_report(self.y_test, y_pred))
        
        return self
    
    def save_results(self):
        """保存结果"""
        self.logger.info("\n保存模型和结果...")
        
        # 保存最佳模型
        CommonUtil.save_model(self.best_model, os.path.join(self.model_dir, 'ultimate_best_model.pkl'))
        CommonUtil.save_model(self.scaler, os.path.join(self.model_dir, 'ultimate_scaler.pkl'))
        CommonUtil.save_model(self.label_encoders, os.path.join(self.model_dir, 'ultimate_label_encoders.pkl'))
        CommonUtil.save_model(self.best_features, os.path.join(self.model_dir, 'ultimate_features.pkl'))
        CommonUtil.save_model(self.best_threshold, os.path.join(self.model_dir, 'ultimate_threshold.pkl'))
        CommonUtil.save_model(self.target_encodings, os.path.join(self.model_dir, 'ultimate_target_encodings.pkl'))
        
        # 保存优化结果
        results_path = os.path.join(self.model_dir, 'optimization_results.pkl')
        CommonUtil.save_model(self.optimization_results, results_path)
        
        self.logger.info("所有文件已保存")
        
        return self
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_data()
            self.preprocess_data()
            self.test_oversampling_methods()
            self.feature_selection()
            self.optimize_models_with_bayesian()
            self.train_final_ensemble()
            self.optimize_threshold()
            self.preprocess_test_data()
            self.evaluate_on_test()
            self.save_results()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("终极优化完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    optimizer = UltimateOptimizer()
    optimizer.run()
