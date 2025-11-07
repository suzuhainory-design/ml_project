"""
使用清洗后的数据训练模型
移除噪声和异常值
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.logUtil import LogUtil
from util.commonUtil import CommonUtil
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import joblib


class CleanDataTrainer:
    """使用清洗后的数据训练模型"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_clean_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("使用清洗后的数据训练模型")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.model_dir)
        CommonUtil.ensure_dir(self.data_dir)
        
    def load_data(self):
        """加载数据"""
        self.logger.info("\n加载数据...")
        
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        self.logger.info(f"原始训练集: {self.train_df.shape}")
        self.logger.info(f"测试集: {self.test_df.shape}")
        
        return self
    
    def identify_outliers(self):
        """识别异常值和噪声"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("识别异常值和噪声")
        self.logger.info("=" * 80)
        
        # 准备数据
        train_df = self.train_df.copy()
        
        # 保存原始索引
        original_indices = train_df.index.tolist()
        
        # 处理Attrition
        train_df['Attrition'] = pd.to_numeric(train_df['Attrition'], errors='coerce')
        train_df['Attrition'] = train_df['Attrition'].fillna(0).astype(int)
        
        y = train_df['Attrition'].values
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        for col in cols_to_drop:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
        
        # 编码类别特征
        categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            train_df[col] = train_df[col].astype(str).fillna('Unknown')
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
        
        # 填充缺失值
        train_df = train_df.fillna(train_df.median())
        
        X = train_df.values
        
        # 1. 异常值检测
        self.logger.info("\n1. 使用机器学习方法检测异常值...")
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_predictions = iso_forest.fit_predict(X)
        
        lof = LocalOutlierFactor(contamination=0.1)
        lof_predictions = lof.fit_predict(X)
        
        try:
            ee = EllipticEnvelope(contamination=0.1, random_state=42)
            ee_predictions = ee.fit_predict(X)
        except:
            ee_predictions = np.ones(len(X))
        
        # 综合判断（至少2个方法认为是异常值）
        outlier_votes = (iso_predictions == -1).astype(int) + \
                       (lof_predictions == -1).astype(int) + \
                       (ee_predictions == -1).astype(int)
        
        outlier_mask = outlier_votes >= 2
        outlier_count = outlier_mask.sum()
        
        self.logger.info(f"  检测到 {outlier_count} 个异常值样本 ({outlier_count/len(X)*100:.2f}%)")
        
        # 2. 标签噪声检测
        self.logger.info("\n2. 使用交叉验证检测标签噪声...")
        
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_proba = cross_val_predict(rf, X, y, cv=5, method='predict_proba')
        
        predicted_labels = cv_proba.argmax(axis=1)
        prediction_confidence = cv_proba.max(axis=1)
        
        # 标签不一致且模型很自信的样本可能是标签错误
        label_noise_mask = (predicted_labels != y) & (prediction_confidence > 0.7)
        label_noise_count = label_noise_mask.sum()
        
        self.logger.info(f"  检测到 {label_noise_count} 个可能的标签错误 ({label_noise_count/len(y)*100:.2f}%)")
        
        # 3. 合并所有噪声样本
        noise_mask = outlier_mask | label_noise_mask
        noise_indices = np.where(noise_mask)[0]
        
        self.logger.info(f"\n总共识别出 {len(noise_indices)} 个噪声样本 ({len(noise_indices)/len(X)*100:.2f}%)")
        
        # 转换回原始索引
        self.noise_indices = [original_indices[i] for i in noise_indices]
        
        return self.noise_indices
    
    def clean_data(self):
        """清洗数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("清洗数据")
        self.logger.info("=" * 80)
        
        # 移除噪声样本
        self.train_clean = self.train_df.drop(index=self.noise_indices).reset_index(drop=True)
        
        self.logger.info(f"\n原始训练集: {len(self.train_df)} 样本")
        self.logger.info(f"移除噪声: {len(self.noise_indices)} 样本")
        self.logger.info(f"清洗后训练集: {len(self.train_clean)} 样本")
        
        # 检查类别分布
        if 'Attrition' in self.train_clean.columns:
            attrition_clean = pd.to_numeric(self.train_clean['Attrition'], errors='coerce').fillna(0).astype(int)
            class_dist = attrition_clean.value_counts().sort_index()
            
            self.logger.info(f"\n清洗后的类别分布:")
            for label, count in class_dist.items():
                pct = count / len(self.train_clean) * 100
                self.logger.info(f"  类别 {label}: {count} ({pct:.2f}%)")
        
        return self
    
    def preprocess_data(self):
        """预处理数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("预处理数据")
        self.logger.info("=" * 80)
        
        # 使用清洗后的训练集
        train_df = self.train_clean.copy()
        
        # 处理Attrition
        train_df['Attrition'] = pd.to_numeric(train_df['Attrition'], errors='coerce')
        train_df['Attrition'] = train_df['Attrition'].fillna(0).astype(int)
        
        self.y_train = train_df['Attrition'].values
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        for col in cols_to_drop:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
        
        # 编码类别特征
        self.label_encoders = {}
        categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            train_df[col] = train_df[col].astype(str).fillna('Unknown')
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            self.label_encoders[col] = le
        
        # 填充缺失值
        train_df = train_df.fillna(train_df.median())
        
        # 标准化
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(train_df.values)
        
        self.feature_names = train_df.columns.tolist()
        
        self.logger.info(f"特征数量: {len(self.feature_names)}")
        self.logger.info(f"训练样本数: {len(self.X_train)}")
        
        return self
    
    def train_models(self):
        """训练模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("训练模型")
        self.logger.info("=" * 80)
        
        # 1. XGBoost（基础版）
        self.logger.info("\n1. 训练 XGBoost (基础版) - 10折CV...")
        
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(xgb, self.X_train, self.y_train, cv=skf, scoring='accuracy')
        
        self.logger.info(f"  CV准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        self.logger.info(f"  各折准确率: {[f'{s:.4f}' for s in cv_scores]}")
        
        # 在全部数据上训练
        xgb.fit(self.X_train, self.y_train)
        
        # 保存模型
        joblib.dump(xgb, os.path.join(self.model_dir, 'clean_xgb_model.pkl'))
        
        self.models = {'XGBoost': (xgb, cv_scores.mean())}
        
        # 2. XGBoost + ADASYN（最佳参数）
        self.logger.info("\n2. 训练 XGBoost+ADASYN (最佳参数) - 10折CV...")
        
        # ADASYN过采样
        adasyn = ADASYN(sampling_strategy=0.5, n_neighbors=5, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(self.X_train, self.y_train)
        
        self.logger.info(f"  过采样后: {len(X_resampled)} 样本")
        
        xgb_adasyn = XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=1,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='logloss'
        )
        
        cv_scores_adasyn = cross_val_score(xgb_adasyn, X_resampled, y_resampled, cv=skf, scoring='accuracy')
        
        self.logger.info(f"  CV准确率: {cv_scores_adasyn.mean():.4f} (+/- {cv_scores_adasyn.std():.4f})")
        self.logger.info(f"  各折准确率: {[f'{s:.4f}' for s in cv_scores_adasyn]}")
        
        # 在全部数据上训练
        xgb_adasyn.fit(X_resampled, y_resampled)
        
        # 保存模型
        joblib.dump(xgb_adasyn, os.path.join(self.model_dir, 'clean_xgb_adasyn_model.pkl'))
        
        self.models['XGBoost+ADASYN'] = (xgb_adasyn, cv_scores_adasyn.mean())
        
        return self
    
    def evaluate_on_test(self):
        """在测试集上评估"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("测试集评估（350条数据）")
        self.logger.info("=" * 80)
        
        # 预处理测试集
        test_df = self.test_df.copy()
        
        # 处理Attrition
        test_df['Attrition'] = pd.to_numeric(test_df['Attrition'], errors='coerce')
        test_df['Attrition'] = test_df['Attrition'].fillna(0).astype(int)
        
        y_test = test_df['Attrition'].values
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        for col in cols_to_drop:
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
        
        # 编码类别特征
        categorical_cols = test_df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if col in self.label_encoders:
                test_df[col] = test_df[col].astype(str).fillna('Unknown')
                # 处理未见过的类别
                le = self.label_encoders[col]
                test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                test_df[col] = le.transform(test_df[col])
        
        # 填充缺失值
        test_df = test_df.fillna(0)
        
        # 标准化
        X_test = self.scaler.transform(test_df.values)
        
        # 评估每个模型
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        results = {}
        
        for name, (model, cv_acc) in self.models.items():
            self.logger.info(f"\n{name}:")
            
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"  CV准确率: {cv_acc:.4f} ({cv_acc*100:.2f}%)")
            self.logger.info(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
            
            overfit = cv_acc - test_acc
            self.logger.info(f"  过拟合程度: {overfit:.4f} ({overfit*100:.2f}%)")
            
            results[name] = {
                'cv_acc': cv_acc,
                'test_acc': test_acc,
                'overfit': overfit
            }
            
            # 分类报告
            self.logger.info(f"\n{name} 分类报告:")
            self.logger.info(f"\n{classification_report(y_test, y_pred, target_names=['未流失', '流失'])}")
            
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info(f"\n混淆矩阵:")
            self.logger.info(f"\n{cm}")
        
        # 找出最佳模型
        best_model_name = max(results.items(), key=lambda x: x[1]['test_acc'])[0]
        best_test_acc = results[best_model_name]['test_acc']
        
        self.logger.info(f"\n" + "=" * 80)
        self.logger.info(f"最佳模型: {best_model_name}")
        self.logger.info(f"测试集准确率: {best_test_acc*100:.2f}%")
        
        if best_test_acc >= 0.90:
            self.logger.info(f"✓ 达到90%目标！")
        else:
            gap = 0.90 - best_test_acc
            self.logger.info(f"✗ 距离90%目标还差: {gap*100:.2f}%")
        
        self.logger.info("=" * 80)
        
        # 保存预处理器
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'clean_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(self.model_dir, 'clean_label_encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(self.model_dir, 'clean_feature_names.pkl'))
        
        return results
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_data()
            self.identify_outliers()
            self.clean_data()
            self.preprocess_data()
            self.train_models()
            results = self.evaluate_on_test()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("数据清洗训练完成！")
            self.logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = CleanDataTrainer()
    trainer.run()
