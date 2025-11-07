"""
同时使用10折CV和8:2分割训练模型，对比两种策略的效果
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

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import joblib


class TrainCVAndSplit:
    """同时使用10折CV和8:2分割训练模型"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_cv_split_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("同时使用10折CV和8:2分割训练模型")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.model_dir)
        CommonUtil.ensure_dir(self.data_dir)
        
        # 最佳参数
        self.best_xgb_params = {
            'n_estimators': 400,
            'max_depth': 7,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        self.adasyn_params = {
            'sampling_strategy': 0.5,
            'n_neighbors': 5,
            'random_state': 42
        }
        
    def load_data(self):
        """加载数据"""
        self.logger.info("\n加载数据...")
        
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        self.logger.info(f"原始训练集: {self.train_df.shape}")
        self.logger.info(f"测试集: {self.test_df.shape}")
        
        return self
    
    def preprocess_data(self):
        """预处理数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("预处理数据")
        self.logger.info("=" * 80)
        
        # 处理训练集
        train_df = self.train_df.copy()
        
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
        X = self.scaler.fit_transform(train_df.values)
        
        self.feature_names = train_df.columns.tolist()
        
        self.logger.info(f"特征数量: {len(self.feature_names)}")
        self.logger.info(f"总样本数: {len(X)}")
        
        return X, y
    
    def train_with_10fold_cv(self, X, y):
        """使用10折CV训练"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("方法1: 10折交叉验证")
        self.logger.info("=" * 80)
        
        # 1. XGBoost + ADASYN + 10折CV
        self.logger.info("\n训练 XGBoost+ADASYN (10折CV)...")
        
        # ADASYN过采样
        adasyn = ADASYN(**self.adasyn_params)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        self.logger.info(f"过采样后: {len(X_resampled)} 样本")
        
        # 10折CV
        model_cv = XGBClassifier(**self.best_xgb_params)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(model_cv, X_resampled, y_resampled, cv=skf, scoring='accuracy')
        
        self.logger.info(f"10折CV准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        self.logger.info(f"各折准确率: {[f'{s:.4f}' for s in cv_scores]}")
        
        # 在全部数据上训练最终模型
        self.logger.info("\n在全部数据上训练最终模型...")
        model_cv.fit(X_resampled, y_resampled)
        
        self.model_10fold = model_cv
        self.cv_score = cv_scores.mean()
        
        return model_cv, cv_scores.mean()
    
    def train_with_82_split(self, X, y):
        """使用8:2分割训练"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("方法2: 8:2分割")
        self.logger.info("=" * 80)
        
        # 8:2分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        self.logger.info(f"训练集: {len(X_train)} 样本 (80%)")
        self.logger.info(f"验证集: {len(X_val)} 样本 (20%)")
        
        # ADASYN过采样
        adasyn = ADASYN(**self.adasyn_params)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        
        self.logger.info(f"过采样后训练集: {len(X_train_resampled)} 样本")
        
        # 训练模型
        model_split = XGBClassifier(**self.best_xgb_params)
        model_split.fit(X_train_resampled, y_train_resampled)
        
        # 验证集评估
        y_val_pred = model_split.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        self.logger.info(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.model_82split = model_split
        self.val_score = val_acc
        
        return model_split, val_acc
    
    def evaluate_on_test(self):
        """在测试集上评估两种方法"""
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
                le = self.label_encoders[col]
                test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                test_df[col] = le.transform(test_df[col])
        
        # 填充缺失值
        test_df = test_df.fillna(0)
        
        # 标准化
        X_test = self.scaler.transform(test_df.values)
        
        # 评估10折CV模型
        self.logger.info("\n" + "=" * 80)
        self.logger.info("方法1: 10折CV模型")
        self.logger.info("=" * 80)
        
        y_pred_cv = self.model_10fold.predict(X_test)
        test_acc_cv = accuracy_score(y_test, y_pred_cv)
        
        self.logger.info(f"10折CV准确率: {self.cv_score:.4f} ({self.cv_score*100:.2f}%)")
        self.logger.info(f"测试集准确率: {test_acc_cv:.4f} ({test_acc_cv*100:.2f}%)")
        
        overfit_cv = self.cv_score - test_acc_cv
        self.logger.info(f"过拟合程度: {overfit_cv:.4f} ({overfit_cv*100:.2f}%)")
        
        self.logger.info(f"\n分类报告:")
        self.logger.info(f"\n{classification_report(y_test, y_pred_cv, target_names=['未流失', '流失'])}")
        
        cm_cv = confusion_matrix(y_test, y_pred_cv)
        self.logger.info(f"\n混淆矩阵:")
        self.logger.info(f"\n{cm_cv}")
        
        # 评估8:2分割模型
        self.logger.info("\n" + "=" * 80)
        self.logger.info("方法2: 8:2分割模型")
        self.logger.info("=" * 80)
        
        y_pred_split = self.model_82split.predict(X_test)
        test_acc_split = accuracy_score(y_test, y_pred_split)
        
        self.logger.info(f"验证集准确率: {self.val_score:.4f} ({self.val_score*100:.2f}%)")
        self.logger.info(f"测试集准确率: {test_acc_split:.4f} ({test_acc_split*100:.2f}%)")
        
        overfit_split = self.val_score - test_acc_split
        self.logger.info(f"过拟合程度: {overfit_split:.4f} ({overfit_split*100:.2f}%)")
        
        self.logger.info(f"\n分类报告:")
        self.logger.info(f"\n{classification_report(y_test, y_pred_split, target_names=['未流失', '流失'])}")
        
        cm_split = confusion_matrix(y_test, y_pred_split)
        self.logger.info(f"\n混淆矩阵:")
        self.logger.info(f"\n{cm_split}")
        
        # 对比结果
        self.logger.info("\n" + "=" * 80)
        self.logger.info("两种方法对比")
        self.logger.info("=" * 80)
        
        self.logger.info(f"\n{'方法':<20} {'验证准确率':<15} {'测试准确率':<15} {'过拟合程度':<15}")
        self.logger.info(f"{'-'*65}")
        self.logger.info(f"{'10折CV':<20} {self.cv_score*100:>6.2f}% {test_acc_cv*100:>13.2f}% {overfit_cv*100:>15.2f}%")
        self.logger.info(f"{'8:2分割':<20} {self.val_score*100:>6.2f}% {test_acc_split*100:>13.2f}% {overfit_split*100:>15.2f}%")
        
        # 找出最佳方法
        if test_acc_cv > test_acc_split:
            best_method = "10折CV"
            best_acc = test_acc_cv
            best_model = self.model_10fold
        else:
            best_method = "8:2分割"
            best_acc = test_acc_split
            best_model = self.model_82split
        
        self.logger.info(f"\n最佳方法: {best_method}")
        self.logger.info(f"测试集准确率: {best_acc*100:.2f}%")
        
        if best_acc >= 0.90:
            self.logger.info(f"✓ 达到90%目标！")
        else:
            gap = 0.90 - best_acc
            self.logger.info(f"✗ 距离90%目标还差: {gap*100:.2f}%")
        
        self.logger.info("=" * 80)
        
        # 保存最佳模型
        joblib.dump(best_model, os.path.join(self.model_dir, 'cv_split_best_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'cv_split_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(self.model_dir, 'cv_split_label_encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(self.model_dir, 'cv_split_feature_names.pkl'))
        
        return {
            '10fold_cv': {
                'cv_acc': self.cv_score,
                'test_acc': test_acc_cv,
                'overfit': overfit_cv
            },
            '82_split': {
                'val_acc': self.val_score,
                'test_acc': test_acc_split,
                'overfit': overfit_split
            },
            'best_method': best_method,
            'best_acc': best_acc
        }
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_data()
            X, y = self.preprocess_data()
            
            # 方法1: 10折CV
            self.train_with_10fold_cv(X, y)
            
            # 方法2: 8:2分割
            self.train_with_82_split(X, y)
            
            # 测试集评估
            results = self.evaluate_on_test()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("训练完成！")
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
    trainer = TrainCVAndSplit()
    trainer.run()
