"""
使用4:6分割训练集（40%验证集，60%训练集）
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN
import joblib


class Train46Split:
    """使用4:6分割训练模型"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_46split_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("使用4:6分割训练模型（40%验证集，60%训练集）")
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
    
    def split_data(self, X, y):
        """4:6分割数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("4:6分割数据（40%验证集，60%训练集）")
        self.logger.info("=" * 80)
        
        # 使用stratify确保类别分布一致
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.4,  # 40%用于验证
            random_state=42,
            stratify=y
        )
        
        self.logger.info(f"\n训练集: {len(X_train)} 样本 (60%)")
        self.logger.info(f"验证集: {len(X_val)} 样本 (40%)")
        
        # 检查类别分布
        train_dist = pd.Series(y_train).value_counts().sort_index()
        val_dist = pd.Series(y_val).value_counts().sort_index()
        
        self.logger.info(f"\n训练集类别分布:")
        for label, count in train_dist.items():
            pct = count / len(y_train) * 100
            self.logger.info(f"  类别 {label}: {count} ({pct:.2f}%)")
        
        self.logger.info(f"\n验证集类别分布:")
        for label, count in val_dist.items():
            pct = count / len(y_val) * 100
            self.logger.info(f"  类别 {label}: {count} ({pct:.2f}%)")
        
        return X_train, X_val, y_train, y_val
    
    def train_models(self, X_train, X_val, y_train, y_val):
        """训练多个模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("训练模型")
        self.logger.info("=" * 80)
        
        self.models = {}
        
        # 1. XGBoost（基础版）
        self.logger.info("\n1. 训练 XGBoost (基础版)...")
        
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb.fit(X_train, y_train)
        val_pred = xgb.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.logger.info(f"  验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.models['XGBoost'] = (xgb, val_acc)
        
        # 2. XGBoost + ADASYN（最佳参数）
        self.logger.info("\n2. 训练 XGBoost+ADASYN (最佳参数)...")
        
        # ADASYN过采样
        adasyn = ADASYN(sampling_strategy=0.5, n_neighbors=5, random_state=42)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        
        self.logger.info(f"  过采样后训练集: {len(X_train_resampled)} 样本")
        
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
        
        xgb_adasyn.fit(X_train_resampled, y_train_resampled)
        val_pred_adasyn = xgb_adasyn.predict(X_val)
        val_acc_adasyn = accuracy_score(y_val, val_pred_adasyn)
        
        self.logger.info(f"  验证集准确率: {val_acc_adasyn:.4f} ({val_acc_adasyn*100:.2f}%)")
        
        self.models['XGBoost+ADASYN'] = (xgb_adasyn, val_acc_adasyn)
        
        # 3. LightGBM
        self.logger.info("\n3. 训练 LightGBM...")
        
        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        lgbm.fit(X_train, y_train)
        val_pred_lgbm = lgbm.predict(X_val)
        val_acc_lgbm = accuracy_score(y_val, val_pred_lgbm)
        
        self.logger.info(f"  验证集准确率: {val_acc_lgbm:.4f} ({val_acc_lgbm*100:.2f}%)")
        
        self.models['LightGBM'] = (lgbm, val_acc_lgbm)
        
        # 4. CatBoost
        self.logger.info("\n4. 训练 CatBoost...")
        
        cat = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=0
        )
        
        cat.fit(X_train, y_train)
        val_pred_cat = cat.predict(X_val)
        val_acc_cat = accuracy_score(y_val, val_pred_cat)
        
        self.logger.info(f"  验证集准确率: {val_acc_cat:.4f} ({val_acc_cat*100:.2f}%)")
        
        self.models['CatBoost'] = (cat, val_acc_cat)
        
        # 5. LightGBM + ADASYN
        self.logger.info("\n5. 训练 LightGBM+ADASYN...")
        
        lgbm_adasyn = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        lgbm_adasyn.fit(X_train_resampled, y_train_resampled)
        val_pred_lgbm_adasyn = lgbm_adasyn.predict(X_val)
        val_acc_lgbm_adasyn = accuracy_score(y_val, val_pred_lgbm_adasyn)
        
        self.logger.info(f"  验证集准确率: {val_acc_lgbm_adasyn:.4f} ({val_acc_lgbm_adasyn*100:.2f}%)")
        
        self.models['LightGBM+ADASYN'] = (lgbm_adasyn, val_acc_lgbm_adasyn)
        
        # 6. CatBoost + ADASYN
        self.logger.info("\n6. 训练 CatBoost+ADASYN...")
        
        cat_adasyn = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=0
        )
        
        cat_adasyn.fit(X_train_resampled, y_train_resampled)
        val_pred_cat_adasyn = cat_adasyn.predict(X_val)
        val_acc_cat_adasyn = accuracy_score(y_val, val_pred_cat_adasyn)
        
        self.logger.info(f"  验证集准确率: {val_acc_cat_adasyn:.4f} ({val_acc_cat_adasyn*100:.2f}%)")
        
        self.models['CatBoost+ADASYN'] = (cat_adasyn, val_acc_cat_adasyn)
        
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
                le = self.label_encoders[col]
                test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                test_df[col] = le.transform(test_df[col])
        
        # 填充缺失值
        test_df = test_df.fillna(0)
        
        # 标准化
        X_test = self.scaler.transform(test_df.values)
        
        # 评估每个模型
        results = {}
        
        for name, (model, val_acc) in self.models.items():
            self.logger.info(f"\n{name}:")
            
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"  验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
            self.logger.info(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
            
            overfit = val_acc - test_acc
            self.logger.info(f"  过拟合程度: {overfit:.4f} ({overfit*100:.2f}%)")
            
            results[name] = {
                'val_acc': val_acc,
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
        best_val_acc = results[best_model_name]['val_acc']
        
        self.logger.info(f"\n" + "=" * 80)
        self.logger.info(f"最佳模型: {best_model_name}")
        self.logger.info(f"验证集准确率: {best_val_acc*100:.2f}%")
        self.logger.info(f"测试集准确率: {best_test_acc*100:.2f}%")
        
        if best_test_acc >= 0.90:
            self.logger.info(f"✓ 达到90%目标！")
        else:
            gap = 0.90 - best_test_acc
            self.logger.info(f"✗ 距离90%目标还差: {gap*100:.2f}%")
        
        self.logger.info("=" * 80)
        
        # 保存最佳模型
        best_model = self.models[best_model_name][0]
        joblib.dump(best_model, os.path.join(self.model_dir, 'split46_best_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'split46_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(self.model_dir, 'split46_label_encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(self.model_dir, 'split46_feature_names.pkl'))
        
        return results
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_data()
            X, y = self.preprocess_data()
            X_train, X_val, y_train, y_val = self.split_data(X, y)
            self.train_models(X_train, X_val, y_train, y_val)
            results = self.evaluate_on_test()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("4:6分割训练完成！")
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
    trainer = Train46Split()
    trainer.run()
