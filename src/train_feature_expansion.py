"""
固定使用8:2分割，对特征进行升维处理
测试多种升维方法：多项式特征、交互特征、PCA等
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import joblib
import itertools


class TrainFeatureExpansion:
    """特征升维训练"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_feature_expansion_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("特征升维训练（固定8:2分割）")
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
        
        self.results = {}
        
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
        
        # 保存原始特征名
        self.original_feature_names = train_df.columns.tolist()
        
        # 标准化
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(train_df.values)
        
        self.logger.info(f"原始特征数量: {len(self.original_feature_names)}")
        self.logger.info(f"总样本数: {len(X)}")
        
        # 8:2分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        self.logger.info(f"训练集: {len(X_train)} 样本 (80%)")
        self.logger.info(f"验证集: {len(X_val)} 样本 (20%)")
        
        return X_train, X_val, y_train, y_val
    
    def create_interaction_features(self, X):
        """创建交互特征"""
        self.logger.info("\n创建交互特征...")
        
        # 选择最重要的特征进行交互
        important_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 前10个特征
        
        interactions = []
        for i, j in itertools.combinations(important_indices, 2):
            interactions.append(X[:, i] * X[:, j])
        
        X_interactions = np.column_stack([X] + interactions)
        
        self.logger.info(f"交互特征数量: {len(interactions)}")
        self.logger.info(f"总特征数量: {X_interactions.shape[1]}")
        
        return X_interactions
    
    def train_baseline(self, X_train, X_val, y_train, y_val):
        """基线模型（无升维）"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("方法1: 基线（无升维）")
        self.logger.info("=" * 80)
        
        # ADASYN过采样
        adasyn = ADASYN(**self.adasyn_params)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        
        # 训练模型
        model = XGBClassifier(**self.best_xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # 验证集评估
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        self.logger.info(f"特征数量: {X_train.shape[1]}")
        self.logger.info(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.results['baseline'] = {
            'model': model,
            'val_acc': val_acc,
            'feature_count': X_train.shape[1],
            'X_train': X_train,
            'X_val': X_val
        }
        
        return model, val_acc
    
    def train_polynomial(self, X_train, X_val, y_train, y_val, degree=2):
        """多项式特征"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"方法2: 多项式特征（degree={degree}）")
        self.logger.info("=" * 80)
        
        # 多项式特征
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        
        self.logger.info(f"原始特征: {X_train.shape[1]}")
        self.logger.info(f"多项式特征: {X_train_poly.shape[1]}")
        
        # ADASYN过采样
        adasyn = ADASYN(**self.adasyn_params)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_poly, y_train)
        
        # 训练模型
        model = XGBClassifier(**self.best_xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # 验证集评估
        y_val_pred = model.predict(X_val_poly)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        self.logger.info(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.results[f'polynomial_{degree}'] = {
            'model': model,
            'val_acc': val_acc,
            'feature_count': X_train_poly.shape[1],
            'poly': poly,
            'X_train': X_train_poly,
            'X_val': X_val_poly
        }
        
        return model, val_acc
    
    def train_interaction(self, X_train, X_val, y_train, y_val):
        """交互特征"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("方法3: 交互特征")
        self.logger.info("=" * 80)
        
        # 创建交互特征
        X_train_inter = self.create_interaction_features(X_train)
        X_val_inter = self.create_interaction_features(X_val)
        
        # ADASYN过采样
        adasyn = ADASYN(**self.adasyn_params)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_inter, y_train)
        
        # 训练模型
        model = XGBClassifier(**self.best_xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # 验证集评估
        y_val_pred = model.predict(X_val_inter)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        self.logger.info(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.results['interaction'] = {
            'model': model,
            'val_acc': val_acc,
            'feature_count': X_train_inter.shape[1],
            'X_train': X_train_inter,
            'X_val': X_val_inter
        }
        
        return model, val_acc
    
    def train_pca(self, X_train, X_val, y_train, y_val, n_components=50):
        """PCA降维后再升维"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"方法4: PCA（n_components={n_components}）")
        self.logger.info("=" * 80)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        
        self.logger.info(f"原始特征: {X_train.shape[1]}")
        self.logger.info(f"PCA特征: {X_train_pca.shape[1]}")
        self.logger.info(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
        
        # ADASYN过采样
        adasyn = ADASYN(**self.adasyn_params)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_pca, y_train)
        
        # 训练模型
        model = XGBClassifier(**self.best_xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # 验证集评估
        y_val_pred = model.predict(X_val_pca)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        self.logger.info(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.results[f'pca_{n_components}'] = {
            'model': model,
            'val_acc': val_acc,
            'feature_count': X_train_pca.shape[1],
            'pca': pca,
            'X_train': X_train_pca,
            'X_val': X_val_pca
        }
        
        return model, val_acc
    
    def train_poly_interaction(self, X_train, X_val, y_train, y_val):
        """多项式+交互特征"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("方法5: 多项式+交互特征")
        self.logger.info("=" * 80)
        
        # 先创建交互特征
        X_train_inter = self.create_interaction_features(X_train)
        X_val_inter = self.create_interaction_features(X_val)
        
        # 再创建多项式特征（degree=2）
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly_inter = poly.fit_transform(X_train_inter)
        X_val_poly_inter = poly.transform(X_val_inter)
        
        self.logger.info(f"原始特征: {X_train.shape[1]}")
        self.logger.info(f"交互特征: {X_train_inter.shape[1]}")
        self.logger.info(f"多项式+交互特征: {X_train_poly_inter.shape[1]}")
        
        # ADASYN过采样
        adasyn = ADASYN(**self.adasyn_params)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_poly_inter, y_train)
        
        # 训练模型
        model = XGBClassifier(**self.best_xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # 验证集评估
        y_val_pred = model.predict(X_val_poly_inter)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        self.logger.info(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.results['poly_interaction'] = {
            'model': model,
            'val_acc': val_acc,
            'feature_count': X_train_poly_inter.shape[1],
            'poly': poly,
            'X_train': X_train_poly_inter,
            'X_val': X_val_poly_inter
        }
        
        return model, val_acc
    
    def evaluate_on_test(self):
        """在测试集上评估所有方法"""
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
        
        # 评估所有方法
        test_results = []
        
        for method_name, method_data in self.results.items():
            self.logger.info(f"\n{'-' * 80}")
            self.logger.info(f"方法: {method_name}")
            self.logger.info(f"{'-' * 80}")
            
            model = method_data['model']
            val_acc = method_data['val_acc']
            feature_count = method_data['feature_count']
            
            # 根据方法转换测试集
            if method_name == 'baseline':
                X_test_transformed = X_test
            elif 'polynomial' in method_name:
                poly = method_data['poly']
                X_test_transformed = poly.transform(X_test)
            elif method_name == 'interaction':
                X_test_transformed = self.create_interaction_features(X_test)
            elif 'pca' in method_name:
                pca = method_data['pca']
                X_test_transformed = pca.transform(X_test)
            elif method_name == 'poly_interaction':
                X_test_inter = self.create_interaction_features(X_test)
                poly = method_data['poly']
                X_test_transformed = poly.transform(X_test_inter)
            
            # 预测
            y_pred = model.predict(X_test_transformed)
            test_acc = accuracy_score(y_test, y_pred)
            
            overfit = val_acc - test_acc
            
            self.logger.info(f"特征数量: {feature_count}")
            self.logger.info(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
            self.logger.info(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
            self.logger.info(f"过拟合程度: {overfit:.4f} ({overfit*100:.2f}%)")
            
            if test_acc >= 0.90:
                self.logger.info(f"✓ 达到90%目标！")
            else:
                gap = 0.90 - test_acc
                self.logger.info(f"✗ 距离90%目标还差: {gap*100:.2f}%")
            
            test_results.append({
                'method': method_name,
                'feature_count': feature_count,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'overfit': overfit
            })
        
        # 汇总结果
        self.logger.info("\n" + "=" * 80)
        self.logger.info("所有方法对比")
        self.logger.info("=" * 80)
        
        # 按测试集准确率排序
        test_results.sort(key=lambda x: x['test_acc'], reverse=True)
        
        self.logger.info(f"\n{'排名':<5} {'方法':<20} {'特征数':<10} {'验证准确率':<12} {'测试准确率':<12} {'过拟合':<10}")
        self.logger.info(f"{'-' * 80}")
        
        for i, result in enumerate(test_results, 1):
            self.logger.info(
                f"{i:<5} {result['method']:<20} {result['feature_count']:<10} "
                f"{result['val_acc']*100:>6.2f}% {result['test_acc']*100:>11.2f}% "
                f"{result['overfit']*100:>10.2f}%"
            )
        
        # 最佳方法
        best_result = test_results[0]
        self.logger.info(f"\n最佳方法: {best_result['method']}")
        self.logger.info(f"特征数量: {best_result['feature_count']}")
        self.logger.info(f"测试集准确率: {best_result['test_acc']*100:.2f}%")
        
        if best_result['test_acc'] >= 0.90:
            self.logger.info(f"✓ 达到90%目标！")
        else:
            gap = 0.90 - best_result['test_acc']
            self.logger.info(f"✗ 距离90%目标还差: {gap*100:.2f}%")
        
        self.logger.info("=" * 80)
        
        return test_results
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_data()
            X_train, X_val, y_train, y_val = self.preprocess_data()
            
            # 方法1: 基线（无升维）
            self.train_baseline(X_train, X_val, y_train, y_val)
            
            # 方法2: 多项式特征（degree=2）
            self.train_polynomial(X_train, X_val, y_train, y_val, degree=2)
            
            # 方法3: 交互特征
            self.train_interaction(X_train, X_val, y_train, y_val)
            
            # 方法4: PCA（跳过，因为PCA是降维不是升维）
            # self.train_pca(X_train, X_val, y_train, y_val, n_components=20)
            
            # 方法5: 多项式+交互特征
            # self.train_poly_interaction(X_train, X_val, y_train, y_val)  # 特征太多，可能导致内存问题
            
            # 测试集评估
            test_results = self.evaluate_on_test()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("训练完成！")
            self.logger.info("=" * 80)
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    trainer = TrainFeatureExpansion()
    trainer.run()
