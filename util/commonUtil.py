"""
通用工具类
提供数据处理、模型保存加载等通用功能
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class CommonUtil:
    """通用工具类，提供常用的辅助功能"""
    
    @staticmethod
    def ensure_dir(directory):
        """
        确保目录存在，不存在则创建
        
        Args:
            directory: 目录路径
        """
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def save_model(model, filepath):
        """
        保存模型到文件
        
        Args:
            model: 模型对象
            filepath: 保存路径
        """
        CommonUtil.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load_model(filepath):
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型对象
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def remove_duplicates_from_train(train_df, test_df, feature_cols, logger=None):
        """
        从训练集中移除与测试集相同的样本
        
        Args:
            train_df: 训练集DataFrame
            test_df: 测试集DataFrame
            feature_cols: 特征列名列表
            logger: 日志记录器
            
        Returns:
            清洗后的训练集DataFrame
        """
        if logger:
            logger.info(f"原始训练集大小: {len(train_df)}")
            logger.info(f"测试集大小: {len(test_df)}")
        
        # 创建特征的字符串表示用于比较
        train_features = train_df[feature_cols].astype(str).agg('_'.join, axis=1)
        test_features = test_df[feature_cols].astype(str).agg('_'.join, axis=1)
        
        # 找出训练集中与测试集重复的样本
        duplicates_mask = train_features.isin(test_features)
        num_duplicates = duplicates_mask.sum()
        
        if logger:
            logger.info(f"发现重复样本数量: {num_duplicates}")
        
        # 移除重复样本
        train_df_cleaned = train_df[~duplicates_mask].reset_index(drop=True)
        
        if logger:
            logger.info(f"清洗后训练集大小: {len(train_df_cleaned)}")
        
        return train_df_cleaned
    
    @staticmethod
    def split_features_target(df, target_col):
        """
        分离特征和目标变量
        
        Args:
            df: DataFrame
            target_col: 目标列名
            
        Returns:
            X, y: 特征和目标变量
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    
    @staticmethod
    def get_numeric_categorical_features(X):
        """
        获取数值型和类别型特征列名
        
        Args:
            X: 特征DataFrame
            
        Returns:
            numeric_features, categorical_features: 数值型和类别型特征列表
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        return numeric_features, categorical_features
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            metrics: 包含各项指标的字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # 如果预测值是概率，计算AUC
        if len(np.unique(y_pred)) > 2:
            metrics['auc'] = roc_auc_score(y_true, y_pred)
        
        return metrics
