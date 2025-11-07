"""
数据质量分析 - 识别噪声和异常值
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

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor


class DataQualityAnalyzer:
    """数据质量分析器"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.output_dir = '../data'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'data_quality_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("数据质量分析 - 识别噪声和异常值")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.output_dir)
        
    def load_data(self):
        """加载数据"""
        self.logger.info("\n加载数据...")
        
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        self.logger.info(f"训练集: {self.train_df.shape}")
        self.logger.info(f"测试集: {self.test_df.shape}")
        
        return self
    
    def analyze_missing_values(self):
        """分析缺失值"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("1. 缺失值分析")
        self.logger.info("=" * 80)
        
        train_missing = self.train_df.isnull().sum()
        train_missing = train_missing[train_missing > 0].sort_values(ascending=False)
        
        if len(train_missing) > 0:
            self.logger.info("\n训练集缺失值:")
            for col, count in train_missing.items():
                pct = count / len(self.train_df) * 100
                self.logger.info(f"  {col}: {count} ({pct:.2f}%)")
        else:
            self.logger.info("\n✓ 训练集无缺失值")
        
        test_missing = self.test_df.isnull().sum()
        test_missing = test_missing[test_missing > 0].sort_values(ascending=False)
        
        if len(test_missing) > 0:
            self.logger.info("\n测试集缺失值:")
            for col, count in test_missing.items():
                pct = count / len(self.test_df) * 100
                self.logger.info(f"  {col}: {count} ({pct:.2f}%)")
        else:
            self.logger.info("\n✓ 测试集无缺失值")
    
    def analyze_duplicates(self):
        """分析重复值"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("2. 重复值分析")
        self.logger.info("=" * 80)
        
        # 完全重复的行
        train_dup = self.train_df.duplicated().sum()
        test_dup = self.test_df.duplicated().sum()
        
        self.logger.info(f"\n完全重复的行:")
        self.logger.info(f"  训练集: {train_dup}")
        self.logger.info(f"  测试集: {test_dup}")
        
        # 特征重复但标签不同（噪声）
        if 'Attrition' in self.train_df.columns:
            feature_cols = [c for c in self.train_df.columns if c not in ['Attrition', 'EmployeeNumber']]
            train_feature_dup = self.train_df[feature_cols].duplicated(keep=False)
            
            if train_feature_dup.sum() > 0:
                dup_groups = self.train_df[train_feature_dup].groupby(feature_cols)['Attrition'].apply(list)
                conflicting = sum(1 for labels in dup_groups if len(set(labels)) > 1)
                
                self.logger.info(f"\n特征重复但标签不同（噪声）:")
                self.logger.info(f"  训练集: {conflicting} 组")
                
                if conflicting > 0:
                    self.logger.info(f"  ⚠️ 这些样本可能是噪声数据")
    
    def analyze_outliers_statistical(self):
        """统计方法分析异常值"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("3. 统计异常值分析（IQR方法）")
        self.logger.info("=" * 80)
        
        numeric_cols = self.train_df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [c for c in numeric_cols if c not in ['Attrition', 'EmployeeNumber']]
        
        outlier_summary = {}
        
        for col in numeric_cols:
            data = self.train_df[col].dropna()
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            if outliers > 0:
                outlier_pct = outliers / len(data) * 100
                outlier_summary[col] = {
                    'count': outliers,
                    'percentage': outlier_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        if outlier_summary:
            self.logger.info(f"\n发现 {len(outlier_summary)} 个特征存在异常值:")
            for col, info in sorted(outlier_summary.items(), key=lambda x: x[1]['count'], reverse=True):
                self.logger.info(f"\n  {col}:")
                self.logger.info(f"    异常值数量: {info['count']} ({info['percentage']:.2f}%)")
                self.logger.info(f"    正常范围: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
        else:
            self.logger.info("\n✓ 未发现统计异常值")
        
        return outlier_summary
    
    def analyze_outliers_ml(self):
        """机器学习方法分析异常值"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("4. 机器学习异常值检测")
        self.logger.info("=" * 80)
        
        # 准备数据
        train_df = self.train_df.copy()
        
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
        
        # 1. Isolation Forest
        self.logger.info("\n使用 Isolation Forest 检测异常值...")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_predictions = iso_forest.fit_predict(X)
        iso_outliers = (iso_predictions == -1).sum()
        iso_pct = iso_outliers / len(X) * 100
        
        self.logger.info(f"  异常值数量: {iso_outliers} ({iso_pct:.2f}%)")
        
        # 2. Local Outlier Factor
        self.logger.info("\n使用 Local Outlier Factor 检测异常值...")
        lof = LocalOutlierFactor(contamination=0.1)
        lof_predictions = lof.fit_predict(X)
        lof_outliers = (lof_predictions == -1).sum()
        lof_pct = lof_outliers / len(X) * 100
        
        self.logger.info(f"  异常值数量: {lof_outliers} ({lof_pct:.2f}%)")
        
        # 3. Elliptic Envelope
        self.logger.info("\n使用 Elliptic Envelope 检测异常值...")
        try:
            ee = EllipticEnvelope(contamination=0.1, random_state=42)
            ee_predictions = ee.fit_predict(X)
            ee_outliers = (ee_predictions == -1).sum()
            ee_pct = ee_outliers / len(X) * 100
            
            self.logger.info(f"  异常值数量: {ee_outliers} ({ee_pct:.2f}%)")
        except Exception as e:
            self.logger.warning(f"  Elliptic Envelope 失败: {str(e)}")
            ee_predictions = np.ones(len(X))
        
        # 综合判断（至少2个方法认为是异常值）
        outlier_votes = (iso_predictions == -1).astype(int) + \
                       (lof_predictions == -1).astype(int) + \
                       (ee_predictions == -1).astype(int)
        
        consensus_outliers = (outlier_votes >= 2).sum()
        consensus_pct = consensus_outliers / len(X) * 100
        
        self.logger.info(f"\n综合判断（至少2个方法认为是异常值）:")
        self.logger.info(f"  异常值数量: {consensus_outliers} ({consensus_pct:.2f}%)")
        
        # 保存异常值索引
        self.outlier_indices = {
            'isolation_forest': np.where(iso_predictions == -1)[0],
            'lof': np.where(lof_predictions == -1)[0],
            'elliptic_envelope': np.where(ee_predictions == -1)[0],
            'consensus': np.where(outlier_votes >= 2)[0]
        }
        
        return self.outlier_indices
    
    def analyze_label_noise(self):
        """分析标签噪声"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("5. 标签噪声分析")
        self.logger.info("=" * 80)
        
        if 'Attrition' not in self.train_df.columns:
            self.logger.warning("训练集中没有Attrition列，跳过标签噪声分析")
            return
        
        # 使用交叉验证预测来识别可能的标签错误
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # 准备数据
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
        categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            train_df[col] = train_df[col].astype(str).fillna('Unknown')
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
        
        # 填充缺失值
        train_df = train_df.fillna(train_df.median())
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(train_df.values)
        
        # 使用随机森林进行交叉验证预测
        self.logger.info("\n使用随机森林进行交叉验证预测...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 获取预测概率
        cv_proba = cross_val_predict(rf, X, y, cv=5, method='predict_proba')
        
        # 找出预测概率与真实标签不一致的样本
        predicted_labels = cv_proba.argmax(axis=1)
        prediction_confidence = cv_proba.max(axis=1)
        
        # 标签不一致且模型很自信的样本可能是标签错误
        mislabeled = (predicted_labels != y) & (prediction_confidence > 0.7)
        mislabeled_count = mislabeled.sum()
        mislabeled_pct = mislabeled_count / len(y) * 100
        
        self.logger.info(f"\n可能的标签错误:")
        self.logger.info(f"  数量: {mislabeled_count} ({mislabeled_pct:.2f}%)")
        self.logger.info(f"  这些样本的真实标签与模型高置信度预测不一致")
        
        if mislabeled_count > 0:
            self.logger.info(f"\n前10个可能的标签错误样本:")
            mislabeled_indices = np.where(mislabeled)[0]
            for i, idx in enumerate(mislabeled_indices[:10]):
                self.logger.info(f"    样本 {idx}: 真实标签={y[idx]}, 预测标签={predicted_labels[idx]}, 置信度={prediction_confidence[idx]:.2f}")
        
        self.label_noise_indices = np.where(mislabeled)[0]
        
        return self.label_noise_indices
    
    def generate_report(self):
        """生成数据质量报告"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("数据质量总结")
        self.logger.info("=" * 80)
        
        self.logger.info("\n发现的问题:")
        
        issues = []
        
        # 缺失值
        train_missing = self.train_df.isnull().sum().sum()
        if train_missing > 0:
            issues.append(f"  1. 训练集存在 {train_missing} 个缺失值")
        
        # 重复值
        train_dup = self.train_df.duplicated().sum()
        if train_dup > 0:
            issues.append(f"  2. 训练集存在 {train_dup} 个完全重复的样本")
        
        # 异常值
        if hasattr(self, 'outlier_indices'):
            consensus_outliers = len(self.outlier_indices['consensus'])
            if consensus_outliers > 0:
                issues.append(f"  3. 训练集存在 {consensus_outliers} 个异常值样本")
        
        # 标签噪声
        if hasattr(self, 'label_noise_indices'):
            label_noise = len(self.label_noise_indices)
            if label_noise > 0:
                issues.append(f"  4. 训练集存在 {label_noise} 个可能的标签错误")
        
        if issues:
            for issue in issues:
                self.logger.info(issue)
        else:
            self.logger.info("  ✓ 数据质量良好，未发现明显问题")
        
        self.logger.info("\n建议:")
        self.logger.info("  1. 移除完全重复的样本")
        self.logger.info("  2. 移除或修正异常值样本")
        self.logger.info("  3. 检查并修正可能的标签错误")
        self.logger.info("  4. 使用鲁棒的缺失值填充方法")
    
    def run(self):
        """运行完整分析"""
        try:
            self.load_data()
            self.analyze_missing_values()
            self.analyze_duplicates()
            self.analyze_outliers_statistical()
            self.analyze_outliers_ml()
            self.analyze_label_noise()
            self.generate_report()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("数据质量分析完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    analyzer = DataQualityAnalyzer()
    analyzer.run()
