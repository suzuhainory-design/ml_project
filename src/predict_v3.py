"""
最终优化版预测脚本 - 使用优化后的模型和阈值
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.logUtil import LogUtil
from util.commonUtil import CommonUtil

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score


class HRAttritionPredictorV2:
    """最终优化版HR员工流失预测器"""
    
    def __init__(self, log_dir='../log', data_dir='../data', model_dir='../model'):
        """初始化预测器"""
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=log_dir, log_name=f'predict_v3_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("最终优化版HR员工流失预测开始")
        self.logger.info("=" * 80)
        
        self.test_df = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = 0.5
    
    def load_model(self):
        """加载模型和预处理器"""
        self.logger.info("加载模型和预处理器...")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.model_dir, 'best_model_v3.pkl')
        best_model_info = CommonUtil.load_model(best_model_path)
        self.best_model = best_model_info['model']
        self.best_model_name = best_model_info['name']
        self.best_threshold = best_model_info['threshold']
        self.logger.info(f"已加载最佳模型: {self.best_model_name}, 阈值: {self.best_threshold:.2f}")
        
        # 加载标准化器
        scaler_path = os.path.join(self.model_dir, 'scaler_v3.pkl')
        self.scaler = CommonUtil.load_model(scaler_path)
        self.logger.info("已加载标准化器")
        
        # 加载标签编码器
        encoders_path = os.path.join(self.model_dir, 'label_encoders_v3.pkl')
        self.label_encoders = CommonUtil.load_model(encoders_path)
        self.logger.info("已加载标签编码器")
        
        # 加载特征名称
        features_path = os.path.join(self.model_dir, 'feature_names_v3.pkl')
        self.feature_names = CommonUtil.load_model(features_path)
        self.logger.info(f"已加载特征名称，共 {len(self.feature_names)} 个特征")
        
        return self
    
    def load_test_data(self):
        """加载测试数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("加载测试数据...")
        self.logger.info("=" * 80)
        
        test_path = os.path.join(self.data_dir, 'test.csv')
        self.test_df = pd.read_csv(test_path)
        
        self.logger.info(f"测试集形状: {self.test_df.shape}")
        self.logger.info(f"测试集列: {list(self.test_df.columns)}")
        
        return self
    
    def _create_features(self, df):
        """创建特征（与训练时相同）"""
        df = df.copy()
        
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
        
        # 工作经验特征
        if 'TotalWorkingYears' in df.columns and 'YearsAtCompany' in df.columns:
            df['ExperienceBeforeCompany'] = df['TotalWorkingYears'] - df['YearsAtCompany']
            df['CompanyTenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
        
        # 职业发展特征
        if 'YearsAtCompany' in df.columns and 'YearsSinceLastPromotion' in df.columns:
            df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
            df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
        
        # 工作稳定性
        if 'YearsAtCompany' in df.columns and 'YearsInCurrentRole' in df.columns:
            df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
        
        if 'YearsAtCompany' in df.columns and 'YearsWithCurrManager' in df.columns:
            df['ManagerStability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
        
        # 薪资特征
        if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
            df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
        
        if 'MonthlyIncome' in df.columns and 'TotalWorkingYears' in df.columns:
            df['IncomePerExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
        
        if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
            df['IncomePerJobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
        
        # 满意度综合指标
        satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                           'RelationshipSatisfaction', 'WorkLifeBalance']
        available_satisfaction = [col for col in satisfaction_cols if col in df.columns]
        if available_satisfaction:
            df['OverallSatisfaction'] = df[available_satisfaction].mean(axis=1)
            df['SatisfactionStd'] = df[available_satisfaction].std(axis=1)
        
        # 跳槽频率
        if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
            df['JobHoppingRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
        
        # 工作投入与满意度交互
        if 'JobInvolvement' in df.columns and 'JobSatisfaction' in df.columns:
            df['InvolvementSatisfactionProduct'] = df['JobInvolvement'] * df['JobSatisfaction']
        
        return df
    
    def preprocess_test_data(self):
        """预处理测试数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("预处理测试数据...")
        self.logger.info("=" * 80)
        
        # 保存真实标签（如果存在）
        if 'Attrition' in self.test_df.columns:
            self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            if self.y_test.isna().any():
                self.test_df['Attrition'] = self.test_df['Attrition'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
                self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            self.y_test = self.y_test.fillna(0).astype(int)
            self.logger.info(f"测试集包含真实标签，样本数: {len(self.y_test)}")
            self.logger.info(f"目标分布 - 流失: {(self.y_test == 1).sum()}, 未流失: {(self.y_test == 0).sum()}")
        else:
            self.logger.info("测试集不包含真实标签")
        
        # 应用特征工程（与训练时相同）
        self.test_df = self._create_features(self.test_df)
        
        # 删除无用列
        cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
        for col in cols_to_drop:
            if col in self.test_df.columns:
                self.test_df = self.test_df.drop(columns=[col])
        
        # 确保数值类型
        numeric_cols = ['Age', 'DistanceFromHome', 'Education', 
                      'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                      'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                      'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                      'StockOptionLevel', 'TotalWorkingYears',
                      'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                      'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        for col in numeric_cols:
            if col in self.test_df.columns:
                self.test_df[col] = pd.to_numeric(self.test_df[col], errors='coerce')
        
        # 处理类别特征
        categorical_cols = self.test_df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.logger.info(f"类别特征: {categorical_cols}")
        
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                self.test_df[col] = self.test_df[col].astype(str)
                self.test_df[col] = self.test_df[col].replace('nan', 'Unknown').replace('None', 'Unknown')
                # 处理测试集中可能出现的新类别
                self.test_df[col] = self.test_df[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                self.test_df[col] = le.transform(self.test_df[col])
        
        # 填充缺失值（分别处理数值型和类别型）
        numeric_cols_test = self.test_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols_test:
            self.test_df[col] = self.test_df[col].fillna(self.test_df[col].median())
        
        categorical_cols_test = self.test_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols_test:
            self.test_df[col] = self.test_df[col].astype(str)
            self.test_df[col] = self.test_df[col].replace('nan', 'Unknown').replace('None', 'Unknown')
        
        # 确保特征顺序与训练时一致
        missing_features = set(self.feature_names) - set(self.test_df.columns)
        if missing_features:
            self.logger.warning(f"测试集缺少特征: {missing_features}")
            for feature in missing_features:
                self.test_df[feature] = 0
        
        self.test_df = self.test_df[self.feature_names]
        
        # 标准化
        self.test_df = pd.DataFrame(
            self.scaler.transform(self.test_df),
            columns=self.feature_names
        )
        
        self.logger.info(f"预处理后测试集形状: {self.test_df.shape}")
        
        return self
    
    def predict(self):
        """进行预测"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始预测...")
        self.logger.info("=" * 80)
        
        # 获取预测概率
        self.y_proba = self.best_model.predict_proba(self.test_df)[:, 1]
        
        # 使用优化的阈值
        self.y_pred = (self.y_proba >= self.best_threshold).astype(int)
        
        self.logger.info(f"预测完成，预测样本数: {len(self.y_pred)}")
        self.logger.info(f"预测为流失的样本数: {(self.y_pred == 1).sum()}")
        self.logger.info(f"预测为未流失的样本数: {(self.y_pred == 0).sum()}")
        
        return self
    
    def evaluate(self):
        """评估预测结果"""
        if self.y_test is None:
            self.logger.warning("测试集不包含真实标签，跳过评估")
            return self
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("评估预测结果...")
        self.logger.info("=" * 80)
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        auc = roc_auc_score(self.y_test, self.y_proba)
        
        self.logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
        self.logger.info(f"精确率 (Precision): {precision:.4f}")
        self.logger.info(f"召回率 (Recall): {recall:.4f}")
        self.logger.info(f"F1分数: {f1:.4f}")
        self.logger.info(f"AUC: {auc:.4f}")
        
        # 分类报告
        self.logger.info("\n分类报告:")
        report = classification_report(self.y_test, self.y_pred, 
                                      target_names=['未流失', '流失'])
        self.logger.info("\n" + report)
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.logger.info("\n混淆矩阵:")
        self.logger.info(f"真负例 (TN): {tn}, 假正例 (FP): {fp}")
        self.logger.info(f"假负例 (FN): {fn}, 真正例 (TP): {tp}")
        
        return self
    
    def plot_results(self):
        """绘制预测结果"""
        if self.y_test is None:
            self.logger.warning("测试集不包含真实标签，跳过绘图")
            return self
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("绘制预测结果...")
        self.logger.info("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 真实值vs预测值
        comparison_df = pd.DataFrame({
            '真实值': self.y_test,
            '预测值': self.y_pred
        })
        
        comparison_counts = comparison_df.groupby(['真实值', '预测值']).size().unstack(fill_value=0)
        comparison_counts.plot(kind='bar', ax=axes[0, 0], alpha=0.8)
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('样本数')
        axes[0, 0].set_title('真实值 vs 预测值')
        axes[0, 0].legend(['预测为未流失', '预测为流失'])
        axes[0, 0].set_xticklabels(['未流失', '流失'], rotation=0)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=['未流失', '流失'],
                   yticklabels=['未流失', '流失'])
        axes[0, 1].set_xlabel('预测标签')
        axes[0, 1].set_ylabel('真实标签')
        axes[0, 1].set_title(f'混淆矩阵 (阈值={self.best_threshold:.2f})')
        
        # ROC曲线
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
        auc = roc_auc_score(self.y_test, self.y_proba)
        
        axes[1, 0].plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.4f})', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='随机猜测')
        axes[1, 0].set_xlabel('假正例率 (FPR)')
        axes[1, 0].set_ylabel('真正例率 (TPR)')
        axes[1, 0].set_title('ROC曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 性能指标对比
        metrics = ['准确率', '精确率', '召回率', 'F1分数']
        values = [
            accuracy_score(self.y_test, self.y_pred),
            precision_score(self.y_test, self.y_pred),
            recall_score(self.y_test, self.y_pred),
            f1_score(self.y_test, self.y_pred)
        ]
        
        bars = axes[1, 1].bar(metrics, values, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].set_title('性能指标')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.4f}',
                          ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.data_dir, 'prediction_results_v3.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"已保存预测结果图: {plot_path}")
        
        plt.close()
        
        return self
    
    def save_predictions(self):
        """保存预测结果"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("保存预测结果...")
        self.logger.info("=" * 80)
        
        results_df = pd.DataFrame({
            '样本索引': range(len(self.y_pred)),
            '预测标签': self.y_pred,
            '流失概率': self.y_proba
        })
        
        if self.y_test is not None:
            results_df['真实标签'] = self.y_test.values
            results_df['预测正确'] = (self.y_pred == self.y_test.values)
        
        output_path = os.path.join(self.data_dir, 'predictions_v3.csv')
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"已保存预测结果: {output_path}")
        
        return self
    
    def run(self):
        """运行完整的预测流程"""
        try:
            self.load_model()
            self.load_test_data()
            self.preprocess_test_data()
            self.predict()
            self.evaluate()
            self.plot_results()
            self.save_predictions()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("预测完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"预测过程中出现错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    predictor = HRAttritionPredictorV2()
    predictor.run()
