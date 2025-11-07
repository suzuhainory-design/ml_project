"""
模型预测脚本
加载训练好的模型对测试集进行预测，并生成评估报告
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

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.logUtil import LogUtil
from util.commonUtil import CommonUtil

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


class HRAttritionPredictor:
    """HR员工流失预测器"""
    
    def __init__(self, log_dir='../log', data_dir='../data', model_dir='../model'):
        """初始化预测器"""
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # 创建日志记录器
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=log_dir, log_name=f'predict_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("HR员工流失预测开始")
        self.logger.info("=" * 80)
        
        # 数据存储
        self.test_df = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
        # 模型和预处理器
        self.model = None
        self.model_name = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
    
    def load_model(self):
        """加载训练好的模型"""
        self.logger.info("加载模型和预处理器...")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.model_dir, 'best_model.pkl')
        best_model_info = CommonUtil.load_model(best_model_path)
        self.model = best_model_info['model']
        self.model_name = best_model_info['name']
        self.logger.info(f"已加载最佳模型: {self.model_name}")
        
        # 加载标准化器
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        self.scaler = CommonUtil.load_model(scaler_path)
        self.logger.info("已加载标准化器")
        
        # 加载标签编码器
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        self.label_encoders = CommonUtil.load_model(encoders_path)
        self.logger.info("已加载标签编码器")
        
        # 加载特征名称
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
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
    
    def preprocess_test_data(self):
        """预处理测试数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("预处理测试数据...")
        self.logger.info("=" * 80)
        
        # 保存真实标签（如果存在）
        if 'Attrition' in self.test_df.columns:
            # 先转换为数值
            self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            # 处理可能的字符串值
            if self.y_test.isna().any():
                self.test_df['Attrition'] = self.test_df['Attrition'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
                self.y_test = pd.to_numeric(self.test_df['Attrition'], errors='coerce')
            # 填充缺失值并转换为整数
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
        
        # 转换数值列
        numeric_cols = ['Age', 'DistanceFromHome', 'Education', 
                       'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                       'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
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
        
        # 选择训练时使用的特征
        self.X_test = self.test_df[self.feature_names]
        
        # 标准化
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        self.logger.info(f"预处理后测试集形状: {self.X_test.shape}")
        
        return self
    
    def _create_features(self, df):
        """创建特征（与训练时相同）"""
        df = df.copy()
        
        # 转换数值列
        numeric_cols = ['Age', 'DistanceFromHome', 'Education', 
                       'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                       'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
                       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. 工作经验相关特征
        if 'TotalWorkingYears' in df.columns and 'YearsAtCompany' in df.columns:
            df['ExperienceBeforeCompany'] = df['TotalWorkingYears'] - df['YearsAtCompany']
            df['CompanyTenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
        
        # 2. 职业发展特征
        if 'YearsAtCompany' in df.columns and 'YearsSinceLastPromotion' in df.columns:
            df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
            df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
        
        # 3. 工作稳定性特征
        if 'YearsAtCompany' in df.columns and 'YearsInCurrentRole' in df.columns:
            df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
        
        if 'YearsAtCompany' in df.columns and 'YearsWithCurrManager' in df.columns:
            df['ManagerStability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
        
        # 4. 薪资相关特征
        if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
            df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
        
        if 'MonthlyIncome' in df.columns and 'TotalWorkingYears' in df.columns:
            df['IncomePerExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
        
        if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
            df['IncomePerJobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
        
        # 5. 工作满意度综合指标
        satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                           'RelationshipSatisfaction', 'WorkLifeBalance']
        available_satisfaction = [col for col in satisfaction_cols if col in df.columns]
        if available_satisfaction:
            df['OverallSatisfaction'] = df[available_satisfaction].mean(axis=1)
            df['SatisfactionStd'] = df[available_satisfaction].std(axis=1)
        
        # 6. 年龄分组
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                                   labels=['Young', 'Middle', 'Senior', 'Veteran'])
        
        # 7. 收入分组
        if 'MonthlyIncome' in df.columns:
            df['IncomeGroup'] = pd.qcut(df['MonthlyIncome'], q=4, 
                                       labels=['Low', 'Medium', 'High', 'VeryHigh'],
                                       duplicates='drop')
        
        # 8. 工作年限分组
        if 'TotalWorkingYears' in df.columns:
            df['ExperienceGroup'] = pd.cut(df['TotalWorkingYears'], 
                                          bins=[-1, 5, 10, 20, 100],
                                          labels=['Entry', 'Mid', 'Senior', 'Expert'])
        
        # 9. 跳槽频率
        if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
            df['JobHoppingRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
        
        # 10. 工作投入与满意度交互
        if 'JobInvolvement' in df.columns and 'JobSatisfaction' in df.columns:
            df['InvolvementSatisfactionProduct'] = df['JobInvolvement'] * df['JobSatisfaction']
        
        return df
    
    def predict(self):
        """进行预测"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始预测...")
        self.logger.info("=" * 80)
        
        # 处理Ultra Ensemble模型
        if self.model_name == 'ultra_ensemble':
            base_models = self.model['base_models']
            meta_learner = self.model['meta_learner']
            
            # 生成元特征
            meta_features = np.column_stack([
                m.predict_proba(self.X_test)[:, 1] for m in base_models.values()
            ])
            
            # 使用元学习器预测
            self.y_pred = meta_learner.predict(meta_features)
            self.y_pred_proba = meta_learner.predict_proba(meta_features)[:, 1]
        else:
            # 标准预测
            self.y_pred = self.model.predict(self.X_test)
            if hasattr(self.model, 'predict_proba'):
                self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            else:
                self.y_pred_proba = self.y_pred
        
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
        
        # 计算各项指标
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)
        
        self.logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
        self.logger.info(f"精确率 (Precision): {precision:.4f}")
        self.logger.info(f"召回率 (Recall): {recall:.4f}")
        self.logger.info(f"F1分数: {f1:.4f}")
        
        # 如果有概率预测，计算AUC
        if self.y_pred_proba is not None and len(np.unique(self.y_pred_proba)) > 2:
            try:
                auc = roc_auc_score(self.y_test, self.y_pred_proba)
                self.logger.info(f"AUC: {auc:.4f}")
            except:
                self.logger.warning("无法计算AUC")
        
        # 分类报告
        self.logger.info("\n分类报告:")
        report = classification_report(self.y_test, self.y_pred, 
                                      target_names=['未流失', '流失'])
        self.logger.info("\n" + report)
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_pred)
        self.logger.info("\n混淆矩阵:")
        self.logger.info(f"真负例 (TN): {cm[0, 0]}, 假正例 (FP): {cm[0, 1]}")
        self.logger.info(f"假负例 (FN): {cm[1, 0]}, 真正例 (TP): {cm[1, 1]}")
        
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
        
        # 1. 真实值 vs 预测值对比
        comparison_df = pd.DataFrame({
            '样本索引': range(len(self.y_test)),
            '真实值': self.y_test.values,
            '预测值': self.y_pred
        })
        
        # 只显示前50个样本
        sample_size = min(50, len(comparison_df))
        comparison_sample = comparison_df.head(sample_size)
        
        x = comparison_sample['样本索引']
        axes[0, 0].scatter(x, comparison_sample['真实值'], alpha=0.6, label='真实值', s=50)
        axes[0, 0].scatter(x, comparison_sample['预测值'], alpha=0.6, label='预测值', s=50, marker='x')
        axes[0, 0].set_xlabel('样本索引')
        axes[0, 0].set_ylabel('标签 (0=未流失, 1=流失)')
        axes[0, 0].set_title(f'真实值 vs 预测值 (前{sample_size}个样本)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 混淆矩阵热图
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=['未流失', '流失'],
                   yticklabels=['未流失', '流失'])
        axes[0, 1].set_xlabel('预测标签')
        axes[0, 1].set_ylabel('真实标签')
        axes[0, 1].set_title('混淆矩阵')
        
        # 3. 预测分布对比
        true_counts = pd.Series(self.y_test).value_counts().sort_index()
        pred_counts = pd.Series(self.y_pred).value_counts().sort_index()
        
        x_pos = np.arange(2)
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, true_counts.values, width, 
                      label='真实分布', alpha=0.8, color='green')
        axes[1, 0].bar(x_pos + width/2, pred_counts.values, width, 
                      label='预测分布', alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('类别')
        axes[1, 0].set_ylabel('样本数量')
        axes[1, 0].set_title('真实分布 vs 预测分布')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(['未流失', '流失'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, (true_v, pred_v) in enumerate(zip(true_counts.values, pred_counts.values)):
            axes[1, 0].text(i - width/2, true_v, str(true_v), ha='center', va='bottom')
            axes[1, 0].text(i + width/2, pred_v, str(pred_v), ha='center', va='bottom')
        
        # 4. ROC曲线（如果有概率预测）
        if self.y_pred_proba is not None and len(np.unique(self.y_pred_proba)) > 2:
            try:
                fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
                auc = roc_auc_score(self.y_test, self.y_pred_proba)
                
                axes[1, 1].plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.4f})', linewidth=2)
                axes[1, 1].plot([0, 1], [0, 1], 'k--', label='随机猜测', linewidth=1)
                axes[1, 1].set_xlabel('假正例率 (FPR)')
                axes[1, 1].set_ylabel('真正例率 (TPR)')
                axes[1, 1].set_title('ROC曲线')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'无法绘制ROC曲线\n{str(e)}', 
                              ha='center', va='center', fontsize=10)
                axes[1, 1].set_title('ROC曲线')
        else:
            # 绘制性能指标柱状图
            metrics = {
                '准确率': accuracy_score(self.y_test, self.y_pred),
                '精确率': precision_score(self.y_test, self.y_pred, zero_division=0),
                '召回率': recall_score(self.y_test, self.y_pred, zero_division=0),
                'F1分数': f1_score(self.y_test, self.y_pred, zero_division=0)
            }
            
            axes[1, 1].bar(metrics.keys(), metrics.values(), alpha=0.8, color='skyblue')
            axes[1, 1].set_ylabel('分数')
            axes[1, 1].set_title('性能指标')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3)
            
            for i, (k, v) in enumerate(metrics.items()):
                axes[1, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图形
        plot_path = os.path.join(self.data_dir, 'prediction_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"已保存预测结果图: {plot_path}")
        
        plt.close()
        
        return self
    
    def save_predictions(self):
        """保存预测结果"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("保存预测结果...")
        self.logger.info("=" * 80)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            '样本索引': range(len(self.y_pred)),
            '预测标签': self.y_pred,
            '预测标签_文本': ['流失' if p == 1 else '未流失' for p in self.y_pred]
        })
        
        if self.y_pred_proba is not None:
            results_df['预测概率'] = self.y_pred_proba
        
        if self.y_test is not None:
            results_df['真实标签'] = self.y_test.values
            results_df['真实标签_文本'] = ['流失' if t == 1 else '未流失' for t in self.y_test.values]
            results_df['预测正确'] = (self.y_pred == self.y_test.values)
        
        # 保存到CSV
        results_path = os.path.join(self.data_dir, 'predictions.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"已保存预测结果: {results_path}")
        
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
    predictor = HRAttritionPredictor()
    predictor.run()
