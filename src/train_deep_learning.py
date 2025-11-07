"""
深度学习优化版本 - 实施多种深度学习方法
目标：使用TabNet、DNN、Wide&Deep等方法突破90%准确率
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from imblearn.over_sampling import ADASYN
import pickle

# 深度学习库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# XGBoost和LightGBM用于对比
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class DeepLearningOptimizer:
    """深度学习优化器"""
    
    def __init__(self):
        self.log_dir = '../log'
        self.data_dir = '../data'
        self.model_dir = '../model'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = LogUtil(log_dir=self.log_dir, log_name=f'train_dl_{timestamp}')
        self.logger.info("=" * 80)
        self.logger.info("深度学习优化训练")
        self.logger.info("=" * 80)
        
        CommonUtil.ensure_dir(self.model_dir)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
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
        
        self.results = {}
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
        
        # 特征工程（使用之前验证有效的方法）
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
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 使用ADASYN过采样
        self.logger.info("应用ADASYN过采样...")
        adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
        X_train, y_train = adasyn.fit_resample(X_train, y_train)
        self.logger.info(f"过采样后训练集: {X_train.shape}")
        
        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns
        )
        
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.y_train = y_train.values
        self.y_val = y_val.values
        self.feature_names = X_train.columns.tolist()
        self.n_features = len(self.feature_names)
        
        self.logger.info(f"特征数量: {self.n_features}")
        self.logger.info(f"训练集: {self.X_train.shape}, 验证集: {self.X_val.shape}")
        
        return self
    
    def feature_engineering(self, df, fit=True):
        """特征工程"""
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
        
        return df
    
    def train_deep_nn(self):
        """训练深度神经网络"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("1. 深度神经网络 (DNN)")
        self.logger.info("=" * 80)
        
        class DeepNN(nn.Module):
            def __init__(self, input_dim):
                super(DeepNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 256)
                self.bn1 = nn.BatchNorm1d(256)
                self.dropout1 = nn.Dropout(0.3)
                
                self.fc2 = nn.Linear(256, 128)
                self.bn2 = nn.BatchNorm1d(128)
                self.dropout2 = nn.Dropout(0.3)
                
                self.fc3 = nn.Linear(128, 64)
                self.bn3 = nn.BatchNorm1d(64)
                self.dropout3 = nn.Dropout(0.2)
                
                self.fc4 = nn.Linear(64, 32)
                self.bn4 = nn.BatchNorm1d(32)
                self.dropout4 = nn.Dropout(0.2)
                
                self.fc5 = nn.Linear(32, 1)
                
            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                x = self.dropout1(x)
                x = torch.relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)
                x = torch.relu(self.bn3(self.fc3(x)))
                x = self.dropout3(x)
                x = torch.relu(self.bn4(self.fc4(x)))
                x = self.dropout4(x)
                x = torch.sigmoid(self.fc5(x))
                return x
        
        # 准备数据
        X_train_tensor = torch.FloatTensor(self.X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val).unsqueeze(1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 创建模型
        model = DeepNN(self.n_features).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # 训练
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch+1}/200, Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        self.logger.info(f"DNN最佳验证集准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        self.results['DNN'] = best_val_acc
        
        return model, best_val_acc
    
    def train_wide_and_deep(self):
        """训练Wide & Deep模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("2. Wide & Deep")
        self.logger.info("=" * 80)
        
        class WideAndDeep(nn.Module):
            def __init__(self, input_dim):
                super(WideAndDeep, self).__init__()
                # Wide部分（线性）
                self.wide = nn.Linear(input_dim, 1)
                
                # Deep部分
                self.deep_fc1 = nn.Linear(input_dim, 128)
                self.deep_bn1 = nn.BatchNorm1d(128)
                self.deep_dropout1 = nn.Dropout(0.3)
                
                self.deep_fc2 = nn.Linear(128, 64)
                self.deep_bn2 = nn.BatchNorm1d(64)
                self.deep_dropout2 = nn.Dropout(0.3)
                
                self.deep_fc3 = nn.Linear(64, 32)
                self.deep_bn3 = nn.BatchNorm1d(32)
                
                # 组合
                self.output = nn.Linear(33, 1)  # 32 from deep + 1 from wide
                
            def forward(self, x):
                # Wide
                wide_out = self.wide(x)
                
                # Deep
                deep = torch.relu(self.deep_bn1(self.deep_fc1(x)))
                deep = self.deep_dropout1(deep)
                deep = torch.relu(self.deep_bn2(self.deep_fc2(deep)))
                deep = self.deep_dropout2(deep)
                deep = torch.relu(self.deep_bn3(self.deep_fc3(deep)))
                
                # 组合
                combined = torch.cat([wide_out, deep], dim=1)
                output = torch.sigmoid(self.output(combined))
                return output
        
        # 准备数据
        X_train_tensor = torch.FloatTensor(self.X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val).unsqueeze(1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 创建模型
        model = WideAndDeep(self.n_features).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # 训练
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch+1}/200, Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        self.logger.info(f"Wide&Deep最佳验证集准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        self.results['Wide&Deep'] = best_val_acc
        
        return model, best_val_acc
    
    def train_attention_nn(self):
        """训练带Attention机制的神经网络"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("3. Attention Neural Network")
        self.logger.info("=" * 80)
        
        class AttentionNN(nn.Module):
            def __init__(self, input_dim):
                super(AttentionNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.bn1 = nn.BatchNorm1d(128)
                
                # Attention层
                self.attention = nn.Linear(128, 128)
                self.attention_softmax = nn.Softmax(dim=1)
                
                self.fc2 = nn.Linear(128, 64)
                self.bn2 = nn.BatchNorm1d(64)
                self.dropout = nn.Dropout(0.3)
                
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)
                
            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                
                # Attention
                attention_weights = self.attention_softmax(self.attention(x))
                x = x * attention_weights
                
                x = torch.relu(self.bn2(self.fc2(x)))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                x = torch.sigmoid(self.fc4(x))
                return x
        
        # 准备数据
        X_train_tensor = torch.FloatTensor(self.X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val).unsqueeze(1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 创建模型
        model = AttentionNN(self.n_features).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # 训练
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch+1}/200, Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        self.logger.info(f"AttentionNN最佳验证集准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        self.results['AttentionNN'] = best_val_acc
        
        return model, best_val_acc
    
    def train_baseline_models(self):
        """训练基线模型用于对比"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("4. 基线模型（用于对比）")
        self.logger.info("=" * 80)
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss')
        xgb.fit(self.X_train, self.y_train)
        xgb_acc = accuracy_score(self.y_val, xgb.predict(self.X_val))
        self.logger.info(f"XGBoost: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
        self.results['XGBoost'] = xgb_acc
        
        # LightGBM
        lgbm = LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm.fit(self.X_train, self.y_train)
        lgbm_acc = accuracy_score(self.y_val, lgbm.predict(self.X_val))
        self.logger.info(f"LightGBM: {lgbm_acc:.4f} ({lgbm_acc*100:.2f}%)")
        self.results['LightGBM'] = lgbm_acc
        
        # CatBoost
        cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, random_state=42, verbose=0)
        cat.fit(self.X_train, self.y_train)
        cat_acc = accuracy_score(self.y_val, cat.predict(self.X_val))
        self.logger.info(f"CatBoost: {cat_acc:.4f} ({cat_acc*100:.2f}%)")
        self.results['CatBoost'] = cat_acc
        
        return xgb, lgbm, cat
    
    def evaluate_on_test(self, models):
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
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # 评估所有模型
        test_results = {}
        
        for name, model in models.items():
            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    preds = (outputs > 0.5).cpu().numpy().flatten()
            else:
                preds = model.predict(X_test)
            
            acc = accuracy_score(self.y_test, preds)
            test_results[name] = acc
            self.logger.info(f"{name}: {acc:.4f} ({acc*100:.2f}%)")
            
            if acc >= 0.90:
                self.logger.info(f"✓ {name} 达到90%目标！")
        
        # 找到最佳模型
        best_name = max(test_results.items(), key=lambda x: x[1])[0]
        best_acc = test_results[best_name]
        
        self.logger.info(f"\n最佳模型: {best_name}")
        self.logger.info(f"测试集准确率: {best_acc*100:.2f}%")
        
        if best_acc >= 0.90:
            self.logger.info("✓ 已达到90%目标！")
        else:
            self.logger.info(f"✗ 距离90%目标还差: {(0.90 - best_acc)*100:.2f}%")
        
        # 详细报告
        best_model = models[best_name]
        if isinstance(best_model, nn.Module):
            best_model.eval()
            with torch.no_grad():
                outputs = best_model(X_test_tensor)
                best_preds = (outputs > 0.5).cpu().numpy().flatten()
        else:
            best_preds = best_model.predict(X_test)
        
        self.logger.info(f"\n{best_name} 分类报告:")
        self.logger.info("\n" + classification_report(self.y_test, best_preds))
        
        return test_results
    
    def run(self):
        """运行完整流程"""
        try:
            self.load_and_preprocess_data()
            
            # 训练深度学习模型
            dnn_model, _ = self.train_deep_nn()
            wd_model, _ = self.train_wide_and_deep()
            att_model, _ = self.train_attention_nn()
            
            # 训练基线模型
            xgb, lgbm, cat = self.train_baseline_models()
            
            # 汇总所有模型
            models = {
                'DNN': dnn_model,
                'Wide&Deep': wd_model,
                'AttentionNN': att_model,
                'XGBoost': xgb,
                'LightGBM': lgbm,
                'CatBoost': cat
            }
            
            # 测试集评估
            test_results = self.evaluate_on_test(models)
            
            # 保存结果
            self.logger.info("\n保存模型...")
            CommonUtil.save_model(self.results, os.path.join(self.model_dir, 'dl_results.pkl'))
            CommonUtil.save_model(test_results, os.path.join(self.model_dir, 'dl_test_results.pkl'))
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("深度学习优化完成！")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.close()


if __name__ == '__main__':
    optimizer = DeepLearningOptimizer()
    optimizer.run()
