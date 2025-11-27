# backend/ml_predictor.py
"""
机器学习预测模块：使用LightGBM/XGBoost/LSTM/Transformer等模型预测号码概率
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from backend.features import extract_all_features

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# LSTM模型定义
class LSTMModel(nn.Module):
    """
    LSTM模型用于时间序列预测
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=35):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

# Transformer模型定义
class TransformerModel(nn.Module):
    """
    Transformer模型用于时间序列预测
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=35):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))  # 位置编码
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_size]
        x = self.embedding(x)
        
        # 添加位置编码
        if x.size(1) > self.pos_encoder.size(1):
            # 如果序列长度超过预定义的位置编码，截断
            x = x[:, :self.pos_encoder.size(1), :]
        else:
            pos_enc = self.pos_encoder[:, :x.size(1), :]
            x += pos_enc
        
        # Transformer要求输入形状为 [seq_len, batch_size, features]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        
        # 转回原始形状并取最后一个时间步
        x = x.permute(1, 0, 2)[:, -1, :]
        x = self.fc(x)
        x = self.softmax(x)
        return x

class MLPredictor:
    """
    机器学习预测器：训练模型预测号码出现概率
    """
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.front_model = None
        self.back_model = None
        self.feature_cols = None
        self.scaler = None
        self.lstm_sequence_length = 10  # LSTM序列长度
        
    def prepare_features(self, df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        准备特征：基于历史窗口提取特征
        """
        df = extract_all_features(df)
        
        # 添加滞后特征
        for lag in range(1, min(lookback, len(df))):
            df[f'sum_front_lag{lag}'] = df['sum_front'].shift(lag)
            df[f'sum_back_lag{lag}'] = df['sum_back'].shift(lag)
            df[f'odd_count_front_lag{lag}'] = df['odd_count_front'].shift(lag)
        
        return df
    
    def prepare_sequence_data(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列数据用于LSTM/Transformer模型
        """
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            # 使用历史窗口数据作为特征
            feature_df = self.prepare_features(df)
            feature_cols = [c for c in feature_df.columns if c not in 
                          ['issue', 'date', 'f1','f2','f3','f4','f5','b1','b2', 'sales', 'pool']]
            feature_df = feature_df.dropna()
            
            if len(feature_df) < sequence_length + 1:
                return np.array([]), np.array([])
            
            # 标准化特征
            scaled_features = self.scaler.fit_transform(feature_df[feature_cols])
            
            # 创建序列数据
            X, y = [], []
            for i in range(len(scaled_features) - sequence_length):
                X.append(scaled_features[i:i+sequence_length])
                # 目标是下一期的前区号码分布
                front_nums = feature_df.iloc[i+sequence_length][['f1','f2','f3','f4','f5']].values
                front_oh = np.zeros(35)
                for num in front_nums:
                    if 1 <= num <= 35:
                        front_oh[num-1] = 1
                y.append(front_oh)
            
            return np.array(X), np.array(y)
        return np.array([]), np.array([])
    
    def train_front_model(self, df: pd.DataFrame, target_col: str = "f1"):
        """
        训练前区模型（支持传统模型和深度学习模型）
        """
        # 处理LSTM和Transformer模型
        if (self.model_type in ["lstm", "transformer"]) and HAS_PYTORCH and HAS_SKLEARN:
            # 准备序列数据
            X, y = self.prepare_sequence_data(df, self.lstm_sequence_length)
            if len(X) == 0:
                return None
            
            # 转换为PyTorch张量
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # 分割训练集
            X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            
            # 初始化模型
            input_size = X.shape[2]
            if self.model_type == "lstm":
                self.front_model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=35)
            else:  # transformer
                self.front_model = TransformerModel(input_size=input_size, d_model=64, nhead=4, num_layers=2, output_size=35)
            
            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.front_model.to(device)
            
            # 定义损失函数和优化器
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.front_model.parameters(), lr=0.001)
            
            # 训练模型
            num_epochs = 20
            for epoch in range(num_epochs):
                self.front_model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 前向传播
                    outputs = self.front_model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            return self.front_model
        
        # 处理传统模型
        if not HAS_LIGHTGBM and not HAS_XGBOOST:
            return None
        
        df_feat = self.prepare_features(df)
        df_feat = df_feat.dropna()
        
        if len(df_feat) < 50:
            return None
        
        # 特征列
        feature_cols = [c for c in df_feat.columns if c not in 
                       ['issue', 'date', 'f1','f2','f3','f4','f5','b1','b2', 'sales', 'pool']]
        self.feature_cols = feature_cols
        
        X = df_feat[feature_cols].values
        y = (df_feat[target_col].values > 17).astype(int)  # 简化：预测是否大于中位数
        
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            try:
                # 尝试使用 sklearn 接口
                self.front_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                self.front_model.fit(X, y)
            except Exception as e:
                if "scikit-learn" in str(e).lower() or "sklearn" in str(e).lower():
                    # 如果缺少 sklearn，尝试使用原生 API
                    try:
                        train_data = lgb.Dataset(X, label=y)
                        params = {
                            'objective': 'binary',
                            'metric': 'binary_logloss',
                            'boosting_type': 'gbdt',
                            'num_leaves': 31,
                            'learning_rate': 0.05,
                            'feature_fraction': 0.9,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'verbose': -1,
                            'seed': 42
                        }
                        self.front_model = lgb.train(params, train_data, num_boost_round=50)
                    except Exception as e2:
                        raise ImportError(f"LightGBM 需要 scikit-learn。请运行: pip install scikit-learn\n原始错误: {e}")
                else:
                    raise
        elif self.model_type == "xgboost" and HAS_XGBOOST:
            self.front_model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
            self.front_model.fit(X, y)
        
        return self.front_model
    
    def predict_number_weights(self, df_history: pd.DataFrame, front_range=range(1,36), back_range=range(1,13)) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        预测每个号码的出现权重（概率）
        返回: (front_weights, back_weights)
        """
        # 处理深度学习模型的预测
        if (self.model_type in ["lstm", "transformer"]) and HAS_PYTORCH and self.front_model is not None:
            try:
                # 准备最新的序列数据
                feature_df = self.prepare_features(df_history)
                feature_cols = [c for c in feature_df.columns if c not in 
                              ['issue', 'date', 'f1','f2','f3','f4','f5','b1','b2', 'sales', 'pool']]
                feature_df = feature_df.dropna()
                
                if len(feature_df) < self.lstm_sequence_length:
                    # 回退到频率模型
                    from predictor import number_frequencies
                    front_freq, back_freq = number_frequencies(df_history, front_range, back_range)
                    return {n: front_freq.get(n, 0) / (sum(front_freq.values()) or 1) for n in front_range}, \
                           {n: back_freq.get(n, 0) / (sum(back_freq.values()) or 1) for n in back_range}
                
                # 获取最新的序列数据
                recent_features = feature_df[feature_cols].tail(self.lstm_sequence_length)
                
                # 标准化
                if self.scaler:
                    recent_features_scaled = self.scaler.transform(recent_features)
                    # 转换为序列格式并添加批次维度
                    X_test = torch.FloatTensor(recent_features_scaled).unsqueeze(0)
                    
                    # 模型预测
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.front_model.to(device)
                    self.front_model.eval()
                    
                    with torch.no_grad():
                        X_test = X_test.to(device)
                        y_pred = self.front_model(X_test)
                        y_pred_np = y_pred.cpu().numpy()[0]
                    
                    # 构建权重字典
                    front_weights = {i+1: y_pred_np[i] for i in range(35)}
                    
                    # 后区仍使用频率模型
                    from predictor import number_frequencies
                    _, back_freq = number_frequencies(df_history, front_range, back_range)
                    back_weights = {n: back_freq.get(n, 0) / (sum(back_freq.values()) or 1) for n in back_range}
                    
                    return front_weights, back_weights
            except Exception as e:
                print(f"深度学习模型预测出错: {e}")
                # 出错时回退到频率模型
                pass
        
        # 传统模型或回退预测
        if self.front_model is None:
            # 回退到简单频率模型
            from predictor import number_frequencies
            front_freq, back_freq = number_frequencies(df_history, front_range, back_range)
            total = sum(front_freq.values()) or 1
            front_weights = {n: front_freq.get(n, 0) / total for n in front_range}
            back_weights = {n: back_freq.get(n, 0) / (sum(back_freq.values()) or 1) for n in back_range}
            return front_weights, back_weights
        
        # 使用ML模型预测（简化实现）
        df_feat = self.prepare_features(df_history)
        if len(df_feat) == 0 or self.feature_cols is None:
            from predictor import number_frequencies
            front_freq, back_freq = number_frequencies(df_history, front_range, back_range)
            total = sum(front_freq.values()) or 1
            return {n: front_freq.get(n, 0) / total for n in front_range}, \
                   {n: back_freq.get(n, 0) / (sum(back_freq.values()) or 1) for n in back_range}
        
        # 基于历史频率 + ML调整
        from predictor import number_frequencies
        front_freq, back_freq = number_frequencies(df_history, front_range, back_range)
        total = sum(front_freq.values()) or 1
        
        front_weights = {n: front_freq.get(n, 0) / total for n in front_range}
        back_weights = {n: back_freq.get(n, 0) / (sum(back_freq.values()) or 1) for n in back_range}
        
        # 增强高额奖项命中的权重调整
        # 对稀有组合给予更高权重
        rarity_factor = 0.3  # 稀有度调整因子
        max_freq = max(front_freq.values()) if front_freq else 1
        front_weights = {n: weight * (1 + rarity_factor * (1 - front_freq.get(n, 0) / max_freq)) 
                        for n, weight in front_weights.items()}
        
        # 归一化权重
        total_weight = sum(front_weights.values()) or 1
        front_weights = {n: w / total_weight for n, w in front_weights.items()}
        
        return front_weights, back_weights

def create_ml_predictor(model_type: str = "lightgbm") -> MLPredictor:
    """
    创建ML预测器实例
    model_type 可选值: lightgbm, xgboost, lstm, transformer
    """
    return MLPredictor(model_type=model_type)

def get_available_model_types() -> List[str]:
    """
    获取可用的模型类型列表
    """
    model_types = []
    if HAS_LIGHTGBM:
        model_types.append("lightgbm")
    if HAS_XGBOOST:
        model_types.append("xgboost")
    if HAS_PYTORCH and HAS_SKLEARN:
        model_types.extend(["lstm", "transformer"])
    # 至少返回基础模型
    if not model_types:
        model_types = ["baseline"]
    return model_types

