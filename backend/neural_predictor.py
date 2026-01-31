# backend/neural_predictor.py
"""
神经网络预测器：使用深度学习技术预测彩票号码
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

class LotteryTransformer(nn.Module):
    """专门为彩票预测设计的Transformer模型"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super(LotteryTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.front_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 35),  # 35个前区号码
            nn.Sigmoid()
        )
        
        self.back_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 12),  # 12个后区号码
            nn.Sigmoid()
        )
        
        # 注意力权重可视化
        self.attention_weights = None
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Transformer编码
        transformer_out = self.transformer(x)
        
        # 使用最后一个时间步的输出
        last_hidden = transformer_out[:, -1, :]
        
        # 预测前区和后区
        front_probs = self.front_predictor(last_hidden)
        back_probs = self.back_predictor(last_hidden)
        
        return front_probs, back_probs

class LotteryLSTM(nn.Module):
    """专门为彩票预测设计的LSTM模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, 
                 dropout: float = 0.2):
        super(LotteryLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 输出层
        self.front_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 35),
            nn.Sigmoid()
        )
        
        self.back_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 12),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 自注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用最后一个时间步的输出
        last_hidden = attn_out[:, -1, :]
        
        # 预测
        front_probs = self.front_predictor(last_hidden)
        back_probs = self.back_predictor(last_hidden)
        
        return front_probs, back_probs

class LotteryGAN(nn.Module):
    """生成对抗网络用于彩票号码生成"""
    
    def __init__(self, noise_dim: int = 100, condition_dim: int = 50):
        super(LotteryGAN, self).__init__()
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        
        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 47),  # 35前区 + 12后区
            nn.Sigmoid()
        )
        
        # 判别器
        self.discriminator = nn.Sequential(
            nn.Linear(47 + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def generate(self, batch_size: int, condition: torch.Tensor):
        noise = torch.randn(batch_size, self.noise_dim)
        condition_expanded = condition.expand(batch_size, -1)
        generator_input = torch.cat([noise, condition_expanded], dim=1)
        
        generated = self.generator(generator_input)
        
        # 分离前区和后区
        front_probs = generated[:, :35]
        back_probs = generated[:, 35:]
        
        return front_probs, back_probs
    
    def discriminate(self, numbers: torch.Tensor, condition: torch.Tensor):
        condition_expanded = condition.expand(numbers.size(0), -1)
        discriminator_input = torch.cat([numbers, condition_expanded], dim=1)
        return self.discriminator(discriminator_input)

class NeuralLotteryPredictor:
    """神经网络彩票预测器"""
    
    def __init__(self, model_type: str = 'transformer', device: str = 'auto'):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch is required for neural network prediction")
        
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.feature_dim = None
        self.sequence_length = 20
        
        # 训练历史
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        logger.info(f"Neural predictor initialized with {model_type} on {self.device}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备神经网络特征"""
        df_features = df.copy()
        
        # 基础特征
        for i in range(1, 6):
            df_features[f'f{i}_norm'] = df_features[f'f{i}'] / 35.0
            df_features[f'f{i}_sin'] = np.sin(2 * np.pi * df_features[f'f{i}'] / 35)
            df_features[f'f{i}_cos'] = np.cos(2 * np.pi * df_features[f'f{i}'] / 35)
        
        for i in range(1, 3):
            df_features[f'b{i}_norm'] = df_features[f'b{i}'] / 12.0
            df_features[f'b{i}_sin'] = np.sin(2 * np.pi * df_features[f'b{i}'] / 12)
            df_features[f'b{i}_cos'] = np.cos(2 * np.pi * df_features[f'b{i}'] / 12)
        
        # 统计特征
        front_cols = [f'f{i}' for i in range(1, 6)]
        df_features['front_sum'] = df_features[front_cols].sum(axis=1)
        df_features['front_mean'] = df_features[front_cols].mean(axis=1)
        df_features['front_std'] = df_features[front_cols].std(axis=1)
        df_features['front_min'] = df_features[front_cols].min(axis=1)
        df_features['front_max'] = df_features[front_cols].max(axis=1)
        df_features['front_span'] = df_features['front_max'] - df_features['front_min']
        
        # 奇偶特征
        df_features['odd_count'] = sum(df_features[f'f{i}'] % 2 for i in range(1, 6))
        df_features['odd_ratio'] = df_features['odd_count'] / 5
        
        # 大小特征
        df_features['large_count'] = sum((df_features[f'f{i}'] > 17).astype(int) for i in range(1, 6))
        df_features['large_ratio'] = df_features['large_count'] / 5
        
        # 时间特征
        if 'date' in df_features.columns:
            df_features['date'] = pd.to_datetime(df_features['date'])
            df_features['weekday'] = df_features['date'].dt.dayofweek
            df_features['month'] = df_features['date'].dt.month
            df_features['day'] = df_features['date'].dt.day
            
            # 周期性编码
            df_features['weekday_sin'] = np.sin(2 * np.pi * df_features['weekday'] / 7)
            df_features['weekday_cos'] = np.cos(2 * np.pi * df_features['weekday'] / 7)
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # 滞后特征
        for lag in range(1, 6):
            df_features[f'front_sum_lag{lag}'] = df_features['front_sum'].shift(lag)
            df_features[f'odd_ratio_lag{lag}'] = df_features['odd_ratio'].shift(lag)
            df_features[f'large_ratio_lag{lag}'] = df_features['large_ratio'].shift(lag)
        
        # 滚动统计特征
        for window in [5, 10, 20]:
            df_features[f'front_sum_ma{window}'] = df_features['front_sum'].rolling(window).mean()
            df_features[f'front_sum_std{window}'] = df_features['front_sum'].rolling(window).std()
        
        return df_features
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """创建序列数据"""
        # 准备特征
        df_features = self.prepare_features(df)
        
        # 选择特征列
        feature_cols = [col for col in df_features.columns 
                       if col not in ['issue', 'date', 'sales', 'pool'] + 
                       [f'f{i}' for i in range(1, 6)] + [f'b{i}' for i in range(1, 3)]]
        
        # 移除包含NaN的行
        df_clean = df_features.dropna()
        
        if len(df_clean) < self.sequence_length + 1:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length + 1} rows")
        
        # 标准化特征
        if self.scaler and HAS_SKLEARN:
            features_scaled = self.scaler.fit_transform(df_clean[feature_cols])
        else:
            features_scaled = df_clean[feature_cols].values
        
        self.feature_dim = features_scaled.shape[1]
        
        # 创建序列
        X, y_front, y_back = [], [], []
        
        for i in range(len(features_scaled) - self.sequence_length):
            # 输入序列
            X.append(features_scaled[i:i + self.sequence_length])
            
            # 目标：下一期的号码（one-hot编码）
            next_row = df_clean.iloc[i + self.sequence_length]
            
            # 前区目标
            front_target = np.zeros(35)
            for j in range(1, 6):
                if 1 <= next_row[f'f{j}'] <= 35:
                    front_target[next_row[f'f{j}'] - 1] = 1
            y_front.append(front_target)
            
            # 后区目标
            back_target = np.zeros(12)
            for j in range(1, 3):
                if 1 <= next_row[f'b{j}'] <= 12:
                    back_target[next_row[f'b{j}'] - 1] = 1
            y_back.append(back_target)
        
        return np.array(X), np.array(y_front), np.array(y_back)
    
    def build_model(self, input_dim: int):
        """构建神经网络模型"""
        if self.model_type == 'transformer':
            self.model = LotteryTransformer(input_dim)
        elif self.model_type == 'lstm':
            self.model = LotteryLSTM(input_dim)
        elif self.model_type == 'gan':
            self.model = LotteryGAN()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        logger.info(f"Built {self.model_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.001, validation_split: float = 0.2):
        """训练神经网络"""
        logger.info("Preparing training data...")
        
        # 创建序列数据
        X, y_front, y_back = self.create_sequences(df)
        
        # 构建模型
        if self.model is None:
            self.build_model(self.feature_dim)
        
        # 分割训练和验证数据
        if HAS_SKLEARN:
            X_train, X_val, y_front_train, y_front_val, y_back_train, y_back_val = train_test_split(
                X, y_front, y_back, test_size=validation_split, random_state=42
            )
        else:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_front_train, y_front_val = y_front[:split_idx], y_front[split_idx:]
            y_back_train, y_back_val = y_back[:split_idx], y_back[split_idx:]
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_front_train_tensor = torch.FloatTensor(y_front_train).to(self.device)
        y_back_train_tensor = torch.FloatTensor(y_back_train).to(self.device)
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_front_val_tensor = torch.FloatTensor(y_front_val).to(self.device)
        y_back_val_tensor = torch.FloatTensor(y_back_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_front_train_tensor, y_back_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_front_acc = 0.0
            train_back_acc = 0.0
            
            for batch_X, batch_y_front, batch_y_back in train_loader:
                optimizer.zero_grad()
                
                # 前向传播
                pred_front, pred_back = self.model(batch_X)
                
                # 计算损失
                loss_front = criterion(pred_front, batch_y_front)
                loss_back = criterion(pred_back, batch_y_back)
                loss = loss_front + loss_back
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 计算准确率（top-5 for front, top-2 for back）
                train_front_acc += self._calculate_topk_accuracy(pred_front, batch_y_front, k=5)
                train_back_acc += self._calculate_topk_accuracy(pred_back, batch_y_back, k=2)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_front_acc = 0.0
            val_back_acc = 0.0
            
            with torch.no_grad():
                pred_front_val, pred_back_val = self.model(X_val_tensor)
                
                loss_front_val = criterion(pred_front_val, y_front_val_tensor)
                loss_back_val = criterion(pred_back_val, y_back_val_tensor)
                val_loss = loss_front_val.item() + loss_back_val.item()
                
                val_front_acc = self._calculate_topk_accuracy(pred_front_val, y_front_val_tensor, k=5)
                val_back_acc = self._calculate_topk_accuracy(pred_back_val, y_back_val_tensor, k=2)
            
            # 记录训练历史
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = (train_front_acc + train_back_acc) / (2 * len(train_loader))
            avg_val_acc = (val_front_acc + val_back_acc) / 2
            
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['accuracy'].append(avg_train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(avg_val_acc)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {avg_train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {avg_val_acc:.4f}")
        
        logger.info("Training completed!")
    
    def _calculate_topk_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """计算Top-K准确率"""
        batch_size = predictions.size(0)
        
        # 获取top-k预测
        _, topk_indices = torch.topk(predictions, k, dim=1)
        
        # 获取真实标签的索引
        target_indices = torch.nonzero(targets, as_tuple=False)
        
        correct = 0
        for i in range(batch_size):
            # 获取该样本的真实标签
            sample_targets = target_indices[target_indices[:, 0] == i, 1]
            # 获取该样本的top-k预测
            sample_preds = topk_indices[i]
            
            # 计算交集
            intersection = len(set(sample_targets.cpu().numpy()) & set(sample_preds.cpu().numpy()))
            if intersection >= min(len(sample_targets), k):
                correct += 1
        
        return correct / batch_size
    
    def predict(self, df: pd.DataFrame, num_predictions: int = 5) -> List[Dict[str, Any]]:
        """预测彩票号码"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Generating {num_predictions} predictions...")
        
        # 准备输入数据
        df_features = self.prepare_features(df)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['issue', 'date', 'sales', 'pool'] + 
                       [f'f{i}' for i in range(1, 6)] + [f'b{i}' for i in range(1, 3)]]
        
        df_clean = df_features.dropna()
        
        if len(df_clean) < self.sequence_length:
            raise ValueError(f"Not enough data for prediction. Need at least {self.sequence_length} rows")
        
        # 使用最近的数据作为输入
        recent_data = df_clean[feature_cols].tail(self.sequence_length)
        
        if self.scaler and HAS_SKLEARN:
            recent_scaled = self.scaler.transform(recent_data)
        else:
            recent_scaled = recent_data.values
        
        # 转换为张量
        input_tensor = torch.FloatTensor(recent_scaled).unsqueeze(0).to(self.device)
        
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_predictions):
                # 预测
                pred_front, pred_back = self.model(input_tensor)
                
                # 转换概率为号码
                front_probs = pred_front.cpu().numpy()[0]
                back_probs = pred_back.cpu().numpy()[0]
                
                # 选择前区号码（top-5）
                front_indices = np.argsort(front_probs)[-5:]
                front_numbers = [idx + 1 for idx in front_indices]
                
                # 选择后区号码（top-2）
                back_indices = np.argsort(back_probs)[-2:]
                back_numbers = [idx + 1 for idx in back_indices]
                
                # 计算置信度
                front_confidence = np.mean([front_probs[idx] for idx in front_indices])
                back_confidence = np.mean([back_probs[idx] for idx in back_indices])
                overall_confidence = (front_confidence * 0.7 + back_confidence * 0.3)
                
                prediction = {
                    'front': sorted(front_numbers),
                    'back': sorted(back_numbers),
                    'front_probabilities': {num: front_probs[num-1] for num in front_numbers},
                    'back_probabilities': {num: back_probs[num-1] for num in back_numbers},
                    'confidence': float(overall_confidence),
                    'model_type': self.model_type,
                    'generation_method': 'neural_network'
                }
                
                predictions.append(prediction)
                
                # 为下一次预测添加一些随机性
                if num_predictions > 1:
                    noise = torch.randn_like(input_tensor) * 0.01
                    input_tensor = input_tensor + noise
        
        return predictions
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'feature_dim': self.feature_dim,
            'sequence_length': self.sequence_length,
            'training_history': self.training_history
        }
        
        if self.scaler and HAS_SKLEARN:
            save_dict['scaler'] = self.scaler
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.feature_dim = checkpoint['feature_dim']
        self.sequence_length = checkpoint['sequence_length']
        self.training_history = checkpoint.get('training_history', {})
        
        # 重建模型
        self.build_model(self.feature_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'scaler' in checkpoint and HAS_SKLEARN:
            self.scaler = checkpoint['scaler']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（简化版本）"""
        if self.model is None:
            return {}
        
        # 这是一个简化的特征重要性计算
        # 在实际应用中，可以使用更复杂的方法如SHAP值
        importance = {}
        
        # 对于Transformer模型，可以分析注意力权重
        if hasattr(self.model, 'attention_weights') and self.model.attention_weights is not None:
            attention_weights = self.model.attention_weights.mean(dim=0).cpu().numpy()
            for i, weight in enumerate(attention_weights):
                importance[f'feature_{i}'] = float(weight)
        
        return importance

def create_neural_predictor(model_type: str = 'transformer', device: str = 'auto') -> NeuralLotteryPredictor:
    """创建神经网络预测器"""
    return NeuralLotteryPredictor(model_type=model_type, device=device)

def get_available_models() -> List[str]:
    """获取可用的模型类型"""
    if HAS_PYTORCH:
        return ['transformer', 'lstm', 'gan']
    else:
        return []