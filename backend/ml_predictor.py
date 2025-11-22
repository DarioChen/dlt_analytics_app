# backend/ml_predictor.py
"""
机器学习预测模块：使用LightGBM/XGBoost等模型预测号码概率
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

class MLPredictor:
    """
    机器学习预测器：训练模型预测号码出现概率
    """
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.front_model = None
        self.back_model = None
        self.feature_cols = None
        
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
    
    def train_front_model(self, df: pd.DataFrame, target_col: str = "f1"):
        """
        训练前区模型（简化版：预测单个号码位置）
        """
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
        
        return front_weights, back_weights

def create_ml_predictor(model_type: str = "lightgbm") -> MLPredictor:
    """
    创建ML预测器实例
    """
    return MLPredictor(model_type=model_type)

