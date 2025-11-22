# backend/features.py
"""
特征工程模块：提取周期性、号码对联动、遗漏等特征
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

def extract_periodic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取周期性特征：周几、月份、季度等
    """
    df = df.copy()
    df['weekday'] = df['date'].dt.dayofweek  # 0=周一, 6=周日
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_month'] = df['date'].dt.day
    return df

def extract_number_pair_features(df: pd.DataFrame, front_range=range(1,36), back_range=range(1,13)) -> Dict:
    """
    计算号码对联动特征：哪些号码经常一起出现
    返回: {('f1','f2'): co_occurrence_count, ...}
    """
    front_cols = ["f1","f2","f3","f4","f5"]
    back_cols = ["b1","b2"]
    
    # 前区号码对
    front_pairs = {}
    for _, row in df.iterrows():
        front_nums = sorted([row[c] for c in front_cols])
        for i in range(len(front_nums)):
            for j in range(i+1, len(front_nums)):
                pair = tuple(sorted([front_nums[i], front_nums[j]]))
                front_pairs[pair] = front_pairs.get(pair, 0) + 1
    
    # 后区号码对
    back_pairs = {}
    for _, row in df.iterrows():
        back_nums = sorted([row[c] for c in back_cols])
        if len(back_nums) >= 2:
            pair = tuple(sorted(back_nums))
            back_pairs[pair] = back_pairs.get(pair, 0) + 1
    
    return {"front_pairs": front_pairs, "back_pairs": back_pairs}

def extract_miss_features(df: pd.DataFrame, front_range=range(1,36), back_range=range(1,13)) -> pd.DataFrame:
    """
    计算遗漏特征：每个号码距离上次出现的期数
    """
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    
    front_cols = ["f1","f2","f3","f4","f5"]
    back_cols = ["b1","b2"]
    
    # 前区遗漏
    front_miss = {n: [] for n in front_range}
    for idx, row in df.iterrows():
        front_nums = set([row[c] for c in front_cols])
        for n in front_range:
            if n in front_nums:
                front_miss[n].append(idx)
    
    # 计算当前遗漏期数
    df['front_miss_avg'] = 0.0
    df['back_miss_avg'] = 0.0
    
    for idx in range(len(df)):
        front_misses = []
        back_misses = []
        
        for n in front_range:
            last_idx = [i for i in front_miss[n] if i < idx]
            miss = idx - last_idx[-1] if last_idx else idx + 1
            front_misses.append(miss)
        
        for n in back_range:
            back_nums = set([df.loc[i, c] for i in range(idx) for c in back_cols])
            miss = idx - max([i for i in range(idx) if n in set([df.loc[i, c] for c in back_cols])], default=-1) if n in back_nums else idx + 1
            back_misses.append(miss)
        
        df.loc[idx, 'front_miss_avg'] = np.mean(front_misses) if front_misses else 0
        df.loc[idx, 'back_miss_avg'] = np.mean(back_misses) if back_misses else 0
    
    return df

def extract_sum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取和值相关特征：前区和值、后区和值、总和、跨度等
    """
    df = df.copy()
    df['sum_front'] = df[["f1","f2","f3","f4","f5"]].sum(axis=1)
    df['sum_back'] = df[["b1","b2"]].sum(axis=1)
    df['sum_all'] = df['sum_front'] + df['sum_back']
    
    # 前区跨度
    df['span_front'] = df[["f1","f2","f3","f4","f5"]].max(axis=1) - df[["f1","f2","f3","f4","f5"]].min(axis=1)
    df['span_back'] = df[["b1","b2"]].max(axis=1) - df[["b1","b2"]].min(axis=1)
    
    # 奇偶比
    df['odd_count_front'] = df[["f1","f2","f3","f4","f5"]].apply(lambda r: sum(x%2 for x in r), axis=1)
    df['odd_count_back'] = df[["b1","b2"]].apply(lambda r: sum(x%2 for x in r), axis=1)
    
    return df

def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取所有特征
    """
    df = extract_periodic_features(df)
    df = extract_sum_features(df)
    df = extract_miss_features(df)
    return df

def predict_number_probability_ml(
    df_history: pd.DataFrame,
    target_numbers: List[int],
    is_front: bool = True
) -> float:
    """
    基于历史数据，使用简单统计模型预测号码出现概率
    这是一个占位函数，后续可以替换为真正的ML模型
    """
    front_cols = ["f1","f2","f3","f4","f5"] if is_front else ["b1","b2"]
    
    # 简单频率模型
    all_nums = df_history[front_cols].values.flatten()
    total_draws = len(df_history)
    
    prob = 0.0
    for num in target_numbers:
        count = np.sum(all_nums == num)
        prob += count / (total_draws * len(front_cols))
    
    return prob / len(target_numbers) if target_numbers else 0.0

