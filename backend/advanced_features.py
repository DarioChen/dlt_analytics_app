# backend/advanced_features.py
"""
高级特征工程模块：提取更深层的彩票号码特征
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

class AdvancedFeatureExtractor:
    """高级特征提取器"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有高级特征
        
        Args:
            df: 历史开奖数据
            
        Returns:
            包含所有特征的DataFrame
        """
        df_features = df.copy()
        
        # 基础特征
        df_features = self._add_basic_features(df_features)
        
        # 分布特征
        df_features = self._add_distribution_features(df_features)
        
        # 间距特征
        df_features = self._add_gap_features(df_features)
        
        # 和值特征
        df_features = self._add_sum_features(df_features)
        
        # 连号特征
        df_features = self._add_consecutive_features(df_features)
        
        # 趋势特征
        df_features = self._add_trend_features(df_features)
        
        # 周期特征
        df_features = self._add_cyclical_features(df_features)
        
        return df_features
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基础特征"""
        # 奇偶比例
        for i in range(1, 6):
            df[f'f{i}_is_odd'] = df[f'f{i}'] % 2
        df['front_odd_count'] = df[['f1_is_odd', 'f2_is_odd', 'f3_is_odd', 'f4_is_odd', 'f5_is_odd']].sum(axis=1)
        df['front_odd_ratio'] = df['front_odd_count'] / 5
        
        # 大小号比例（1-17为小号，18-35为大号）
        for i in range(1, 6):
            df[f'f{i}_is_large'] = (df[f'f{i}'] > 17).astype(int)
        df['front_large_count'] = df[['f1_is_large', 'f2_is_large', 'f3_is_large', 'f4_is_large', 'f5_is_large']].sum(axis=1)
        df['front_large_ratio'] = df['front_large_count'] / 5
        
        # 后区奇偶和大小
        for i in range(1, 3):
            df[f'b{i}_is_odd'] = df[f'b{i}'] % 2
            df[f'b{i}_is_large'] = (df[f'b{i}'] > 6).astype(int)
        df['back_odd_count'] = df[['b1_is_odd', 'b2_is_odd']].sum(axis=1)
        df['back_large_count'] = df[['b1_is_large', 'b2_is_large']].sum(axis=1)
        
        return df
    
    def _add_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加分布特征"""
        # 前区分布
        front_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        # 区域分布（1-7, 8-14, 15-21, 22-28, 29-35）
        zones = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
        for i, (start, end) in enumerate(zones):
            zone_count = 0
            for col in front_cols:
                zone_count += ((df[col] >= start) & (df[col] <= end)).astype(int)
            df[f'front_zone_{i+1}_count'] = zone_count
        
        # 号码跨度
        df['front_span'] = df[front_cols].max(axis=1) - df[front_cols].min(axis=1)
        df['back_span'] = df[['b1', 'b2']].max(axis=1) - df[['b1', 'b2']].min(axis=1)
        
        # 号码方差
        df['front_variance'] = df[front_cols].var(axis=1)
        
        return df
    
    def _add_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加间距特征"""
        front_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        # 计算相邻号码间距
        for i in range(4):
            df[f'front_gap_{i+1}'] = df[f'f{i+2}'] - df[f'f{i+1}']
        
        # 间距统计
        gap_cols = [f'front_gap_{i+1}' for i in range(4)]
        df['front_gap_mean'] = df[gap_cols].mean(axis=1)
        df['front_gap_std'] = df[gap_cols].std(axis=1)
        df['front_gap_max'] = df[gap_cols].max(axis=1)
        df['front_gap_min'] = df[gap_cols].min(axis=1)
        
        # 后区间距
        df['back_gap'] = df['b2'] - df['b1']
        
        return df
    
    def _add_sum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加和值特征"""
        # 前区和值
        df['front_sum'] = df[['f1', 'f2', 'f3', 'f4', 'f5']].sum(axis=1)
        df['back_sum'] = df[['b1', 'b2']].sum(axis=1)
        df['total_sum'] = df['front_sum'] + df['back_sum']
        
        # 和值的滚动统计
        window = 10
        df['front_sum_ma'] = df['front_sum'].rolling(window=window).mean()
        df['front_sum_std'] = df['front_sum'].rolling(window=window).std()
        
        # 和值偏离度
        df['front_sum_deviation'] = (df['front_sum'] - df['front_sum_ma']) / df['front_sum_std']
        
        return df
    
    def _add_consecutive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加连号特征"""
        def count_consecutive_groups(row):
            """计算连号组数"""
            numbers = sorted([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
            groups = 0
            in_group = False
            
            for i in range(1, len(numbers)):
                if numbers[i] == numbers[i-1] + 1:
                    if not in_group:
                        groups += 1
                        in_group = True
                else:
                    in_group = False
            
            return groups
        
        def max_consecutive_length(row):
            """计算最大连号长度"""
            numbers = sorted([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
            max_length = 1
            current_length = 1
            
            for i in range(1, len(numbers)):
                if numbers[i] == numbers[i-1] + 1:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    current_length = 1
            
            return max_length
        
        df['consecutive_groups'] = df.apply(count_consecutive_groups, axis=1)
        df['max_consecutive_length'] = df.apply(max_consecutive_length, axis=1)
        
        # 后区连号
        df['back_consecutive'] = ((df['b2'] - df['b1']) == 1).astype(int)
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势特征"""
        # 号码出现频率的变化趋势
        window = 20
        
        for num in range(1, 36):
            # 计算每个号码在滑动窗口内的出现频率
            freq_col = f'freq_{num}'
            df[freq_col] = 0
            
            for i in range(len(df)):
                start_idx = max(0, i - window + 1)
                window_data = df.iloc[start_idx:i+1]
                
                # 计算该号码在窗口内的出现次数
                count = 0
                for _, row in window_data.iterrows():
                    if num in [row['f1'], row['f2'], row['f3'], row['f4'], row['f5']]:
                        count += 1
                
                df.loc[df.index[i], freq_col] = count / len(window_data)
        
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加周期特征"""
        # 确保date列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # 周几
            df['weekday'] = df['date'].dt.dayofweek
            
            # 月份
            df['month'] = df['date'].dt.month
            
            # 季度
            df['quarter'] = df['date'].dt.quarter
            
            # 年份
            df['year'] = df['date'].dt.year
            
            # 周期性编码（使用sin/cos变换）
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def get_pattern_features(self, df: pd.DataFrame, pattern_length: int = 3) -> Dict:
        """
        提取模式特征
        
        Args:
            df: 历史数据
            pattern_length: 模式长度
            
        Returns:
            模式特征字典
        """
        patterns = defaultdict(int)
        
        # 提取前区号码模式
        for i in range(len(df) - pattern_length + 1):
            pattern = []
            for j in range(pattern_length):
                row = df.iloc[i + j]
                front_nums = sorted([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
                pattern.append(tuple(front_nums))
            
            patterns[tuple(pattern)] += 1
        
        return dict(patterns)
    
    def calculate_number_relationships(self, df: pd.DataFrame) -> Dict:
        """
        计算号码间的关系
        
        Returns:
            号码关系字典
        """
        relationships = {
            'co_occurrence': defaultdict(int),  # 共现次数
            'follow_probability': defaultdict(dict),  # 跟随概率
            'gap_distribution': defaultdict(list)  # 间隔分布
        }
        
        # 计算共现和跟随关系
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            current_front = set([current_row['f1'], current_row['f2'], current_row['f3'], 
                                current_row['f4'], current_row['f5']])
            next_front = set([next_row['f1'], next_row['f2'], next_row['f3'], 
                             next_row['f4'], next_row['f5']])
            
            # 共现关系
            for num1 in current_front:
                for num2 in current_front:
                    if num1 != num2:
                        relationships['co_occurrence'][(num1, num2)] += 1
            
            # 跟随关系
            for num1 in current_front:
                for num2 in next_front:
                    if num1 not in relationships['follow_probability']:
                        relationships['follow_probability'][num1] = defaultdict(int)
                    relationships['follow_probability'][num1][num2] += 1
        
        # 转换跟随计数为概率
        for num1, followers in relationships['follow_probability'].items():
            total = sum(followers.values())
            for num2 in followers:
                followers[num2] = followers[num2] / total
        
        return relationships


class SmartFilter:
    """智能过滤器"""
    
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter_func, weight: float = 1.0):
        """添加过滤器"""
        self.filters.append((filter_func, weight))
    
    def apply_filters(self, candidates: List[Dict], historical_data: pd.DataFrame) -> List[Dict]:
        """应用所有过滤器"""
        scored_candidates = []
        
        for candidate in candidates:
            total_score = 0
            total_weight = 0
            
            for filter_func, weight in self.filters:
                score = filter_func(candidate, historical_data)
                total_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_score = total_score / total_weight
                candidate['filter_score'] = avg_score
                scored_candidates.append(candidate)
        
        # 按分数排序，返回前50%
        scored_candidates.sort(key=lambda x: x['filter_score'], reverse=True)
        return scored_candidates[:len(scored_candidates)//2]


def historical_avoidance_filter(candidate: Dict, historical_data: pd.DataFrame, 
                               avoid_periods: int = 5) -> float:
    """历史回避过滤器"""
    recent_data = historical_data.tail(avoid_periods)
    candidate_front = set(candidate['front'])
    
    # 计算与最近开奖的相似度
    similarity_scores = []
    for _, row in recent_data.iterrows():
        historical_front = set([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
        intersection = len(candidate_front & historical_front)
        similarity = intersection / 5  # 相似度
        similarity_scores.append(similarity)
    
    # 返回反向分数（相似度越低，分数越高）
    avg_similarity = np.mean(similarity_scores)
    return 1.0 - avg_similarity


def pattern_consistency_filter(candidate: Dict, historical_data: pd.DataFrame) -> float:
    """模式一致性过滤器"""
    # 检查候选号码是否符合历史模式
    front_nums = sorted(candidate['front'])
    
    # 计算特征
    odd_count = sum(1 for n in front_nums if n % 2 == 1)
    large_count = sum(1 for n in front_nums if n > 17)
    span = max(front_nums) - min(front_nums)
    
    # 与历史分布比较
    recent_data = historical_data.tail(50)
    
    # 历史奇数比例分布
    hist_odd_ratios = []
    hist_large_ratios = []
    hist_spans = []
    
    for _, row in recent_data.iterrows():
        hist_front = [row['f1'], row['f2'], row['f3'], row['f4'], row['f5']]
        hist_odd_count = sum(1 for n in hist_front if n % 2 == 1)
        hist_large_count = sum(1 for n in hist_front if n > 17)
        hist_span = max(hist_front) - min(hist_front)
        
        hist_odd_ratios.append(hist_odd_count / 5)
        hist_large_ratios.append(hist_large_count / 5)
        hist_spans.append(hist_span)
    
    # 计算候选号码特征与历史分布的匹配度
    odd_ratio = odd_count / 5
    large_ratio = large_count / 5
    
    odd_score = 1.0 - abs(odd_ratio - np.mean(hist_odd_ratios)) / np.std(hist_odd_ratios) if np.std(hist_odd_ratios) > 0 else 1.0
    large_score = 1.0 - abs(large_ratio - np.mean(hist_large_ratios)) / np.std(hist_large_ratios) if np.std(hist_large_ratios) > 0 else 1.0
    span_score = 1.0 - abs(span - np.mean(hist_spans)) / np.std(hist_spans) if np.std(hist_spans) > 0 else 1.0
    
    # 综合分数
    return (odd_score + large_score + span_score) / 3


def statistical_boundary_filter(candidate: Dict, historical_data: pd.DataFrame) -> float:
    """统计边界过滤器"""
    front_nums = candidate['front']
    front_sum = sum(front_nums)
    
    # 历史和值分布
    recent_data = historical_data.tail(100)
    hist_sums = []
    for _, row in recent_data.iterrows():
        hist_sum = sum([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
        hist_sums.append(hist_sum)
    
    # 计算和值的正常范围（均值±2标准差）
    mean_sum = np.mean(hist_sums)
    std_sum = np.std(hist_sums)
    
    lower_bound = mean_sum - 2 * std_sum
    upper_bound = mean_sum + 2 * std_sum
    
    # 如果和值在正常范围内，给高分
    if lower_bound <= front_sum <= upper_bound:
        # 越接近均值，分数越高
        distance_from_mean = abs(front_sum - mean_sum) / std_sum
        return max(0, 1.0 - distance_from_mean / 2)
    else:
        # 超出范围，给低分
        return 0.1