# backend/markov_model.py
"""
马尔可夫链数学模型：基于历史状态转移预测号码出现概率
"""
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import pickle
import os

class MarkovChainModel:
    """
    马尔可夫链模型用于号码预测
    支持多阶马尔可夫链和不同的状态表示方法
    """
    
    def __init__(self, order: int = 2, state_type: str = "number"):
        """
        初始化马尔可夫链模型
        
        Args:
            order: 马尔可夫链的阶数（考虑前N期的影响）
            state_type: 状态类型 - "number"(单号码), "pair"(号码对), "block"(区块), "pattern"(模式)
        """
        self.order = order
        self.state_type = state_type
        self.transition_matrix = {}  # 状态转移矩阵
        self.state_frequencies = {}  # 状态频率
        self.trained = False
        
    def _extract_states(self, df: pd.DataFrame, is_front: bool = True) -> List[List]:
        """
        从历史数据中提取状态序列
        
        Args:
            df: 历史开奖数据
            is_front: 是否为前区数据
            
        Returns:
            状态序列列表
        """
        cols = ["f1", "f2", "f3", "f4", "f5"] if is_front else ["b1", "b2"]
        states = []
        
        for _, row in df.iterrows():
            numbers = sorted([row[col] for col in cols])
            
            if self.state_type == "number":
                # 单号码状态：每个号码作为一个状态
                states.append(numbers)
            elif self.state_type == "pair":
                # 号码对状态：相邻号码对作为状态
                pairs = []
                for i in range(len(numbers) - 1):
                    pairs.append((numbers[i], numbers[i + 1]))
                states.append(pairs)
            elif self.state_type == "block":
                # 区块状态：号码所属区块作为状态
                if is_front:
                    blocks = self._numbers_to_blocks(numbers, front=True)
                else:
                    blocks = self._numbers_to_blocks(numbers, front=False)
                states.append(blocks)
            elif self.state_type == "pattern":
                # 模式状态：号码分布模式作为状态
                pattern = self._numbers_to_pattern(numbers, is_front)
                states.append([pattern])
                
        return states
    
    def _numbers_to_blocks(self, numbers: List[int], front: bool = True) -> List[str]:
        """将号码转换为区块标识"""
        if front:
            # 前区区块：1-5, 6-10, 11-15, 16-20, 21-25, 26-30, 31-35
            bins = [(1,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35)]
            labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35"]
        else:
            # 后区区块：1-2, 3-4, 5-6, 7-8, 9-10, 11-12
            bins = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]
            labels = ["1-2", "3-4", "5-6", "7-8", "9-10", "11-12"]
        
        blocks = []
        for num in numbers:
            for i, (lo, hi) in enumerate(bins):
                if lo <= num <= hi:
                    blocks.append(labels[i])
                    break
        return blocks
    
    def _numbers_to_pattern(self, numbers: List[int], is_front: bool = True) -> str:
        """将号码转换为分布模式"""
        if not numbers:
            return "empty"
            
        # 计算模式特征
        span = max(numbers) - min(numbers)
        gaps = [numbers[i] - numbers[i-1] for i in range(1, len(numbers))]
        avg_gap = np.mean(gaps) if gaps else 0
        
        # 奇偶分布
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        
        # 连号情况
        consecutive_count = sum(1 for gap in gaps if gap == 1)
        
        # 生成模式字符串
        pattern = f"span_{span//5}_gap_{int(avg_gap)}_odd_{odd_count}_cons_{consecutive_count}"
        return pattern
    
    def train(self, df: pd.DataFrame, front_range: range = range(1, 36), 
              back_range: range = range(1, 13)):
        """
        训练马尔可夫链模型
        
        Args:
            df: 历史开奖数据，按时间排序
            front_range: 前区号码范围
            back_range: 后区号码范围
        """
        print(f"训练马尔可夫链模型 (order={self.order}, state_type={self.state_type})")
        
        # 分别处理前区和后区
        self.front_model = self._train_single_area(df, is_front=True)
        self.back_model = self._train_single_area(df, is_front=False)
        
        self.trained = True
        print("马尔可夫链模型训练完成")
    
    def _train_single_area(self, df: pd.DataFrame, is_front: bool = True) -> Dict:
        """训练单个区域（前区或后区）的马尔可夫链"""
        # 提取状态序列
        states_sequence = self._extract_states(df, is_front)
        
        # 构建转移矩阵
        transition_counts = defaultdict(lambda: defaultdict(int))
        state_counts = defaultdict(int)
        
        # 统计状态转移
        for i in range(self.order, len(states_sequence)):
            # 当前状态：前order期的状态组合
            current_states = []
            for j in range(i - self.order, i):
                current_states.extend(states_sequence[j])
            current_state = tuple(sorted(current_states))
            
            # 下一状态：当前期的状态
            next_states = tuple(sorted(states_sequence[i]))
            
            # 更新计数
            transition_counts[current_state][next_states] += 1
            state_counts[current_state] += 1
        
        # 计算转移概率
        transition_probs = {}
        for from_state, to_states in transition_counts.items():
            total_count = state_counts[from_state]
            transition_probs[from_state] = {
                to_state: count / total_count 
                for to_state, count in to_states.items()
            }
        
        return {
            'transition_probs': transition_probs,
            'state_counts': dict(state_counts),
            'total_sequences': len(states_sequence)
        }
    
    def predict_probabilities(self, recent_history: pd.DataFrame, 
                            front_range: range = range(1, 36),
                            back_range: range = range(1, 13)) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        基于最近历史预测号码出现概率
        
        Args:
            recent_history: 最近的历史数据（至少order期）
            front_range: 前区号码范围
            back_range: 后区号码范围
            
        Returns:
            (前区号码概率字典, 后区号码概率字典)
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        if len(recent_history) < self.order:
            # 历史数据不足，返回均匀分布
            front_probs = {n: 1.0 / len(front_range) for n in front_range}
            back_probs = {n: 1.0 / len(back_range) for n in back_range}
            return front_probs, back_probs
        
        # 预测前区
        front_probs = self._predict_single_area(recent_history, is_front=True, number_range=front_range)
        
        # 预测后区
        back_probs = self._predict_single_area(recent_history, is_front=False, number_range=back_range)
        
        return front_probs, back_probs
    
    def _predict_single_area(self, recent_history: pd.DataFrame, is_front: bool, 
                           number_range: range) -> Dict[int, float]:
        """预测单个区域的号码概率"""
        model = self.front_model if is_front else self.back_model
        
        # 提取最近的状态
        recent_states = self._extract_states(recent_history.tail(self.order), is_front)
        
        # 构建当前状态
        current_states = []
        for states in recent_states:
            current_states.extend(states)
        current_state = tuple(sorted(current_states))
        
        # 查找转移概率
        if current_state in model['transition_probs']:
            next_state_probs = model['transition_probs'][current_state]
        else:
            # 当前状态未见过，使用平滑处理
            next_state_probs = self._smooth_unseen_state(model, current_state)
        
        # 将状态概率转换为号码概率
        number_probs = self._state_probs_to_number_probs(next_state_probs, number_range, is_front)
        
        return number_probs
    
    def _smooth_unseen_state(self, model: Dict, unseen_state: Tuple) -> Dict:
        """对未见过的状态进行平滑处理"""
        # 使用拉普拉斯平滑
        all_next_states = set()
        for transitions in model['transition_probs'].values():
            all_next_states.update(transitions.keys())
        
        smoothed_probs = {}
        alpha = 0.01  # 平滑参数
        uniform_prob = alpha / len(all_next_states) if all_next_states else 0.01
        
        for next_state in all_next_states:
            smoothed_probs[next_state] = uniform_prob
            
        return smoothed_probs
    
    def _state_probs_to_number_probs(self, state_probs: Dict, number_range: range, 
                                   is_front: bool) -> Dict[int, float]:
        """将状态概率转换为号码概率"""
        number_probs = {n: 0.0 for n in number_range}
        
        for state, prob in state_probs.items():
            if self.state_type == "number":
                # 直接映射
                for num in state:
                    if num in number_range:
                        number_probs[num] += prob / len(state)
            elif self.state_type == "block":
                # 从区块映射到号码
                for block in state:
                    numbers_in_block = self._block_to_numbers(block, is_front)
                    for num in numbers_in_block:
                        if num in number_range:
                            number_probs[num] += prob / (len(state) * len(numbers_in_block))
            elif self.state_type == "pattern":
                # 模式状态需要特殊处理，这里简化为均匀分布
                uniform_prob = prob / len(number_range)
                for num in number_range:
                    number_probs[num] += uniform_prob
        
        # 归一化
        total_prob = sum(number_probs.values())
        if total_prob > 0:
            number_probs = {n: p / total_prob for n, p in number_probs.items()}
        else:
            # 回退到均匀分布
            uniform_prob = 1.0 / len(number_range)
            number_probs = {n: uniform_prob for n in number_range}
        
        return number_probs
    
    def _block_to_numbers(self, block: str, is_front: bool) -> List[int]:
        """将区块标识转换为号码列表"""
        if is_front:
            block_map = {
                "1-5": list(range(1, 6)),
                "6-10": list(range(6, 11)),
                "11-15": list(range(11, 16)),
                "16-20": list(range(16, 21)),
                "21-25": list(range(21, 26)),
                "26-30": list(range(26, 31)),
                "31-35": list(range(31, 36))
            }
        else:
            block_map = {
                "1-2": [1, 2],
                "3-4": [3, 4],
                "5-6": [5, 6],
                "7-8": [7, 8],
                "9-10": [9, 10],
                "11-12": [11, 12]
            }
        
        return block_map.get(block, [])
    
    def save_model(self, filepath: str):
        """保存训练好的模型"""
        model_data = {
            'order': self.order,
            'state_type': self.state_type,
            'front_model': self.front_model if hasattr(self, 'front_model') else None,
            'back_model': self.back_model if hasattr(self, 'back_model') else None,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载训练好的模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.order = model_data['order']
        self.state_type = model_data['state_type']
        self.front_model = model_data['front_model']
        self.back_model = model_data['back_model']
        self.trained = model_data['trained']
        print(f"模型已从 {filepath} 加载")


class BigDataAnalyzer:
    """
    大数据规律分析器：分析历史数据中的各种规律和模式
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        分析时间相关的模式
        
        Args:
            df: 历史开奖数据，包含date列
            
        Returns:
            时间模式分析结果
        """
        print("分析时间相关模式...")
        
        # 确保date列是datetime类型
        df = df.copy()
        if 'date' not in df.columns:
            raise ValueError("数据中缺少date列")
        
        df['date'] = pd.to_datetime(df['date'])
        df['weekday'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        results = {}
        
        # 1. 周几的号码分布规律
        weekday_patterns = {}
        for weekday in range(7):
            weekday_data = df[df['weekday'] == weekday]
            if len(weekday_data) > 0:
                front_nums = []
                back_nums = []
                for _, row in weekday_data.iterrows():
                    front_nums.extend([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
                    back_nums.extend([row['b1'], row['b2']])
                
                weekday_patterns[weekday] = {
                    'front_freq': Counter(front_nums),
                    'back_freq': Counter(back_nums),
                    'count': len(weekday_data)
                }
        
        results['weekday_patterns'] = weekday_patterns
        
        # 2. 月份的号码分布规律
        month_patterns = {}
        for month in range(1, 13):
            month_data = df[df['month'] == month]
            if len(month_data) > 0:
                front_nums = []
                back_nums = []
                for _, row in month_data.iterrows():
                    front_nums.extend([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
                    back_nums.extend([row['b1'], row['b2']])
                
                month_patterns[month] = {
                    'front_freq': Counter(front_nums),
                    'back_freq': Counter(back_nums),
                    'count': len(month_data)
                }
        
        results['month_patterns'] = month_patterns
        
        # 3. 季节性规律
        season_patterns = {}
        season_map = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
                     6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 
                     11: 'autumn', 12: 'winter'}
        
        df['season'] = df['month'].map(season_map)
        
        for season in ['spring', 'summer', 'autumn', 'winter']:
            season_data = df[df['season'] == season]
            if len(season_data) > 0:
                front_nums = []
                back_nums = []
                for _, row in season_data.iterrows():
                    front_nums.extend([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
                    back_nums.extend([row['b1'], row['b2']])
                
                season_patterns[season] = {
                    'front_freq': Counter(front_nums),
                    'back_freq': Counter(back_nums),
                    'count': len(season_data)
                }
        
        results['season_patterns'] = season_patterns
        
        return results
    
    def analyze_number_correlations(self, df: pd.DataFrame) -> Dict:
        """
        分析号码之间的关联性
        
        Args:
            df: 历史开奖数据
            
        Returns:
            号码关联性分析结果
        """
        print("分析号码关联性...")
        
        results = {}
        
        # 1. 前区号码共现矩阵
        front_cooccurrence = np.zeros((35, 35))  # 1-35号码
        
        for _, row in df.iterrows():
            front_nums = [row['f1'], row['f2'], row['f3'], row['f4'], row['f5']]
            for i, num1 in enumerate(front_nums):
                for j, num2 in enumerate(front_nums):
                    if i != j:  # 不同位置的号码
                        front_cooccurrence[num1-1][num2-1] += 1
        
        results['front_cooccurrence'] = front_cooccurrence
        
        # 2. 后区号码共现矩阵
        back_cooccurrence = np.zeros((12, 12))  # 1-12号码
        
        for _, row in df.iterrows():
            back_nums = [row['b1'], row['b2']]
            for i, num1 in enumerate(back_nums):
                for j, num2 in enumerate(back_nums):
                    if i != j:
                        back_cooccurrence[num1-1][num2-1] += 1
        
        results['back_cooccurrence'] = back_cooccurrence
        
        # 3. 号码间距分析
        gap_patterns = defaultdict(int)
        
        for _, row in df.iterrows():
            front_nums = sorted([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
            gaps = [front_nums[i] - front_nums[i-1] for i in range(1, len(front_nums))]
            
            for gap in gaps:
                gap_patterns[gap] += 1
        
        results['gap_patterns'] = dict(gap_patterns)
        
        # 4. 和值分布分析
        sum_patterns = []
        
        for _, row in df.iterrows():
            front_sum = sum([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
            back_sum = sum([row['b1'], row['b2']])
            sum_patterns.append({
                'front_sum': front_sum,
                'back_sum': back_sum,
                'total_sum': front_sum + back_sum
            })
        
        results['sum_patterns'] = sum_patterns
        
        return results
    
    def analyze_cyclical_patterns(self, df: pd.DataFrame, cycle_length: int = 10) -> Dict:
        """
        分析周期性规律
        
        Args:
            df: 历史开奖数据
            cycle_length: 周期长度
            
        Returns:
            周期性分析结果
        """
        print(f"分析周期性规律 (周期长度={cycle_length})...")
        
        results = {}
        
        # 1. 号码出现的周期性
        front_cycles = {i: [] for i in range(1, 36)}
        back_cycles = {i: [] for i in range(1, 13)}
        
        for idx, row in df.iterrows():
            cycle_pos = idx % cycle_length
            
            front_nums = [row['f1'], row['f2'], row['f3'], row['f4'], row['f5']]
            back_nums = [row['b1'], row['b2']]
            
            for num in front_nums:
                front_cycles[num].append(cycle_pos)
            
            for num in back_nums:
                back_cycles[num].append(cycle_pos)
        
        # 计算每个号码在周期中的分布
        front_cycle_dist = {}
        for num, positions in front_cycles.items():
            if positions:
                cycle_counter = Counter(positions)
                total = len(positions)
                front_cycle_dist[num] = {pos: count/total for pos, count in cycle_counter.items()}
        
        back_cycle_dist = {}
        for num, positions in back_cycles.items():
            if positions:
                cycle_counter = Counter(positions)
                total = len(positions)
                back_cycle_dist[num] = {pos: count/total for pos, count in cycle_counter.items()}
        
        results['front_cycle_distribution'] = front_cycle_dist
        results['back_cycle_distribution'] = back_cycle_dist
        results['cycle_length'] = cycle_length
        
        return results
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """
        生成综合洞察报告
        
        Args:
            df: 历史开奖数据
            
        Returns:
            综合分析洞察
        """
        print("生成综合洞察报告...")
        
        insights = {}
        
        # 运行所有分析
        temporal_results = self.analyze_temporal_patterns(df)
        correlation_results = self.analyze_number_correlations(df)
        cyclical_results = self.analyze_cyclical_patterns(df)
        
        # 1. 热门号码识别
        front_nums_all = []
        back_nums_all = []
        
        for _, row in df.iterrows():
            front_nums_all.extend([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
            back_nums_all.extend([row['b1'], row['b2']])
        
        front_freq = Counter(front_nums_all)
        back_freq = Counter(back_nums_all)
        
        # 识别热门和冷门号码
        front_hot = [num for num, count in front_freq.most_common(10)]
        front_cold = [num for num, count in front_freq.most_common()[-10:]]
        
        back_hot = [num for num, count in back_freq.most_common(5)]
        back_cold = [num for num, count in back_freq.most_common()[-5:]]
        
        insights['hot_numbers'] = {
            'front': front_hot,
            'back': back_hot
        }
        
        insights['cold_numbers'] = {
            'front': front_cold,
            'back': back_cold
        }
        
        # 2. 最佳号码组合
        # 基于共现矩阵找出最常一起出现的号码对
        front_cooccurrence = correlation_results['front_cooccurrence']
        best_front_pairs = []
        
        for i in range(35):
            for j in range(i+1, 35):
                count = front_cooccurrence[i][j] + front_cooccurrence[j][i]
                if count > 0:
                    best_front_pairs.append(((i+1, j+1), count))
        
        best_front_pairs.sort(key=lambda x: x[1], reverse=True)
        insights['best_front_pairs'] = best_front_pairs[:20]
        
        # 3. 时间相关的推荐
        current_date = datetime.now()
        current_weekday = current_date.weekday()
        current_month = current_date.month
        
        weekday_patterns = temporal_results.get('weekday_patterns', {})
        month_patterns = temporal_results.get('month_patterns', {})
        
        if current_weekday in weekday_patterns:
            weekday_front_freq = weekday_patterns[current_weekday]['front_freq']
            weekday_back_freq = weekday_patterns[current_weekday]['back_freq']
            
            insights['weekday_recommendations'] = {
                'front': [num for num, count in weekday_front_freq.most_common(10)],
                'back': [num for num, count in weekday_back_freq.most_common(5)]
            }
        
        if current_month in month_patterns:
            month_front_freq = month_patterns[current_month]['front_freq']
            month_back_freq = month_patterns[current_month]['back_freq']
            
            insights['month_recommendations'] = {
                'front': [num for num, count in month_front_freq.most_common(10)],
                'back': [num for num, count in month_back_freq.most_common(5)]
            }
        
        # 保存分析结果
        self.analysis_results = {
            'temporal': temporal_results,
            'correlation': correlation_results,
            'cyclical': cyclical_results,
            'insights': insights
        }
        
        return insights


def create_markov_models(df: pd.DataFrame) -> Dict[str, MarkovChainModel]:
    """
    创建多个不同配置的马尔可夫链模型
    
    Args:
        df: 历史开奖数据
        
    Returns:
        模型字典
    """
    models = {}
    
    # 不同配置的模型
    configs = [
        {'order': 1, 'state_type': 'number', 'name': '一阶号码模型'},
        {'order': 2, 'state_type': 'number', 'name': '二阶号码模型'},
        {'order': 1, 'state_type': 'block', 'name': '一阶区块模型'},
        {'order': 2, 'state_type': 'block', 'name': '二阶区块模型'},
        {'order': 1, 'state_type': 'pattern', 'name': '一阶模式模型'},
    ]
    
    for config in configs:
        print(f"创建并训练 {config['name']}...")
        model = MarkovChainModel(order=config['order'], state_type=config['state_type'])
        model.train(df)
        models[config['name']] = model
    
    return models