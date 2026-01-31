# backend/evolutionary_optimizer.py
"""
进化优化器：使用进化算法和深度学习技术大幅提升中奖概率
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvolutionaryConfig:
    """进化算法配置"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    diversity_threshold: float = 0.3
    adaptive_mutation: bool = True
    multi_objective: bool = True

class AdvancedPatternAnalyzer:
    """高级模式分析器：发现深层隐藏规律"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.sequence_patterns = defaultdict(list)
        self.correlation_matrix = None
        self.temporal_patterns = {}
        
    def analyze_deep_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """深度模式分析"""
        patterns = {
            'sequential_patterns': self._analyze_sequential_patterns(df),
            'correlation_patterns': self._analyze_correlation_patterns(df),
            'temporal_cycles': self._analyze_temporal_cycles(df),
            'number_relationships': self._analyze_number_relationships(df),
            'distribution_anomalies': self._detect_distribution_anomalies(df),
            'hidden_periodicities': self._find_hidden_periodicities(df)
        }
        return patterns
    
    def _analyze_sequential_patterns(self, df: pd.DataFrame) -> Dict:
        """分析序列模式"""
        patterns = {}
        
        # 分析连续期数的号码变化模式
        for window_size in [3, 5, 7, 10]:
            sequence_patterns = []
            for i in range(len(df) - window_size + 1):
                window = df.iloc[i:i+window_size]
                
                # 提取序列特征
                front_sequences = []
                for _, row in window.iterrows():
                    front_nums = sorted([row[f'f{j}'] for j in range(1, 6)])
                    front_sequences.append(front_nums)
                
                # 计算序列特征
                seq_features = {
                    'sum_trend': [sum(seq) for seq in front_sequences],
                    'span_trend': [max(seq) - min(seq) for seq in front_sequences],
                    'odd_count_trend': [sum(1 for n in seq if n % 2 == 1) for seq in front_sequences]
                }
                
                sequence_patterns.append(seq_features)
            
            patterns[f'window_{window_size}'] = sequence_patterns
        
        return patterns
    
    def _analyze_correlation_patterns(self, df: pd.DataFrame) -> Dict:
        """分析号码相关性模式"""
        # 构建号码共现矩阵
        cooccurrence_matrix = np.zeros((35, 35))
        
        for _, row in df.iterrows():
            front_nums = [row[f'f{i}'] - 1 for i in range(1, 6)]  # 转为0-34索引
            for i in range(len(front_nums)):
                for j in range(i+1, len(front_nums)):
                    cooccurrence_matrix[front_nums[i]][front_nums[j]] += 1
                    cooccurrence_matrix[front_nums[j]][front_nums[i]] += 1
        
        # 计算相关性强度
        correlation_strength = {}
        for i in range(35):
            for j in range(i+1, 35):
                strength = cooccurrence_matrix[i][j]
                if strength > 0:
                    correlation_strength[(i+1, j+1)] = strength
        
        # 找出强相关和负相关的号码对
        sorted_correlations = sorted(correlation_strength.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'cooccurrence_matrix': cooccurrence_matrix.tolist(),
            'strong_correlations': sorted_correlations[:20],
            'weak_correlations': sorted_correlations[-20:],
            'correlation_clusters': self._find_correlation_clusters(cooccurrence_matrix)
        }
    
    def _analyze_temporal_cycles(self, df: pd.DataFrame) -> Dict:
        """分析时间周期模式"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        cycles = {}
        
        # 分析不同时间周期的模式
        for cycle_type in ['weekday', 'month', 'quarter', 'year']:
            if cycle_type == 'weekday':
                df['cycle'] = df['date'].dt.dayofweek
                cycle_range = range(7)
            elif cycle_type == 'month':
                df['cycle'] = df['date'].dt.month
                cycle_range = range(1, 13)
            elif cycle_type == 'quarter':
                df['cycle'] = df['date'].dt.quarter
                cycle_range = range(1, 5)
            else:  # year
                df['cycle'] = df['date'].dt.year
                cycle_range = df['cycle'].unique()
            
            cycle_patterns = {}
            for cycle_val in cycle_range:
                cycle_data = df[df['cycle'] == cycle_val]
                if len(cycle_data) > 0:
                    # 计算该周期的号码分布特征
                    front_nums = []
                    for _, row in cycle_data.iterrows():
                        front_nums.extend([row[f'f{i}'] for i in range(1, 6)])
                    
                    cycle_patterns[cycle_val] = {
                        'frequency': pd.Series(front_nums).value_counts().to_dict(),
                        'avg_sum': np.mean([sum([row[f'f{i}'] for i in range(1, 6)]) for _, row in cycle_data.iterrows()]),
                        'avg_span': np.mean([max([row[f'f{i}'] for i in range(1, 6)]) - min([row[f'f{i}'] for i in range(1, 6)]) for _, row in cycle_data.iterrows()])
                    }
            
            cycles[cycle_type] = cycle_patterns
        
        return cycles
    
    def _analyze_number_relationships(self, df: pd.DataFrame) -> Dict:
        """分析号码间的复杂关系"""
        relationships = {
            'follow_patterns': defaultdict(lambda: defaultdict(int)),
            'gap_patterns': defaultdict(list),
            'position_preferences': defaultdict(lambda: defaultdict(int))
        }
        
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            current_front = set([current_row[f'f{j}'] for j in range(1, 6)])
            next_front = set([next_row[f'f{j}'] for j in range(1, 6)])
            
            # 分析跟随模式
            for num in current_front:
                for next_num in next_front:
                    relationships['follow_patterns'][num][next_num] += 1
            
            # 分析间隔模式
            for num in range(1, 36):
                if num in current_front:
                    # 找到该号码下次出现的间隔
                    gap = 1
                    for j in range(i + 1, min(i + 20, len(df))):  # 最多看20期
                        future_row = df.iloc[j]
                        future_front = set([future_row[f'f{k}'] for k in range(1, 6)])
                        if num in future_front:
                            relationships['gap_patterns'][num].append(gap)
                            break
                        gap += 1
        
        return dict(relationships)
    
    def _detect_distribution_anomalies(self, df: pd.DataFrame) -> Dict:
        """检测分布异常"""
        anomalies = {}
        
        # 计算各种分布指标
        sums = [sum([row[f'f{i}'] for i in range(1, 6)]) for _, row in df.iterrows()]
        spans = [max([row[f'f{i}'] for i in range(1, 6)]) - min([row[f'f{i}'] for i in range(1, 6)]) for _, row in df.iterrows()]
        odd_counts = [sum(1 for i in range(1, 6) if row[f'f{i}'] % 2 == 1) for _, row in df.iterrows()]
        
        # 使用Z-score检测异常
        def detect_outliers(data, threshold=2.5):
            mean_val = np.mean(data)
            std_val = np.std(data)
            z_scores = [(x - mean_val) / std_val for x in data]
            return [i for i, z in enumerate(z_scores) if abs(z) > threshold]
        
        anomalies['sum_outliers'] = detect_outliers(sums)
        anomalies['span_outliers'] = detect_outliers(spans)
        anomalies['odd_count_outliers'] = detect_outliers(odd_counts)
        
        return anomalies
    
    def _find_hidden_periodicities(self, df: pd.DataFrame) -> Dict:
        """寻找隐藏的周期性"""
        periodicities = {}
        
        # 分析不同特征的周期性
        features = {
            'sum': [sum([row[f'f{i}'] for i in range(1, 6)]) for _, row in df.iterrows()],
            'max_num': [max([row[f'f{i}'] for i in range(1, 6)]) for _, row in df.iterrows()],
            'min_num': [min([row[f'f{i}'] for i in range(1, 6)]) for _, row in df.iterrows()]
        }
        
        for feature_name, feature_data in features.items():
            # 使用简单的自相关分析寻找周期
            autocorrelations = []
            for lag in range(1, min(50, len(feature_data) // 2)):
                if len(feature_data) > lag:
                    corr = np.corrcoef(feature_data[:-lag], feature_data[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrelations.append((lag, corr))
            
            # 找出显著的周期
            significant_periods = [(lag, corr) for lag, corr in autocorrelations if abs(corr) > 0.3]
            periodicities[feature_name] = significant_periods
        
        return periodicities
    
    def _find_correlation_clusters(self, cooccurrence_matrix: np.ndarray) -> List[List[int]]:
        """寻找相关性聚类"""
        # 简单的聚类算法
        clusters = []
        used_numbers = set()
        
        for i in range(35):
            if i in used_numbers:
                continue
            
            cluster = [i + 1]  # 转回1-35
            used_numbers.add(i)
            
            # 找出与当前号码强相关的其他号码
            for j in range(35):
                if j != i and j not in used_numbers:
                    if cooccurrence_matrix[i][j] > np.mean(cooccurrence_matrix) + np.std(cooccurrence_matrix):
                        cluster.append(j + 1)
                        used_numbers.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters

class EvolutionaryNumberGenerator:
    """进化号码生成器：使用进化算法优化号码组合"""
    
    def __init__(self, config: EvolutionaryConfig = None):
        self.config = config or EvolutionaryConfig()
        self.pattern_analyzer = AdvancedPatternAnalyzer()
        self.fitness_history = deque(maxlen=1000)
        self.best_individuals = []
        
    def evolve_optimal_numbers(self, df: pd.DataFrame, target_count: int = 5, 
                              objectives: List[str] = None) -> Dict[str, Any]:
        """使用进化算法生成最优号码组合"""
        
        if objectives is None:
            objectives = ['hit_probability', 'diversity', 'pattern_match']
        
        logger.info(f"开始进化算法优化，目标：{objectives}")
        
        # 分析历史模式
        patterns = self.pattern_analyzer.analyze_deep_patterns(df)
        
        # 初始化种群
        population = self._initialize_population(df, patterns)
        
        best_fitness_history = []
        diversity_history = []
        
        for generation in range(self.config.generations):
            # 评估适应度
            fitness_scores = self._evaluate_population(population, df, patterns, objectives)
            
            # 记录最佳适应度
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # 计算种群多样性
            diversity = self._calculate_diversity(population)
            diversity_history.append(diversity)
            
            # 选择精英
            elite_size = int(self.config.population_size * self.config.elite_ratio)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # 生成新种群
            new_population = elite.copy()
            
            while len(new_population) < self.config.population_size:
                # 选择父代
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # 交叉
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异
                mutation_rate = self._adaptive_mutation_rate(generation, diversity)
                child1 = self._mutate(child1, mutation_rate, patterns)
                child2 = self._mutate(child2, mutation_rate, patterns)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.config.population_size]
            
            if generation % 10 == 0:
                logger.info(f"第{generation}代：最佳适应度={best_fitness:.4f}, 多样性={diversity:.4f}")
        
        # 选择最终结果
        final_fitness = self._evaluate_population(population, df, patterns, objectives)
        best_individual = population[np.argmax(final_fitness)]
        
        return {
            'best_numbers': {
                'front': sorted(best_individual['front']),
                'back': sorted(best_individual['back'])
            },
            'fitness_score': max(final_fitness),
            'generation_stats': {
                'fitness_history': best_fitness_history,
                'diversity_history': diversity_history
            },
            'patterns_used': patterns,
            'evolution_config': self.config.__dict__
        }
    
    def _initialize_population(self, df: pd.DataFrame, patterns: Dict) -> List[Dict]:
        """初始化种群"""
        population = []
        
        # 基于不同策略初始化
        strategies = [
            'random',
            'frequency_based',
            'pattern_based',
            'correlation_based',
            'temporal_based'
        ]
        
        strategy_counts = self.config.population_size // len(strategies)
        
        for strategy in strategies:
            for _ in range(strategy_counts):
                individual = self._create_individual(df, patterns, strategy)
                population.append(individual)
        
        # 填充剩余个体
        while len(population) < self.config.population_size:
            strategy = random.choice(strategies)
            individual = self._create_individual(df, patterns, strategy)
            population.append(individual)
        
        return population
    
    def _create_individual(self, df: pd.DataFrame, patterns: Dict, strategy: str) -> Dict:
        """创建个体"""
        if strategy == 'random':
            front = random.sample(range(1, 36), 5)
            back = random.sample(range(1, 13), 2)
        
        elif strategy == 'frequency_based':
            # 基于频率的选择
            front_freq = defaultdict(int)
            back_freq = defaultdict(int)
            
            for _, row in df.tail(100).iterrows():  # 使用最近100期
                for i in range(1, 6):
                    front_freq[row[f'f{i}']] += 1
                for i in range(1, 3):
                    back_freq[row[f'b{i}']] += 1
            
            # 反向选择（冷号策略）
            front_weights = {n: 1.0 / (front_freq.get(n, 0) + 1) for n in range(1, 36)}
            back_weights = {n: 1.0 / (back_freq.get(n, 0) + 1) for n in range(1, 13)}
            
            front = self._weighted_sample(front_weights, 5)
            back = self._weighted_sample(back_weights, 2)
        
        elif strategy == 'pattern_based':
            # 基于模式的选择
            front, back = self._pattern_based_selection(df, patterns)
        
        elif strategy == 'correlation_based':
            # 基于相关性的选择
            front, back = self._correlation_based_selection(patterns)
        
        else:  # temporal_based
            # 基于时间模式的选择
            front, back = self._temporal_based_selection(df, patterns)
        
        return {
            'front': front,
            'back': back,
            'strategy': strategy,
            'generation': 0
        }
    
    def _weighted_sample(self, weights: Dict[int, float], k: int) -> List[int]:
        """加权采样"""
        numbers = list(weights.keys())
        weight_values = list(weights.values())
        
        # 归一化权重
        total_weight = sum(weight_values)
        if total_weight > 0:
            weight_values = [w / total_weight for w in weight_values]
        else:
            weight_values = [1.0 / len(numbers)] * len(numbers)
        
        selected = []
        remaining_numbers = numbers.copy()
        remaining_weights = weight_values.copy()
        
        for _ in range(k):
            if not remaining_numbers:
                break
            
            # 重新归一化权重
            total = sum(remaining_weights)
            if total > 0:
                probs = [w / total for w in remaining_weights]
            else:
                probs = [1.0 / len(remaining_numbers)] * len(remaining_numbers)
            
            # 选择一个号码
            chosen_idx = np.random.choice(len(remaining_numbers), p=probs)
            selected.append(remaining_numbers[chosen_idx])
            
            # 移除已选择的号码
            remaining_numbers.pop(chosen_idx)
            remaining_weights.pop(chosen_idx)
        
        return selected
    
    def _pattern_based_selection(self, df: pd.DataFrame, patterns: Dict) -> Tuple[List[int], List[int]]:
        """基于模式的选择"""
        # 使用序列模式预测
        recent_data = df.tail(10)
        
        # 分析最近的趋势
        recent_sums = [sum([row[f'f{i}'] for i in range(1, 6)]) for _, row in recent_data.iterrows()]
        recent_spans = [max([row[f'f{i}'] for i in range(1, 6)]) - min([row[f'f{i}'] for i in range(1, 6)]) for _, row in recent_data.iterrows()]
        
        # 预测下一期的特征
        predicted_sum = np.mean(recent_sums) + np.random.normal(0, np.std(recent_sums) * 0.5)
        predicted_span = np.mean(recent_spans) + np.random.normal(0, np.std(recent_spans) * 0.5)
        
        # 生成符合预测特征的号码
        front = self._generate_numbers_with_constraints(
            target_sum=predicted_sum,
            target_span=predicted_span,
            count=5,
            number_range=(1, 35)
        )
        
        back = random.sample(range(1, 13), 2)
        
        return front, back
    
    def _correlation_based_selection(self, patterns: Dict) -> Tuple[List[int], List[int]]:
        """基于相关性的选择"""
        correlation_patterns = patterns.get('correlation_patterns', {})
        clusters = correlation_patterns.get('correlation_clusters', [])
        
        if clusters:
            # 从不同的聚类中选择号码
            front = []
            used_clusters = set()
            
            while len(front) < 5 and len(used_clusters) < len(clusters):
                cluster_idx = random.randint(0, len(clusters) - 1)
                if cluster_idx not in used_clusters:
                    cluster = clusters[cluster_idx]
                    if cluster:
                        selected_num = random.choice(cluster)
                        if selected_num not in front:
                            front.append(selected_num)
                    used_clusters.add(cluster_idx)
            
            # 如果还不够5个，随机补充
            while len(front) < 5:
                num = random.randint(1, 35)
                if num not in front:
                    front.append(num)
        else:
            front = random.sample(range(1, 36), 5)
        
        back = random.sample(range(1, 13), 2)
        return front, back
    
    def _temporal_based_selection(self, df: pd.DataFrame, patterns: Dict) -> Tuple[List[int], List[int]]:
        """基于时间模式的选择"""
        temporal_cycles = patterns.get('temporal_cycles', {})
        
        # 获取当前时间信息
        current_date = datetime.now()
        current_weekday = current_date.weekday()
        current_month = current_date.month
        
        # 使用周几模式
        weekday_patterns = temporal_cycles.get('weekday', {})
        if current_weekday in weekday_patterns:
            weekday_freq = weekday_patterns[current_weekday].get('frequency', {})
            if weekday_freq:
                # 基于该周几的历史频率选择
                front_weights = {n: weekday_freq.get(n, 1) for n in range(1, 36)}
                front = self._weighted_sample(front_weights, 5)
            else:
                front = random.sample(range(1, 36), 5)
        else:
            front = random.sample(range(1, 36), 5)
        
        back = random.sample(range(1, 13), 2)
        return front, back
    
    def _generate_numbers_with_constraints(self, target_sum: float, target_span: float, 
                                         count: int, number_range: Tuple[int, int]) -> List[int]:
        """生成满足约束条件的号码"""
        min_num, max_num = number_range
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            numbers = random.sample(range(min_num, max_num + 1), count)
            current_sum = sum(numbers)
            current_span = max(numbers) - min(numbers)
            
            # 检查是否接近目标
            sum_diff = abs(current_sum - target_sum)
            span_diff = abs(current_span - target_span)
            
            if sum_diff <= target_sum * 0.2 and span_diff <= target_span * 0.3:
                return numbers
            
            attempts += 1
        
        # 如果无法满足约束，返回随机选择
        return random.sample(range(min_num, max_num + 1), count)
    
    def _evaluate_population(self, population: List[Dict], df: pd.DataFrame, 
                           patterns: Dict, objectives: List[str]) -> List[float]:
        """评估种群适应度"""
        fitness_scores = []
        
        for individual in population:
            score = self._calculate_fitness(individual, df, patterns, objectives)
            fitness_scores.append(score)
        
        return fitness_scores
    
    def _calculate_fitness(self, individual: Dict, df: pd.DataFrame, 
                          patterns: Dict, objectives: List[str]) -> float:
        """计算个体适应度"""
        scores = {}
        
        for objective in objectives:
            if objective == 'hit_probability':
                scores[objective] = self._calculate_hit_probability(individual, df)
            elif objective == 'diversity':
                scores[objective] = self._calculate_diversity_score(individual, df)
            elif objective == 'pattern_match':
                scores[objective] = self._calculate_pattern_match_score(individual, patterns)
            elif objective == 'rarity':
                scores[objective] = self._calculate_rarity_score(individual, df)
            elif objective == 'balance':
                scores[objective] = self._calculate_balance_score(individual)
        
        # 多目标优化：加权平均
        if self.config.multi_objective:
            weights = {
                'hit_probability': 0.4,
                'diversity': 0.2,
                'pattern_match': 0.2,
                'rarity': 0.1,
                'balance': 0.1
            }
            total_score = sum(scores.get(obj, 0) * weights.get(obj, 0) for obj in objectives)
        else:
            total_score = scores.get(objectives[0], 0)
        
        return total_score
    
    def _calculate_hit_probability(self, individual: Dict, df: pd.DataFrame) -> float:
        """计算命中概率"""
        front_nums = set(individual['front'])
        back_nums = set(individual['back'])
        
        # 基于历史数据计算命中概率
        recent_data = df.tail(50)  # 使用最近50期
        hit_scores = []
        
        for _, row in recent_data.iterrows():
            historical_front = set([row[f'f{i}'] for i in range(1, 6)])
            historical_back = set([row[f'b{i}'] for i in range(1, 3)])
            
            front_hits = len(front_nums & historical_front)
            back_hits = len(back_nums & historical_back)
            
            # 计算命中分数（权重：前区70%，后区30%）
            hit_score = (front_hits / 5) * 0.7 + (back_hits / 2) * 0.3
            hit_scores.append(hit_score)
        
        return np.mean(hit_scores)
    
    def _calculate_diversity_score(self, individual: Dict, df: pd.DataFrame) -> float:
        """计算多样性分数"""
        front_nums = individual['front']
        
        # 计算号码分布的多样性
        diversity_factors = []
        
        # 1. 奇偶分布
        odd_count = sum(1 for n in front_nums if n % 2 == 1)
        odd_ratio = odd_count / 5
        # 理想的奇偶比例是2:3或3:2
        odd_diversity = 1 - abs(odd_ratio - 0.5) * 2
        diversity_factors.append(odd_diversity)
        
        # 2. 大小分布
        large_count = sum(1 for n in front_nums if n > 17)
        large_ratio = large_count / 5
        large_diversity = 1 - abs(large_ratio - 0.5) * 2
        diversity_factors.append(large_diversity)
        
        # 3. 区域分布
        zones = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
        zone_counts = [sum(1 for n in front_nums if start <= n <= end) for start, end in zones]
        zone_diversity = 1 - np.std(zone_counts) / np.mean(zone_counts) if np.mean(zone_counts) > 0 else 0
        diversity_factors.append(zone_diversity)
        
        return np.mean(diversity_factors)
    
    def _calculate_pattern_match_score(self, individual: Dict, patterns: Dict) -> float:
        """计算模式匹配分数"""
        front_nums = individual['front']
        
        # 检查是否符合发现的模式
        pattern_scores = []
        
        # 1. 相关性模式匹配
        correlation_patterns = patterns.get('correlation_patterns', {})
        strong_correlations = correlation_patterns.get('strong_correlations', [])
        
        correlation_score = 0
        for (num1, num2), strength in strong_correlations[:10]:  # 检查前10个强相关对
            if num1 in front_nums and num2 in front_nums:
                correlation_score += 1
        
        pattern_scores.append(correlation_score / 10)  # 归一化
        
        # 2. 时间模式匹配
        current_weekday = datetime.now().weekday()
        temporal_cycles = patterns.get('temporal_cycles', {})
        weekday_patterns = temporal_cycles.get('weekday', {})
        
        if current_weekday in weekday_patterns:
            weekday_freq = weekday_patterns[current_weekday].get('frequency', {})
            if weekday_freq:
                # 计算与该周几模式的匹配度
                total_freq = sum(weekday_freq.values())
                match_score = sum(weekday_freq.get(n, 0) for n in front_nums) / total_freq if total_freq > 0 else 0
                pattern_scores.append(match_score)
        
        return np.mean(pattern_scores) if pattern_scores else 0.5
    
    def _calculate_rarity_score(self, individual: Dict, df: pd.DataFrame) -> float:
        """计算稀有度分数"""
        front_nums = set(individual['front'])
        
        # 计算组合的稀有度
        combination_freq = 0
        recent_data = df.tail(100)
        
        for _, row in recent_data.iterrows():
            historical_front = set([row[f'f{i}'] for i in range(1, 6)])
            overlap = len(front_nums & historical_front)
            if overlap >= 3:  # 如果有3个或更多号码重复
                combination_freq += 1
        
        # 稀有度分数：出现频率越低，分数越高
        rarity_score = 1 - (combination_freq / len(recent_data))
        return max(0, rarity_score)
    
    def _calculate_balance_score(self, individual: Dict) -> float:
        """计算平衡性分数"""
        front_nums = individual['front']
        
        # 计算号码的平衡性
        balance_factors = []
        
        # 1. 和值平衡
        total_sum = sum(front_nums)
        ideal_sum = (1 + 35) * 5 / 2  # 理论平均和值
        sum_balance = 1 - abs(total_sum - ideal_sum) / ideal_sum
        balance_factors.append(sum_balance)
        
        # 2. 跨度平衡
        span = max(front_nums) - min(front_nums)
        ideal_span = 20  # 理想跨度
        span_balance = 1 - abs(span - ideal_span) / ideal_span
        balance_factors.append(span_balance)
        
        # 3. 间距平衡
        sorted_nums = sorted(front_nums)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(4)]
        gap_std = np.std(gaps)
        gap_balance = 1 - gap_std / np.mean(gaps) if np.mean(gaps) > 0 else 0
        balance_factors.append(gap_balance)
        
        return np.mean(balance_factors)
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """交叉操作"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 前区交叉
        if random.random() < 0.5:
            # 单点交叉
            crossover_point = random.randint(1, 4)
            child1_front = parent1['front'][:crossover_point] + parent2['front'][crossover_point:]
            child2_front = parent2['front'][:crossover_point] + parent1['front'][crossover_point:]
        else:
            # 均匀交叉
            child1_front = []
            child2_front = []
            for i in range(5):
                if random.random() < 0.5:
                    child1_front.append(parent1['front'][i])
                    child2_front.append(parent2['front'][i])
                else:
                    child1_front.append(parent2['front'][i])
                    child2_front.append(parent1['front'][i])
        
        # 确保没有重复号码
        child1['front'] = self._fix_duplicates(child1_front, range(1, 36))
        child2['front'] = self._fix_duplicates(child2_front, range(1, 36))
        
        # 后区交叉
        if random.random() < 0.5:
            child1['back'], child2['back'] = parent2['back'].copy(), parent1['back'].copy()
        
        return child1, child2
    
    def _fix_duplicates(self, numbers: List[int], valid_range: range) -> List[int]:
        """修复重复号码"""
        unique_numbers = []
        used = set()
        
        for num in numbers:
            if num not in used and num in valid_range:
                unique_numbers.append(num)
                used.add(num)
        
        # 补充缺失的号码
        while len(unique_numbers) < len(numbers):
            candidate = random.choice(list(valid_range))
            if candidate not in used:
                unique_numbers.append(candidate)
                used.add(candidate)
        
        return unique_numbers[:len(numbers)]
    
    def _mutate(self, individual: Dict, mutation_rate: float, patterns: Dict) -> Dict:
        """变异操作"""
        mutated = individual.copy()
        
        # 前区变异
        for i in range(5):
            if random.random() < mutation_rate:
                # 智能变异：基于模式选择新号码
                if random.random() < 0.3 and patterns:
                    # 基于相关性变异
                    correlation_patterns = patterns.get('correlation_patterns', {})
                    strong_correlations = correlation_patterns.get('strong_correlations', [])
                    
                    if strong_correlations:
                        # 寻找与当前号码相关的号码
                        current_num = mutated['front'][i]
                        related_nums = []
                        for (num1, num2), strength in strong_correlations:
                            if num1 == current_num and num2 not in mutated['front']:
                                related_nums.append(num2)
                            elif num2 == current_num and num1 not in mutated['front']:
                                related_nums.append(num1)
                        
                        if related_nums:
                            mutated['front'][i] = random.choice(related_nums)
                        else:
                            # 随机变异
                            new_num = random.randint(1, 35)
                            while new_num in mutated['front']:
                                new_num = random.randint(1, 35)
                            mutated['front'][i] = new_num
                    else:
                        # 随机变异
                        new_num = random.randint(1, 35)
                        while new_num in mutated['front']:
                            new_num = random.randint(1, 35)
                        mutated['front'][i] = new_num
                else:
                    # 随机变异
                    new_num = random.randint(1, 35)
                    while new_num in mutated['front']:
                        new_num = random.randint(1, 35)
                    mutated['front'][i] = new_num
        
        # 后区变异
        for i in range(2):
            if random.random() < mutation_rate:
                new_num = random.randint(1, 12)
                while new_num in mutated['back']:
                    new_num = random.randint(1, 12)
                mutated['back'][i] = new_num
        
        return mutated
    
    def _adaptive_mutation_rate(self, generation: int, diversity: float) -> float:
        """自适应变异率"""
        if not self.config.adaptive_mutation:
            return self.config.mutation_rate
        
        # 基于代数和多样性调整变异率
        base_rate = self.config.mutation_rate
        
        # 如果多样性过低，增加变异率
        if diversity < self.config.diversity_threshold:
            diversity_factor = 2.0
        else:
            diversity_factor = 1.0
        
        # 随着代数增加，逐渐降低变异率
        generation_factor = max(0.5, 1.0 - generation / self.config.generations)
        
        return base_rate * diversity_factor * generation_factor
    
    def _calculate_diversity(self, population: List[Dict]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 1.0
        
        diversity_sum = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # 计算两个个体的差异
                front1 = set(population[i]['front'])
                front2 = set(population[j]['front'])
                
                # Jaccard距离
                intersection = len(front1 & front2)
                union = len(front1 | front2)
                
                if union > 0:
                    similarity = intersection / union
                    diversity_sum += (1 - similarity)
                    comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 1.0

def create_evolutionary_optimizer(config: EvolutionaryConfig = None) -> EvolutionaryNumberGenerator:
    """创建进化优化器实例"""
    return EvolutionaryNumberGenerator(config)