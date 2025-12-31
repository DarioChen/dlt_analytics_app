# backend/secondary_generator.py
"""
二次生成器：基于第一轮生成的号码进行二次分析和生成
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import random

class SecondaryGenerator:
    """
    二次生成器：分析第一轮生成的号码特征，生成第二轮号码
    """
    
    def __init__(self):
        self.first_round_analysis = {}
    
    def analyze_first_round(self, first_round_candidates: List[Dict]) -> Dict:
        """
        分析第一轮生成的号码特征
        
        Args:
            first_round_candidates: 第一轮生成的候选号码
            
        Returns:
            分析结果字典
        """
        if not first_round_candidates:
            return {}
        
        analysis = {
            'number_frequency': {},
            'pattern_analysis': {},
            'statistical_features': {},
            'recommendations': {}
        }
        
        # 1. 号码频率分析
        front_numbers = []
        back_numbers = []
        
        for candidate in first_round_candidates:
            front_numbers.extend(candidate['front'])
            back_numbers.extend(candidate['back'])
        
        analysis['number_frequency'] = {
            'front': Counter(front_numbers),
            'back': Counter(back_numbers)
        }
        
        # 2. 模式分析
        analysis['pattern_analysis'] = self._analyze_patterns(first_round_candidates)
        
        # 3. 统计特征分析
        analysis['statistical_features'] = self._analyze_statistical_features(first_round_candidates)
        
        # 4. 生成推荐
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        self.first_round_analysis = analysis
        return analysis
    
    def _analyze_patterns(self, candidates: List[Dict]) -> Dict:
        """分析号码模式"""
        patterns = {
            'odd_even_distribution': [],
            'size_distribution': [],
            'consecutive_patterns': [],
            'gap_patterns': [],
            'sum_patterns': []
        }
        
        for candidate in candidates:
            front_nums = sorted(candidate['front'])
            
            # 奇偶分布
            odd_count = sum(1 for n in front_nums if n % 2 == 1)
            patterns['odd_even_distribution'].append(odd_count)
            
            # 大小分布（1-17为小，18-35为大）
            large_count = sum(1 for n in front_nums if n > 17)
            patterns['size_distribution'].append(large_count)
            
            # 连号模式
            consecutive_count = self._count_consecutive_groups(front_nums)
            patterns['consecutive_patterns'].append(consecutive_count)
            
            # 间距模式
            gaps = [front_nums[i] - front_nums[i-1] for i in range(1, len(front_nums))]
            avg_gap = np.mean(gaps)
            patterns['gap_patterns'].append(avg_gap)
            
            # 和值模式
            front_sum = sum(front_nums)
            patterns['sum_patterns'].append(front_sum)
        
        # 计算模式统计
        pattern_stats = {}
        for pattern_name, values in patterns.items():
            if values:
                pattern_stats[pattern_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'most_common': Counter(values).most_common(3)
                }
        
        return pattern_stats
    
    def _count_consecutive_groups(self, numbers: List[int]) -> int:
        """计算连号组数"""
        if len(numbers) < 2:
            return 0
        
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
    
    def _analyze_statistical_features(self, candidates: List[Dict]) -> Dict:
        """分析统计特征"""
        features = {
            'front_spans': [],
            'back_spans': [],
            'front_variances': [],
            'zone_distributions': []
        }
        
        for candidate in candidates:
            front_nums = sorted(candidate['front'])
            back_nums = sorted(candidate['back'])
            
            # 跨度
            features['front_spans'].append(max(front_nums) - min(front_nums))
            features['back_spans'].append(max(back_nums) - min(back_nums))
            
            # 方差
            features['front_variances'].append(np.var(front_nums))
            
            # 区域分布（1-7, 8-14, 15-21, 22-28, 29-35）
            zones = [0, 0, 0, 0, 0]  # 5个区域
            for num in front_nums:
                zone_idx = min(4, (num - 1) // 7)
                zones[zone_idx] += 1
            features['zone_distributions'].append(zones)
        
        # 计算统计信息
        stats = {}
        for feature_name, values in features.items():
            if feature_name == 'zone_distributions':
                # 区域分布需要特殊处理
                zone_stats = []
                for zone_idx in range(5):
                    zone_counts = [dist[zone_idx] for dist in values]
                    zone_stats.append({
                        'mean': np.mean(zone_counts),
                        'total': sum(zone_counts)
                    })
                stats[feature_name] = zone_stats
            else:
                if values:
                    stats[feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values)
                    }
        
        return stats
    
    def _generate_recommendations(self, analysis: Dict) -> Dict:
        """基于分析结果生成推荐"""
        recommendations = {
            'hot_numbers': {},
            'preferred_patterns': {},
            'avoid_patterns': {},
            'generation_strategy': {}
        }
        
        # 热门号码推荐
        front_freq = analysis['number_frequency']['front']
        back_freq = analysis['number_frequency']['back']
        
        # 选择出现频率最高的号码作为热门号码
        hot_front = [num for num, count in front_freq.most_common(10)]
        hot_back = [num for num, count in back_freq.most_common(6)]
        
        recommendations['hot_numbers'] = {
            'front': hot_front,
            'back': hot_back
        }
        
        # 模式推荐
        pattern_analysis = analysis['pattern_analysis']
        
        if 'odd_even_distribution' in pattern_analysis:
            odd_stats = pattern_analysis['odd_even_distribution']
            preferred_odd_count = round(odd_stats['mean'])
            recommendations['preferred_patterns']['odd_count'] = preferred_odd_count
        
        if 'size_distribution' in pattern_analysis:
            size_stats = pattern_analysis['size_distribution']
            preferred_large_count = round(size_stats['mean'])
            recommendations['preferred_patterns']['large_count'] = preferred_large_count
        
        if 'sum_patterns' in pattern_analysis:
            sum_stats = pattern_analysis['sum_patterns']
            preferred_sum_range = (
                max(50, sum_stats['mean'] - sum_stats['std']),
                min(175, sum_stats['mean'] + sum_stats['std'])
            )
            recommendations['preferred_patterns']['sum_range'] = preferred_sum_range
        
        # 生成策略
        recommendations['generation_strategy'] = {
            'emphasize_hot_numbers': True,
            'maintain_pattern_consistency': True,
            'slight_variation': True,  # 在保持模式的基础上适度变化
            'avoid_exact_duplication': True
        }
        
        return recommendations
    
    def generate_second_round(self, 
                            first_round_candidates: List[Dict],
                            count: int = 5,
                            variation_strength: float = 0.3,
                            use_recombination: bool = False) -> List[Dict]:
        """
        基于第一轮候选号码生成第二轮号码
        
        Args:
            first_round_candidates: 第一轮候选号码
            count: 生成数量
            variation_strength: 变化强度 (0-1, 0表示完全基于第一轮，1表示完全随机)
            use_recombination: 是否使用重组模式（仅从第一轮号码中重新组合）
            
        Returns:
            第二轮候选号码
        """
        if not first_round_candidates:
            return []
        
        if use_recombination:
            # 使用重组模式：仅从第一轮号码中重新组合
            return self.generate_recombined_numbers(first_round_candidates, count, variation_strength)
        else:
            # 使用原有的生成模式
            return self._generate_traditional_second_round(first_round_candidates, count, variation_strength)
    
    def _generate_traditional_second_round(self, 
                                        first_round_candidates: List[Dict],
                                        count: int = 5,
                                        variation_strength: float = 0.3) -> List[Dict]:
        """
        传统的第二轮生成方法（原有逻辑）
        """
        # 分析第一轮号码
        analysis = self.analyze_first_round(first_round_candidates)
        
        second_round_candidates = []
        
        # 尝试生成指定数量的候选号码
        for i in range(count):
            # 生成第二轮号码
            candidate = self._generate_single_candidate(analysis, variation_strength, i)
            if candidate:
                candidate['generation_method'] = 'secondary'
                candidate['round'] = 2
                candidate['variation_strength'] = variation_strength
                candidate['based_on_first_round'] = True
                second_round_candidates.append(candidate)
        
        # 如果生成数量不足，使用随机生成补充
        if len(second_round_candidates) < count:
            print(f"第二轮传统生成不足({len(second_round_candidates)}<{count})，使用随机生成补充")
            
            import random
            
            for i in range(count - len(second_round_candidates)):
                # 基于第一轮特征生成随机候选
                front_nums = sorted(random.sample(range(1, 36), 5))
                back_nums = sorted(random.sample(range(1, 13), 2))
                
                second_round_candidates.append({
                    'front': front_nums,
                    'back': back_nums,
                    'generation_method': 'secondary_random',
                    'round': 2,
                    'variation_strength': variation_strength,
                    'based_on_first_round': True
                })
        
        return second_round_candidates[:count]
    
    def generate_recombined_numbers(self, 
                                  first_round_candidates: List[Dict],
                                  count: int = 5,
                                  variation_strength: float = 0.3) -> List[Dict]:
        """
        重组生成：仅从第一轮生成的号码中重新组合生成第二轮号码
        
        Args:
            first_round_candidates: 第一轮候选号码
            count: 生成数量
            variation_strength: 变化强度 (影响组合策略)
            
        Returns:
            重组后的第二轮候选号码
        """
        if not first_round_candidates:
            return []
        
        # 提取第一轮所有号码
        all_front_numbers = set()
        all_back_numbers = set()
        
        for candidate in first_round_candidates:
            all_front_numbers.update(candidate['front'])
            all_back_numbers.update(candidate['back'])
        
        front_pool = list(all_front_numbers)
        back_pool = list(all_back_numbers)
        
        print(f"重组号码池 - 前区: {sorted(front_pool)} ({len(front_pool)}个)")
        print(f"重组号码池 - 后区: {sorted(back_pool)} ({len(back_pool)}个)")
        
        # 检查号码池是否足够
        if len(front_pool) < 5:
            print(f"警告: 前区号码池不足({len(front_pool)}<5)，无法生成完整组合")
            return []
        
        if len(back_pool) < 2:
            print(f"警告: 后区号码池不足({len(back_pool)}<2)，无法生成完整组合")
            return []
        
        # 分析第一轮号码特征
        analysis = self.analyze_first_round(first_round_candidates)
        
        # 生成重组候选号码
        recombined_candidates = []
        
        # 使用不同的重组策略
        strategies = self._get_recombination_strategies(variation_strength)
        
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            candidate = self._generate_recombined_candidate(
                front_pool, back_pool, analysis, strategy, i
            )
            
            if candidate:
                candidate['generation_method'] = 'recombination'
                candidate['round'] = 2
                candidate['variation_strength'] = variation_strength
                candidate['recombination_strategy'] = strategy['name']
                candidate['based_on_first_round'] = True
                recombined_candidates.append(candidate)
        
        # 如果重组生成数量不足，使用简单随机重组补充
        if len(recombined_candidates) < count:
            print(f"重组生成不足({len(recombined_candidates)}<{count})，使用简单随机重组补充")
            
            import random
            
            for i in range(count - len(recombined_candidates)):
                # 从号码池中随机选择
                if len(front_pool) >= 5 and len(back_pool) >= 2:
                    front_nums = sorted(random.sample(front_pool, 5))
                    back_nums = sorted(random.sample(back_pool, 2))
                    
                    recombined_candidates.append({
                        'front': front_nums,
                        'back': back_nums,
                        'generation_method': 'recombination_random',
                        'round': 2,
                        'variation_strength': variation_strength,
                        'recombination_strategy': 'random_fallback',
                        'based_on_first_round': True
                    })
        
        return recombined_candidates[:count]
    
    def _get_recombination_strategies(self, variation_strength: float) -> List[Dict]:
        """
        获取重组策略列表
        
        Args:
            variation_strength: 变化强度
            
        Returns:
            策略列表
        """
        strategies = [
            {
                'name': 'frequency_based',
                'description': '基于频率的重组',
                'weight_hot': 1.0 - variation_strength * 0.5,
                'randomness': variation_strength * 0.3
            },
            {
                'name': 'pattern_preserved',
                'description': '保持模式的重组',
                'weight_hot': 0.8 - variation_strength * 0.3,
                'randomness': variation_strength * 0.4
            },
            {
                'name': 'balanced_mix',
                'description': '平衡混合重组',
                'weight_hot': 0.6,
                'randomness': 0.5
            },
            {
                'name': 'diverse_selection',
                'description': '多样化选择重组',
                'weight_hot': 0.4 + variation_strength * 0.2,
                'randomness': 0.6 + variation_strength * 0.3
            },
            {
                'name': 'edge_exploration',
                'description': '边缘探索重组',
                'weight_hot': 0.3,
                'randomness': 0.7 + variation_strength * 0.2
            }
        ]
        
        return strategies
    
    def _generate_recombined_candidate(self, 
                                     front_pool: List[int], 
                                     back_pool: List[int],
                                     analysis: Dict,
                                     strategy: Dict,
                                     seed: int) -> Optional[Dict]:
        """
        生成单个重组候选号码
        
        Args:
            front_pool: 前区号码池
            back_pool: 后区号码池
            analysis: 第一轮分析结果
            strategy: 重组策略
            seed: 随机种子
            
        Returns:
            重组后的候选号码
        """
        random.seed(seed + 2000)  # 确保可重现性
        
        # 获取频率信息
        front_freq = analysis.get('number_frequency', {}).get('front', {})
        back_freq = analysis.get('number_frequency', {}).get('back', {})
        
        # 生成前区号码
        front_nums = self._select_recombined_front_numbers(
            front_pool, front_freq, strategy, analysis
        )
        
        if len(front_nums) != 5:
            return None
        
        # 生成后区号码
        back_nums = self._select_recombined_back_numbers(
            back_pool, back_freq, strategy
        )
        
        if len(back_nums) != 2:
            return None
        
        return {
            'front': sorted(front_nums),
            'back': sorted(back_nums)
        }
    
    def _select_recombined_front_numbers(self, 
                                       front_pool: List[int],
                                       front_freq: Dict[int, int],
                                       strategy: Dict,
                                       analysis: Dict) -> List[int]:
        """
        从前区号码池中选择5个号码进行重组
        """
        if len(front_pool) < 5:
            return []
        
        selected = set()
        weight_hot = strategy.get('weight_hot', 0.5)
        randomness = strategy.get('randomness', 0.5)
        
        # 计算每个号码的选择权重
        weights = {}
        max_freq = max(front_freq.values()) if front_freq else 1
        
        for num in front_pool:
            freq = front_freq.get(num, 0)
            # 基础权重：频率权重 + 随机权重
            freq_weight = (freq / max_freq) * weight_hot
            random_weight = random.random() * randomness
            weights[num] = freq_weight + random_weight + 0.1  # 保证最小权重
        
        # 根据策略调整权重
        if strategy['name'] == 'pattern_preserved':
            # 保持模式策略：考虑奇偶、大小分布
            weights = self._adjust_weights_for_pattern(weights, analysis, front_pool)
        elif strategy['name'] == 'diverse_selection':
            # 多样化策略：降低高频号码权重
            for num in front_pool:
                if front_freq.get(num, 0) > 2:  # 出现超过2次的号码
                    weights[num] *= 0.7
        elif strategy['name'] == 'edge_exploration':
            # 边缘探索策略：提高低频号码权重
            for num in front_pool:
                if front_freq.get(num, 0) <= 1:  # 出现1次或0次的号码
                    weights[num] *= 1.5
        
        # 选择号码
        attempts = 0
        while len(selected) < 5 and attempts < 50:
            attempts += 1
            
            # 计算当前可选号码的权重
            available = [num for num in front_pool if num not in selected]
            if not available:
                break
            
            available_weights = [weights[num] for num in available]
            
            # 加权随机选择
            if sum(available_weights) > 0:
                chosen = random.choices(available, weights=available_weights, k=1)[0]
            else:
                chosen = random.choice(available)
            
            selected.add(chosen)
        
        return list(selected)
    
    def _adjust_weights_for_pattern(self, 
                                  weights: Dict[int, float],
                                  analysis: Dict,
                                  front_pool: List[int]) -> Dict[int, float]:
        """
        根据模式分析调整权重
        """
        adjusted_weights = weights.copy()
        
        # 获取偏好模式
        recommendations = analysis.get('recommendations', {})
        preferred_patterns = recommendations.get('preferred_patterns', {})
        
        # 奇偶分布调整
        if 'odd_count' in preferred_patterns:
            preferred_odd = preferred_patterns['odd_count']
            
            # 计算当前奇偶分布
            odd_nums = [n for n in front_pool if n % 2 == 1]
            even_nums = [n for n in front_pool if n % 2 == 0]
            
            # 根据需要调整奇偶号码权重
            if len(odd_nums) > preferred_odd:
                # 需要更多偶数
                for num in even_nums:
                    adjusted_weights[num] *= 1.2
                for num in odd_nums:
                    adjusted_weights[num] *= 0.8
            elif len(odd_nums) < preferred_odd:
                # 需要更多奇数
                for num in odd_nums:
                    adjusted_weights[num] *= 1.2
                for num in even_nums:
                    adjusted_weights[num] *= 0.8
        
        # 大小分布调整
        if 'large_count' in preferred_patterns:
            preferred_large = preferred_patterns['large_count']
            
            large_nums = [n for n in front_pool if n > 17]
            small_nums = [n for n in front_pool if n <= 17]
            
            if len(large_nums) > preferred_large:
                # 需要更多小号
                for num in small_nums:
                    adjusted_weights[num] *= 1.2
                for num in large_nums:
                    adjusted_weights[num] *= 0.8
            elif len(large_nums) < preferred_large:
                # 需要更多大号
                for num in large_nums:
                    adjusted_weights[num] *= 1.2
                for num in small_nums:
                    adjusted_weights[num] *= 0.8
        
        return adjusted_weights
    
    def _select_recombined_back_numbers(self, 
                                      back_pool: List[int],
                                      back_freq: Dict[int, int],
                                      strategy: Dict) -> List[int]:
        """
        从后区号码池中选择2个号码进行重组
        """
        if len(back_pool) < 2:
            return []
        
        weight_hot = strategy.get('weight_hot', 0.5)
        randomness = strategy.get('randomness', 0.5)
        
        # 计算权重
        weights = {}
        max_freq = max(back_freq.values()) if back_freq else 1
        
        for num in back_pool:
            freq = back_freq.get(num, 0)
            freq_weight = (freq / max_freq) * weight_hot
            random_weight = random.random() * randomness
            weights[num] = freq_weight + random_weight + 0.1
        
        # 根据策略调整
        if strategy['name'] == 'diverse_selection':
            for num in back_pool:
                if back_freq.get(num, 0) > 2:
                    weights[num] *= 0.7
        elif strategy['name'] == 'edge_exploration':
            for num in back_pool:
                if back_freq.get(num, 0) <= 1:
                    weights[num] *= 1.5
        
        # 选择2个号码
        selected = set()
        attempts = 0
        
        while len(selected) < 2 and attempts < 20:
            attempts += 1
            
            available = [num for num in back_pool if num not in selected]
            if not available:
                break
            
            available_weights = [weights[num] for num in available]
            
            if sum(available_weights) > 0:
                chosen = random.choices(available, weights=available_weights, k=1)[0]
            else:
                chosen = random.choice(available)
            
            selected.add(chosen)
        
        return list(selected)
    
    def _generate_single_candidate(self, analysis: Dict, variation_strength: float, seed: int) -> Optional[Dict]:
        """生成单个第二轮候选号码"""
        random.seed(seed + 1000)  # 确保可重现性
        
        recommendations = analysis.get('recommendations', {})
        hot_numbers = recommendations.get('hot_numbers', {})
        preferred_patterns = recommendations.get('preferred_patterns', {})
        
        # 生成前区号码
        front_nums = self._generate_front_numbers(hot_numbers, preferred_patterns, variation_strength)
        if len(front_nums) != 5:
            return None
        
        # 生成后区号码
        back_nums = self._generate_back_numbers(hot_numbers, variation_strength)
        if len(back_nums) != 2:
            return None
        
        return {
            'front': sorted(front_nums),
            'back': sorted(back_nums)
        }
    
    def _generate_front_numbers(self, hot_numbers: Dict, preferred_patterns: Dict, variation_strength: float) -> List[int]:
        """生成前区号码"""
        hot_front = hot_numbers.get('front', [])
        
        # 计算使用热门号码的数量
        hot_count = max(1, int(5 * (1 - variation_strength)))
        random_count = 5 - hot_count
        
        selected_numbers = set()
        
        # 从热门号码中选择
        if hot_front and hot_count > 0:
            available_hot = [n for n in hot_front if n not in selected_numbers]
            selected_hot = random.sample(available_hot, min(hot_count, len(available_hot)))
            selected_numbers.update(selected_hot)
        
        # 随机选择剩余号码
        while len(selected_numbers) < 5:
            remaining_pool = [n for n in range(1, 36) if n not in selected_numbers]
            if not remaining_pool:
                break
            
            # 根据偏好模式调整选择概率
            weights = self._calculate_selection_weights(remaining_pool, selected_numbers, preferred_patterns)
            
            if sum(weights) > 0:
                chosen = random.choices(remaining_pool, weights=weights, k=1)[0]
                selected_numbers.add(chosen)
            else:
                chosen = random.choice(remaining_pool)
                selected_numbers.add(chosen)
        
        return list(selected_numbers)
    
    def _calculate_selection_weights(self, candidates: List[int], selected: Set[int], preferred_patterns: Dict) -> List[float]:
        """计算候选号码的选择权重"""
        weights = [1.0] * len(candidates)
        
        if not preferred_patterns:
            return weights
        
        # 根据奇偶偏好调整权重
        if 'odd_count' in preferred_patterns:
            preferred_odd = preferred_patterns['odd_count']
            current_odd = sum(1 for n in selected if n % 2 == 1)
            remaining_slots = 5 - len(selected)
            
            if remaining_slots > 0:
                needed_odd = preferred_odd - current_odd
                
                for i, num in enumerate(candidates):
                    if num % 2 == 1:  # 奇数
                        if needed_odd > 0:
                            weights[i] *= 1.5
                        elif needed_odd < 0:
                            weights[i] *= 0.5
                    else:  # 偶数
                        if needed_odd < remaining_slots:
                            weights[i] *= 1.5
                        else:
                            weights[i] *= 0.5
        
        # 根据大小偏好调整权重
        if 'large_count' in preferred_patterns:
            preferred_large = preferred_patterns['large_count']
            current_large = sum(1 for n in selected if n > 17)
            remaining_slots = 5 - len(selected)
            
            if remaining_slots > 0:
                needed_large = preferred_large - current_large
                
                for i, num in enumerate(candidates):
                    if num > 17:  # 大号
                        if needed_large > 0:
                            weights[i] *= 1.3
                        elif needed_large < 0:
                            weights[i] *= 0.7
                    else:  # 小号
                        if needed_large < remaining_slots:
                            weights[i] *= 1.3
                        else:
                            weights[i] *= 0.7
        
        return weights
    
    def _generate_back_numbers(self, hot_numbers: Dict, variation_strength: float) -> List[int]:
        """生成后区号码"""
        hot_back = hot_numbers.get('back', [])
        
        # 计算使用热门号码的数量
        hot_count = max(0, int(2 * (1 - variation_strength)))
        
        selected_numbers = set()
        
        # 从热门号码中选择
        if hot_back and hot_count > 0:
            available_hot = [n for n in hot_back if n not in selected_numbers]
            selected_hot = random.sample(available_hot, min(hot_count, len(available_hot)))
            selected_numbers.update(selected_hot)
        
        # 随机选择剩余号码
        while len(selected_numbers) < 2:
            remaining_pool = [n for n in range(1, 13) if n not in selected_numbers]
            if not remaining_pool:
                break
            chosen = random.choice(remaining_pool)
            selected_numbers.add(chosen)
        
        return list(selected_numbers)
    
    def compare_rounds(self, first_round: List[Dict], second_round: List[Dict], 
                      actual_result: Optional[Dict] = None) -> Dict:
        """
        对比两轮生成结果
        
        Args:
            first_round: 第一轮候选号码
            second_round: 第二轮候选号码
            actual_result: 实际开奖结果（可选）
            
        Returns:
            对比分析结果
        """
        comparison = {
            'round_analysis': {},
            'overlap_analysis': {},
            'pattern_comparison': {},
            'hit_analysis': {}
        }
        
        # 1. 轮次分析
        comparison['round_analysis'] = {
            'first_round': self._analyze_round(first_round, 1),
            'second_round': self._analyze_round(second_round, 2)
        }
        
        # 2. 重叠分析
        comparison['overlap_analysis'] = self._analyze_overlap(first_round, second_round)
        
        # 3. 模式对比
        comparison['pattern_comparison'] = self._compare_patterns(first_round, second_round)
        
        # 4. 命中分析（如果有实际结果）
        if actual_result:
            comparison['hit_analysis'] = self._analyze_hits(first_round, second_round, actual_result)
        
        return comparison
    
    def _analyze_round(self, candidates: List[Dict], round_num: int) -> Dict:
        """分析单轮结果"""
        if not candidates:
            return {}
        
        front_numbers = []
        back_numbers = []
        
        for candidate in candidates:
            front_numbers.extend(candidate['front'])
            back_numbers.extend(candidate['back'])
        
        return {
            'round': round_num,
            'candidate_count': len(candidates),
            'unique_front_numbers': len(set(front_numbers)),
            'unique_back_numbers': len(set(back_numbers)),
            'front_frequency': dict(Counter(front_numbers)),
            'back_frequency': dict(Counter(back_numbers)),
            'most_popular_front': Counter(front_numbers).most_common(5),
            'most_popular_back': Counter(back_numbers).most_common(3)
        }
    
    def _analyze_overlap(self, first_round: List[Dict], second_round: List[Dict]) -> Dict:
        """分析两轮重叠情况"""
        first_front = set()
        first_back = set()
        second_front = set()
        second_back = set()
        
        for candidate in first_round:
            first_front.update(candidate['front'])
            first_back.update(candidate['back'])
        
        for candidate in second_round:
            second_front.update(candidate['front'])
            second_back.update(candidate['back'])
        
        return {
            'front_overlap': list(first_front & second_front),
            'back_overlap': list(first_back & second_back),
            'front_overlap_ratio': len(first_front & second_front) / len(first_front | second_front) if first_front | second_front else 0,
            'back_overlap_ratio': len(first_back & second_back) / len(first_back | second_back) if first_back | second_back else 0,
            'unique_to_first_front': list(first_front - second_front),
            'unique_to_second_front': list(second_front - first_front),
            'unique_to_first_back': list(first_back - second_back),
            'unique_to_second_back': list(second_back - first_back)
        }
    
    def _compare_patterns(self, first_round: List[Dict], second_round: List[Dict]) -> Dict:
        """对比两轮的模式特征"""
        first_patterns = self._analyze_patterns(first_round)
        second_patterns = self._analyze_patterns(second_round)
        
        comparison = {}
        
        for pattern_name in first_patterns:
            if pattern_name in second_patterns:
                first_mean = first_patterns[pattern_name]['mean']
                second_mean = second_patterns[pattern_name]['mean']
                
                comparison[pattern_name] = {
                    'first_round_mean': first_mean,
                    'second_round_mean': second_mean,
                    'difference': second_mean - first_mean,
                    'change_percentage': ((second_mean - first_mean) / first_mean * 100) if first_mean != 0 else 0
                }
        
        return comparison
    
    def _analyze_hits(self, first_round: List[Dict], second_round: List[Dict], actual_result: Dict) -> Dict:
        """分析命中情况"""
        actual_front = set(actual_result.get('front', []))
        actual_back = set(actual_result.get('back', []))
        
        def calculate_hits(candidates, round_name):
            hits = []
            for i, candidate in enumerate(candidates):
                front_hits = len(set(candidate['front']) & actual_front)
                back_hits = len(set(candidate['back']) & actual_back)
                total_hits = front_hits + back_hits
                
                hits.append({
                    'candidate_index': i,
                    'front_hits': front_hits,
                    'back_hits': back_hits,
                    'total_hits': total_hits,
                    'front_numbers': candidate['front'],
                    'back_numbers': candidate['back']
                })
            
            return {
                'round': round_name,
                'individual_hits': hits,
                'best_candidate': max(hits, key=lambda x: x['total_hits']) if hits else None,
                'average_front_hits': np.mean([h['front_hits'] for h in hits]) if hits else 0,
                'average_back_hits': np.mean([h['back_hits'] for h in hits]) if hits else 0,
                'average_total_hits': np.mean([h['total_hits'] for h in hits]) if hits else 0,
                'total_unique_front_hits': len(set().union(*[set(c['front']) for c in candidates]) & actual_front),
                'total_unique_back_hits': len(set().union(*[set(c['back']) for c in candidates]) & actual_back)
            }
        
        first_hits = calculate_hits(first_round, 'first')
        second_hits = calculate_hits(second_round, 'second')
        
        return {
            'first_round': first_hits,
            'second_round': second_hits,
            'comparison': {
                'better_round': 'second' if second_hits['average_total_hits'] > first_hits['average_total_hits'] else 'first',
                'improvement': second_hits['average_total_hits'] - first_hits['average_total_hits'],
                'improvement_percentage': ((second_hits['average_total_hits'] - first_hits['average_total_hits']) / first_hits['average_total_hits'] * 100) if first_hits['average_total_hits'] > 0 else 0
            }
        }