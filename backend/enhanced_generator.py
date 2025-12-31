# backend/enhanced_generator.py
"""
增强的号码生成器：集成马尔可夫链模型和大数据规律分析
"""
from typing import Dict, List, Tuple, Optional, Set
import random
import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime

from .markov_model import MarkovChainModel, BigDataAnalyzer
from .generator import consecutive_pairs_count, consecutive_groups_count
from .ensemble_predictor import EnsemblePredictor
from .advanced_features import AdvancedFeatureExtractor, SmartFilter
from .secondary_generator import SecondaryGenerator

class EnhancedNumberGenerator:
    """
    增强的号码生成器，集成多种数学模型和大数据分析
    """
    
    def __init__(self):
        self.markov_models = {}
        self.big_data_analyzer = BigDataAnalyzer()
        self.analysis_cache = {}
        
        # 新增：集成预测器和高级特征提取器
        self.ensemble_predictor = EnsemblePredictor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.secondary_generator = SecondaryGenerator()
        self.use_ensemble = False  # 是否使用集成预测
        
    def initialize_models(self, df: pd.DataFrame, use_ensemble: bool = True, force_reinit: bool = False):
        """
        初始化所有数学模型
        
        Args:
            df: 历史开奖数据
            use_ensemble: 是否使用集成学习
            force_reinit: 是否强制重新初始化
        """
        # 检查是否已经初始化且不需要强制重新初始化
        if not force_reinit and (self.markov_models or self.use_ensemble):
            print("增强号码生成器已初始化，跳过重复初始化")
            return
        
        print("🔄 正在初始化增强号码生成器...")
        
        # 简化特征提取（仅提取必要特征）
        print("📊 提取基础特征...")
        try:
            df_enhanced = self._extract_basic_features_only(df)
        except Exception as e:
            print(f"特征提取失败，使用原始数据: {e}")
            df_enhanced = df
        
        if use_ensemble:
            # 使用集成预测器（简化版本）
            print("🤖 初始化轻量级集成预测器...")
            try:
                self.ensemble_predictor.train_lightweight(df_enhanced)
                self.use_ensemble = True
                print("✅ 集成预测器初始化成功")
            except Exception as e:
                print(f"集成预测器初始化失败，回退到马尔可夫链: {e}")
                use_ensemble = False
        
        if not use_ensemble:
            # 使用简化的马尔可夫链模型
            print("🔗 初始化轻量级马尔可夫链模型...")
            self.markov_models = {
                'number_order1': MarkovChainModel(order=1, state_type='number'),
                'number_order2': MarkovChainModel(order=2, state_type='number'),
            }
            
            # 训练核心模型
            for name, model in self.markov_models.items():
                try:
                    print(f"🎯 训练模型: {name}")
                    model.train(df_enhanced)
                except Exception as e:
                    print(f"模型 {name} 训练失败: {e}")
            
            self.use_ensemble = False
            print("✅ 马尔可夫链模型初始化成功")
        
        # 简化大数据分析
        print("📈 运行轻量级大数据分析...")
        try:
            self.big_data_insights = self._generate_basic_insights(df_enhanced)
            print("✅ 大数据分析完成")
        except Exception as e:
            print(f"大数据分析失败: {e}")
            self.big_data_insights = {}
        
        print("🎉 增强号码生成器初始化完成")
    
    def _extract_basic_features_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        仅提取基础特征，避免复杂计算
        """
        df_basic = df.copy()
        
        # 只添加最基本的特征
        for i in range(1, 6):  # 前区
            if f'f{i}' in df_basic.columns:
                df_basic[f'f{i}_is_odd'] = df_basic[f'f{i}'] % 2
                df_basic[f'f{i}_is_large'] = (df_basic[f'f{i}'] > 17).astype(int)
        
        for i in range(1, 3):  # 后区
            if f'b{i}' in df_basic.columns:
                df_basic[f'b{i}_is_odd'] = df_basic[f'b{i}'] % 2
        
        return df_basic
    
    def _generate_basic_insights(self, df: pd.DataFrame) -> Dict:
        """
        生成基础洞察，避免复杂分析
        """
        insights = {
            'data_summary': {
                'total_records': len(df),
                'date_range': {
                    'start': df['date'].min() if 'date' in df.columns else None,
                    'end': df['date'].max() if 'date' in df.columns else None
                }
            },
            'basic_stats': {}
        }
        
        # 基础统计
        for col in ['f1', 'f2', 'f3', 'f4', 'f5', 'b1', 'b2']:
            if col in df.columns:
                insights['basic_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': int(df[col].min()),
                    'max': int(df[col].max())
                }
        
        return insights
    
    def generate_enhanced_numbers(
        self,
        count: int = 5,
        rules: Optional[Dict] = None,
        front_blocks: Optional[Dict[str, List[int]]] = None,
        back_blocks: Optional[Dict[str, List[int]]] = None,
        front_weights: Optional[Dict[str, float]] = None,
        back_weights: Optional[Dict[str, float]] = None,
        selected_front_blocks: Optional[List[str]] = None,
        selected_back_blocks: Optional[List[str]] = None,
        historical_data: Optional[pd.DataFrame] = None,
        use_markov: bool = True,
        use_big_data: bool = True,
        markov_weight: float = 0.4,
        big_data_weight: float = 0.3,
        traditional_weight: float = 0.3
    ) -> List[Dict]:
        """
        使用增强算法生成号码
        
        Args:
            count: 生成号码组数
            rules: 生成规则
            front_blocks: 前区区块定义
            back_blocks: 后区区块定义
            front_weights: 前区权重
            back_weights: 后区权重
            selected_front_blocks: 选中的前区区块
            selected_back_blocks: 选中的后区区块
            historical_data: 历史数据
            use_markov: 是否使用马尔可夫链
            use_big_data: 是否使用大数据分析
            markov_weight: 马尔可夫链权重
            big_data_weight: 大数据分析权重
            traditional_weight: 传统方法权重
            
        Returns:
            生成的号码组合列表
        """
        # 检查是否使用集成预测器
        if self.use_ensemble and hasattr(self.ensemble_predictor, 'predictors'):
            return self._generate_with_ensemble(
                count=count,
                rules=rules,
                historical_data=historical_data,
                front_blocks=front_blocks,
                back_blocks=back_blocks,
                selected_front_blocks=selected_front_blocks,
                selected_back_blocks=selected_back_blocks
            )
        
        # 原有的生成逻辑
        if not self.markov_models and not self.use_ensemble:
            raise ValueError("模型未初始化，请先调用initialize_models()或启用集成学习")
        
        # 调试信息
        print(f"增强生成器调试信息:")
        print(f"  - historical_data类型: {type(historical_data)}")
        if historical_data is not None:
            print(f"  - historical_data形状: {historical_data.shape}")
            print(f"  - historical_data列: {list(historical_data.columns)}")
            print(f"  - historical_data为空: {historical_data.empty}")
        
        try:
            # 计算综合权重
            enhanced_front_weights, enhanced_back_weights = self._calculate_enhanced_weights(
                historical_data=historical_data,
                front_weights=front_weights,
                back_weights=back_weights,
                use_markov=use_markov,
                use_big_data=use_big_data,
                markov_weight=markov_weight,
                big_data_weight=big_data_weight,
                traditional_weight=traditional_weight
            )
        except Exception as e:
            print(f"权重计算失败: {e}")
            raise e
        
        # 使用增强权重生成号码
        results = self._generate_with_enhanced_weights(
            count=count,
            rules=rules,
            front_blocks=front_blocks,
            back_blocks=back_blocks,
            enhanced_front_weights=enhanced_front_weights,
            enhanced_back_weights=enhanced_back_weights,
            selected_front_blocks=selected_front_blocks,
            selected_back_blocks=selected_back_blocks,
            historical_data=historical_data
        )
        
        return results
    
    def _generate_with_ensemble(
        self,
        count: int,
        rules: Optional[Dict],
        historical_data: Optional[pd.DataFrame],
        front_blocks: Optional[Dict[str, List[int]]],
        back_blocks: Optional[Dict[str, List[int]]],
        selected_front_blocks: Optional[List[str]],
        selected_back_blocks: Optional[List[str]]
    ) -> List[Dict]:
        """
        使用集成预测器生成号码
        """
        print("使用集成预测器生成号码...")
        
        if historical_data is None or historical_data.empty:
            print("历史数据不足，回退到传统方法")
            return []
        
        try:
            # 使用集成预测器生成候选号码
            candidates = self.ensemble_predictor.predict_enhanced(historical_data, count)
            
            # 应用规则过滤
            if rules:
                candidates = self._apply_rules_to_candidates(candidates, rules)
            
            # 添加生成方法标识
            for candidate in candidates:
                candidate['generation_method'] = 'ensemble'
                candidate['ensemble_confidence'] = candidate.get('ensemble_confidence', 0.5)
                candidate['filter_score'] = candidate.get('filter_score', 0.5)
            
            print(f"集成预测器生成了 {len(candidates)} 个候选号码")
            return candidates
            
        except Exception as e:
            print(f"集成预测器生成失败: {e}")
            return []
    
    def _apply_rules_to_candidates(self, candidates: List[Dict], rules: Dict) -> List[Dict]:
        """
        对候选号码应用规则过滤
        """
        filtered_candidates = []
        
        for candidate in candidates:
            if self._check_candidate_rules(candidate, rules):
                filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def _check_candidate_rules(self, candidate: Dict, rules: Dict) -> bool:
        """
        检查候选号码是否符合规则
        """
        front_nums = candidate['front']
        back_nums = candidate['back']
        
        # 和值检查
        sum_range = rules.get("sum_front_range", [None, None])
        front_sum = sum(front_nums)
        smin, smax = sum_range
        if (smin is not None and front_sum < smin) or (smax is not None and front_sum > smax):
            return False
        
        # 奇偶检查
        odd_even_range = rules.get("odd_even_front", [0, 5])
        odd_count = sum(1 for n in front_nums if n % 2 == 1)
        min_odd, max_odd = odd_even_range
        if odd_count < min_odd or odd_count > max_odd:
            return False
        
        # 包含检查
        front_include = set(rules.get("front_include", []))
        back_include = set(rules.get("back_include", []))
        if front_include and not front_include.issubset(set(front_nums)):
            return False
        if back_include and not back_include.issubset(set(back_nums)):
            return False
        
        # 排除检查
        front_exclude = set(rules.get("front_exclude", []))
        back_exclude = set(rules.get("back_exclude", []))
        if front_exclude and front_exclude & set(front_nums):
            return False
        if back_exclude and back_exclude & set(back_nums):
            return False
        
        # 连号检查
        consecutive_count = rules.get("consecutive_count", 0)
        consecutive_mode = rules.get("consecutive_mode", "exact")
        consecutive_check_type = rules.get("consecutive_check_type", "groups")
        
        if consecutive_check_type == "groups":
            cons_count = consecutive_groups_count(front_nums)
        else:
            cons_count = consecutive_pairs_count(front_nums)
        
        if consecutive_mode == "exact" and cons_count != consecutive_count:
            return False
        elif consecutive_mode == "min" and cons_count < consecutive_count:
            return False
        elif consecutive_mode == "max" and cons_count > consecutive_count:
            return False
        
        return True
    
    def generate_two_rounds(
        self,
        count: int = 5,
        rules: Optional[Dict] = None,
        front_blocks: Optional[Dict[str, List[int]]] = None,
        back_blocks: Optional[Dict[str, List[int]]] = None,
        front_weights: Optional[Dict[str, float]] = None,
        back_weights: Optional[Dict[str, float]] = None,
        selected_front_blocks: Optional[List[str]] = None,
        selected_back_blocks: Optional[List[str]] = None,
        historical_data: Optional[pd.DataFrame] = None,
        use_markov: bool = True,
        use_big_data: bool = True,
        markov_weight: float = 0.4,
        big_data_weight: float = 0.3,
        traditional_weight: float = 0.3,
        variation_strength: float = 0.3,
        use_recombination: bool = False
    ) -> Dict:
        """
        生成两轮号码并进行对比分析
        
        Args:
            count: 每轮生成号码组数
            rules: 生成规则
            front_blocks: 前区区块定义
            back_blocks: 后区区块定义
            front_weights: 前区权重
            back_weights: 后区权重
            selected_front_blocks: 选中的前区区块
            selected_back_blocks: 选中的后区区块
            historical_data: 历史数据
            use_markov: 是否使用马尔可夫链
            use_big_data: 是否使用大数据分析
            markov_weight: 马尔可夫链权重
            big_data_weight: 大数据分析权重
            traditional_weight: 传统方法权重
            variation_strength: 第二轮变化强度 (0-1)
            use_recombination: 是否使用重组模式（仅从第一轮号码中重新组合）
            
        Returns:
            包含两轮结果和分析的字典
        """
        print("开始两轮号码生成...")
        print(f"重组模式: {'启用' if use_recombination else '禁用'}")
        
        # 检查模型是否已初始化
        if not self.markov_models and not self.use_ensemble:
            return {
                'first_round': [],
                'second_round': [],
                'analysis': {},
                'comparison': {},
                'error': '模型未初始化，请先调用initialize_models()或启用集成学习'
            }
        
        # 第一轮生成 - 确保生成准确数量
        print(f"生成第一轮号码（目标：{count}个）...")
        
        # 使用更可靠的生成方法确保第一轮生成准确数量
        first_round = self._generate_first_round_reliable(
            count=count,
            rules=rules,
            front_blocks=front_blocks,
            back_blocks=back_blocks,
            front_weights=front_weights,
            back_weights=back_weights,
            selected_front_blocks=selected_front_blocks,
            selected_back_blocks=selected_back_blocks,
            historical_data=historical_data,
            use_markov=use_markov,
            use_big_data=use_big_data,
            markov_weight=markov_weight,
            big_data_weight=big_data_weight,
            traditional_weight=traditional_weight
        )
        
        print(f"第一轮实际生成：{len(first_round)}个候选号码")
        
        if len(first_round) != count:
            print(f"警告：第一轮生成数量({len(first_round)})与期望数量({count})不符")
        
        if not first_round:
            return {
                'first_round': [],
                'second_round': [],
                'analysis': {},
                'comparison': {},
                'error': '第一轮生成失败'
            }
        
        # 第二轮生成
        if use_recombination:
            print("基于第一轮结果进行重组生成第二轮号码...")
        else:
            print("基于第一轮结果生成第二轮号码...")
        
        second_round = self.secondary_generator.generate_second_round(
            first_round_candidates=first_round,
            count=count,
            variation_strength=variation_strength,
            use_recombination=use_recombination
        )
        
        # 分析第一轮结果
        print("分析第一轮号码特征...")
        first_round_analysis = self.secondary_generator.analyze_first_round(first_round)
        
        # 对比两轮结果
        print("对比两轮生成结果...")
        comparison = self.secondary_generator.compare_rounds(first_round, second_round)
        
        return {
            'first_round': first_round,
            'second_round': second_round,
            'first_round_analysis': first_round_analysis,
            'comparison': comparison,
            'generation_params': {
                'count': count,
                'variation_strength': variation_strength,
                'use_ensemble': self.use_ensemble
            }
        }
    
    def _generate_first_round_reliable(
        self,
        count: int,
        rules: Optional[Dict] = None,
        front_blocks: Optional[Dict[str, List[int]]] = None,
        back_blocks: Optional[Dict[str, List[int]]] = None,
        front_weights: Optional[Dict[str, float]] = None,
        back_weights: Optional[Dict[str, float]] = None,
        selected_front_blocks: Optional[List[str]] = None,
        selected_back_blocks: Optional[List[str]] = None,
        historical_data: Optional[pd.DataFrame] = None,
        use_markov: bool = True,
        use_big_data: bool = True,
        markov_weight: float = 0.4,
        big_data_weight: float = 0.3,
        traditional_weight: float = 0.3
    ) -> List[Dict]:
        """
        可靠的第一轮生成方法，确保生成准确数量的候选号码
        """
        print(f"使用可靠生成方法，目标生成 {count} 个候选号码")
        
        # 首先尝试使用增强生成
        try:
            enhanced_candidates = self.generate_enhanced_numbers(
                count=count,
                rules=rules,
                front_blocks=front_blocks,
                back_blocks=back_blocks,
                front_weights=front_weights,
                back_weights=back_weights,
                selected_front_blocks=selected_front_blocks,
                selected_back_blocks=selected_back_blocks,
                historical_data=historical_data,
                use_markov=use_markov,
                use_big_data=use_big_data,
                markov_weight=markov_weight,
                big_data_weight=big_data_weight,
                traditional_weight=traditional_weight
            )
            
            print(f"增强生成方法生成了 {len(enhanced_candidates)} 个候选号码")
            
            # 如果增强生成的数量足够，直接返回
            if len(enhanced_candidates) >= count:
                return enhanced_candidates[:count]
            
        except Exception as e:
            print(f"增强生成失败: {e}")
            enhanced_candidates = []
        
        # 如果增强生成不足，使用传统生成方法补充
        if len(enhanced_candidates) < count:
            print(f"增强生成不足，使用传统方法补充 {count - len(enhanced_candidates)} 个候选号码")
            
            try:
                # 导入传统生成器
                from . import generator as genmod
                
                traditional_candidates = genmod.gen_numbers(
                    count=count - len(enhanced_candidates),
                    rules=rules or {},
                    front_blocks=front_blocks or {},
                    back_blocks=back_blocks or {},
                    front_weights=front_weights or {},
                    back_weights=back_weights or {},
                    selected_front_blocks=selected_front_blocks or [],
                    selected_back_blocks=selected_back_blocks or []
                )
                
                print(f"传统生成方法补充了 {len(traditional_candidates)} 个候选号码")
                
                # 为传统生成的候选号码添加标识
                for candidate in traditional_candidates:
                    candidate['generation_method'] = 'traditional_fallback'
                    candidate['markov_confidence'] = 0.5
                    candidate['big_data_score'] = 0.5
                
                enhanced_candidates.extend(traditional_candidates)
                
            except Exception as e:
                print(f"传统生成也失败: {e}")
        
        # 如果还是不足，使用随机生成补充
        if len(enhanced_candidates) < count:
            print(f"仍然不足，使用随机生成补充 {count - len(enhanced_candidates)} 个候选号码")
            
            import random
            
            for i in range(count - len(enhanced_candidates)):
                # 生成随机候选号码
                front_nums = sorted(random.sample(range(1, 36), 5))
                back_nums = sorted(random.sample(range(1, 13), 2))
                
                enhanced_candidates.append({
                    'front': front_nums,
                    'back': back_nums,
                    'generation_method': 'random_fallback',
                    'markov_confidence': 0.3,
                    'big_data_score': 0.3
                })
        
        # 确保返回准确数量
        result = enhanced_candidates[:count]
        print(f"最终第一轮生成了 {len(result)} 个候选号码")
        
        return result
    
    def analyze_hit_performance(
        self,
        first_round: List[Dict],
        second_round: List[Dict],
        actual_result: Dict
    ) -> Dict:
        """
        分析两轮号码的中奖表现
        
        Args:
            first_round: 第一轮候选号码
            second_round: 第二轮候选号码
            actual_result: 实际开奖结果 {'front': [1,2,3,4,5], 'back': [1,2]}
            
        Returns:
            详细的中奖分析结果
        """
        return self.secondary_generator.compare_rounds(
            first_round, second_round, actual_result
        )
    
    def _calculate_enhanced_weights(
        self,
        historical_data: Optional[pd.DataFrame],
        front_weights: Optional[Dict[str, float]],
        back_weights: Optional[Dict[str, float]],
        use_markov: bool,
        use_big_data: bool,
        markov_weight: float,
        big_data_weight: float,
        traditional_weight: float
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        计算增强的号码权重
        """
        front_range = range(1, 36)
        back_range = range(1, 13)
        
        # 初始化权重
        enhanced_front_weights = {n: 0.0 for n in front_range}
        enhanced_back_weights = {n: 0.0 for n in back_range}
        
        total_weight = 0.0
        
        # 1. 传统权重（基于区块权重）
        if traditional_weight > 0 and front_weights and back_weights:
            traditional_front, traditional_back = self._convert_block_weights_to_number_weights(
                front_weights, back_weights, front_range, back_range
            )
            
            for n in front_range:
                enhanced_front_weights[n] += traditional_front.get(n, 0) * traditional_weight
            for n in back_range:
                enhanced_back_weights[n] += traditional_back.get(n, 0) * traditional_weight
            
            total_weight += traditional_weight
        
        # 2. 马尔可夫链权重
        if use_markov and markov_weight > 0 and historical_data is not None and len(historical_data) > 0:
            try:
                markov_front, markov_back = self._get_markov_weights(historical_data, front_range, back_range)
                
                for n in front_range:
                    enhanced_front_weights[n] += markov_front.get(n, 0) * markov_weight
                for n in back_range:
                    enhanced_back_weights[n] += markov_back.get(n, 0) * markov_weight
                
                total_weight += markov_weight
            except Exception as e:
                print(f"马尔可夫链权重计算失败: {e}")
                # 跳过马尔可夫链权重，继续其他权重计算
        
        # 3. 大数据分析权重
        if use_big_data and big_data_weight > 0:
            big_data_front, big_data_back = self._get_big_data_weights(front_range, back_range)
            
            for n in front_range:
                enhanced_front_weights[n] += big_data_front.get(n, 0) * big_data_weight
            for n in back_range:
                enhanced_back_weights[n] += big_data_back.get(n, 0) * big_data_weight
            
            total_weight += big_data_weight
        
        # 归一化权重
        if total_weight > 0:
            for n in front_range:
                enhanced_front_weights[n] /= total_weight
            for n in back_range:
                enhanced_back_weights[n] /= total_weight
        else:
            # 回退到均匀分布
            uniform_front = 1.0 / len(front_range)
            uniform_back = 1.0 / len(back_range)
            enhanced_front_weights = {n: uniform_front for n in front_range}
            enhanced_back_weights = {n: uniform_back for n in back_range}
        
        return enhanced_front_weights, enhanced_back_weights
    
    def _convert_block_weights_to_number_weights(
        self,
        front_weights: Dict[str, float],
        back_weights: Dict[str, float],
        front_range: range,
        back_range: range
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        将区块权重转换为号码权重
        """
        # 前区区块映射
        front_block_map = {
            "1-5": list(range(1, 6)),
            "6-10": list(range(6, 11)),
            "11-15": list(range(11, 16)),
            "16-20": list(range(16, 21)),
            "21-25": list(range(21, 26)),
            "26-30": list(range(26, 31)),
            "31-35": list(range(31, 36))
        }
        
        # 后区区块映射
        back_block_map = {
            "1-2": [1, 2],
            "3-4": [3, 4],
            "5-6": [5, 6],
            "7-8": [7, 8],
            "9-10": [9, 10],
            "11-12": [11, 12]
        }
        
        front_number_weights = {n: 0.0 for n in front_range}
        back_number_weights = {n: 0.0 for n in back_range}
        
        # 转换前区权重
        for block, weight in front_weights.items():
            if block in front_block_map:
                numbers = front_block_map[block]
                weight_per_number = weight / len(numbers)
                for num in numbers:
                    if num in front_range:
                        front_number_weights[num] += weight_per_number
        
        # 转换后区权重
        for block, weight in back_weights.items():
            if block in back_block_map:
                numbers = back_block_map[block]
                weight_per_number = weight / len(numbers)
                for num in numbers:
                    if num in back_range:
                        back_number_weights[num] += weight_per_number
        
        return front_number_weights, back_number_weights
    
    def _get_markov_weights(
        self,
        historical_data: pd.DataFrame,
        front_range: range,
        back_range: range
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        从马尔可夫链模型获取权重
        """
        # 验证历史数据
        if historical_data is None or historical_data.empty:
            # 回退到均匀分布
            front_avg = {n: 1.0 / len(front_range) for n in front_range}
            back_avg = {n: 1.0 / len(back_range) for n in back_range}
            return front_avg, back_avg
        
        # 使用多个模型的平均预测
        front_probs_sum = {n: 0.0 for n in front_range}
        back_probs_sum = {n: 0.0 for n in back_range}
        model_count = 0
        
        for name, model in self.markov_models.items():
            try:
                # 确保有足够的历史数据
                required_length = max(2, getattr(model, 'order', 2))
                if len(historical_data) < required_length:
                    continue
                    
                front_probs, back_probs = model.predict_probabilities(
                    historical_data.tail(required_length), front_range, back_range
                )
                
                for n in front_range:
                    front_probs_sum[n] += front_probs.get(n, 0)
                for n in back_range:
                    back_probs_sum[n] += back_probs.get(n, 0)
                
                model_count += 1
            except Exception as e:
                print(f"模型 {name} 预测失败: {e}")
                continue
        
        # 计算平均概率
        if model_count > 0:
            front_avg = {n: prob / model_count for n, prob in front_probs_sum.items()}
            back_avg = {n: prob / model_count for n, prob in back_probs_sum.items()}
        else:
            # 回退到均匀分布
            front_avg = {n: 1.0 / len(front_range) for n in front_range}
            back_avg = {n: 1.0 / len(back_range) for n in back_range}
        
        return front_avg, back_avg
    
    def _get_big_data_weights(
        self,
        front_range: range,
        back_range: range
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        从大数据分析获取权重
        """
        front_weights = {n: 1.0 / len(front_range) for n in front_range}  # 默认均匀分布
        back_weights = {n: 1.0 / len(back_range) for n in back_range}
        
        if not hasattr(self, 'big_data_insights'):
            return front_weights, back_weights
        
        insights = self.big_data_insights
        
        # 1. 基于热门号码调整权重
        if 'hot_numbers' in insights:
            hot_front = insights['hot_numbers'].get('front', [])
            hot_back = insights['hot_numbers'].get('back', [])
            
            # 给热门号码更高权重
            for num in hot_front:
                if num in front_weights:
                    front_weights[num] *= 1.2
            
            for num in hot_back:
                if num in back_weights:
                    back_weights[num] *= 1.2
        
        # 2. 基于时间模式调整权重
        current_weekday = datetime.now().weekday()
        current_month = datetime.now().month
        
        if 'weekday_recommendations' in insights:
            weekday_front = insights['weekday_recommendations'].get('front', [])
            weekday_back = insights['weekday_recommendations'].get('back', [])
            
            for num in weekday_front[:5]:  # 取前5个推荐
                if num in front_weights:
                    front_weights[num] *= 1.1
            
            for num in weekday_back[:3]:  # 取前3个推荐
                if num in back_weights:
                    back_weights[num] *= 1.1
        
        if 'month_recommendations' in insights:
            month_front = insights['month_recommendations'].get('front', [])
            month_back = insights['month_recommendations'].get('back', [])
            
            for num in month_front[:5]:
                if num in front_weights:
                    front_weights[num] *= 1.1
            
            for num in month_back[:3]:
                if num in back_weights:
                    back_weights[num] *= 1.1
        
        # 3. 基于冷门号码调整（适度降低权重，但不完全排除）
        if 'cold_numbers' in insights:
            cold_front = insights['cold_numbers'].get('front', [])
            cold_back = insights['cold_numbers'].get('back', [])
            
            for num in cold_front[-5:]:  # 最冷的5个
                if num in front_weights:
                    front_weights[num] *= 0.8
            
            for num in cold_back[-3:]:  # 最冷的3个
                if num in back_weights:
                    back_weights[num] *= 0.8
        
        # 归一化权重
        front_total = sum(front_weights.values())
        back_total = sum(back_weights.values())
        
        if front_total > 0:
            front_weights = {n: w / front_total for n, w in front_weights.items()}
        if back_total > 0:
            back_weights = {n: w / back_total for n, w in back_weights.items()}
        
        return front_weights, back_weights
    
    def _generate_with_enhanced_weights(
        self,
        count: int,
        rules: Optional[Dict],
        front_blocks: Optional[Dict[str, List[int]]],
        back_blocks: Optional[Dict[str, List[int]]],
        enhanced_front_weights: Dict[int, float],
        enhanced_back_weights: Dict[int, float],
        selected_front_blocks: Optional[List[str]],
        selected_back_blocks: Optional[List[str]],
        historical_data: Optional[pd.DataFrame]
    ) -> List[Dict]:
        """
        使用增强权重生成号码
        """
        rng = random.Random()
        rules = rules or {}
        results = []
        
        # 基础规则配置
        sum_range = rules.get("sum_front_range", [None, None])
        min_odd_count = rules.get("odd_even_front", [0])[0]
        max_odd_count = rules.get("odd_even_front", [0, 5])[1]
        cons_req = rules.get("consecutive_count", 0)
        cons_mode = rules.get("consecutive_mode", "exact")
        cons_check_type = rules.get("consecutive_check_type", "groups")
        
        # 排除规则
        front_include = set(rules.get("front_include", []))
        front_exclude = set(rules.get("front_exclude", []))
        back_include = set(rules.get("back_include", []))
        back_exclude = set(rules.get("back_exclude", []))
        
        max_tries = 50000  # 增加最大尝试次数
        tries = 0
        
        print(f"开始生成 {count} 个候选号码...")
        
        while len(results) < count and tries < max_tries:
            tries += 1
            
            if tries % 5000 == 0:
                print(f"已尝试 {tries} 次，生成了 {len(results)} 个候选号码")
            
            # 生成前区号码（基于增强权重）
            front_candidates = [n for n in range(1, 36) if n not in front_exclude]
            front_weights_list = [enhanced_front_weights.get(n, 0) for n in front_candidates]
            
            # 确保权重总和大于0
            if sum(front_weights_list) == 0:
                front_weights_list = [1.0] * len(front_candidates)
            
            try:
                f_selected = rng.choices(front_candidates, weights=front_weights_list, k=5)
                f_selected = sorted(list(set(f_selected)))  # 去重并排序
                
                # 如果去重后数量不足，补充
                while len(f_selected) < 5:
                    remaining = [n for n in front_candidates if n not in f_selected]
                    if not remaining:
                        break
                    remaining_weights = [enhanced_front_weights.get(n, 0) for n in remaining]
                    if sum(remaining_weights) == 0:
                        remaining_weights = [1.0] * len(remaining)
                    additional = rng.choices(remaining, weights=remaining_weights, k=1)[0]
                    f_selected.append(additional)
                    f_selected = sorted(f_selected)
                
                if len(f_selected) < 5:
                    continue
                    
            except Exception:
                continue
            
            # 生成后区号码（基于增强权重）
            back_candidates = [n for n in range(1, 13) if n not in back_exclude]
            back_weights_list = [enhanced_back_weights.get(n, 0) for n in back_candidates]
            
            if sum(back_weights_list) == 0:
                back_weights_list = [1.0] * len(back_candidates)
            
            try:
                b_selected = rng.choices(back_candidates, weights=back_weights_list, k=2)
                b_selected = sorted(list(set(b_selected)))
                
                # 如果去重后数量不足，补充
                while len(b_selected) < 2:
                    remaining = [n for n in back_candidates if n not in b_selected]
                    if not remaining:
                        break
                    remaining_weights = [enhanced_back_weights.get(n, 0) for n in remaining]
                    if sum(remaining_weights) == 0:
                        remaining_weights = [1.0] * len(remaining)
                    additional = rng.choices(remaining, weights=remaining_weights, k=1)[0]
                    b_selected.append(additional)
                    b_selected = sorted(b_selected)
                
                if len(b_selected) < 2:
                    continue
                    
            except Exception:
                continue
            
            # 应用规则检查
            ok = True
            
            # 包含检查
            if front_include and not front_include.issubset(f_selected):
                ok = False
            if back_include and not back_include.issubset(b_selected):
                ok = False
            
            # 和值检查
            s = sum(f_selected)
            smin, smax = sum_range
            if (smin is not None and s < smin) or (smax is not None and s > smax):
                ok = False
            
            # 奇偶检查
            odd_count = sum(1 for n in f_selected if n % 2 == 1)
            if odd_count < min_odd_count or (max_odd_count is not None and odd_count > max_odd_count):
                ok = False
            
            # 连号检查
            if cons_check_type == "groups":
                cons_count = consecutive_groups_count(f_selected)
            else:
                cons_count = consecutive_pairs_count(f_selected)
            
            if cons_mode == "exact" and cons_count != cons_req:
                ok = False
            elif cons_mode == "min" and cons_count < cons_req:
                ok = False
            elif cons_mode == "max" and cons_count > cons_req:
                ok = False
            
            if ok:
                results.append({
                    "front": f_selected,
                    "back": b_selected,
                    "generation_method": "enhanced",
                    "markov_confidence": self._calculate_markov_confidence(f_selected, b_selected, historical_data),
                    "big_data_score": self._calculate_big_data_score(f_selected, b_selected)
                })
        
        # 如果生成的候选号码不足，使用宽松规则补充
        if len(results) < count:
            print(f"警告：严格规则只生成了 {len(results)} 个候选号码，使用宽松规则补充到 {count} 个")
            
            # 使用更宽松的规则补充
            relaxed_tries = 0
            max_relaxed_tries = 20000
            
            while len(results) < count and relaxed_tries < max_relaxed_tries:
                relaxed_tries += 1
                
                # 生成前区号码（使用更宽松的规则）
                front_candidates = [n for n in range(1, 36) if n not in front_exclude]
                front_weights_list = [enhanced_front_weights.get(n, 0) for n in front_candidates]
                
                if sum(front_weights_list) == 0:
                    front_weights_list = [1.0] * len(front_candidates)
                
                try:
                    f_selected = rng.choices(front_candidates, weights=front_weights_list, k=5)
                    f_selected = sorted(list(set(f_selected)))
                    
                    while len(f_selected) < 5:
                        remaining = [n for n in front_candidates if n not in f_selected]
                        if not remaining:
                            break
                        additional = rng.choice(remaining)
                        f_selected.append(additional)
                        f_selected = sorted(f_selected)
                    
                    if len(f_selected) < 5:
                        continue
                        
                except Exception:
                    continue
                
                # 生成后区号码
                back_candidates = [n for n in range(1, 13) if n not in back_exclude]
                back_weights_list = [enhanced_back_weights.get(n, 0) for n in back_candidates]
                
                if sum(back_weights_list) == 0:
                    back_weights_list = [1.0] * len(back_candidates)
                
                try:
                    b_selected = rng.choices(back_candidates, weights=back_weights_list, k=2)
                    b_selected = sorted(list(set(b_selected)))
                    
                    while len(b_selected) < 2:
                        remaining = [n for n in back_candidates if n not in b_selected]
                        if not remaining:
                            break
                        additional = rng.choice(remaining)
                        b_selected.append(additional)
                        b_selected = sorted(b_selected)
                    
                    if len(b_selected) < 2:
                        continue
                        
                except Exception:
                    continue
                
                # 只检查基本规则（包含、排除、和值）
                ok = True
                
                # 包含检查
                if front_include and not front_include.issubset(f_selected):
                    ok = False
                if back_include and not back_include.issubset(b_selected):
                    ok = False
                
                # 和值检查（放宽范围）
                s = sum(f_selected)
                smin, smax = sum_range
                if smin is not None and smax is not None:
                    # 放宽和值范围 ±10
                    relaxed_min = max(15, smin - 10)
                    relaxed_max = min(175, smax + 10)
                    if s < relaxed_min or s > relaxed_max:
                        ok = False
                elif (smin is not None and s < smin - 10) or (smax is not None and s > smax + 10):
                    ok = False
                
                if ok:
                    results.append({
                        "front": f_selected,
                        "back": b_selected,
                        "generation_method": "enhanced_relaxed",
                        "markov_confidence": self._calculate_markov_confidence(f_selected, b_selected, historical_data),
                        "big_data_score": self._calculate_big_data_score(f_selected, b_selected)
                    })
        
        print(f"最终生成了 {len(results)} 个候选号码")
        return results
    
    def _calculate_markov_confidence(self, front_nums: List[int], back_nums: List[int], 
                                   historical_data: Optional[pd.DataFrame]) -> float:
        """
        计算马尔可夫链预测的置信度
        """
        if historical_data is None or historical_data.empty or len(historical_data) < 2:
            return 0.5
        
        try:
            # 使用最佳模型计算置信度
            model = self.markov_models.get('number_order2')
            if not model:
                return 0.5
            
            # 确保有足够的数据
            required_length = max(2, getattr(model, 'order', 2))
            if len(historical_data) < required_length:
                return 0.5
            
            front_probs, back_probs = model.predict_probabilities(
                historical_data.tail(required_length), range(1, 36), range(1, 13)
            )
            
            # 计算选中号码的概率
            front_prob = sum(front_probs.get(n, 0) for n in front_nums) / len(front_nums)
            back_prob = sum(back_probs.get(n, 0) for n in back_nums) / len(back_nums)
            
            return (front_prob + back_prob) / 2
        except Exception as e:
            print(f"马尔可夫置信度计算失败: {e}")
            return 0.5
    
    def _calculate_big_data_score(self, front_nums: List[int], back_nums: List[int]) -> float:
        """
        计算大数据分析评分
        """
        if not hasattr(self, 'big_data_insights'):
            return 0.5
        
        score = 0.0
        factors = 0
        
        insights = self.big_data_insights
        
        # 热门号码加分
        if 'hot_numbers' in insights:
            hot_front = set(insights['hot_numbers'].get('front', []))
            hot_back = set(insights['hot_numbers'].get('back', []))
            
            front_hot_count = len(set(front_nums) & hot_front)
            back_hot_count = len(set(back_nums) & hot_back)
            
            score += (front_hot_count / len(front_nums)) * 0.3
            score += (back_hot_count / len(back_nums)) * 0.3
            factors += 0.6
        
        # 时间相关推荐加分
        if 'weekday_recommendations' in insights:
            weekday_front = set(insights['weekday_recommendations'].get('front', [])[:10])
            weekday_back = set(insights['weekday_recommendations'].get('back', [])[:5])
            
            front_weekday_count = len(set(front_nums) & weekday_front)
            back_weekday_count = len(set(back_nums) & weekday_back)
            
            score += (front_weekday_count / len(front_nums)) * 0.2
            score += (back_weekday_count / len(back_nums)) * 0.2
            factors += 0.4
        
        return score / factors if factors > 0 else 0.5
    
    def get_model_status(self) -> Dict:
        """
        获取模型状态信息
        """
        status = {
            'models_initialized': len(self.markov_models) > 0,
            'markov_models': list(self.markov_models.keys()),
            'big_data_analyzed': hasattr(self, 'big_data_insights'),
            'analysis_cache_size': len(self.analysis_cache)
        }
        
        if hasattr(self, 'big_data_insights'):
            insights = self.big_data_insights
            status['insights_summary'] = {
                'hot_front_numbers': insights.get('hot_numbers', {}).get('front', [])[:5],
                'hot_back_numbers': insights.get('hot_numbers', {}).get('back', [])[:3],
                'has_weekday_recommendations': 'weekday_recommendations' in insights,
                'has_month_recommendations': 'month_recommendations' in insights
            }
        
        return status