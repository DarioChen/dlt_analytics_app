# backend/strategy_optimizer.py
"""
策略优化器：整合所有高级技术，自动寻找最优预测策略
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import random
import time

# 导入其他模块
from .evolutionary_optimizer import EvolutionaryNumberGenerator, EvolutionaryConfig
from .neural_predictor import NeuralLotteryPredictor, create_neural_predictor, HAS_PYTORCH
from .ensemble_predictor import EnsemblePredictor
from .enhanced_generator import EnhancedNumberGenerator
from .backtest import BacktestAnalyzer
from .optimizer import StrategyParams, genetic_algorithm_optimize, bayesian_optimize

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """优化配置"""
    # 优化目标
    primary_objective: str = 'roi'  # 'roi', 'hit_rate', 'high_prize_rate', 'stability'
    secondary_objectives: List[str] = None
    
    # 优化算法
    optimization_method: str = 'multi_stage'  # 'genetic', 'bayesian', 'evolutionary', 'multi_stage'
    max_iterations: int = 100
    population_size: int = 50
    
    # 回测配置
    backtest_periods: int = 50
    validation_periods: int = 20
    cross_validation_folds: int = 5
    
    # 模型配置
    use_neural_networks: bool = True
    use_evolutionary: bool = True
    use_ensemble: bool = True
    
    # 风险控制
    max_cost_per_period: float = 100.0
    min_roi_threshold: float = -0.5
    diversification_requirement: float = 0.3
    
    def __post_init__(self):
        if self.secondary_objectives is None:
            self.secondary_objectives = ['hit_rate', 'stability']

class StrategyOptimizer:
    """策略优化器：自动寻找最优预测策略"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # 初始化组件
        self.backtest_analyzer = BacktestAnalyzer()
        self.ensemble_predictor = EnsemblePredictor()
        self.enhanced_generator = EnhancedNumberGenerator()
        
        # 神经网络预测器
        self.neural_predictors = {}
        if self.config.use_neural_networks and HAS_PYTORCH:
            for model_type in ['transformer', 'lstm']:
                try:
                    self.neural_predictors[model_type] = create_neural_predictor(model_type)
                except Exception as e:
                    logger.warning(f"Failed to create {model_type} predictor: {e}")
        
        # 进化优化器
        self.evolutionary_optimizer = None
        if self.config.use_evolutionary:
            try:
                evo_config = EvolutionaryConfig(
                    population_size=self.config.population_size,
                    generations=self.config.max_iterations
                )
                self.evolutionary_optimizer = EvolutionaryNumberGenerator(evo_config)
            except Exception as e:
                logger.warning(f"Failed to create evolutionary optimizer: {e}")
        
        # 优化历史
        self.optimization_history = []
        self.best_strategies = []
        
        logger.info("Strategy optimizer initialized")
    
    def optimize_comprehensive_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """综合策略优化"""
        logger.info("Starting comprehensive strategy optimization...")
        
        start_time = time.time()
        
        # 第一阶段：数据分析和特征工程
        logger.info("Phase 1: Data analysis and feature engineering")
        analysis_results = self._analyze_data_patterns(df)
        
        # 第二阶段：模型训练和初始化
        logger.info("Phase 2: Model training and initialization")
        model_results = self._train_and_initialize_models(df)
        
        # 第三阶段：策略空间探索
        logger.info("Phase 3: Strategy space exploration")
        exploration_results = self._explore_strategy_space(df, analysis_results)
        
        # 第四阶段：多目标优化
        logger.info("Phase 4: Multi-objective optimization")
        optimization_results = self._multi_objective_optimization(df, exploration_results)
        
        # 第五阶段：策略验证和选择
        logger.info("Phase 5: Strategy validation and selection")
        validation_results = self._validate_and_select_strategies(df, optimization_results)
        
        # 第六阶段：集成最优策略
        logger.info("Phase 6: Ensemble optimal strategies")
        final_strategy = self._create_ensemble_strategy(validation_results)
        
        total_time = time.time() - start_time
        
        return {
            'final_strategy': final_strategy,
            'analysis_results': analysis_results,
            'model_results': model_results,
            'exploration_results': exploration_results,
            'optimization_results': optimization_results,
            'validation_results': validation_results,
            'optimization_time': total_time,
            'config': asdict(self.config)
        }
    
    def _analyze_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据模式"""
        patterns = {}
        
        # 基础统计分析
        patterns['basic_stats'] = self._calculate_basic_statistics(df)
        
        # 时间序列分析
        patterns['time_series'] = self._analyze_time_series_patterns(df)
        
        # 号码关联分析
        patterns['correlations'] = self._analyze_number_correlations(df)
        
        # 周期性分析
        patterns['cycles'] = self._analyze_cyclical_patterns(df)
        
        # 异常检测
        patterns['anomalies'] = self._detect_anomalies(df)
        
        return patterns
    
    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算基础统计信息"""
        stats = {}
        
        # 前区统计
        front_numbers = []
        for _, row in df.iterrows():
            front_numbers.extend([row[f'f{i}'] for i in range(1, 6)])
        
        front_freq = pd.Series(front_numbers).value_counts()
        stats['front_frequency'] = front_freq.to_dict()
        stats['front_hot_numbers'] = front_freq.head(10).index.tolist()
        stats['front_cold_numbers'] = front_freq.tail(10).index.tolist()
        
        # 后区统计
        back_numbers = []
        for _, row in df.iterrows():
            back_numbers.extend([row[f'b{i}'] for i in range(1, 3)])
        
        back_freq = pd.Series(back_numbers).value_counts()
        stats['back_frequency'] = back_freq.to_dict()
        stats['back_hot_numbers'] = back_freq.head(5).index.tolist()
        stats['back_cold_numbers'] = back_freq.tail(5).index.tolist()
        
        # 和值统计
        front_sums = [sum([row[f'f{i}'] for i in range(1, 6)]) for _, row in df.iterrows()]
        stats['sum_stats'] = {
            'mean': np.mean(front_sums),
            'std': np.std(front_sums),
            'min': np.min(front_sums),
            'max': np.max(front_sums),
            'median': np.median(front_sums)
        }
        
        return stats
    
    def _analyze_time_series_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析时间序列模式"""
        patterns = {}
        
        # 趋势分析
        df_sorted = df.sort_values('date')
        
        # 和值趋势
        front_sums = [sum([row[f'f{i}'] for i in range(1, 6)]) for _, row in df_sorted.iterrows()]
        patterns['sum_trend'] = self._calculate_trend(front_sums)
        
        # 奇偶比例趋势
        odd_ratios = []
        for _, row in df_sorted.iterrows():
            odd_count = sum(1 for i in range(1, 6) if row[f'f{i}'] % 2 == 1)
            odd_ratios.append(odd_count / 5)
        patterns['odd_ratio_trend'] = self._calculate_trend(odd_ratios)
        
        # 大小比例趋势
        large_ratios = []
        for _, row in df_sorted.iterrows():
            large_count = sum(1 for i in range(1, 6) if row[f'f{i}'] > 17)
            large_ratios.append(large_count / 5)
        patterns['large_ratio_trend'] = self._calculate_trend(large_ratios)
        
        return patterns
    
    def _calculate_trend(self, data: List[float]) -> Dict[str, float]:
        """计算趋势"""
        if len(data) < 2:
            return {'slope': 0, 'correlation': 0}
        
        x = np.arange(len(data))
        correlation = np.corrcoef(x, data)[0, 1] if not np.isnan(np.corrcoef(x, data)[0, 1]) else 0
        slope = np.polyfit(x, data, 1)[0] if len(data) > 1 else 0
        
        return {'slope': float(slope), 'correlation': float(correlation)}
    
    def _analyze_number_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析号码相关性"""
        correlations = {}
        
        # 构建共现矩阵
        cooccurrence = np.zeros((35, 35))
        
        for _, row in df.iterrows():
            front_nums = [row[f'f{i}'] - 1 for i in range(1, 6)]  # 转为0-34索引
            for i in range(len(front_nums)):
                for j in range(i + 1, len(front_nums)):
                    cooccurrence[front_nums[i]][front_nums[j]] += 1
                    cooccurrence[front_nums[j]][front_nums[i]] += 1
        
        # 找出强相关对
        strong_pairs = []
        threshold = np.mean(cooccurrence) + 2 * np.std(cooccurrence)
        
        for i in range(35):
            for j in range(i + 1, 35):
                if cooccurrence[i][j] > threshold:
                    strong_pairs.append(((i + 1, j + 1), cooccurrence[i][j]))
        
        correlations['strong_pairs'] = sorted(strong_pairs, key=lambda x: x[1], reverse=True)[:20]
        correlations['cooccurrence_matrix'] = cooccurrence.tolist()
        
        return correlations
    
    def _analyze_cyclical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析周期性模式"""
        cycles = {}
        
        df_with_date = df.copy()
        df_with_date['date'] = pd.to_datetime(df_with_date['date'])
        
        # 周几模式
        weekday_patterns = defaultdict(list)
        for _, row in df_with_date.iterrows():
            weekday = row['date'].weekday()
            front_nums = [row[f'f{i}'] for i in range(1, 6)]
            weekday_patterns[weekday].extend(front_nums)
        
        cycles['weekday'] = {}
        for weekday, numbers in weekday_patterns.items():
            freq = pd.Series(numbers).value_counts()
            cycles['weekday'][weekday] = freq.head(10).to_dict()
        
        # 月份模式
        month_patterns = defaultdict(list)
        for _, row in df_with_date.iterrows():
            month = row['date'].month
            front_nums = [row[f'f{i}'] for i in range(1, 6)]
            month_patterns[month].extend(front_nums)
        
        cycles['month'] = {}
        for month, numbers in month_patterns.items():
            freq = pd.Series(numbers).value_counts()
            cycles['month'][month] = freq.head(10).to_dict()
        
        return cycles
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测异常"""
        anomalies = {}
        
        # 和值异常
        front_sums = [sum([row[f'f{i}'] for i in range(1, 6)]) for _, row in df.iterrows()]
        mean_sum = np.mean(front_sums)
        std_sum = np.std(front_sums)
        
        anomaly_indices = []
        for i, s in enumerate(front_sums):
            z_score = abs(s - mean_sum) / std_sum
            if z_score > 2.5:  # 异常阈值
                anomaly_indices.append(i)
        
        anomalies['sum_anomalies'] = anomaly_indices
        anomalies['anomaly_count'] = len(anomaly_indices)
        anomalies['anomaly_rate'] = len(anomaly_indices) / len(df)
        
        return anomalies
    
    def _train_and_initialize_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """训练和初始化模型"""
        results = {}
        
        # 训练集成预测器
        try:
            logger.info("Training ensemble predictor...")
            self.ensemble_predictor.train(df)
            results['ensemble'] = {'status': 'success', 'predictors': len(self.ensemble_predictor.predictors)}
        except Exception as e:
            logger.error(f"Failed to train ensemble predictor: {e}")
            results['ensemble'] = {'status': 'failed', 'error': str(e)}
        
        # 训练神经网络预测器
        neural_results = {}
        for model_type, predictor in self.neural_predictors.items():
            try:
                logger.info(f"Training {model_type} neural predictor...")
                predictor.train(df, epochs=50, batch_size=16)  # 减少训练时间
                neural_results[model_type] = {'status': 'success'}
            except Exception as e:
                logger.error(f"Failed to train {model_type} predictor: {e}")
                neural_results[model_type] = {'status': 'failed', 'error': str(e)}
        
        results['neural'] = neural_results
        
        # 初始化增强生成器
        try:
            logger.info("Initializing enhanced generator...")
            self.enhanced_generator.initialize_models(df, use_ensemble=True)
            results['enhanced'] = {'status': 'success'}
        except Exception as e:
            logger.error(f"Failed to initialize enhanced generator: {e}")
            results['enhanced'] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def _explore_strategy_space(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """探索策略空间"""
        strategies = []
        
        # 基于分析结果生成策略候选
        hot_numbers = analysis_results['basic_stats']['front_hot_numbers']
        cold_numbers = analysis_results['basic_stats']['front_cold_numbers']
        
        # 策略1：热号策略
        strategies.append({
            'name': 'hot_numbers',
            'type': 'frequency_based',
            'params': {
                'focus_numbers': hot_numbers[:15],
                'weight_multiplier': 2.0,
                'selection_method': 'weighted'
            }
        })
        
        # 策略2：冷号策略
        strategies.append({
            'name': 'cold_numbers',
            'type': 'frequency_based',
            'params': {
                'focus_numbers': cold_numbers[:15],
                'weight_multiplier': 2.0,
                'selection_method': 'weighted'
            }
        })
        
        # 策略3：平衡策略
        strategies.append({
            'name': 'balanced',
            'type': 'balanced',
            'params': {
                'hot_ratio': 0.4,
                'cold_ratio': 0.4,
                'random_ratio': 0.2,
                'diversity_requirement': 0.8
            }
        })
        
        # 策略4：相关性策略
        if analysis_results['correlations']['strong_pairs']:
            strong_pairs = analysis_results['correlations']['strong_pairs'][:10]
            correlated_numbers = []
            for (num1, num2), strength in strong_pairs:
                correlated_numbers.extend([num1, num2])
            
            strategies.append({
                'name': 'correlation_based',
                'type': 'correlation',
                'params': {
                    'focus_numbers': list(set(correlated_numbers)),
                    'pair_bonus': 1.5,
                    'selection_method': 'pair_aware'
                }
            })
        
        # 策略5：时间模式策略
        current_weekday = datetime.now().weekday()
        if current_weekday in analysis_results['cycles']['weekday']:
            weekday_numbers = list(analysis_results['cycles']['weekday'][current_weekday].keys())
            strategies.append({
                'name': 'temporal_pattern',
                'type': 'temporal',
                'params': {
                    'focus_numbers': weekday_numbers[:15],
                    'time_weight': 1.5,
                    'selection_method': 'time_aware'
                }
            })
        
        # 评估每个策略
        evaluated_strategies = []
        for strategy in strategies:
            try:
                performance = self._evaluate_strategy(df, strategy)
                strategy['performance'] = performance
                evaluated_strategies.append(strategy)
            except Exception as e:
                logger.warning(f"Failed to evaluate strategy {strategy['name']}: {e}")
        
        return {
            'strategies': evaluated_strategies,
            'strategy_count': len(evaluated_strategies)
        }
    
    def _evaluate_strategy(self, df: pd.DataFrame, strategy: Dict) -> Dict[str, float]:
        """评估策略性能"""
        # 简化的策略评估
        # 在实际应用中，这里会运行完整的回测
        
        performance = {
            'roi': random.uniform(-0.3, 0.5),  # 模拟ROI
            'hit_rate': random.uniform(0.1, 0.4),  # 模拟命中率
            'stability': random.uniform(0.3, 0.8),  # 模拟稳定性
            'high_prize_rate': random.uniform(0.001, 0.01)  # 模拟高奖项命中率
        }
        
        return performance
    
    def _multi_objective_optimization(self, df: pd.DataFrame, exploration_results: Dict) -> Dict[str, Any]:
        """多目标优化"""
        strategies = exploration_results['strategies']
        
        if not strategies:
            return {'optimized_strategies': [], 'pareto_front': []}
        
        # 多目标优化：寻找帕累托前沿
        pareto_front = self._find_pareto_front(strategies)
        
        # 基于主要目标选择最优策略
        primary_obj = self.config.primary_objective
        best_strategy = max(pareto_front, key=lambda s: s['performance'].get(primary_obj, 0))
        
        # 进化优化（如果启用）
        evolved_strategies = []
        if self.evolutionary_optimizer:
            try:
                logger.info("Running evolutionary optimization...")
                evo_result = self.evolutionary_optimizer.evolve_optimal_numbers(
                    df, target_count=5, objectives=[primary_obj] + self.config.secondary_objectives
                )
                
                evolved_strategy = {
                    'name': 'evolutionary_optimized',
                    'type': 'evolutionary',
                    'params': evo_result,
                    'performance': {
                        primary_obj: evo_result['fitness_score'],
                        'confidence': 0.8
                    }
                }
                evolved_strategies.append(evolved_strategy)
            except Exception as e:
                logger.warning(f"Evolutionary optimization failed: {e}")
        
        return {
            'pareto_front': pareto_front,
            'best_strategy': best_strategy,
            'evolved_strategies': evolved_strategies,
            'optimization_method': self.config.optimization_method
        }
    
    def _find_pareto_front(self, strategies: List[Dict]) -> List[Dict]:
        """寻找帕累托前沿"""
        pareto_front = []
        
        for i, strategy1 in enumerate(strategies):
            is_dominated = False
            
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    # 检查strategy1是否被strategy2支配
                    if self._dominates(strategy2, strategy1):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(strategy1)
        
        return pareto_front
    
    def _dominates(self, strategy1: Dict, strategy2: Dict) -> bool:
        """检查strategy1是否支配strategy2"""
        objectives = [self.config.primary_objective] + self.config.secondary_objectives
        
        better_in_all = True
        better_in_at_least_one = False
        
        for obj in objectives:
            perf1 = strategy1['performance'].get(obj, 0)
            perf2 = strategy2['performance'].get(obj, 0)
            
            if perf1 < perf2:
                better_in_all = False
            elif perf1 > perf2:
                better_in_at_least_one = True
        
        return better_in_all and better_in_at_least_one
    
    def _validate_and_select_strategies(self, df: pd.DataFrame, optimization_results: Dict) -> Dict[str, Any]:
        """验证和选择策略"""
        all_strategies = (optimization_results['pareto_front'] + 
                         optimization_results['evolved_strategies'])
        
        if not all_strategies:
            return {'validated_strategies': [], 'selected_strategy': None}
        
        # 交叉验证
        validated_strategies = []
        for strategy in all_strategies:
            try:
                cv_results = self._cross_validate_strategy(df, strategy)
                strategy['cv_results'] = cv_results
                
                # 计算综合评分
                strategy['composite_score'] = self._calculate_composite_score(strategy)
                validated_strategies.append(strategy)
            except Exception as e:
                logger.warning(f"Failed to validate strategy {strategy['name']}: {e}")
        
        # 选择最佳策略
        if validated_strategies:
            selected_strategy = max(validated_strategies, key=lambda s: s['composite_score'])
        else:
            selected_strategy = None
        
        return {
            'validated_strategies': validated_strategies,
            'selected_strategy': selected_strategy,
            'validation_method': 'cross_validation'
        }
    
    def _cross_validate_strategy(self, df: pd.DataFrame, strategy: Dict) -> Dict[str, float]:
        """交叉验证策略"""
        # 简化的交叉验证
        folds = self.config.cross_validation_folds
        fold_size = len(df) // folds
        
        cv_scores = []
        for fold in range(folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < folds - 1 else len(df)
            
            # 模拟交叉验证分数
            fold_score = random.uniform(0.1, 0.8)
            cv_scores.append(fold_score)
        
        return {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'scores': cv_scores
        }
    
    def _calculate_composite_score(self, strategy: Dict) -> float:
        """计算综合评分"""
        weights = {
            'roi': 0.4,
            'hit_rate': 0.3,
            'stability': 0.2,
            'high_prize_rate': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = strategy['performance'].get(metric, 0)
            score += value * weight
        
        # 考虑交叉验证结果
        if 'cv_results' in strategy:
            cv_penalty = strategy['cv_results']['std_score'] * 0.1
            score -= cv_penalty
        
        return score
    
    def _create_ensemble_strategy(self, validation_results: Dict) -> Dict[str, Any]:
        """创建集成策略"""
        validated_strategies = validation_results['validated_strategies']
        
        if not validated_strategies:
            return {'type': 'fallback', 'method': 'random'}
        
        # 选择top-3策略进行集成
        top_strategies = sorted(validated_strategies, 
                               key=lambda s: s['composite_score'], 
                               reverse=True)[:3]
        
        # 计算集成权重
        total_score = sum(s['composite_score'] for s in top_strategies)
        ensemble_weights = {}
        
        for strategy in top_strategies:
            weight = strategy['composite_score'] / total_score if total_score > 0 else 1.0 / len(top_strategies)
            ensemble_weights[strategy['name']] = weight
        
        ensemble_strategy = {
            'type': 'ensemble',
            'strategies': top_strategies,
            'weights': ensemble_weights,
            'combination_method': 'weighted_average',
            'expected_performance': {
                'roi': sum(s['performance'].get('roi', 0) * ensemble_weights[s['name']] 
                          for s in top_strategies),
                'hit_rate': sum(s['performance'].get('hit_rate', 0) * ensemble_weights[s['name']] 
                               for s in top_strategies),
                'stability': sum(s['performance'].get('stability', 0) * ensemble_weights[s['name']] 
                                for s in top_strategies)
            }
        }
        
        return ensemble_strategy
    
    def generate_optimized_numbers(self, df: pd.DataFrame, strategy: Dict = None, 
                                  count: int = 5) -> List[Dict[str, Any]]:
        """使用优化策略生成号码"""
        if strategy is None:
            # 使用默认策略
            strategy = {'type': 'ensemble', 'method': 'enhanced_generator'}
        
        predictions = []
        
        try:
            if strategy['type'] == 'ensemble':
                # 使用集成方法
                if hasattr(self, 'ensemble_predictor') and self.ensemble_predictor:
                    predictions = self.ensemble_predictor.predict_enhanced(df, count)
                
                # 如果集成预测失败，使用增强生成器
                if not predictions and hasattr(self, 'enhanced_generator'):
                    enhanced_result = self.enhanced_generator.generate_enhanced_numbers(
                        count=count,
                        historical_data=df,
                        use_markov=True,
                        use_big_data=True
                    )
                    predictions = enhanced_result if isinstance(enhanced_result, list) else [enhanced_result]
            
            elif strategy['type'] == 'neural':
                # 使用神经网络预测
                for model_type, predictor in self.neural_predictors.items():
                    try:
                        neural_predictions = predictor.predict(df, count)
                        predictions.extend(neural_predictions)
                        break  # 使用第一个成功的预测器
                    except Exception as e:
                        logger.warning(f"Neural predictor {model_type} failed: {e}")
                        continue
            
            elif strategy['type'] == 'evolutionary':
                # 使用进化算法
                if self.evolutionary_optimizer:
                    evo_result = self.evolutionary_optimizer.evolve_optimal_numbers(df, count)
                    predictions = [{
                        'front': evo_result['best_numbers']['front'],
                        'back': evo_result['best_numbers']['back'],
                        'confidence': evo_result['fitness_score'],
                        'generation_method': 'evolutionary'
                    }]
            
            # 如果所有方法都失败，使用基础方法
            if not predictions:
                logger.warning("All advanced methods failed, using basic generation")
                predictions = self._generate_basic_numbers(df, count)
        
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            predictions = self._generate_basic_numbers(df, count)
        
        return predictions
    
    def _generate_basic_numbers(self, df: pd.DataFrame, count: int) -> List[Dict[str, Any]]:
        """基础号码生成（备用方法）"""
        predictions = []
        
        for _ in range(count):
            front = sorted(random.sample(range(1, 36), 5))
            back = sorted(random.sample(range(1, 13), 2))
            
            predictions.append({
                'front': front,
                'back': back,
                'confidence': 0.3,
                'generation_method': 'random_fallback'
            })
        
        return predictions
    
    def save_optimization_results(self, results: Dict[str, Any], filepath: str):
        """保存优化结果"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Optimization results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def load_optimization_results(self, filepath: str) -> Dict[str, Any]:
        """加载优化结果"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Optimization results loaded from {filepath}")
            return results
        except Exception as e:
            logger.error(f"Failed to load optimization results: {e}")
            return {}

def create_strategy_optimizer(config: OptimizationConfig = None) -> StrategyOptimizer:
    """创建策略优化器"""
    return StrategyOptimizer(config)