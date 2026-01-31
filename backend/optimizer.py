# backend/optimizer.py
"""
参数优化模块：使用遗传算法/贝叶斯优化自动寻找最优策略参数
"""
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import random
from dataclasses import dataclass

@dataclass
class StrategyParams:
    """策略参数配置"""
    recent_n: int = 2
    span: int = 1
    top_n_blocks: int = 4
    max_per_block: int = 2
    random_blocks_count: int = 3
    random_back_blocks_count: int = 2
    min_consec: int = 1
    min_odd: int = 2
    exclude_top_n: bool = False
    exclude_front_n: int = 3
    exclude_back_n: int = 2
    consecutive_mode: str = "max"  # "exact", "min", "max"
    consecutive_check_type: str = "groups"  # "groups", "pairs"

def genetic_algorithm_optimize(
    fitness_func: Callable[[StrategyParams], float],
    param_ranges: Dict[str, Tuple],
    population_size: int = 20,
    generations: int = 10,
    mutation_rate: float = 0.1,
    elite_size: int = 4
) -> Tuple[StrategyParams, float]:
    """
    遗传算法优化策略参数
    
    Args:
        fitness_func: 适应度函数，输入StrategyParams，返回分数（越高越好）
        param_ranges: 参数范围，如 {"recent_n": (1, 20), "span": (1, 5)}
        population_size: 种群大小
        generations: 迭代代数
        mutation_rate: 变异率
        elite_size: 精英保留数量
    
    Returns:
        (最优参数, 最优分数)
    """
    # 初始化种群
    population = []
    for _ in range(population_size):
        params = StrategyParams()
        for key, (min_val, max_val) in param_ranges.items():
            if hasattr(params, key):
                if isinstance(min_val, int):
                    setattr(params, key, random.randint(min_val, max_val))
                else:
                    setattr(params, key, random.uniform(min_val, max_val))
        population.append(params)
    
    best_params = None
    best_score = float('-inf')
    
    for generation in range(generations):
        # 评估适应度
        scores = []
        for params in population:
            try:
                score = fitness_func(params)
                scores.append(score)
            except Exception as e:
                scores.append(float('-inf'))
        
        # 记录最佳
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = population[best_idx]
        
        # 选择精英
        elite_indices = np.argsort(scores)[-elite_size:][::-1]
        elite = [population[i] for i in elite_indices]
        
        # 生成新种群
        new_population = elite.copy()
        
        while len(new_population) < population_size:
            # 选择父代（轮盘赌）- 修复权重为零的问题
            # 将分数归一化到正数范围
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 1
            
            if max_score == min_score:
                # 所有分数相同，使用均匀随机选择
                weights = [1.0] * len(population)
            else:
                # 归一化到 [0.1, 1.0] 范围，确保所有权重为正
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
                weights = [max(0.1, s) for s in normalized_scores]  # 最小权重0.1
            
            # 确保权重总和大于零
            total_weight = sum(weights)
            if total_weight <= 0:
                weights = [1.0] * len(population)
            
            parent1 = population[random.choices(range(len(population)), weights=weights)[0]]
            parent2 = population[random.choices(range(len(population)), weights=weights)[0]]
            
            # 交叉
            child = StrategyParams()
            for key in param_ranges.keys():
                if hasattr(parent1, key):
                    if random.random() < 0.5:
                        setattr(child, key, getattr(parent1, key))
                    else:
                        setattr(child, key, getattr(parent2, key))
            
            # 变异
            if random.random() < mutation_rate:
                key = random.choice(list(param_ranges.keys()))
                if hasattr(child, key):
                    min_val, max_val = param_ranges[key]
                    if isinstance(min_val, int):
                        setattr(child, key, random.randint(min_val, max_val))
                    else:
                        setattr(child, key, random.uniform(min_val, max_val))
            
            new_population.append(child)
        
        population = new_population
    
    return best_params, best_score

def bayesian_optimize(
    fitness_func: Callable[[StrategyParams], float],
    param_ranges: Dict[str, Tuple],
    n_iterations: int = 20
) -> Tuple[StrategyParams, float]:
    """
    贝叶斯优化（简化版：随机搜索）
    完整实现需要scikit-optimize库
    """
    best_params = None
    best_score = float('-inf')
    
    for _ in range(n_iterations):
        params = StrategyParams()
        for key, (min_val, max_val) in param_ranges.items():
            if hasattr(params, key):
                if isinstance(min_val, int):
                    setattr(params, key, random.randint(min_val, max_val))
                else:
                    setattr(params, key, random.uniform(min_val, max_val))
        
        try:
            score = fitness_func(params)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            continue
    
    return best_params, best_score

