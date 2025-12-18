# backend/optimizer.py
"""
参数优化模块：使用遗传算法/贝叶斯优化自动寻找最优策略参数
"""
from typing import Dict, List, Tuple, Callable, Optional, Any
import numpy as np
import random
from dataclasses import dataclass
import time

# 检查是否有scikit-optimize库
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    HAS_SKO = True
except ImportError:
    HAS_SKO = False

# 检查是否有hyperopt库
try:
    from hyperopt import hp, fmin, tpe, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

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
    n_iterations: int = 100,
    algorithm: str = "auto"
) -> Tuple[StrategyParams, float]:
    """
    贝叶斯优化实现
    
    Args:
        fitness_func: 适应度函数，输入StrategyParams，返回分数（越高越好）
        param_ranges: 参数范围，如 {"recent_n": (1, 20), "span": (1, 5)}
        n_iterations: 迭代次数
        algorithm: 优化算法，可选 "auto", "skopt", "hyperopt", "random"
    
    Returns:
        (最优参数, 最优分数)
    """
    # 确定使用哪种算法
    if algorithm == "auto":
        if HAS_SKO:
            algorithm = "skopt"
        elif HAS_HYPEROPT:
            algorithm = "hyperopt"
        else:
            algorithm = "random"
    
    print(f"使用贝叶斯优化算法: {algorithm}")
    
    if algorithm == "skopt" and HAS_SKO:
        return _bayesian_optimize_skopt(fitness_func, param_ranges, n_iterations)
    elif algorithm == "hyperopt" and HAS_HYPEROPT:
        return _bayesian_optimize_hyperopt(fitness_func, param_ranges, n_iterations)
    else:
        # 降级到随机搜索（带进度跟踪）
        return _random_search_with_progress(fitness_func, param_ranges, n_iterations)

def _bayesian_optimize_skopt(
    fitness_func: Callable[[StrategyParams], float],
    param_ranges: Dict[str, Tuple],
    n_iterations: int = 100
) -> Tuple[StrategyParams, float]:
    """
    使用scikit-optimize实现贝叶斯优化
    """
    # 创建搜索空间
    dimensions = []
    param_names = []
    
    for key, (min_val, max_val) in param_ranges.items():
        if hasattr(StrategyParams(), key):
            param_names.append(key)
            if isinstance(min_val, int):
                dimensions.append(Integer(min_val, max_val, name=key))
            else:
                dimensions.append(Real(min_val, max_val, name=key))
    
    # 定义目标函数（最大化问题转为最小化）
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        strategy_params = StrategyParams()
        for key, value in params.items():
            if hasattr(strategy_params, key):
                # 确保整数参数正确转换
                if key in param_ranges and isinstance(param_ranges[key][0], int):
                    setattr(strategy_params, key, int(value))
                else:
                    setattr(strategy_params, key, value)
        
        try:
            # 最大化问题转为最小化
            return -fitness_func(strategy_params)
        except Exception:
            return float('inf')
    
    # 运行贝叶斯优化
    # 先进行一些随机采样作为初始点
    n_initial_points = min(10, n_iterations // 5)
    result = gp_minimize(
        objective,
        dimensions,
        n_calls=n_iterations,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=True
    )
    
    # 转换结果
    best_params = StrategyParams()
    for i, name in enumerate(param_names):
        if hasattr(best_params, name):
            # 确保整数参数正确转换
            if name in param_ranges and isinstance(param_ranges[name][0], int):
                setattr(best_params, name, int(result.x[i]))
            else:
                setattr(best_params, name, result.x[i])
    
    best_score = -result.fun
    return best_params, best_score

def _bayesian_optimize_hyperopt(
    fitness_func: Callable[[StrategyParams], float],
    param_ranges: Dict[str, Tuple],
    n_iterations: int = 100
) -> Tuple[StrategyParams, float]:
    """
    使用hyperopt实现贝叶斯优化
    """
    # 创建搜索空间
    space = {}
    for key, (min_val, max_val) in param_ranges.items():
        if hasattr(StrategyParams(), key):
            if isinstance(min_val, int):
                space[key] = hp.quniform(key, min_val, max_val, q=1)
            else:
                space[key] = hp.uniform(key, min_val, max_val)
    
    # 定义目标函数（最大化问题转为最小化）
    def objective(params):
        strategy_params = StrategyParams()
        for key, value in params.items():
            if hasattr(strategy_params, key):
                # 确保整数参数正确转换
                if key in param_ranges and isinstance(param_ranges[key][0], int):
                    setattr(strategy_params, key, int(value))
                else:
                    setattr(strategy_params, key, value)
        
        try:
            # 最大化问题转为最小化
            return -fitness_func(strategy_params)
        except Exception:
            return float('inf')
    
    # 运行贝叶斯优化
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_iterations,
        trials=trials,
        rstate=np.random.RandomState(42)
    )
    
    # 转换结果
    best_params = StrategyParams()
    for key, (min_val, max_val) in param_ranges.items():
        if hasattr(best_params, key):
            if key in best:
                # 确保整数参数正确转换
                if isinstance(min_val, int):
                    setattr(best_params, key, int(best[key]))
                else:
                    setattr(best_params, key, best[key])
    
    # 找到最佳分数
    best_score = -min(trials.losses())
    return best_params, best_score

def _random_search_with_progress(
    fitness_func: Callable[[StrategyParams], float],
    param_ranges: Dict[str, Tuple],
    n_iterations: int = 100
) -> Tuple[StrategyParams, float]:
    """
    带进度跟踪的随机搜索
    当没有高级优化库时使用
    """
    best_params = None
    best_score = float('-inf')
    start_time = time.time()
    
    for i in range(n_iterations):
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
                
                # 每找到更好的参数就输出进度
                elapsed = time.time() - start_time
                print(f"找到更好的参数：分数={best_score:.2f}, 迭代={i+1}/{n_iterations}, 用时={elapsed:.2f}秒")
        except Exception as e:
            print(f"迭代 {i+1} 出错: {e}")
            continue
        
        # 每10次迭代输出一次进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"进度: {i+1}/{n_iterations}, 当前最佳分数={best_score:.2f}, 用时={elapsed:.2f}秒")
    
    return best_params, best_score

def get_optimization_methods() -> Dict[str, Dict[str, Any]]:
    """
    获取可用的优化方法
    
    Returns:
        方法名称到详情的映射
    """
    methods = {
        "genetic": {
            "name": "遗传算法",
            "available": True,
            "description": "基于自然选择的全局优化算法",
            "requires_lib": None
        },
        "bayesian_skopt": {
            "name": "贝叶斯优化 (scikit-optimize)",
            "available": HAS_SKO,
            "description": "基于高斯过程的贝叶斯优化",
            "requires_lib": "scikit-optimize"
        },
        "bayesian_hyperopt": {
            "name": "贝叶斯优化 (hyperopt)",
            "available": HAS_HYPEROPT,
            "description": "基于树的Parzen估计器的贝叶斯优化",
            "requires_lib": "hyperopt"
        },
        "random": {
            "name": "随机搜索",
            "available": True,
            "description": "简单但有效的参数空间随机采样",
            "requires_lib": None
        }
    }
    return methods

