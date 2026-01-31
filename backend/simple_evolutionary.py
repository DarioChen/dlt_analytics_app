# backend/simple_evolutionary.py
"""
简化版进化优化器：不依赖复杂库的基础实现
"""
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class SimpleEvolutionaryConfig:
    """简化进化算法配置"""
    population_size: int = 30
    generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2

class SimpleEvolutionaryOptimizer:
    """简化版进化优化器"""
    
    def __init__(self, config: SimpleEvolutionaryConfig = None):
        self.config = config or SimpleEvolutionaryConfig()
        
    def evolve_optimal_numbers(self, df: pd.DataFrame, target_count: int = 5, 
                              objectives: List[str] = None) -> Dict[str, Any]:
        """使用简化进化算法生成最优号码组合"""
        
        if objectives is None:
            objectives = ['hit_probability', 'diversity']
        
        # 初始化种群
        population = self._initialize_population()
        
        best_fitness_history = []
        
        for generation in range(self.config.generations):
            # 评估适应度
            fitness_scores = [self._calculate_fitness(individual, df) for individual in population]
            
            # 记录最佳适应度
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # 选择精英
            elite_size = int(self.config.population_size * self.config.elite_ratio)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
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
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.config.population_size]
        
        # 选择最终结果
        final_fitness = [self._calculate_fitness(individual, df) for individual in population]
        best_individual = population[final_fitness.index(max(final_fitness))]
        
        return {
            'best_numbers': {
                'front': sorted(best_individual['front']),
                'back': sorted(best_individual['back'])
            },
            'fitness_score': max(final_fitness),
            'generation_stats': {
                'fitness_history': best_fitness_history,
                'diversity_history': [0.5] * len(best_fitness_history)  # 简化的多样性
            },
            'patterns_used': {'simple_evolution': True},
            'evolution_config': self.config.__dict__
        }
    
    def _initialize_population(self) -> List[Dict]:
        """初始化种群"""
        population = []
        
        for _ in range(self.config.population_size):
            individual = {
                'front': random.sample(range(1, 36), 5),
                'back': random.sample(range(1, 13), 2),
                'strategy': 'random'
            }
            population.append(individual)
        
        return population
    
    def _calculate_fitness(self, individual: Dict, df: pd.DataFrame) -> float:
        """计算个体适应度"""
        front_nums = set(individual['front'])
        back_nums = set(individual['back'])
        
        # 简化的适应度计算
        fitness = 0.0
        
        # 1. 多样性分数
        odd_count = sum(1 for n in front_nums if n % 2 == 1)
        odd_ratio = odd_count / 5
        diversity_score = 1 - abs(odd_ratio - 0.5) * 2
        fitness += diversity_score * 0.3
        
        # 2. 分布平衡分数
        zones = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
        zone_counts = [sum(1 for n in front_nums if start <= n <= end) for start, end in zones]
        balance_score = 1 - np.std(zone_counts) / np.mean(zone_counts) if np.mean(zone_counts) > 0 else 0
        fitness += balance_score * 0.3
        
        # 3. 和值合理性
        front_sum = sum(front_nums)
        ideal_sum = (1 + 35) * 5 / 2  # 理论平均和值
        sum_score = 1 - abs(front_sum - ideal_sum) / ideal_sum
        fitness += sum_score * 0.2
        
        # 4. 随机性奖励
        fitness += random.uniform(0, 0.2)
        
        return max(0, fitness)
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
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
    
    def _mutate(self, individual: Dict) -> Dict:
        """变异操作"""
        mutated = individual.copy()
        
        # 前区变异
        for i in range(5):
            if random.random() < self.config.mutation_rate:
                new_num = random.randint(1, 35)
                while new_num in mutated['front']:
                    new_num = random.randint(1, 35)
                mutated['front'][i] = new_num
        
        # 后区变异
        for i in range(2):
            if random.random() < self.config.mutation_rate:
                new_num = random.randint(1, 12)
                while new_num in mutated['back']:
                    new_num = random.randint(1, 12)
                mutated['back'][i] = new_num
        
        return mutated

def create_simple_evolutionary_optimizer(config: SimpleEvolutionaryConfig = None) -> SimpleEvolutionaryOptimizer:
    """创建简化版进化优化器"""
    return SimpleEvolutionaryOptimizer(config)