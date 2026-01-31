# backend/exclusion_pool_generator.py
"""
排除池生成器 - 通过排除已生成的号码组合来提高高等奖中奖率
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from backend import generator as genmod
from backend.enhanced_generator import EnhancedNumberGenerator
from backend.exclusion_pool_db import exclusion_pool_db
import json
from datetime import datetime


class ExclusionPoolGenerator:
    """排除池生成器"""
    
    def __init__(self):
        self.enhanced_generator = EnhancedNumberGenerator()
        self.exclusion_history = []  # 记录排除池历史
        self.analysis_results = []   # 记录分析结果
    
    def generate_with_dynamic_exclusion_pool(self, 
                                            exclusion_pool_size: int,
                                            target_count: int,
                                            rules: Dict,
                                            front_blocks: Dict,
                                            back_blocks: Dict,
                                            front_weights: Dict,
                                            back_weights: Dict,
                                            selected_front_blocks: List[str],
                                            selected_back_blocks: List[str],
                                            historical_data: Optional[pd.DataFrame] = None,
                                            use_enhanced: bool = True,
                                            max_attempts: int = 10000,
                                            save_to_db: bool = True,
                                            predicted_issue: str = None,
                                            strategy_name: str = None,
                                            generation_method: str = "dynamic_exclusion_pool") -> Dict:
        """
        使用动态排除池策略生成号码
        
        核心逻辑：
        1. 使用当前生成算法生成N组号码作为动态排除池
        2. 在排除这些号码的基础上再生成Y组最终号码
        3. 这样可以避免生成算法的"热门偏好"，增加独特号码的概率
        
        Args:
            exclusion_pool_size: 动态排除池大小（N组号码）
            target_count: 目标生成数量（Y组号码）
            rules: 生成规则
            front_blocks: 前区区块
            back_blocks: 后区区块
            front_weights: 前区权重
            back_weights: 后区权重
            selected_front_blocks: 选中的前区区块
            selected_back_blocks: 选中的后区区块
            historical_data: 历史数据
            use_enhanced: 是否使用增强生成器
            max_attempts: 最大尝试次数
            save_to_db: 是否保存到数据库
            predicted_issue: 预测期号
            strategy_name: 策略名称
            generation_method: 生成方法标识
            
        Returns:
            生成结果字典
        """
        
        print(f"开始动态排除池生成：排除池大小={exclusion_pool_size}，目标数量={target_count}")
        
        try:
            # 第一步：使用当前生成算法生成动态排除池（N组号码）
            print("第一步：生成动态排除池（使用当前算法的热门倾向）...")
            
            if use_enhanced and historical_data is not None:
                # 使用增强生成器生成动态排除池
                try:
                    dynamic_exclusion_pool = self.enhanced_generator.generate_enhanced_numbers(
                        count=exclusion_pool_size,
                        rules=rules,
                        front_blocks=front_blocks,
                        back_blocks=back_blocks,
                        front_weights=front_weights,
                        back_weights=back_weights,
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks,
                        historical_data=historical_data,
                        use_markov=True,
                        use_big_data=True,
                        markov_weight=0.4,
                        big_data_weight=0.3,
                        traditional_weight=0.3
                    )
                except Exception as e:
                    print(f"增强生成器生成动态排除池失败，回退到传统方法: {e}")
                    # 回退到传统生成器
                    dynamic_exclusion_pool = genmod.gen_numbers(
                        count=exclusion_pool_size,
                        rules=rules,
                        front_blocks=front_blocks,
                        back_blocks=back_blocks,
                        front_weights=front_weights,
                        back_weights=back_weights,
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks
                    )
            else:
                # 使用传统生成器生成动态排除池
                dynamic_exclusion_pool = genmod.gen_numbers(
                    count=exclusion_pool_size,
                    rules=rules,
                    front_blocks=front_blocks,
                    back_blocks=back_blocks,
                    front_weights=front_weights,
                    back_weights=back_weights,
                    selected_front_blocks=selected_front_blocks,
                    selected_back_blocks=selected_back_blocks
                )
            
            if not dynamic_exclusion_pool:
                return {"error": "动态排除池生成失败"}
            
            print(f"动态排除池生成完成，共{len(dynamic_exclusion_pool)}组号码")
            
            # 将动态排除池转换为集合，便于快速查找
            exclusion_set = set()
            for combo in dynamic_exclusion_pool:
                front_tuple = tuple(sorted(combo['front']))
                back_tuple = tuple(sorted(combo['back']))
                exclusion_set.add((front_tuple, back_tuple))
            
            print(f"动态排除池转换完成，排除{len(exclusion_set)}个唯一组合")
            
            # 第二步：在排除动态排除池的基础上生成目标号码（Y组号码）
            print("第二步：生成目标号码（避开算法热门倾向）...")
            target_numbers = []
            attempts = 0
            
            # 为了提高生成效率，我们使用批量生成和筛选的策略
            batch_size = min(100, target_count * 10)  # 每批生成的数量
            
            while len(target_numbers) < target_count and attempts < max_attempts:
                attempts += 1
                
                # 批量生成候选号码
                if use_enhanced and historical_data is not None:
                    try:
                        candidates = self.enhanced_generator.generate_enhanced_numbers(
                            count=batch_size,
                            rules=rules,
                            front_blocks=front_blocks,
                            back_blocks=back_blocks,
                            front_weights=front_weights,
                            back_weights=back_weights,
                            selected_front_blocks=selected_front_blocks,
                            selected_back_blocks=selected_back_blocks,
                            historical_data=historical_data,
                            use_markov=True,
                            use_big_data=True,
                            markov_weight=0.4,
                            big_data_weight=0.3,
                            traditional_weight=0.3
                        )
                    except Exception as e:
                        print(f"增强生成器生成候选号码失败，回退到传统方法: {e}")
                        candidates = genmod.gen_numbers(
                            count=batch_size,
                            rules=rules,
                            front_blocks=front_blocks,
                            back_blocks=back_blocks,
                            front_weights=front_weights,
                            back_weights=back_weights,
                            selected_front_blocks=selected_front_blocks,
                            selected_back_blocks=selected_back_blocks
                        )
                else:
                    candidates = genmod.gen_numbers(
                        count=batch_size,
                        rules=rules,
                        front_blocks=front_blocks,
                        back_blocks=back_blocks,
                        front_weights=front_weights,
                        back_weights=back_weights,
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks
                    )
                
                # 筛选不在动态排除池中的号码
                for candidate in candidates:
                    if len(target_numbers) >= target_count:
                        break
                    
                    front_tuple = tuple(sorted(candidate['front']))
                    back_tuple = tuple(sorted(candidate['back']))
                    combo_tuple = (front_tuple, back_tuple)
                    
                    # 检查是否在动态排除池中
                    if combo_tuple not in exclusion_set:
                        # 检查是否已经在目标列表中
                        already_exists = False
                        for existing in target_numbers:
                            existing_front = tuple(sorted(existing['front']))
                            existing_back = tuple(sorted(existing['back']))
                            if (existing_front, existing_back) == combo_tuple:
                                already_exists = True
                                break
                        
                        if not already_exists:
                            # 添加生成方法标识
                            candidate['generation_method'] = 'dynamic_exclusion_target'
                            target_numbers.append(candidate)
                
                if attempts % 100 == 0:
                    print(f"尝试{attempts}次，已生成{len(target_numbers)}/{target_count}个目标号码")
            
            if len(target_numbers) < target_count:
                print(f"警告：只生成了{len(target_numbers)}/{target_count}个目标号码（尝试{attempts}次）")
            else:
                print(f"目标号码生成完成，共{len(target_numbers)}组")
            
            # 记录生成历史
            generation_record = {
                "timestamp": datetime.now().isoformat(),
                "exclusion_pool_size": exclusion_pool_size,
                "target_count": target_count,
                "actual_generated": len(target_numbers),
                "attempts": attempts,
                "success_rate": len(target_numbers) / target_count if target_count > 0 else 0,
                "use_enhanced": use_enhanced,
                "generation_type": "dynamic_exclusion_pool"
            }
            self.exclusion_history.append(generation_record)
            
            # 准备返回结果
            result = {
                "dynamic_exclusion_pool": dynamic_exclusion_pool,
                "target_numbers": target_numbers,
                "exclusion_pool_size": len(dynamic_exclusion_pool),
                "target_count_actual": len(target_numbers),
                "generation_attempts": attempts,
                "generation_record": generation_record,
                "success": True,
                "generation_type": "dynamic_exclusion_pool"
            }
            
            # 保存到数据库
            if save_to_db:
                try:
                    # 计算投注成本（假设每注2元）
                    investment_cost = len(target_numbers) * 2.0
                    
                    # 准备数据库记录数据
                    db_data = {
                        'exclusion_pool_size': exclusion_pool_size,
                        'target_count': target_count,
                        'generation_method': generation_method,
                        'actual_generated': len(target_numbers),
                        'generation_attempts': attempts,
                        'success_rate': len(target_numbers) / target_count if target_count > 0 else 0,
                        'exclusion_pool_data': [
                            {
                                'front': c['front'],
                                'back': c['back'],
                                'method': c.get('generation_method', 'dynamic_exclusion_pool')
                            } for c in dynamic_exclusion_pool
                        ],
                        'target_numbers_data': [
                            {
                                'front': c['front'],
                                'back': c['back'],
                                'method': c.get('generation_method', 'dynamic_exclusion_target')
                            } for c in target_numbers
                        ],
                        'predicted_issue': predicted_issue,
                        'investment_cost': investment_cost,
                        'use_enhanced': use_enhanced,
                        'use_two_round': False,
                        'strategy_name': strategy_name,
                        'generation_config': {
                            'rules': rules,
                            'selected_front_blocks': selected_front_blocks,
                            'selected_back_blocks': selected_back_blocks,
                            'max_attempts': max_attempts,
                            'generation_type': 'dynamic_exclusion_pool'
                        }
                    }
                    
                    # 保存到数据库
                    record_id = exclusion_pool_db.save_generation_result(db_data)
                    if record_id:
                        result['db_record_id'] = record_id
                        print(f"动态排除池生成结果已保存到数据库，记录ID: {record_id}")
                    else:
                        print("保存到数据库失败")
                        
                except Exception as e:
                    print(f"数据库保存过程中出现错误: {e}")
            
            return result
            
        except Exception as e:
            print(f"动态排除池生成过程中出现错误: {e}")
            return {"error": str(e)}
    def generate_with_exclusion_pool(self, 
                                   exclusion_pool_size: int,
                                   target_count: int,
                                   rules: Dict,
                                   front_blocks: Dict,
                                   back_blocks: Dict,
                                   front_weights: Dict,
                                   back_weights: Dict,
                                   selected_front_blocks: List[str],
                                   selected_back_blocks: List[str],
                                   historical_data: Optional[pd.DataFrame] = None,
                                   use_enhanced: bool = True,
                                   max_attempts: int = 10000,
                                   save_to_db: bool = True,
                                   predicted_issue: str = None,
                                   strategy_name: str = None,
                                   generation_method: str = "dynamic_exclusion_pool",
                                   use_dynamic_pool: bool = True) -> Dict:
        """
        使用排除池策略生成号码（默认使用动态排除池）
        
        Args:
            exclusion_pool_size: 排除池大小（N组号码）
            target_count: 目标生成数量（Y组号码）
            rules: 生成规则
            front_blocks: 前区区块
            back_blocks: 后区区块
            front_weights: 前区权重
            back_weights: 后区权重
            selected_front_blocks: 选中的前区区块
            selected_back_blocks: 选中的后区区块
            historical_data: 历史数据
            use_enhanced: 是否使用增强生成器
            max_attempts: 最大尝试次数
            save_to_db: 是否保存到数据库
            predicted_issue: 预测期号
            strategy_name: 策略名称
            generation_method: 生成方法标识
            use_dynamic_pool: 是否使用动态排除池（默认True）
            
        Returns:
            生成结果字典
        """
        
        if use_dynamic_pool:
            # 使用动态排除池策略（推荐）
            return self.generate_with_dynamic_exclusion_pool(
                exclusion_pool_size=exclusion_pool_size,
                target_count=target_count,
                rules=rules,
                front_blocks=front_blocks,
                back_blocks=back_blocks,
                front_weights=front_weights,
                back_weights=back_weights,
                selected_front_blocks=selected_front_blocks,
                selected_back_blocks=selected_back_blocks,
                historical_data=historical_data,
                use_enhanced=use_enhanced,
                max_attempts=max_attempts,
                save_to_db=save_to_db,
                predicted_issue=predicted_issue,
                strategy_name=strategy_name,
                generation_method=generation_method
            )
        else:
            # 使用静态排除池策略（兼容性保留）
            return self.generate_with_static_exclusion_pool(
                exclusion_pool_size=exclusion_pool_size,
                target_count=target_count,
                rules=rules,
                front_blocks=front_blocks,
                back_blocks=back_blocks,
                front_weights=front_weights,
                back_weights=back_weights,
                selected_front_blocks=selected_front_blocks,
                selected_back_blocks=selected_back_blocks,
                historical_data=historical_data,
                use_enhanced=use_enhanced,
                max_attempts=max_attempts,
                save_to_db=save_to_db,
                predicted_issue=predicted_issue,
                strategy_name=strategy_name,
                generation_method=generation_method
            )
    
    def generate_with_static_exclusion_pool(self, 
                                          exclusion_pool_size: int,
                                          target_count: int,
                                          rules: Dict,
                                          front_blocks: Dict,
                                          back_blocks: Dict,
                                          front_weights: Dict,
                                          back_weights: Dict,
                                          selected_front_blocks: List[str],
                                          selected_back_blocks: List[str],
                                          historical_data: Optional[pd.DataFrame] = None,
                                          use_enhanced: bool = True,
                                          max_attempts: int = 10000,
                                          save_to_db: bool = True,
                                          predicted_issue: str = None,
                                          strategy_name: str = None,
                                          generation_method: str = "static_exclusion_pool") -> Dict:
        """
        使用静态排除池策略生成号码（原始方法，兼容性保留）
        
        Args:
            exclusion_pool_size: 静态排除池大小（N组号码）
            target_count: 目标生成数量（Y组号码）
            其他参数同generate_with_exclusion_pool
            
        Returns:
            生成结果字典
        """
        
        print(f"开始静态排除池生成：排除池大小={exclusion_pool_size}，目标数量={target_count}")
        
        try:
            # 第一步：生成静态排除池（N组号码）
            print("第一步：生成静态排除池...")
            if use_enhanced and historical_data is not None:
                # 使用增强生成器
                try:
                    exclusion_pool = self.enhanced_generator.generate_enhanced_numbers(
                        count=exclusion_pool_size,
                        rules=rules,
                        front_blocks=front_blocks,
                        back_blocks=back_blocks,
                        front_weights=front_weights,
                        back_weights=back_weights,
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks,
                        historical_data=historical_data,
                        use_markov=True,
                        use_big_data=True,
                        markov_weight=0.4,
                        big_data_weight=0.3,
                        traditional_weight=0.3
                    )
                except Exception as e:
                    print(f"增强生成器生成静态排除池失败，回退到传统方法: {e}")
                    exclusion_pool = genmod.gen_numbers(
                        count=exclusion_pool_size,
                        rules=rules,
                        front_blocks=front_blocks,
                        back_blocks=back_blocks,
                        front_weights=front_weights,
                        back_weights=back_weights,
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks
                    )
            else:
                # 使用传统生成器
                exclusion_pool = genmod.gen_numbers(
                    count=exclusion_pool_size,
                    rules=rules,
                    front_blocks=front_blocks,
                    back_blocks=back_blocks,
                    front_weights=front_weights,
                    back_weights=back_weights,
                    selected_front_blocks=selected_front_blocks,
                    selected_back_blocks=selected_back_blocks
                )
            
            if not exclusion_pool:
                return {"error": "静态排除池生成失败"}
            
            print(f"静态排除池生成完成，共{len(exclusion_pool)}组号码")
            
            # 将排除池转换为集合，便于快速查找
            exclusion_set = set()
            for combo in exclusion_pool:
                front_tuple = tuple(sorted(combo['front']))
                back_tuple = tuple(sorted(combo['back']))
                exclusion_set.add((front_tuple, back_tuple))
            
            print(f"静态排除池转换完成，排除{len(exclusion_set)}个唯一组合")
            
            # 第二步：生成目标号码（Y组号码），确保不与排除池重复
            print("第二步：生成目标号码...")
            target_numbers = []
            attempts = 0
            
            while len(target_numbers) < target_count and attempts < max_attempts:
                attempts += 1
                
                # 生成候选号码
                if use_enhanced and historical_data is not None:
                    try:
                        candidates = self.enhanced_generator.generate_enhanced_numbers(
                            count=min(50, target_count * 2),  # 批量生成以提高效率
                            rules=rules,
                            front_blocks=front_blocks,
                            back_blocks=back_blocks,
                            front_weights=front_weights,
                            back_weights=back_weights,
                            selected_front_blocks=selected_front_blocks,
                            selected_back_blocks=selected_back_blocks,
                            historical_data=historical_data,
                            use_markov=True,
                            use_big_data=True,
                            markov_weight=0.4,
                            big_data_weight=0.3,
                            traditional_weight=0.3
                        )
                    except Exception as e:
                        print(f"增强生成器生成候选号码失败，回退到传统方法: {e}")
                        candidates = genmod.gen_numbers(
                            count=min(50, target_count * 2),
                            rules=rules,
                            front_blocks=front_blocks,
                            back_blocks=back_blocks,
                            front_weights=front_weights,
                            back_weights=back_weights,
                            selected_front_blocks=selected_front_blocks,
                            selected_back_blocks=selected_back_blocks
                        )
                else:
                    candidates = genmod.gen_numbers(
                        count=min(50, target_count * 2),
                        rules=rules,
                        front_blocks=front_blocks,
                        back_blocks=back_blocks,
                        front_weights=front_weights,
                        back_weights=back_weights,
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks
                    )
                
                # 筛选不在排除池中的号码
                for candidate in candidates:
                    if len(target_numbers) >= target_count:
                        break
                    
                    front_tuple = tuple(sorted(candidate['front']))
                    back_tuple = tuple(sorted(candidate['back']))
                    combo_tuple = (front_tuple, back_tuple)
                    
                    # 检查是否在排除池中
                    if combo_tuple not in exclusion_set:
                        # 检查是否已经在目标列表中
                        already_exists = False
                        for existing in target_numbers:
                            existing_front = tuple(sorted(existing['front']))
                            existing_back = tuple(sorted(existing['back']))
                            if (existing_front, existing_back) == combo_tuple:
                                already_exists = True
                                break
                        
                        if not already_exists:
                            candidate['generation_method'] = 'static_exclusion_target'
                            target_numbers.append(candidate)
                
                if attempts % 1000 == 0:
                    print(f"尝试{attempts}次，已生成{len(target_numbers)}/{target_count}个目标号码")
            
            if len(target_numbers) < target_count:
                print(f"警告：只生成了{len(target_numbers)}/{target_count}个目标号码（尝试{attempts}次）")
            else:
                print(f"目标号码生成完成，共{len(target_numbers)}组")
            
            # 记录生成历史
            generation_record = {
                "timestamp": datetime.now().isoformat(),
                "exclusion_pool_size": exclusion_pool_size,
                "target_count": target_count,
                "actual_generated": len(target_numbers),
                "attempts": attempts,
                "success_rate": len(target_numbers) / target_count if target_count > 0 else 0,
                "use_enhanced": use_enhanced,
                "generation_type": "static_exclusion_pool"
            }
            self.exclusion_history.append(generation_record)
            
            # 准备返回结果
            result = {
                "exclusion_pool": exclusion_pool,
                "target_numbers": target_numbers,
                "exclusion_pool_size": len(exclusion_pool),
                "target_count_actual": len(target_numbers),
                "generation_attempts": attempts,
                "generation_record": generation_record,
                "success": True,
                "generation_type": "static_exclusion_pool"
            }
            
            # 保存到数据库
            if save_to_db:
                try:
                    # 计算投注成本（假设每注2元）
                    investment_cost = len(target_numbers) * 2.0
                    
                    # 准备数据库记录数据
                    db_data = {
                        'exclusion_pool_size': exclusion_pool_size,
                        'target_count': target_count,
                        'generation_method': generation_method,
                        'actual_generated': len(target_numbers),
                        'generation_attempts': attempts,
                        'success_rate': len(target_numbers) / target_count if target_count > 0 else 0,
                        'exclusion_pool_data': [
                            {
                                'front': c['front'],
                                'back': c['back'],
                                'method': c.get('generation_method', 'static_exclusion_pool')
                            } for c in exclusion_pool
                        ],
                        'target_numbers_data': [
                            {
                                'front': c['front'],
                                'back': c['back'],
                                'method': c.get('generation_method', 'static_exclusion_target')
                            } for c in target_numbers
                        ],
                        'predicted_issue': predicted_issue,
                        'investment_cost': investment_cost,
                        'use_enhanced': use_enhanced,
                        'use_two_round': False,
                        'strategy_name': strategy_name,
                        'generation_config': {
                            'rules': rules,
                            'selected_front_blocks': selected_front_blocks,
                            'selected_back_blocks': selected_back_blocks,
                            'max_attempts': max_attempts,
                            'generation_type': 'static_exclusion_pool'
                        }
                    }
                    
                    # 保存到数据库
                    record_id = exclusion_pool_db.save_generation_result(db_data)
                    if record_id:
                        result['db_record_id'] = record_id
                        print(f"静态排除池生成结果已保存到数据库，记录ID: {record_id}")
                    else:
                        print("保存到数据库失败")
                        
                except Exception as e:
                    print(f"数据库保存过程中出现错误: {e}")
            
            return result
            
        except Exception as e:
            print(f"静态排除池生成过程中出现错误: {e}")
            return {"error": str(e)}
    
    def analyze_exclusion_effectiveness(self, 
                                      historical_data: pd.DataFrame,
                                      exclusion_pool_sizes: List[int],
                                      target_count: int,
                                      test_periods: int,
                                      rules: Dict,
                                      generation_context: Dict,
                                      selected_front_blocks: List[str],
                                      selected_back_blocks: List[str],
                                      prize_structure: Dict) -> Dict:
        """
        分析不同排除池大小对高等奖中奖率的影响
        
        Args:
            historical_data: 历史数据
            exclusion_pool_sizes: 要测试的排除池大小列表
            target_count: 目标生成数量
            test_periods: 测试期数
            rules: 生成规则
            generation_context: 生成上下文
            selected_front_blocks: 选中的前区区块
            selected_back_blocks: 选中的后区区块
            prize_structure: 奖金结构
            
        Returns:
            分析结果
        """
        
        print(f"开始排除池效果分析：测试{len(exclusion_pool_sizes)}种排除池大小，{test_periods}期数据")
        
        analysis_results = []
        
        # 准备测试数据
        if len(historical_data) < test_periods + 100:
            raise ValueError(f"历史数据不足，需要至少{test_periods + 100}期数据")
        
        test_data = historical_data.tail(test_periods).reset_index(drop=True)
        
        for pool_size in exclusion_pool_sizes:
            print(f"\n测试排除池大小: {pool_size}")
            
            pool_results = {
                "exclusion_pool_size": pool_size,
                "target_count": target_count,
                "total_cost": 0.0,
                "total_return": 0.0,
                "high_prize_hits": 0,  # 高等奖（一二三等奖）命中次数
                "total_hits": 0,
                "periods_tested": 0,
                "generation_success_rate": 0.0,
                "detailed_results": []
            }
            
            successful_generations = 0
            
            for i in range(test_periods):
                # 获取训练数据（当前测试期之前的数据）
                train_end_idx = len(historical_data) - test_periods + i
                train_data = historical_data.iloc[:train_end_idx]
                
                if len(train_data) < 50:
                    continue
                
                # 获取实际开奖结果
                test_row = test_data.iloc[i]
                actual_front = test_row[['f1', 'f2', 'f3', 'f4', 'f5']].tolist()
                actual_back = test_row[['b1', 'b2']].tolist()
                
                try:
                    # 使用排除池策略生成号码
                    generation_result = self.generate_with_exclusion_pool(
                        exclusion_pool_size=pool_size,
                        target_count=target_count,
                        rules=rules,
                        front_blocks=generation_context["front_blocks"],
                        back_blocks=generation_context["back_blocks"],
                        front_weights=generation_context["front_weights"],
                        back_weights=generation_context["back_weights"],
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks,
                        historical_data=train_data.tail(100),
                        use_enhanced=True,
                        max_attempts=5000  # 减少尝试次数以提高速度
                    )
                    
                    if 'error' in generation_result:
                        print(f"第{i+1}期生成失败: {generation_result['error']}")
                        continue
                    
                    target_numbers = generation_result.get('target_numbers', [])
                    if not target_numbers:
                        continue
                    
                    successful_generations += 1
                    
                    # 计算本期成本和收益
                    period_cost = len(target_numbers) * 2.0  # 假设每注2元
                    period_return = 0.0
                    period_hits = 0
                    high_prize_hits_period = 0
                    
                    period_details = {
                        "period": i + 1,
                        "issue": test_row.get('issue', f'期{i+1}'),
                        "actual_front": actual_front,
                        "actual_back": actual_back,
                        "generated_count": len(target_numbers),
                        "prizes": []
                    }
                    
                    for candidate in target_numbers:
                        # 检查中奖情况
                        prize_name = self._check_prize(
                            candidate['front'], candidate['back'],
                            actual_front, actual_back
                        )
                        
                        prize_amount = prize_structure.get(prize_name, 0)
                        period_return += prize_amount
                        
                        if prize_name != "未中奖":
                            period_hits += 1
                            
                            # 检查是否为高等奖
                            if prize_name in ["一等奖", "二等奖", "三等奖"]:
                                high_prize_hits_period += 1
                        
                        period_details["prizes"].append({
                            "front": candidate['front'],
                            "back": candidate['back'],
                            "prize": prize_name,
                            "amount": prize_amount
                        })
                    
                    pool_results["total_cost"] += period_cost
                    pool_results["total_return"] += period_return
                    pool_results["total_hits"] += period_hits
                    pool_results["high_prize_hits"] += high_prize_hits_period
                    pool_results["periods_tested"] += 1
                    
                    period_details.update({
                        "cost": period_cost,
                        "return": period_return,
                        "profit": period_return - period_cost,
                        "hits": period_hits,
                        "high_prize_hits": high_prize_hits_period
                    })
                    
                    pool_results["detailed_results"].append(period_details)
                    
                except Exception as e:
                    print(f"第{i+1}期测试失败: {e}")
                    continue
            
            # 计算统计指标
            if pool_results["periods_tested"] > 0:
                pool_results["generation_success_rate"] = successful_generations / test_periods
                pool_results["net_profit"] = pool_results["total_return"] - pool_results["total_cost"]
                pool_results["roi"] = (pool_results["net_profit"] / pool_results["total_cost"]) if pool_results["total_cost"] > 0 else 0
                pool_results["hit_rate"] = pool_results["total_hits"] / (pool_results["periods_tested"] * target_count) if pool_results["periods_tested"] > 0 else 0
                pool_results["high_prize_rate"] = pool_results["high_prize_hits"] / (pool_results["periods_tested"] * target_count) if pool_results["periods_tested"] > 0 else 0
                pool_results["avg_cost_per_period"] = pool_results["total_cost"] / pool_results["periods_tested"]
                pool_results["avg_return_per_period"] = pool_results["total_return"] / pool_results["periods_tested"]
            
            analysis_results.append(pool_results)
            
            print(f"排除池大小{pool_size}测试完成:")
            print(f"  - 成功生成率: {pool_results['generation_success_rate']:.2%}")
            print(f"  - 高等奖命中率: {pool_results.get('high_prize_rate', 0):.4%}")
            print(f"  - 总体命中率: {pool_results.get('hit_rate', 0):.2%}")
            print(f"  - ROI: {pool_results.get('roi', 0):.2%}")
        
        # 保存分析结果
        analysis_summary = {
            "timestamp": datetime.now().isoformat(),
            "test_periods": test_periods,
            "target_count": target_count,
            "exclusion_pool_sizes_tested": exclusion_pool_sizes,
            "results": analysis_results
        }
        
        self.analysis_results.append(analysis_summary)
        
        return analysis_summary
    
    def _check_prize(self, front_nums: List[int], back_nums: List[int], 
                    win_front: List[int], win_back: List[int]) -> str:
        """检查中奖等级"""
        front_match = len(set(front_nums) & set(win_front))
        back_match = len(set(back_nums) & set(win_back))
        
        # 奖项规则
        if front_match == 5 and back_match == 2:
            return "一等奖"
        elif front_match == 5 and back_match == 1:
            return "二等奖"
        elif front_match == 5 and back_match == 0:
            return "三等奖"
        elif front_match >= 4 and back_match == 2:
            return "四等奖"
        elif front_match >= 4 and back_match == 1:
            return "五等奖"
        elif front_match >= 3 and back_match == 2:
            return "六等奖"
        elif front_match >= 4 and back_match == 0:
            return "七等奖"
        elif (front_match >= 3 and back_match >= 1) or (front_match == 2 and back_match == 2):
            return "八等奖"
        elif front_match >= 3 or (front_match == 1 and back_match == 2) or (front_match == 2 and back_match == 1) or back_match == 2:
            return "九等奖"
        else:
            return "未中奖"
    
    def get_optimal_exclusion_pool_size(self, analysis_results: Dict) -> Dict:
        """
        基于分析结果推荐最优排除池大小
        
        Args:
            analysis_results: 分析结果
            
        Returns:
            推荐结果
        """
        
        if not analysis_results.get("results"):
            return {"error": "没有分析结果"}
        
        results = analysis_results["results"]
        
        # 按不同指标排序
        by_high_prize_rate = sorted(results, key=lambda x: x.get("high_prize_rate", 0), reverse=True)
        by_roi = sorted(results, key=lambda x: x.get("roi", -999), reverse=True)
        by_hit_rate = sorted(results, key=lambda x: x.get("hit_rate", 0), reverse=True)
        
        # 综合评分（高等奖命中率权重最高）
        for result in results:
            high_prize_score = result.get("high_prize_rate", 0) * 100  # 高等奖命中率权重100
            roi_score = max(0, result.get("roi", 0)) * 10  # ROI权重10（只考虑正ROI）
            hit_score = result.get("hit_rate", 0) * 5  # 总命中率权重5
            generation_score = result.get("generation_success_rate", 0) * 2  # 生成成功率权重2
            
            result["composite_score"] = high_prize_score + roi_score + hit_score + generation_score
        
        by_composite = sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True)
        
        recommendations = {
            "best_for_high_prize": by_high_prize_rate[0] if by_high_prize_rate else None,
            "best_for_roi": by_roi[0] if by_roi else None,
            "best_for_hit_rate": by_hit_rate[0] if by_hit_rate else None,
            "best_overall": by_composite[0] if by_composite else None,
            "analysis_summary": {
                "total_tested": len(results),
                "best_high_prize_rate": by_high_prize_rate[0].get("high_prize_rate", 0) if by_high_prize_rate else 0,
                "best_roi": by_roi[0].get("roi", 0) if by_roi else 0,
                "best_hit_rate": by_hit_rate[0].get("hit_rate", 0) if by_hit_rate else 0
            }
        }
        
        return recommendations
    
    def save_analysis_results(self, filepath: str):
        """保存分析结果到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "exclusion_history": self.exclusion_history,
                    "analysis_results": self.analysis_results
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存分析结果失败: {e}")
            return False
    
    def load_analysis_results(self, filepath: str):
        """从文件加载分析结果"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.exclusion_history = data.get("exclusion_history", [])
                self.analysis_results = data.get("analysis_results", [])
            return True
        except Exception as e:
            print(f"加载分析结果失败: {e}")
            return False


# 创建全局实例
exclusion_pool_generator = ExclusionPoolGenerator()