#!/usr/bin/env python3
# test_exclusion_pool.py
"""
排除池生成器测试脚本
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.exclusion_pool_generator import exclusion_pool_generator

def create_test_data(num_periods=100):
    """创建测试用的历史数据"""
    data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(num_periods):
        # 生成随机的开奖号码
        front_nums = sorted(np.random.choice(range(1, 36), 5, replace=False))
        back_nums = sorted(np.random.choice(range(1, 13), 2, replace=False))
        
        data.append({
            'issue': f'23{i+1:03d}',
            'date': base_date + timedelta(days=i*3),
            'f1': front_nums[0], 'f2': front_nums[1], 'f3': front_nums[2], 
            'f4': front_nums[3], 'f5': front_nums[4],
            'b1': back_nums[0], 'b2': back_nums[1],
            'sales': 10000000 + np.random.randint(-1000000, 1000000),
            'pool': 50000000 + np.random.randint(-5000000, 5000000)
        })
    
    return pd.DataFrame(data)

def test_basic_generation():
    """测试基本的排除池生成功能"""
    print("=" * 50)
    print("测试基本排除池生成功能")
    print("=" * 50)
    
    # 创建测试数据
    df = create_test_data(50)
    
    # 基本参数
    rules = {
        "sum_front_range": [0, 999],
        "odd_even_front": [0, 5],
        "front_include": [],
        "front_exclude": [],
        "back_include": [],
        "back_exclude": [],
        "consecutive_count": 0,
        "consecutive_mode": "min"
    }
    
    # 区块定义
    front_blocks = {
        "1-5": [1, 2, 3, 4, 5],
        "6-10": [6, 7, 8, 9, 10],
        "11-15": [11, 12, 13, 14, 15],
        "16-20": [16, 17, 18, 19, 20],
        "21-25": [21, 22, 23, 24, 25],
        "26-30": [26, 27, 28, 29, 30],
        "31-35": [31, 32, 33, 34, 35]
    }
    
    back_blocks = {
        "1-2": [1, 2],
        "3-4": [3, 4],
        "5-6": [5, 6],
        "7-8": [7, 8],
        "9-10": [9, 10],
        "11-12": [11, 12]
    }
    
    # 权重（均匀分布）
    front_weights = {k: 1.0 for k in front_blocks.keys()}
    back_weights = {k: 1.0 for k in back_blocks.keys()}
    
    # 选中的区块
    selected_front_blocks = list(front_blocks.keys())
    selected_back_blocks = list(back_blocks.keys())
    
    # 测试不同的排除池大小
    test_cases = [
        {"pool_size": 10, "target_count": 5},
        {"pool_size": 20, "target_count": 5},
        {"pool_size": 50, "target_count": 10}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: 排除池大小={case['pool_size']}, 目标数量={case['target_count']}")
        
        try:
            result = exclusion_pool_generator.generate_with_exclusion_pool(
                exclusion_pool_size=case['pool_size'],
                target_count=case['target_count'],
                rules=rules,
                front_blocks=front_blocks,
                back_blocks=back_blocks,
                front_weights=front_weights,
                back_weights=back_weights,
                selected_front_blocks=selected_front_blocks,
                selected_back_blocks=selected_back_blocks,
                historical_data=df,
                use_enhanced=False,  # 使用传统生成器进行测试
                max_attempts=1000
            )
            
            if 'error' in result:
                print(f"  ❌ 生成失败: {result['error']}")
            else:
                print(f"  ✅ 生成成功!")
                print(f"     排除池: {result['exclusion_pool_size']} 组")
                print(f"     目标号码: {result['target_count_actual']} 组")
                print(f"     尝试次数: {result['generation_attempts']}")
                print(f"     成功率: {result['target_count_actual']/case['target_count']:.2%}")
                
                # 显示前3组目标号码
                target_numbers = result.get('target_numbers', [])
                if target_numbers:
                    print("     前3组目标号码:")
                    for j, num in enumerate(target_numbers[:3], 1):
                        front_str = ",".join(map(str, num['front']))
                        back_str = ",".join(map(str, num['back']))
                        print(f"       {j}. 前区: {front_str} | 后区: {back_str}")
        
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")

def test_uniqueness_check():
    """测试排除池的唯一性检查"""
    print("\n" + "=" * 50)
    print("测试排除池唯一性检查")
    print("=" * 50)
    
    # 创建一个简单的测试案例
    exclusion_pool = [
        {'front': [1, 2, 3, 4, 5], 'back': [1, 2]},
        {'front': [6, 7, 8, 9, 10], 'back': [3, 4]},
        {'front': [11, 12, 13, 14, 15], 'back': [5, 6]}
    ]
    
    # 转换为集合
    exclusion_set = set()
    for combo in exclusion_pool:
        front_tuple = tuple(sorted(combo['front']))
        back_tuple = tuple(sorted(combo['back']))
        exclusion_set.add((front_tuple, back_tuple))
    
    print(f"排除池大小: {len(exclusion_set)}")
    print("排除池内容:")
    for i, combo in enumerate(exclusion_set, 1):
        print(f"  {i}. 前区: {combo[0]} | 后区: {combo[1]}")
    
    # 测试重复检查
    test_candidates = [
        {'front': [1, 2, 3, 4, 5], 'back': [1, 2]},  # 应该被排除
        {'front': [5, 4, 3, 2, 1], 'back': [2, 1]},  # 应该被排除（顺序不同但内容相同）
        {'front': [16, 17, 18, 19, 20], 'back': [7, 8]},  # 应该通过
    ]
    
    print("\n测试候选号码:")
    for i, candidate in enumerate(test_candidates, 1):
        front_tuple = tuple(sorted(candidate['front']))
        back_tuple = tuple(sorted(candidate['back']))
        combo_tuple = (front_tuple, back_tuple)
        
        is_excluded = combo_tuple in exclusion_set
        status = "❌ 被排除" if is_excluded else "✅ 通过"
        
        front_str = ",".join(map(str, candidate['front']))
        back_str = ",".join(map(str, candidate['back']))
        print(f"  {i}. 前区: {front_str} | 后区: {back_str} -> {status}")

def test_prize_check():
    """测试中奖检查功能"""
    print("\n" + "=" * 50)
    print("测试中奖检查功能")
    print("=" * 50)
    
    # 模拟开奖号码
    win_front = [5, 12, 18, 25, 32]
    win_back = [3, 9]
    
    print(f"开奖号码: 前区 {win_front} | 后区 {win_back}")
    
    # 测试不同的中奖情况
    test_cases = [
        {'front': [5, 12, 18, 25, 32], 'back': [3, 9], 'expected': '一等奖'},
        {'front': [5, 12, 18, 25, 32], 'back': [3, 10], 'expected': '二等奖'},
        {'front': [5, 12, 18, 25, 32], 'back': [7, 10], 'expected': '三等奖'},
        {'front': [5, 12, 18, 25, 30], 'back': [3, 9], 'expected': '四等奖'},
        {'front': [5, 12, 18, 25, 30], 'back': [3, 10], 'expected': '五等奖'},
        {'front': [5, 12, 18, 30, 35], 'back': [3, 9], 'expected': '六等奖'},
        {'front': [1, 2, 3, 4, 6], 'back': [1, 2], 'expected': '未中奖'},
    ]
    
    print("\n测试中奖检查:")
    for i, case in enumerate(test_cases, 1):
        result = exclusion_pool_generator._check_prize(
            case['front'], case['back'], win_front, win_back
        )
        
        status = "✅" if result == case['expected'] else "❌"
        front_str = ",".join(map(str, case['front']))
        back_str = ",".join(map(str, case['back']))
        
        print(f"  {i}. 前区: {front_str} | 后区: {back_str}")
        print(f"     预期: {case['expected']} | 实际: {result} {status}")

def main():
    """主测试函数"""
    print("🎯 排除池生成器测试")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 运行各项测试
        test_basic_generation()
        test_uniqueness_check()
        test_prize_check()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()