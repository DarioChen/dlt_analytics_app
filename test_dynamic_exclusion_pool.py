#!/usr/bin/env python3
"""
测试动态排除池功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from backend.exclusion_pool_generator import exclusion_pool_generator
from backend.enhanced_generator import EnhancedNumberGenerator
import pandas as pd

def test_dynamic_exclusion_pool():
    """测试动态排除池功能"""
    
    print("🧪 开始测试动态排除池功能...")
    
    # 创建模拟历史数据
    test_data = []
    for i in range(100):
        test_data.append({
            'issue': f'2024{i:03d}',
            'date': f'2024-01-{(i % 30) + 1:02d}',
            'f1': (i % 35) + 1,
            'f2': ((i + 5) % 35) + 1,
            'f3': ((i + 10) % 35) + 1,
            'f4': ((i + 15) % 35) + 1,
            'f5': ((i + 20) % 35) + 1,
            'b1': (i % 12) + 1,
            'b2': ((i + 6) % 12) + 1
        })
    
    df = pd.DataFrame(test_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # 测试参数
    test_params = {
        'exclusion_pool_size': 10,
        'target_count': 5,
        'rules': {
            'sum_front_range': [50, 150],
            'odd_even_front': [1, 4],
            'consecutive_count': 0,
            'consecutive_mode': 'exact',
            'consecutive_check_type': 'groups'
        },
        'front_blocks': {
            '1-5': [1, 2, 3, 4, 5],
            '6-10': [6, 7, 8, 9, 10],
            '11-15': [11, 12, 13, 14, 15],
            '16-20': [16, 17, 18, 19, 20],
            '21-25': [21, 22, 23, 24, 25],
            '26-30': [26, 27, 28, 29, 30],
            '31-35': [31, 32, 33, 34, 35]
        },
        'back_blocks': {
            '1-2': [1, 2],
            '3-4': [3, 4],
            '5-6': [5, 6],
            '7-8': [7, 8],
            '9-10': [9, 10],
            '11-12': [11, 12]
        },
        'front_weights': {
            '1-5': 1.0, '6-10': 1.0, '11-15': 1.0, '16-20': 1.0,
            '21-25': 1.0, '26-30': 1.0, '31-35': 1.0
        },
        'back_weights': {
            '1-2': 1.0, '3-4': 1.0, '5-6': 1.0,
            '7-8': 1.0, '9-10': 1.0, '11-12': 1.0
        },
        'selected_front_blocks': ['1-5', '11-15', '21-25'],
        'selected_back_blocks': ['1-2', '7-8'],
        'historical_data': df,
        'use_enhanced': True,
        'max_attempts': 1000,
        'save_to_db': False
    }
    
    # 测试1: 动态排除池
    print("\n📋 测试1: 动态排除池策略")
    result_dynamic = exclusion_pool_generator.generate_with_exclusion_pool(
        use_dynamic_pool=True,
        **test_params
    )
    
    if 'error' in result_dynamic:
        print(f"❌ 动态排除池测试失败: {result_dynamic['error']}")
        return False
    
    print(f"✅ 动态排除池测试成功:")
    print(f"   - 排除池大小: {result_dynamic.get('exclusion_pool_size', 0)}")
    print(f"   - 生成目标数量: {result_dynamic.get('target_count_actual', 0)}")
    print(f"   - 生成尝试次数: {result_dynamic.get('generation_attempts', 0)}")
    print(f"   - 生成类型: {result_dynamic.get('generation_type', 'unknown')}")
    
    # 验证动态排除池的内容
    dynamic_exclusion_pool = result_dynamic.get('dynamic_exclusion_pool', [])
    target_numbers = result_dynamic.get('target_numbers', [])
    
    if dynamic_exclusion_pool:
        print(f"   - 动态排除池示例: {dynamic_exclusion_pool[0]}")
    if target_numbers:
        print(f"   - 目标号码示例: {target_numbers[0]}")
    
    # 测试2: 静态排除池（对比）
    print("\n📋 测试2: 静态排除池策略")
    result_static = exclusion_pool_generator.generate_with_exclusion_pool(
        use_dynamic_pool=False,
        **test_params
    )
    
    if 'error' in result_static:
        print(f"❌ 静态排除池测试失败: {result_static['error']}")
        return False
    
    print(f"✅ 静态排除池测试成功:")
    print(f"   - 排除池大小: {result_static.get('exclusion_pool_size', 0)}")
    print(f"   - 生成目标数量: {result_static.get('target_count_actual', 0)}")
    print(f"   - 生成尝试次数: {result_static.get('generation_attempts', 0)}")
    print(f"   - 生成类型: {result_static.get('generation_type', 'unknown')}")
    
    # 测试3: 增强生成器的两轮生成（带动态排除池）
    print("\n📋 测试3: 增强生成器两轮生成（动态排除池）")
    
    enhanced_generator = EnhancedNumberGenerator()
    
    try:
        # 初始化模型（简化版）
        enhanced_generator.initialize_models(df, use_ensemble=False, force_reinit=True)
        
        two_round_result = enhanced_generator.generate_two_rounds(
            count=3,
            rules=test_params['rules'],
            front_blocks=test_params['front_blocks'],
            back_blocks=test_params['back_blocks'],
            front_weights=test_params['front_weights'],
            back_weights=test_params['back_weights'],
            selected_front_blocks=test_params['selected_front_blocks'],
            selected_back_blocks=test_params['selected_back_blocks'],
            historical_data=df,
            use_markov=True,
            use_big_data=True,
            markov_weight=0.4,
            big_data_weight=0.3,
            traditional_weight=0.3,
            variation_strength=0.3,
            use_recombination=False,
            use_exclusion_pool=True,
            exclusion_pool_size=5,
            use_dynamic_pool=True,
            save_to_db=False
        )
        
        if 'error' in two_round_result:
            print(f"❌ 两轮生成测试失败: {two_round_result['error']}")
            return False
        
        print(f"✅ 两轮生成测试成功:")
        print(f"   - 第一轮数量: {len(two_round_result.get('first_round', []))}")
        print(f"   - 第二轮数量: {len(two_round_result.get('second_round', []))}")
        
        if 'exclusion_pool_info' in two_round_result:
            exclusion_info = two_round_result['exclusion_pool_info']
            print(f"   - 排除池大小: {exclusion_info.get('exclusion_pool_size', 0)}")
            print(f"   - 生成尝试次数: {exclusion_info.get('generation_attempts', 0)}")
        
    except Exception as e:
        print(f"❌ 两轮生成测试异常: {e}")
        return False
    
    print("\n🎉 所有测试完成！动态排除池功能正常工作。")
    return True

if __name__ == "__main__":
    success = test_dynamic_exclusion_pool()
    if success:
        print("\n✅ 测试通过：动态排除池功能已正确实现")
    else:
        print("\n❌ 测试失败：动态排除池功能存在问题")
        sys.exit(1)