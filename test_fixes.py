#!/usr/bin/env python3
"""
测试修复后的两轮生成功能 - 更新版
"""
import pandas as pd
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.enhanced_generator import EnhancedNumberGenerator
from backend.db import init_db, session_scope, Draw
from backend.analysis import dataframe_from_draws

def test_two_round_generation():
    """测试两轮生成功能"""
    print("🧪 开始测试两轮生成功能（更新版）...")
    
    # 初始化数据库并获取数据
    init_db()
    with session_scope() as s:
        rows = [dict(
            issue=d.issue, date=d.date.isoformat(),
            f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4, f5=d.f5,
            b1=d.b1, b2=d.b2,
            sales=d.sales, pool=d.pool
        ) for d in s.query(Draw).order_by(Draw.issue.desc()).limit(100).all()]
    
    if not rows:
        print("❌ 没有找到历史数据，无法进行测试")
        return False
    
    df = dataframe_from_draws(rows)
    print(f"✅ 加载了 {len(df)} 条历史数据")
    
    # 初始化增强生成器
    enhanced_generator = EnhancedNumberGenerator()
    
    try:
        print("🔄 初始化增强生成器...")
        enhanced_generator.initialize_models(df, use_ensemble=True)
        print("✅ 增强生成器初始化成功")
    except Exception as e:
        print(f"❌ 增强生成器初始化失败: {e}")
        return False
    
    # 测试不同的count值
    test_counts = [3, 5, 7]
    
    for count in test_counts:
        print(f"\n🎯 测试生成 {count} 个候选号码...")
        
        # 测试传统模式两轮生成
        print(f"  传统模式测试（目标：{count}个）...")
        try:
            result_traditional = enhanced_generator.generate_two_rounds(
                count=count,
                rules={
                    "sum_front_range": [70, 140],
                    "odd_even_front": [2, 3],
                    "consecutive_count": 0,
                    "consecutive_mode": "max"
                },
                historical_data=df.tail(20),
                use_recombination=False,
                variation_strength=0.3
            )
            
            if 'error' in result_traditional:
                print(f"    ❌ 传统模式生成失败: {result_traditional['error']}")
                continue
            
            first_round = result_traditional.get('first_round', [])
            second_round = result_traditional.get('second_round', [])
            
            print(f"    ✅ 第一轮生成: {len(first_round)} 个候选（期望：{count}）")
            print(f"    ✅ 第二轮生成: {len(second_round)} 个候选（期望：{count}）")
            
            # 验证数量是否正确
            if len(first_round) == count and len(second_round) == count:
                print(f"    🎉 传统模式数量验证通过！")
            else:
                print(f"    ⚠️  传统模式数量验证失败：第一轮{len(first_round)}，第二轮{len(second_round)}，期望{count}")
            
        except Exception as e:
            print(f"    ❌ 传统模式测试失败: {e}")
            continue
        
        # 测试重组模式两轮生成
        print(f"  重组模式测试（目标：{count}个）...")
        try:
            result_recombination = enhanced_generator.generate_two_rounds(
                count=count,
                rules={
                    "sum_front_range": [70, 140],
                    "odd_even_front": [2, 3],
                    "consecutive_count": 0,
                    "consecutive_mode": "max"
                },
                historical_data=df.tail(20),
                use_recombination=True,
                variation_strength=0.3
            )
            
            if 'error' in result_recombination:
                print(f"    ❌ 重组模式生成失败: {result_recombination['error']}")
                continue
            
            first_round_r = result_recombination.get('first_round', [])
            second_round_r = result_recombination.get('second_round', [])
            
            print(f"    ✅ 第一轮生成: {len(first_round_r)} 个候选（期望：{count}）")
            print(f"    ✅ 第二轮生成: {len(second_round_r)} 个候选（期望：{count}）")
            
            # 验证数量是否正确
            if len(first_round_r) == count and len(second_round_r) == count:
                print(f"    🎉 重组模式数量验证通过！")
            else:
                print(f"    ⚠️  重组模式数量验证失败：第一轮{len(first_round_r)}，第二轮{len(second_round_r)}，期望{count}")
            
            # 验证重组模式的号码池限制
            if first_round_r and second_round_r:
                first_front_pool = set()
                first_back_pool = set()
                for candidate in first_round_r:
                    first_front_pool.update(candidate['front'])
                    first_back_pool.update(candidate['back'])
                
                second_front_nums = set()
                second_back_nums = set()
                for candidate in second_round_r:
                    second_front_nums.update(candidate['front'])
                    second_back_nums.update(candidate['back'])
                
                # 检查第二轮号码是否都来自第一轮
                if second_front_nums.issubset(first_front_pool) and second_back_nums.issubset(first_back_pool):
                    print("    ✅ 重组模式号码池验证通过")
                else:
                    print("    ⚠️  重组模式号码池验证失败")
            
        except Exception as e:
            print(f"    ❌ 重组模式测试失败: {e}")
            continue
    
    print("\n🎉 所有测试完成！")
    return True

if __name__ == "__main__":
    success = test_two_round_generation()
    if success:
        print("✅ 测试通过")
        exit(0)
    else:
        print("❌ 测试失败")
        exit(1)