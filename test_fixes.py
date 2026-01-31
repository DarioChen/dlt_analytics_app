#!/usr/bin/env python3
"""
测试修复后的两轮生成功能 - 严格验证版
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
    """测试两轮生成功能 - 严格验证第二轮号码来源"""
    print("🧪 开始测试两轮生成功能（严格验证版）...")
    
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
    
    # 测试不同的count值和模式
    test_cases = [
        {"count": 5, "mode": "traditional", "name": "传统模式"},
        {"count": 5, "mode": "recombination", "name": "重组模式"},
        {"count": 3, "mode": "traditional", "name": "传统模式(3个)"},
        {"count": 3, "mode": "recombination", "name": "重组模式(3个)"},
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        count = test_case["count"]
        use_recombination = test_case["mode"] == "recombination"
        test_name = test_case["name"]
        
        print(f"\n🎯 测试 {test_name}（目标：{count}个）...")
        
        try:
            result = enhanced_generator.generate_two_rounds(
                count=count,
                rules={
                    "sum_front_range": [70, 140],
                    "odd_even_front": [2, 3],
                    "consecutive_count": 0,
                    "consecutive_mode": "max"
                },
                historical_data=df.tail(20),
                use_recombination=use_recombination,
                variation_strength=0.3
            )
            
            if 'error' in result:
                print(f"    ❌ 生成失败: {result['error']}")
                all_passed = False
                continue
            
            first_round = result.get('first_round', [])
            second_round = result.get('second_round', [])
            
            print(f"    ✅ 第一轮生成: {len(first_round)} 个候选（期望：{count}）")
            print(f"    ✅ 第二轮生成: {len(second_round)} 个候选（期望：{count}）")
            
            # 验证数量是否正确
            if len(first_round) != count or len(second_round) != count:
                print(f"    ❌ 数量验证失败：第一轮{len(first_round)}，第二轮{len(second_round)}，期望{count}")
                all_passed = False
                continue
            
            # 提取第一轮所有号码
            first_front_pool = set()
            first_back_pool = set()
            for candidate in first_round:
                first_front_pool.update(candidate['front'])
                first_back_pool.update(candidate['back'])
            
            print(f"    📊 第一轮号码池 - 前区: {sorted(first_front_pool)} ({len(first_front_pool)}个)")
            print(f"    📊 第一轮号码池 - 后区: {sorted(first_back_pool)} ({len(first_back_pool)}个)")
            
            # 验证第二轮每个候选号码
            validation_passed = True
            for i, candidate in enumerate(second_round, 1):
                front_nums = set(candidate['front'])
                back_nums = set(candidate['back'])
                
                # 检查前区号码是否都在第一轮号码池中
                front_outside = front_nums - first_front_pool
                back_outside = back_nums - first_back_pool
                
                if front_outside:
                    print(f"    ❌ 第二轮候选{i}前区包含第一轮之外的号码: {sorted(front_outside)}")
                    print(f"        候选前区: {sorted(candidate['front'])}")
                    print(f"        第一轮前区池: {sorted(first_front_pool)}")
                    validation_passed = False
                
                if back_outside:
                    print(f"    ❌ 第二轮候选{i}后区包含第一轮之外的号码: {sorted(back_outside)}")
                    print(f"        候选后区: {sorted(candidate['back'])}")
                    print(f"        第一轮后区池: {sorted(first_back_pool)}")
                    validation_passed = False
                
                if not front_outside and not back_outside:
                    print(f"    ✅ 第二轮候选{i}号码池验证通过")
            
            if validation_passed:
                print(f"    🎉 {test_name}严格验证通过！")
            else:
                print(f"    ❌ {test_name}严格验证失败！")
                all_passed = False
            
            # 显示第一轮和第二轮的详细号码
            print(f"    📋 详细号码对比:")
            print(f"        第一轮:")
            for i, candidate in enumerate(first_round, 1):
                print(f"          {i}. 前区: {candidate['front']}, 后区: {candidate['back']}")
            print(f"        第二轮:")
            for i, candidate in enumerate(second_round, 1):
                print(f"          {i}. 前区: {candidate['front']}, 后区: {candidate['back']}")
            
        except Exception as e:
            print(f"    ❌ {test_name}测试失败: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    print("\n🎉 所有测试完成！")
    return all_passed

if __name__ == "__main__":
    success = test_two_round_generation()
    if success:
        print("✅ 严格验证测试通过")
        exit(0)
    else:
        print("❌ 严格验证测试失败")
        exit(1)