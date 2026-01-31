#!/usr/bin/env python3
"""
测试进化优化器的简单脚本
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_simple_evolutionary():
    """测试简化版进化优化器"""
    print("测试简化版进化优化器...")
    
    try:
        from backend.simple_evolutionary import create_simple_evolutionary_optimizer, SimpleEvolutionaryConfig
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'f1': [1, 5, 10, 15, 20, 25, 30, 2, 7, 12],
            'f2': [2, 6, 11, 16, 21, 26, 31, 3, 8, 13],
            'f3': [3, 7, 12, 17, 22, 27, 32, 4, 9, 14],
            'f4': [4, 8, 13, 18, 23, 28, 33, 5, 10, 15],
            'f5': [5, 9, 14, 19, 24, 29, 34, 6, 11, 16],
            'b1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'date': [datetime.now() - timedelta(days=i) for i in range(10)]
        })
        
        # 创建优化器
        config = SimpleEvolutionaryConfig(
            population_size=20,
            generations=10,
            mutation_rate=0.15
        )
        optimizer = create_simple_evolutionary_optimizer(config)
        
        # 运行优化
        result = optimizer.evolve_optimal_numbers(test_data, target_count=1)
        
        print("✅ 简化版进化优化器测试成功！")
        print(f"最优号码: {result['best_numbers']}")
        print(f"适应度分数: {result['fitness_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简化版进化优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_evolutionary():
    """测试完整版进化优化器"""
    print("测试完整版进化优化器...")
    
    try:
        from backend.evolutionary_optimizer import create_evolutionary_optimizer, EvolutionaryConfig
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'f1': [1, 5, 10, 15, 20, 25, 30, 2, 7, 12],
            'f2': [2, 6, 11, 16, 21, 26, 31, 3, 8, 13],
            'f3': [3, 7, 12, 17, 22, 27, 32, 4, 9, 14],
            'f4': [4, 8, 13, 18, 23, 28, 33, 5, 10, 15],
            'f5': [5, 9, 14, 19, 24, 29, 34, 6, 11, 16],
            'b1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'date': [datetime.now() - timedelta(days=i) for i in range(10)]
        })
        
        # 创建优化器
        config = EvolutionaryConfig(
            population_size=20,
            generations=10,
            mutation_rate=0.15,
            multi_objective=True
        )
        optimizer = create_evolutionary_optimizer(config)
        
        if optimizer is None:
            print("⚠️ 完整版进化优化器创建失败，可能缺少依赖")
            return False
        
        # 运行优化
        result = optimizer.evolve_optimal_numbers(test_data, target_count=1)
        
        print("✅ 完整版进化优化器测试成功！")
        print(f"最优号码: {result['best_numbers']}")
        print(f"适应度分数: {result['fitness_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整版进化优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧬 进化优化器测试开始...")
    print("=" * 50)
    
    # 测试简化版
    simple_success = test_simple_evolutionary()
    print()
    
    # 测试完整版
    full_success = test_full_evolutionary()
    print()
    
    # 总结
    print("=" * 50)
    print("📊 测试结果总结:")
    print(f"简化版进化优化器: {'✅ 成功' if simple_success else '❌ 失败'}")
    print(f"完整版进化优化器: {'✅ 成功' if full_success else '❌ 失败'}")
    
    if simple_success or full_success:
        print("🎉 至少有一个版本可以正常工作！")
        return True
    else:
        print("😞 所有版本都测试失败，请检查代码和依赖")
        return False

if __name__ == "__main__":
    main()