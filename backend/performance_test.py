# backend/performance_test.py
"""
系统性能测试模块：评估优化后各组件的性能表现
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('performance_test')

class PerformanceTester:
    """
    性能测试器，用于测量各组件的执行时间和资源消耗
    """
    
    def __init__(self):
        self.results = {
            "tests": {},
            "summary": {},
            "comparison": {}
        }
        self.baseline_times = {}
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        测量函数执行时间
        
        Args:
            func: 要测试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            包含执行时间和结果的字典
        """
        start_time = time.time()
        start_cpu = time.process_time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_cpu = time.process_time()
        
        return {
            "wall_time": end_time - start_time,
            "cpu_time": end_cpu - start_cpu,
            "success": success,
            "error": error,
            "result": result
        }
    
    def run_benchmark(self, test_name: str, func: Callable, iterations: int = 10, 
                     *args, **kwargs) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            test_name: 测试名称
            func: 测试函数
            iterations: 迭代次数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            基准测试结果
        """
        logger.info(f"Running benchmark: {test_name} for {iterations} iterations")
        
        times = []
        cpu_times = []
        success_count = 0
        
        for i in range(iterations):
            logger.debug(f"Iteration {i+1}/{iterations}")
            result = self.measure_execution_time(func, *args, **kwargs)
            
            if result["success"]:
                times.append(result["wall_time"])
                cpu_times.append(result["cpu_time"])
                success_count += 1
            else:
                logger.error(f"Iteration {i+1} failed: {result['error']}")
        
        # 计算统计数据
        stats = {
            "test_name": test_name,
            "iterations": iterations,
            "successful_runs": success_count,
            "wall_time": {
                "mean": np.mean(times) if times else 0,
                "median": np.median(times) if times else 0,
                "std": np.std(times) if times else 0,
                "min": np.min(times) if times else 0,
                "max": np.max(times) if times else 0
            },
            "cpu_time": {
                "mean": np.mean(cpu_times) if cpu_times else 0,
                "median": np.median(cpu_times) if cpu_times else 0,
                "std": np.std(cpu_times) if cpu_times else 0,
                "min": np.min(cpu_times) if cpu_times else 0,
                "max": np.max(cpu_times) if cpu_times else 0
            },
            "success_rate": success_count / iterations * 100
        }
        
        self.results["tests"][test_name] = stats
        logger.info(f"Benchmark completed: {test_name} - Success rate: {stats['success_rate']}%, Mean time: {stats['wall_time']['mean']:.4f}s")
        
        return stats
    
    def compare_versions(self, test_name: str, old_func: Callable, new_func: Callable,
                        iterations: int = 10, *args, **kwargs) -> Dict[str, Any]:
        """
        比较两个版本的性能
        
        Args:
            test_name: 测试名称
            old_func: 旧版本函数
            new_func: 新版本函数
            iterations: 迭代次数
            *args: 共同参数
            **kwargs: 共同关键字参数
            
        Returns:
            比较结果
        """
        logger.info(f"Comparing versions for: {test_name}")
        
        # 运行旧版本测试
        old_stats = self.run_benchmark(f"{test_name}_old", old_func, iterations, *args, **kwargs)
        
        # 运行新版本测试
        new_stats = self.run_benchmark(f"{test_name}_new", new_func, iterations, *args, **kwargs)
        
        # 计算改进
        improvement = {
            "wall_time": (1 - new_stats["wall_time"]["mean"] / old_stats["wall_time"]["mean"]) * 100 if old_stats["wall_time"]["mean"] > 0 else 0,
            "cpu_time": (1 - new_stats["cpu_time"]["mean"] / old_stats["cpu_time"]["mean"]) * 100 if old_stats["cpu_time"]["mean"] > 0 else 0,
            "success_rate_diff": new_stats["success_rate"] - old_stats["success_rate"]
        }
        
        comparison = {
            "test_name": test_name,
            "old_version": old_stats,
            "new_version": new_stats,
            "improvement": improvement
        }
        
        self.results["comparison"][test_name] = comparison
        logger.info(f"Comparison completed: {test_name} - Wall time improvement: {improvement['wall_time']:.2f}%")
        
        return comparison
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        生成测试总结
        """
        all_tests = list(self.results["tests"].values())
        
        if not all_tests:
            return {"message": "No tests run"}
        
        # 计算总体统计
        total_mean_time = np.mean([t["wall_time"]["mean"] for t in all_tests])
        total_success_rate = np.mean([t["success_rate"] for t in all_tests])
        
        # 找出最快和最慢的测试
        fastest_test = min(all_tests, key=lambda x: x["wall_time"]["mean"])
        slowest_test = max(all_tests, key=lambda x: x["wall_time"]["mean"])
        
        # 找出成功率最高的测试
        most_reliable_test = max(all_tests, key=lambda x: x["success_rate"])
        
        summary = {
            "total_tests": len(all_tests),
            "total_mean_execution_time": total_mean_time,
            "average_success_rate": total_success_rate,
            "fastest_test": {
                "name": fastest_test["test_name"],
                "mean_time": fastest_test["wall_time"]["mean"]
            },
            "slowest_test": {
                "name": slowest_test["test_name"],
                "mean_time": slowest_test["wall_time"]["mean"]
            },
            "most_reliable_test": {
                "name": most_reliable_test["test_name"],
                "success_rate": most_reliable_test["success_rate"]
            },
            "comparison_count": len(self.results["comparison"]),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加比较结果总结
        if self.results["comparison"]:
            avg_improvements = {
                "wall_time": np.mean([c["improvement"]["wall_time"] for c in self.results["comparison"].values()]),
                "cpu_time": np.mean([c["improvement"]["cpu_time"] for c in self.results["comparison"].values()]),
                "success_rate": np.mean([c["improvement"]["success_rate_diff"] for c in self.results["comparison"].values()])
            }
            summary["average_improvements"] = avg_improvements
        
        self.results["summary"] = summary
        return summary
    
    def generate_report(self, output_file: str = None) -> str:
        """
        生成性能测试报告
        
        Args:
            output_file: 输出文件路径（可选）
            
        Returns:
            报告内容
        """
        # 生成总结
        summary = self.generate_summary()
        
        # 构建报告
        report = []
        report.append("# 系统性能测试报告")
        report.append(f"生成时间: {summary['date']}")
        report.append("")
        
        # 总结部分
        report.append("## 总体性能总结")
        report.append(f"- 测试总数: {summary['total_tests']}")
        report.append(f"- 平均执行时间: {summary['total_mean_execution_time']:.4f} 秒")
        report.append(f"- 平均成功率: {summary['average_success_rate']:.2f}%")
        report.append(f"- 最快测试: {summary['fastest_test']['name']} ({summary['fastest_test']['mean_time']:.4f} 秒)")
        report.append(f"- 最慢测试: {summary['slowest_test']['name']} ({summary['slowest_test']['mean_time']:.4f} 秒)")
        report.append(f"- 最可靠测试: {summary['most_reliable_test']['name']} ({summary['most_reliable_test']['success_rate']:.2f}%)")
        report.append("")
        
        # 性能改进总结
        if "average_improvements" in summary:
            report.append("## 性能改进总结")
            report.append(f"- 平均执行时间改进: {summary['average_improvements']['wall_time']:.2f}%")
            report.append(f"- 平均CPU时间改进: {summary['average_improvements']['cpu_time']:.2f}%")
            report.append(f"- 平均成功率变化: {summary['average_improvements']['success_rate']:+.2f}%")
            report.append("")
        
        # 详细测试结果
        report.append("## 详细测试结果")
        for test_name, stats in self.results["tests"].items():
            report.append(f"### {test_name}")
            report.append(f"- 迭代次数: {stats['iterations']}")
            report.append(f"- 成功运行: {stats['successful_runs']} ({stats['success_rate']:.2f}%)")
            report.append(f"- 执行时间 (秒): 平均={stats['wall_time']['mean']:.4f}, 中位数={stats['wall_time']['median']:.4f}, 标准差={stats['wall_time']['std']:.4f}")
            report.append(f"- CPU时间 (秒): 平均={stats['cpu_time']['mean']:.4f}, 中位数={stats['cpu_time']['median']:.4f}")
            report.append("")
        
        # 版本比较
        if self.results["comparison"]:
            report.append("## 版本性能比较")
            for test_name, comparison in self.results["comparison"].items():
                report.append(f"### {test_name}")
                report.append(f"- 旧版本平均执行时间: {comparison['old_version']['wall_time']['mean']:.4f} 秒")
                report.append(f"- 新版本平均执行时间: {comparison['new_version']['wall_time']['mean']:.4f} 秒")
                report.append(f"- 性能改进: {comparison['improvement']['wall_time']:+.2f}%")
                report.append(f"- 成功率变化: {comparison['improvement']['success_rate_diff']:+.2f}%")
                report.append("")
        
        # 生成文本报告
        report_text = "\n".join(report)
        
        # 保存到文件
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Performance report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report_text

# 针对大乐透分析应用的特定测试
def create_app_performance_tests(tester: PerformanceTester, app_components: Dict) -> Dict[str, Any]:
    """
    创建大乐透分析应用的性能测试
    
    Args:
        tester: 性能测试器实例
        app_components: 应用组件字典，包含需要测试的函数和参数
        
    Returns:
        测试结果
    """
    # 确保必要的组件可用
    required_components = [
        "generate_numbers", 
        "compute_weights", 
        "predict_next_numbers",
        "optimize_parameters",
        "run_backtest"
    ]
    
    missing_components = [comp for comp in required_components if comp not in app_components]
    if missing_components:
        logger.error(f"Missing required components: {missing_components}")
        return {"error": f"Missing components: {missing_components}"}
    
    # 测试号码生成性能
    generate_params = app_components.get("generate_params", {})
    generate_stats = tester.run_benchmark(
        "号码生成性能", 
        app_components["generate_numbers"],
        iterations=50,  # 增加迭代次数以获得更准确的结果
        **generate_params
    )
    
    # 测试权重计算性能
    weights_params = app_components.get("weights_params", {})
    weights_stats = tester.run_benchmark(
        "权重计算性能",
        app_components["compute_weights"],
        iterations=30,
        **weights_params
    )
    
    # 测试预测性能
    predict_params = app_components.get("predict_params", {})
    predict_stats = tester.run_benchmark(
        "号码预测性能",
        app_components["predict_next_numbers"],
        iterations=20,
        **predict_params
    )
    
    # 测试参数优化性能
    optimize_params = app_components.get("optimize_params", {})
    # 为了快速测试，可以使用较少的迭代次数
    optimize_stats = tester.run_benchmark(
        "参数优化性能",
        app_components["optimize_parameters"],
        iterations=5,
        **optimize_params
    )
    
    # 测试回测性能
    backtest_params = app_components.get("backtest_params", {})
    backtest_stats = tester.run_benchmark(
        "回测分析性能",
        app_components["run_backtest"],
        iterations=10,
        **backtest_params
    )
    
    # 如果提供了优化前后的版本，进行比较
    comparison_results = {}
    
    # 例如，比较不同的权重计算方法
    if "compute_weights_old" in app_components and "compute_weights_new" in app_components:
        compare_weights = tester.compare_versions(
            "权重计算方法比较",
            app_components["compute_weights_old"],
            app_components["compute_weights_new"],
            iterations=20,
            **weights_params
        )
        comparison_results["weights"] = compare_weights
    
    # 比较不同的优化策略
    if "optimize_parameters_old" in app_components and "optimize_parameters_new" in app_components:
        compare_optimize = tester.compare_versions(
            "参数优化策略比较",
            app_components["optimize_parameters_old"],
            app_components["optimize_parameters_new"],
            iterations=3,  # 贝叶斯优化可能比较耗时
            **optimize_params
        )
        comparison_results["optimize"] = compare_optimize
    
    return {
        "individual_tests": {
            "generate": generate_stats,
            "weights": weights_stats,
            "predict": predict_stats,
            "optimize": optimize_stats,
            "backtest": backtest_stats
        },
        "comparisons": comparison_results,
        "summary": tester.generate_summary()
    }

def run_comprehensive_performance_test(app_components: Dict, 
                                       output_report: str = "performance_report.md") -> Dict[str, Any]:
    """
    运行综合性能测试
    
    Args:
        app_components: 应用组件字典
        output_report: 输出报告文件路径
        
    Returns:
        完整的测试结果
    """
    tester = PerformanceTester()
    
    logger.info("Starting comprehensive performance test")
    
    # 运行应用特定测试
    results = create_app_performance_tests(tester, app_components)
    
    # 生成并保存报告
    report = tester.generate_report(output_report)
    
    logger.info(f"Performance test completed. Report saved to {output_report}")
    
    return {
        "test_results": results,
        "report_path": output_report,
        "report_text": report
    }
