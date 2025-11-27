# backend/backtest.py
"""
回测与验证模块：增强的回测分析功能，重点关注高额奖项命中分析
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 奖项规则定义
PRIZE_RULES = [
    ("一等奖", lambda fc, bc: fc == 5 and bc == 2),
    ("二等奖", lambda fc, bc: fc == 5 and bc == 1),
    ("三等奖", lambda fc, bc: fc == 5 and bc == 0),
    ("四等奖", lambda fc, bc: fc >= 4 and bc == 2),
    ("五等奖", lambda fc, bc: fc >= 4 and bc == 1),
    ("六等奖", lambda fc, bc: fc >= 3 and bc == 2),
    ("七等奖", lambda fc, bc: fc >= 4 and bc == 0),
    ("八等奖", lambda fc, bc: (fc >= 3 and bc >= 1) or (fc == 2 and bc == 2)),
    ("九等奖", lambda fc, bc: (fc >= 3) or (fc == 1 and bc == 2) or (fc == 2 and bc == 1) or (bc == 2))
]

# 默认奖金额度
PRIZE_PAYOUT_DEFAULT = {
    "一等奖": 5000000.0,
    "二等奖": 1500000.0,
    "三等奖": 10000.0,
    "四等奖": 3000.0,
    "五等奖": 300.0,
    "六等奖": 200.0,
    "七等奖": 100.0,
    "八等奖": 15.0,
    "九等奖": 5.0,
    "未中奖": 0.0
}

# 高额奖项列表
HIGH_PRIZE_LEVELS = ["一等奖", "二等奖", "三等奖"]

class BacktestAnalyzer:
    """增强的回测分析器"""
    
    def __init__(self, prize_amounts: Optional[Dict[str, float]] = None):
        """
        初始化回测分析器
        
        Args:
            prize_amounts: 奖项金额字典，默认为PRIZE_PAYOUT_DEFAULT
        """
        self.prize_amounts = prize_amounts or PRIZE_PAYOUT_DEFAULT
    
    def check_prize(self, front_nums: List[int], back_nums: List[int], 
                    win_front: List[int], win_back: List[int]) -> str:
        """
        检查中奖等级
        
        Args:
            front_nums: 预测的前区号码
            back_nums: 预测的后区号码
            win_front: 实际中奖的前区号码
            win_back: 实际中奖的后区号码
        
        Returns:
            奖项名称
        """
        front_match = len(set(front_nums) & set(win_front))
        back_match = len(set(back_nums) & set(win_back))
        
        for name, condition in PRIZE_RULES:
            if condition(front_match, back_match):
                return name
        
        return "未中奖"
    
    def run_backtest(self, historical_data: pd.DataFrame, predictions: List[Dict],
                    ticket_cost: float = 2.0, test_periods: int = 50) -> Dict:
        """
        运行回测并生成详细报告
        
        Args:
            historical_data: 历史开奖数据
            predictions: 预测结果列表，格式为[{"front": [], "back": []}, ...]
            ticket_cost: 单注成本
            test_periods: 测试期数
            
        Returns:
            回测结果字典
        """
        # 准备测试数据
        test_data = historical_data.tail(test_periods).reset_index(drop=True)
        
        # 确保预测数量与测试期数匹配
        if len(predictions) < len(test_data):
            print(f"警告：预测数量({len(predictions)})少于测试期数({len(test_data)})")
            predictions = predictions[:len(test_data)]
        elif len(predictions) > len(test_data):
            predictions = predictions[:len(test_data)]
        
        results = {
            "periods": len(test_data),
            "total_bets": 0,
            "total_cost": 0.0,
            "total_return": 0.0,
            "prize_distribution": {p: 0 for p in self.prize_amounts},
            "high_prize_hits": [],  # 高额奖项命中详情
            "hit_details": [],  # 每期命中详情
            "profit_curve": [],  # 盈亏曲线
            "metrics": {}
        }
        
        current_profit = 0.0
        
        for i, (_, row) in enumerate(test_data.iterrows()):
            if i >= len(predictions):
                break
            
            pred = predictions[i]
            win_front = [row[f"f{i+1}"] for i in range(5)]
            win_back = [row["b1"], row["b2"]]
            
            # 统计每注的结果
            for j, (front, back) in enumerate(zip(pred.get("front", []), pred.get("back", []))):
                prize = self.check_prize(front, back, win_front, win_back)
                prize_amount = self.prize_amounts.get(prize, 0.0)
                
                results["total_bets"] += 1
                results["total_cost"] += ticket_cost
                results["total_return"] += prize_amount
                results["prize_distribution"][prize] += 1
                
                current_profit += (prize_amount - ticket_cost)
                
                # 记录高额奖项命中
                if prize in HIGH_PRIZE_LEVELS:
                    hit_record = {
                        "period": row["issue"],
                        "date": row["date"],
                        "prize": prize,
                        "amount": prize_amount,
                        "prediction": {"front": front, "back": back},
                        "winning": {"front": win_front, "back": win_back}
                    }
                    results["high_prize_hits"].append(hit_record)
                
                # 记录命中详情
                hit_detail = {
                    "period": row["issue"],
                    "date": row["date"],
                    "prediction_index": j,
                    "front_match": len(set(front) & set(win_front)),
                    "back_match": len(set(back) & set(win_back)),
                    "prize": prize,
                    "profit": prize_amount - ticket_cost
                }
                results["hit_details"].append(hit_detail)
            
            # 更新盈亏曲线
            results["profit_curve"].append({
                "period": row["issue"],
                "date": row["date"],
                "cumulative_profit": current_profit
            })
        
        # 计算核心指标
        results["metrics"] = self._calculate_metrics(results)
        
        return results
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """
        计算回测指标
        """
        metrics = {}
        
        # 基础指标
        metrics["total_bets"] = results["total_bets"]
        metrics["total_cost"] = results["total_cost"]
        metrics["total_return"] = results["total_return"]
        metrics["profit"] = results["total_return"] - results["total_cost"]
        
        # ROI相关
        if results["total_cost"] > 0:
            metrics["roi"] = metrics["profit"] / results["total_cost"] * 100
            metrics["roi_per_bet"] = metrics["profit"] / results["total_bets"]
        else:
            metrics["roi"] = 0.0
            metrics["roi_per_bet"] = 0.0
        
        # 命中率
        metrics["hit_rate"] = sum(1 for p, cnt in results["prize_distribution"].items() 
                                 if p != "未中奖" and cnt > 0) / results["total_bets"] * 100 if results["total_bets"] > 0 else 0.0
        
        # 高额奖项指标
        high_prize_count = sum(results["prize_distribution"].get(p, 0) for p in HIGH_PRIZE_LEVELS)
        metrics["high_prize_count"] = high_prize_count
        metrics["high_prize_rate"] = high_prize_count / results["total_bets"] * 100 if results["total_bets"] > 0 else 0.0
        
        # 最大回撤
        if results["profit_curve"]:
            profits = [p["cumulative_profit"] for p in results["profit_curve"]]
            metrics["max_drawdown"] = self._calculate_max_drawdown(profits)
        
        # 各奖项详细统计
        metrics["prize_stats"] = {}
        for prize, count in results["prize_distribution"].items():
            if count > 0:
                prize_profit = (self.prize_amounts.get(prize, 0.0) * count) - (count * 2.0)  # 假设单注2元
                metrics["prize_stats"][prize] = {
                    "count": count,
                    "percentage": count / results["total_bets"] * 100,
                    "total_return": self.prize_amounts.get(prize, 0.0) * count,
                    "profit": prize_profit,
                    "roi": prize_profit / (count * 2.0) * 100 if count > 0 else 0.0
                }
        
        return metrics
    
    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """
        计算最大回撤
        """
        if not profits:
            return 0.0
        
        max_drawdown = 0.0
        peak = profits[0]
        
        for profit in profits:
            if profit > peak:
                peak = profit
            drawdown = (peak - profit) / (peak if peak > 0 else 1) * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def analyze_high_prize_patterns(self, results: Dict) -> Dict:
        """
        分析高额奖项命中模式
        
        Args:
            results: 回测结果
            
        Returns:
            高额奖项分析结果
        """
        high_prize_analysis = {
            "overview": {},
            "patterns": {},
            "number_frequency": {"front": {}, "back": {}},
            "block_analysis": {},
            "recommendations": []
        }
        
        # 基本统计
        high_prize_analysis["overview"] = {
            "total_high_prizes": results["metrics"]["high_prize_count"],
            "hit_rate": results["metrics"]["high_prize_rate"],
            "by_level": {p: results["prize_distribution"].get(p, 0) for p in HIGH_PRIZE_LEVELS}
        }
        
        # 分析命中的号码模式
        if results["high_prize_hits"]:
            # 统计高频出现的号码
            front_freq = {}
            back_freq = {}
            
            for hit in results["high_prize_hits"]:
                for num in hit["prediction"]["front"]:
                    front_freq[num] = front_freq.get(num, 0) + 1
                for num in hit["prediction"]["back"]:
                    back_freq[num] = back_freq.get(num, 0) + 1
            
            high_prize_analysis["number_frequency"]["front"] = front_freq
            high_prize_analysis["number_frequency"]["back"] = back_freq
            
            # 生成推荐
            high_front_nums = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)[:8]
            high_back_nums = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)[:4]
            
            if high_front_nums:
                high_prize_analysis["recommendations"].append(
                    f"高额奖项中高频出现的前区号码（推荐关注）：{[num for num, cnt in high_front_nums]}"
                )
            if high_back_nums:
                high_prize_analysis["recommendations"].append(
                    f"高额奖项中高频出现的后区号码（推荐关注）：{[num for num, cnt in high_back_nums]}"
                )
        
        # 生成策略建议
        if results["metrics"]["high_prize_count"] > 0:
            high_prize_analysis["recommendations"].append(
                f"当前策略高额奖项命中率为{results["metrics"]["high_prize_rate"]:.4f}%，建议继续优化参数以提高稳定性"
            )
        else:
            high_prize_analysis["recommendations"].append(
                "未命中高额奖项，建议调整权重策略，增加稀有号码和区间分布多样性"
            )
        
        return high_prize_analysis
    
    def generate_visualization_data(self, results: Dict) -> Dict:
        """
        生成可视化所需的数据
        
        Args:
            results: 回测结果
            
        Returns:
            可视化数据
        """
        viz_data = {
            "prize_distribution": {
                "labels": [p for p in results["prize_distribution"].keys() if p != "未中奖" and results["prize_distribution"][p] > 0],
                "values": [results["prize_distribution"][p] for p in results["prize_distribution"].keys() 
                          if p != "未中奖" and results["prize_distribution"][p] > 0]
            },
            "profit_curve": {
                "dates": [p["date"] for p in results["profit_curve"]],
                "profits": [p["cumulative_profit"] for p in results["profit_curve"]]
            },
            "hit_matrix": self._generate_hit_matrix(results),
            "metrics_summary": results["metrics"]
        }
        
        return viz_data
    
    def _generate_hit_matrix(self, results: Dict) -> List[List[int]]:
        """
        生成命中矩阵（前区命中数 vs 后区命中数）
        """
        # 初始化10x3的矩阵（前区0-5中5个，后区0-2中2个）
        matrix = [[0 for _ in range(3)] for _ in range(6)]  # [front_match][back_match]
        
        for detail in results["hit_details"]:
            front_match = min(detail["front_match"], 5)
            back_match = min(detail["back_match"], 2)
            matrix[front_match][back_match] += 1
        
        return matrix

# 工具函数
def run_optimized_backtest(historical_data: pd.DataFrame, strategy_params: Dict,
                          prediction_func, periods: int = 50, trials: int = 10) -> Dict:
    """
    运行优化后的回测，针对高额奖项进行分析
    
    Args:
        historical_data: 历史数据
        strategy_params: 策略参数
        prediction_func: 预测函数
        periods: 回测期数
        trials: 每个周期的测试次数
        
    Returns:
        综合回测结果
    """
    analyzer = BacktestAnalyzer()
    all_predictions = []
    
    # 为每个周期生成预测
    for _ in range(periods):
        # 生成多组预测（模拟多次购买）
        pred = prediction_func(strategy_params, count=trials)
        all_predictions.append(pred)
    
    # 运行回测
    results = analyzer.run_backtest(historical_data, all_predictions, test_periods=periods)
    
    # 分析高额奖项
    high_prize_analysis = analyzer.analyze_high_prize_patterns(results)
    
    # 生成可视化数据
    viz_data = analyzer.generate_visualization_data(results)
    
    return {
        "backtest_results": results,
        "high_prize_analysis": high_prize_analysis,
        "visualization_data": viz_data
    }

def create_backtest_report(results: Dict) -> str:
    """
    创建回测报告文本
    
    Args:
        results: 回测结果
        
    Returns:
        格式化的报告文本
    """
    metrics = results["backtest_results"]["metrics"]
    high_analysis = results["high_prize_analysis"]
    
    report = []
    report.append("\n===== 回测分析报告 =====\n")
    
    # 概览
    report.append("## 回测概览")
    report.append(f"- 测试期数: {results['backtest_results']['periods']}")
    report.append(f"- 总投注数: {metrics['total_bets']}")
    report.append(f"- 总成本: ¥{metrics['total_cost']:.2f}")
    report.append(f"- 总收益: ¥{metrics['total_return']:.2f}")
    report.append(f"- 净利润: ¥{metrics['profit']:.2f}")
    report.append(f"- ROI: {metrics['roi']:.2f}%")
    report.append(f"- 平均单注收益: ¥{metrics['roi_per_bet']:.4f}")
    report.append("\n")
    
    # 高额奖项分析
    report.append("## 高额奖项分析")
    report.append(f"- 高额奖项总数: {high_analysis['overview']['total_high_prizes']}")
    report.append(f"- 高额奖项命中率: {high_analysis['overview']['hit_rate']:.4f}%")
    report.append("  - 按等级分布:")
    for prize, count in high_analysis['overview']['by_level'].items():
        rate = count / metrics['total_bets'] * 100 if metrics['total_bets'] > 0 else 0
        report.append(f"    - {prize}: {count}次 ({rate:.4f}%)")
    
    if results['backtest_results']['high_prize_hits']:
        report.append("\n- 命中详情:")
        for i, hit in enumerate(results['backtest_results']['high_prize_hits'], 1):
            report.append(f"  {i}. {hit['prize']} (期号: {hit['period']}, 日期: {hit['date']})")
            report.append(f"     预测: 前区{hit['prediction']['front']}, 后区{hit['prediction']['back']}")
            report.append(f"     开奖: 前区{hit['winning']['front']}, 后区{hit['winning']['back']}")
    report.append("\n")
    
    # 策略建议
    report.append("## 策略建议")
    for rec in high_analysis['recommendations']:
        report.append(f"- {rec}")
    
    return "\n".join(report)
