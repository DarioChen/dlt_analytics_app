# predictor.py
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

def compute_weights_from_history_ewma(
    df,
    front_blocks: Dict[str, List[int]],
    back_blocks: Dict[str, List[int]],
    recent_n: int = 0,
    front_range=range(1,36),
    back_range=range(1,13),
    out_min: float = 0.2,
    out_max: float = 1.5,
    span: int = 5,  # EWMA span
    adaptive: bool = False,  # 是否使用自适应权重
    performance_history: Optional[Dict] = None,  # 历史表现记录
    high_prize_focus: bool = False  # 是否优先高额奖项
):
    """
    使用 EWMA 计算前区/后区区块权重，支持自适应调整
    
    Parameters:
    -----------
    adaptive: 是否启用自适应权重调整
    performance_history: 历史表现记录，格式为 {block_label: {"hit_rate": float, "high_prize_contrib": float}}
    high_prize_focus: 是否增加对高额奖项贡献的权重
    """
    if recent_n and recent_n > 0:
        df_use = df.sort_values("date", ascending=True).tail(recent_n)
    else:
        df_use = df.sort_values("date", ascending=True)

    front_cols = ["f1","f2","f3","f4","f5"]
    back_cols = ["b1","b2"]

    front_hist = pd.DataFrame([{num: 1 if num in row[front_cols].values else 0 for num in front_range}
                               for _, row in df_use.iterrows()])
    back_hist = pd.DataFrame([{num: 1 if num in row[back_cols].values else 0 for num in back_range}
                              for _, row in df_use.iterrows()])

    # EWMA
    front_ewma = front_hist.ewm(span=span, adjust=False).mean().iloc[-1].to_dict()
    back_ewma = back_hist.ewm(span=span, adjust=False).mean().iloc[-1].to_dict()

    # 增加稀有号码权重提升因子
    front_rare_factor = calculate_rare_factor(front_ewma, front_range, high_prize_focus)
    back_rare_factor = calculate_rare_factor(back_ewma, back_range, high_prize_focus)
    
    # 应用稀有因子
    front_ewma_adjusted = {n: v * front_rare_factor.get(n, 1.0) for n, v in front_ewma.items()}
    back_ewma_adjusted = {n: v * back_rare_factor.get(n, 1.0) for n, v in back_ewma.items()}

    front_block_avg = block_average_freq(front_blocks, front_ewma_adjusted)
    back_block_avg = block_average_freq(back_blocks, back_ewma_adjusted)

    # 基础权重归一化
    front_block_weights = normalize_block_weights(front_block_avg, out_min, out_max)
    back_block_weights = normalize_block_weights(back_block_avg, out_min, out_max)
    
    # 应用自适应调整
    if adaptive and performance_history:
        front_block_weights = adapt_weights(front_block_weights, 
                                          performance_history.get('front_blocks', {}),
                                          high_prize_focus)
        back_block_weights = adapt_weights(back_block_weights, 
                                         performance_history.get('back_blocks', {}),
                                         high_prize_focus)

    return front_block_weights, back_block_weights, front_ewma_adjusted, back_ewma_adjusted



def number_frequencies(df: pd.DataFrame, front_range=range(1,36), back_range=range(1,13)) -> Tuple[Dict[int,int], Dict[int,int]]:
    """
    统计历史出现次数（在 df 中，列名 f1..f5, b1..b2）
    返回 (front_freq, back_freq) 映射 number -> count
    """
    front_cols = ["f1","f2","f3","f4","f5"]
    back_cols = ["b1","b2"]

    front_vals = df[front_cols].values.flatten()
    back_vals = df[back_cols].values.flatten()

    front_counts = pd.Series(front_vals).value_counts().to_dict()
    back_counts = pd.Series(back_vals).value_counts().to_dict()

    # fill zeros for missing numbers
    front_freq = {n: int(front_counts.get(n, 0)) for n in front_range}
    back_freq = {n: int(back_counts.get(n, 0)) for n in back_range}
    return front_freq, back_freq

def block_average_freq(blocks: Dict[str, List[int]], freq_map: Dict[int,int]) -> Dict[str, float]:
    """
    计算每个区块内号码的平均出现次数
    blocks: {"1-5": [1,2,3,4,5], ...}
    freq_map: {num: count}
    返回: {"1-5": avg, ...}
    """
    block_avg = {}
    for label, nums in blocks.items():
        vals = [freq_map.get(n, 0) for n in nums]
        block_avg[label] = float(np.mean(vals)) if vals else 0.0
    return block_avg

def normalize_block_weights(block_avg: Dict[str,float], out_min: float = 0.2, out_max: float = 1.5) -> Dict[str,float]:
    """
    将 block 平均频率映射为权重（线性归一化到 [out_min, out_max]）
    若所有值相同，则返回所有权重 = 1.0
    """
    vals = list(block_avg.values())
    if not vals:
        return {k: 1.0 for k in block_avg}
    vmin = min(vals)
    vmax = max(vals)
    if vmax == vmin:
        return {k: 1.0 for k in block_avg}
    weights = {}
    for k,v in block_avg.items():
        # normalized to 0..1
        norm = (v - vmin) / (vmax - vmin)
        # map to out_min..out_max
        weights[k] = float(out_min + norm * (out_max - out_min))
    return weights

def calculate_rare_factor(freq_map: Dict[int, float], num_range, high_prize_focus: bool = False) -> Dict[int, float]:
    """
    计算稀有号码的权重提升因子
    对于低频号码给予额外权重，特别是对高额奖项有利的组合
    """
    rare_factor = {}
    # 计算频率分位数
    freq_values = list(freq_map.values())
    q1 = np.percentile(freq_values, 25)  # 25%分位数
    q3 = np.percentile(freq_values, 75)  # 75%分位数
    iqr = q3 - q1
    
    # 计算号码之间的距离矩阵
    dist_matrix = calculate_distance_matrix(list(num_range))
    
    for num in num_range:
        freq = freq_map.get(num, 0)
        # 基础稀有因子
        if freq < q1 - 0.5 * iqr:  # 明显低于平均水平
            base_factor = 1.3
        elif freq < q1:  # 低于平均水平
            base_factor = 1.1
        else:
            base_factor = 1.0
        
        # 对于高额奖项，增加对稀有组合的权重
        if high_prize_focus:
            # 计算该号码与其他低频号码的平均距离
            low_freq_nums = [n for n, f in freq_map.items() if f < q1]
            if low_freq_nums:
                avg_dist = np.mean([dist_matrix[num][n] for n in low_freq_nums if n in dist_matrix.get(num, {})])
                # 距离较远的号码组合更有可能产生高额奖项
                if avg_dist > 10:  # 距离较远
                    base_factor *= 1.2
        
        rare_factor[num] = base_factor
    
    return rare_factor

def calculate_distance_matrix(nums: List[int]) -> Dict[int, Dict[int, int]]:
    """
    计算号码之间的距离矩阵
    """
    matrix = {}
    for num1 in nums:
        matrix[num1] = {}
        for num2 in nums:
            matrix[num1][num2] = abs(num1 - num2)
    return matrix

def adapt_weights(base_weights: Dict[str, float], performance: Dict[str, Dict], 
                 high_prize_focus: bool = False) -> Dict[str, float]:
    """
    根据历史表现自适应调整权重
    
    Parameters:
    -----------
    base_weights: 基础权重
    performance: 历史表现数据 {block_label: {"hit_rate": float, "high_prize_contrib": float}}
    high_prize_focus: 是否更注重高额奖项贡献
    """
    adapted_weights = base_weights.copy()
    
    # 计算表现统计
    hit_rates = [p.get("hit_rate", 0.0) for p in performance.values()]
    prize_contribs = [p.get("high_prize_contrib", 0.0) for p in performance.values()]
    
    if not hit_rates:
        return adapted_weights
    
    avg_hit_rate = np.mean(hit_rates)
    max_hit_rate = max(hit_rates) if hit_rates else 0
    
    if prize_contribs:
        avg_prize_contrib = np.mean(prize_contribs)
        max_prize_contrib = max(prize_contribs) if prize_contribs else 0
    else:
        avg_prize_contrib = 0
        max_prize_contrib = 0
    
    for block, weight in base_weights.items():
        if block in performance:
            perf = performance[block]
            hit_rate = perf.get("hit_rate", 0.0)
            prize_contrib = perf.get("high_prize_contrib", 0.0)
            
            # 表现调整因子
            hit_factor = 1.0
            if hit_rate > avg_hit_rate * 1.5 and max_hit_rate > 0:
                hit_factor = 1.1 + (hit_rate / max_hit_rate) * 0.2
            elif hit_rate < avg_hit_rate * 0.5:
                hit_factor = 0.9
            
            # 高额奖项贡献调整因子
            prize_factor = 1.0
            if high_prize_focus and prize_contribs:
                if prize_contrib > avg_prize_contrib * 1.5 and max_prize_contrib > 0:
                    prize_factor = 1.2 + (prize_contrib / max_prize_contrib) * 0.3
            
            # 综合调整
            if high_prize_focus:
                # 高额奖项模式下，更注重奖项贡献
                final_factor = hit_factor * 0.4 + prize_factor * 0.6
            else:
                # 普通模式下，更注重命中率
                final_factor = hit_factor * 0.7 + prize_factor * 0.3
            
            adapted_weights[block] = min(weight * final_factor, 2.0)  # 上限为2.0
    
    # 重新归一化，保持权重范围
    min_weight = min(adapted_weights.values())
    max_weight = max(adapted_weights.values())
    if max_weight > min_weight:
        scale_factor = (1.5 - 0.2) / (max_weight - min_weight)
        for block in adapted_weights:
            adapted_weights[block] = 0.2 + (adapted_weights[block] - min_weight) * scale_factor
    
    return adapted_weights

def compute_weights_from_history(
    df,
    front_blocks: Dict[str, List[int]],
    back_blocks: Dict[str, List[int]],
    recent_n: int = 0,
    front_range=range(1,36),
    back_range=range(1,13),
    out_min: float = 0.2,
    out_max: float = 1.5,
    adaptive: bool = False,
    performance_history: Optional[Dict] = None,
    high_prize_focus: bool = False
):
    """
    主流程：从历史数据计算前后区块权重，支持自适应调整
    
    Parameters:
    -----------
    recent_n: 若 >0 则只统计最近 N 期
    adaptive: 是否启用自适应权重
    performance_history: 历史表现记录
    high_prize_focus: 是否优先高额奖项
    
    返回: front_block_weights, back_block_weights, front_freq_map, back_freq_map
    """
    if recent_n and recent_n > 0:
        df_use = df.sort_values("date", ascending=False).head(recent_n)
    else:
        df_use = df

    front_freq, back_freq = number_frequencies(df_use, front_range, back_range)
    
    # 增加稀有号码权重
    front_rare_factor = calculate_rare_factor(
        {k: float(v) for k, v in front_freq.items()}, 
        front_range, 
        high_prize_focus
    )
    back_rare_factor = calculate_rare_factor(
        {k: float(v) for k, v in back_freq.items()}, 
        back_range, 
        high_prize_focus
    )
    
    # 应用稀有因子
    front_freq_adjusted = {n: v * front_rare_factor.get(n, 1.0) for n, v in front_freq.items()}
    back_freq_adjusted = {n: v * back_rare_factor.get(n, 1.0) for n, v in back_freq.items()}

    front_block_avg = block_average_freq(front_blocks, front_freq_adjusted)
    back_block_avg = block_average_freq(back_blocks, back_freq_adjusted)

    front_block_weights = normalize_block_weights(front_block_avg, out_min, out_max)
    back_block_weights = normalize_block_weights(back_block_avg, out_min, out_max)
    
    # 应用自适应调整
    if adaptive and performance_history:
        front_block_weights = adapt_weights(front_block_weights, 
                                          performance_history.get('front_blocks', {}),
                                          high_prize_focus)
        back_block_weights = adapt_weights(back_block_weights, 
                                         performance_history.get('back_blocks', {}),
                                         high_prize_focus)

    return front_block_weights, back_block_weights, front_freq_adjusted, back_freq_adjusted

def track_block_performance(
    generated_nums: List[Dict],
    winning_nums: Dict,
    front_blocks: Dict[str, List[int]],
    back_blocks: Dict[str, List[int]],
    current_performance: Optional[Dict] = None
) -> Dict:
    """
    跟踪区块表现，更新表现历史
    
    Parameters:
    -----------
    generated_nums: 生成的号码列表 [{"front": [...], "back": [...]}]
    winning_nums: 中奖号码 {"front": [...], "back": [...]}
    front_blocks: 前区区块映射
    back_blocks: 后区区块映射
    current_performance: 当前表现历史
    
    Returns:
    --------
    更新后的表现历史
    """
    if current_performance is None:
        current_performance = {
            'front_blocks': defaultdict(lambda: {'hit_rate': 0.0, 'high_prize_contrib': 0.0, 'count': 0}),
            'back_blocks': defaultdict(lambda: {'hit_rate': 0.0, 'high_prize_contrib': 0.0, 'count': 0})
        }
    
    # 转换为defaultdict以便更新
    front_perf = defaultdict(lambda: {'hit_rate': 0.0, 'high_prize_contrib': 0.0, 'count': 0}, 
                           current_performance.get('front_blocks', {}))
    back_perf = defaultdict(lambda: {'hit_rate': 0.0, 'high_prize_contrib': 0.0, 'count': 0}, 
                          current_performance.get('back_blocks', {}))
    
    # 确定哪些区块包含中奖号码
    winning_front_blocks = set()
    for num in winning_nums.get('front', []):
        for block, nums in front_blocks.items():
            if num in nums:
                winning_front_blocks.add(block)
                break
    
    winning_back_blocks = set()
    for num in winning_nums.get('back', []):
        for block, nums in back_blocks.items():
            if num in nums:
                winning_back_blocks.add(block)
                break
    
    # 分析每个生成的号码组合
    for gen_num in generated_nums:
        # 计算前区命中情况
        front_hit_blocks = set()
        front_hits = 0
        for num in gen_num.get('front', []):
            for block, nums in front_blocks.items():
                if num in nums:
                    if num in winning_nums.get('front', []):
                        front_hit_blocks.add(block)
                        front_hits += 1
                    break
        
        # 计算后区命中情况
        back_hit_blocks = set()
        back_hits = 0
        for num in gen_num.get('back', []):
            for block, nums in back_blocks.items():
                if num in nums:
                    if num in winning_nums.get('back', []):
                        back_hit_blocks.add(block)
                        back_hits += 1
                    break
        
        # 计算高额奖项贡献（根据命中情况估算）
        high_prize_score = 0
        if front_hits >= 4 and back_hits >= 1:
            high_prize_score = 2.0  # 接近一等奖
        elif front_hits >= 3 and back_hits >= 2:
            high_prize_score = 1.5  # 二等奖或三等奖
        elif front_hits >= 3 or (front_hits >= 2 and back_hits >= 1):
            high_prize_score = 0.5  # 中低等奖项
        
        # 更新前区区块表现
        for block in front_blocks.keys():
            front_perf[block]['count'] += 1
            if block in front_hit_blocks:
                front_perf[block]['hit_rate'] = ((front_perf[block]['hit_rate'] * 
                                               (front_perf[block]['count'] - 1)) + 1) / front_perf[block]['count']
                if high_prize_score > 0:
                    front_perf[block]['high_prize_contrib'] = ((front_perf[block]['high_prize_contrib'] * 
                                                           (front_perf[block]['count'] - 1)) + high_prize_score) / front_perf[block]['count']
            else:
                front_perf[block]['hit_rate'] = (front_perf[block]['hit_rate'] * 
                                               (front_perf[block]['count'] - 1)) / front_perf[block]['count']
        
        # 更新后区区块表现
        for block in back_blocks.keys():
            back_perf[block]['count'] += 1
            if block in back_hit_blocks:
                back_perf[block]['hit_rate'] = ((back_perf[block]['hit_rate'] * 
                                              (back_perf[block]['count'] - 1)) + 1) / back_perf[block]['count']
                if high_prize_score > 0:
                    back_perf[block]['high_prize_contrib'] = ((back_perf[block]['high_prize_contrib'] * 
                                                            (back_perf[block]['count'] - 1)) + high_prize_score) / back_perf[block]['count']
            else:
                back_perf[block]['hit_rate'] = (back_perf[block]['hit_rate'] * 
                                              (back_perf[block]['count'] - 1)) / back_perf[block]['count']
    
    return {
        'front_blocks': dict(front_perf),
        'back_blocks': dict(back_perf)
    }

def prepare_generator_inputs(
    front_blocks_labels: List[str],
    front_bins: List[Tuple[int,int]],
    back_blocks_labels: List[str],
    back_bins: List[Tuple[int,int]],
    selected_front_blocks: List[str],
    selected_back_blocks: List[str],
    block_front_weights: Dict[str,float],
    block_back_weights: Dict[str,float],
):
    """
    从 labels & bins 构造 generator 所需的 front_blocks/back_blocks 映射（label -> [nums]）
    并返回 block 权重映射（与 label 一致）
    """
    front_blocks_map = {label: list(range(lo, hi+1)) for label, (lo,hi) in zip(front_blocks_labels, front_bins)}
    back_blocks_map = {label: list(range(lo, hi+1)) for label, (lo,hi) in zip(back_blocks_labels, back_bins)}
    # supply weights but ensure all labels exist
    front_weights = {label: float(block_front_weights.get(label, 1.0)) for label in front_blocks_map.keys()}
    back_weights = {label: float(block_back_weights.get(label, 1.0)) for label in back_blocks_map.keys()}

    return front_blocks_map, back_blocks_map, front_weights, back_weights
