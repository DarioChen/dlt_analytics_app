# predictor.py
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

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

def compute_weights_from_history(
    df,
    front_blocks: Dict[str, List[int]],
    back_blocks: Dict[str, List[int]],
    recent_n: int = 0,
    front_range=range(1,36),
    back_range=range(1,13),
    out_min: float = 0.2,
    out_max: float = 1.5,
):
    """
    主流程：从历史数据计算前后区块权重
    recent_n: 若 >0 则只统计最近 N 期
    返回: front_block_weights, back_block_weights, front_freq_map, back_freq_map
    """
    if recent_n and recent_n > 0:
        df_use = df.sort_values("date", ascending=False).head(recent_n)
    else:
        df_use = df

    front_freq, back_freq = number_frequencies(df_use, front_range, back_range)
    front_block_avg = block_average_freq(front_blocks, front_freq)
    back_block_avg = block_average_freq(back_blocks, back_freq)

    front_block_weights = normalize_block_weights(front_block_avg, out_min, out_max)
    back_block_weights = normalize_block_weights(back_block_avg, out_min, out_max)

    return front_block_weights, back_block_weights, front_freq, back_freq

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
