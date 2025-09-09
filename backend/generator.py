# backend/generator.py
from __future__ import annotations
from typing import List, Dict, Optional
import random

def gen_numbers(
    count: int = 5,
    rules: Optional[Dict] = None,
    rng: Optional[random.Random] = None,
    front_pool_user: Optional[List[int]] = None,
    back_pool_user: Optional[List[int]] = None,
    front_blocks: Optional[Dict[str,List[int]]] = None,
    back_blocks: Optional[Dict[str,List[int]]] = None,
    front_weights: Optional[Dict[str,float]] = None,
    back_weights: Optional[Dict[str,float]] = None,
    use_block_weight: bool = False
) -> List[Dict]:
    rng = rng or random.Random()
    rules = rules or {}

    def consecutive_pairs_count(front_sorted: List[int]) -> int:
        cnt = 0
        for i in range(1,len(front_sorted)):
            if front_sorted[i] == front_sorted[i-1]+1:
                cnt += 1
        return cnt

    def choose_from_weighted_blocks(blocks: Dict[str, List[int]], weights: Dict[str, float], num_needed: int,
                                    exclude_nums: set = None) -> List[int]:
        exclude_nums = exclude_nums or set()
        # 过滤每个区块里的排除数字
        valid_blocks = {b: [n for n in nums if n not in exclude_nums] for b, nums in blocks.items() if
                        weights.get(b, 0) > 0}
        # 如果过滤后没有数字，退回原始区块
        if all(len(v) == 0 for v in valid_blocks.values()):
            valid_blocks = {b: [n for n in nums if n not in exclude_nums] for b, nums in blocks.items()}

        total_w = sum(weights.get(b, 1.0) for b in valid_blocks)
        norm_weights = {b: weights.get(b, 1.0) / total_w for b in valid_blocks}

        counts = {b: int(norm_weights[b] * num_needed) for b in valid_blocks}
        total_selected = sum(counts.values())
        remaining = num_needed - total_selected

        block_names = list(valid_blocks.keys())
        probs = [norm_weights[b] for b in block_names]
        for _ in range(remaining):
            chosen_block = random.choices(block_names, probs, k=1)[0]
            counts[chosen_block] += 1

        result = []
        for b, c in counts.items():
            nums = valid_blocks[b].copy()
            random.shuffle(nums)
            result.extend(nums[:c])
        random.shuffle(result)
        return sorted(result[:num_needed])

    results: List[Dict] = []
    tries = 0
    max_tries = max(5000,count*2000)

    front_exclude = set(rules.get("front_exclude",[]))
    front_include = set(rules.get("front_include",[]))
    back_exclude = set(rules.get("back_exclude",[]))
    back_include = set(rules.get("back_include",[]))
    sum_range = rules.get("sum_front_range",[None,None])
    odd_even = rules.get("odd_even_front",None)
    cons_req = rules.get("consecutive_count",None)
    cons_mode = rules.get("consecutive_mode","exact")

    while len(results)<count and tries<max_tries:
        tries += 1
        front_pool = front_pool_user if front_pool_user is not None else [n for n in range(1,36)]
        back_pool = back_pool_user if back_pool_user is not None else [n for n in range(1,13)]

        front_pool = [n for n in front_pool if n not in front_exclude]
        back_pool = [n for n in back_pool if n not in back_exclude]

        if len(front_pool)<5 or len(back_pool)<2:
            break

        if use_block_weight and front_blocks and front_weights:
            f = choose_from_weighted_blocks(front_blocks, front_weights, 5, exclude_nums=front_exclude)

        else:
            f = sorted(rng.sample(front_pool,5))

        if use_block_weight and back_blocks and back_weights:
            b = choose_from_weighted_blocks(back_blocks, back_weights, 2, exclude_nums=back_exclude)
        else:
            b = sorted(rng.sample(back_pool,2))

        ok = True
        if front_include and not front_include.issubset(set(f)):
            ok=False
        if back_include and not back_include.issubset(set(b)):
            ok=False

        s=sum(f)
        smin,smax=sum_range if sum_range is not None else (None,None)
        if smin is not None and s<smin:
            ok=False
        if smax is not None and s>smax:
            ok=False

        if odd_even:
            odd_need,even_need=odd_even[0],odd_even[1]
            odd_actual=sum(1 for x in f if x%2==1)
            even_actual=5-odd_actual
            if odd_actual!=odd_need or even_actual!=even_need:
                ok=False

        if cons_req is not None:
            cnt=consecutive_pairs_count(f)
            if cons_mode=="exact" and cnt!=cons_req:
                ok=False
            elif cons_mode=="min" and cnt<cons_req:
                ok=False

        if ok:
            results.append({"front":f,"back":b})

    return results
