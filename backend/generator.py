# backend/generator.py v1.3
from __future__ import annotations
from typing import List, Dict, Optional
import random

def gen_numbers(
    count: int = 5,
    rules: Optional[Dict] = None,
    rng: Optional[random.Random] = None,
    front_pool_user: Optional[List[int]] = None,
    back_pool_user: Optional[List[int]] = None,
    front_blocks: Optional[Dict[str, List[int]]] = None,
    back_blocks: Optional[Dict[str, List[int]]] = None,
    front_weights: Optional[Dict[str, float]] = None,
    back_weights: Optional[Dict[str, float]] = None,
    use_block_weight: bool = False
) -> List[Dict]:
    rng = rng or random.Random()
    rules = rules or {}

    def consecutive_pairs_count(front_sorted: List[int]) -> int:
        cnt = 0
        for i in range(1, len(front_sorted)):
            if front_sorted[i] == front_sorted[i - 1] + 1:
                cnt += 1
        return cnt

    def choose_numbers(blocks: Dict[str, List[int]], weights: Dict[str, float], num_needed: int, rng: random.Random) -> List[int]:
        """
        按权重选择 num_needed 个号码，剔除空区块。
        已剔除的区块不会被选中，权重重新归一化。
        """
        available_blocks = {b: nums.copy() for b, nums in blocks.items() if nums}
        if not available_blocks:
            available_blocks = {b: nums.copy() for b, nums in blocks.items()}

        result = []

        for _ in range(num_needed):
            total_weight = sum(weights.get(b, 1.0) for b in available_blocks)
            block_names = list(available_blocks.keys())
            probs = [weights.get(b, 1.0)/total_weight for b in block_names]

            chosen_block = rng.choices(block_names, weights=probs, k=1)[0]
            num = rng.choice(available_blocks[chosen_block])
            result.append(num)

            available_blocks[chosen_block].remove(num)
            if not available_blocks[chosen_block]:
                del available_blocks[chosen_block]

        return sorted(result)

    results: List[Dict] = []
    tries = 0
    max_tries = max(5000, count * 2000)

    front_exclude = set(rules.get("front_exclude", []))
    front_include = set(rules.get("front_include", []))
    back_exclude = set(rules.get("back_exclude", []))
    back_include = set(rules.get("back_include", []))
    sum_range = rules.get("sum_front_range", [None, None])
    odd_even = rules.get("odd_even_front", None)
    cons_req = rules.get("consecutive_count", None)
    cons_mode = rules.get("consecutive_mode", "exact")

    while len(results) < count and tries < max_tries:
        tries += 1

        front_pool = front_pool_user if front_pool_user is not None else [n for n in range(1, 36)]
        back_pool = back_pool_user if back_pool_user is not None else [n for n in range(1, 13)]

        front_pool = [n for n in front_pool if n not in front_exclude]
        back_pool = [n for n in back_pool if n not in back_exclude]

        if len(front_pool) < 5 or len(back_pool) < 2:
            break

        # ----------------- 前区 -----------------
        if use_block_weight and front_blocks and front_weights:
            # 过滤掉剔除的区块
            filtered_front_blocks = {b: [n for n in nums if n in front_pool] for b, nums in front_blocks.items()}
            f = choose_numbers(filtered_front_blocks, front_weights, 5, rng)
        else:
            f = sorted(rng.sample(front_pool, 5))

        # ----------------- 后区 -----------------
        if use_block_weight and back_blocks and back_weights:
            filtered_back_blocks = {b: [n for n in nums if n in back_pool] for b, nums in back_blocks.items()}
            b = choose_numbers(filtered_back_blocks, back_weights, 2, rng)
        else:
            b = sorted(rng.sample(back_pool, 2))

        ok = True

        if front_include and not front_include.issubset(set(f)):
            ok = False
        if back_include and not back_include.issubset(set(b)):
            ok = False

        s = sum(f)
        smin, smax = sum_range if sum_range is not None else (None, None)
        if smin is not None and s < smin:
            ok = False
        if smax is not None and s > smax:
            ok = False

        if odd_even:
            odd_need, even_need = odd_even[0], odd_even[1]
            odd_actual = sum(1 for x in f if x % 2 == 1)
            even_actual = 5 - odd_actual
            if odd_actual != odd_need or even_actual != even_need:
                ok = False

        if cons_req is not None:
            cnt = consecutive_pairs_count(f)
            if cons_mode == "exact" and cnt != cons_req:
                ok = False
            elif cons_mode == "min" and cnt < cons_req:
                ok = False

        if ok:
            results.append({"front": f, "back": b})

    return results
