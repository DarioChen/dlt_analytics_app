from typing import List, Dict, Optional
import random

def gen_numbers(
    count: int = 5,
    rules: Optional[Dict] = None,
    front_blocks: Optional[Dict[str, List[int]]] = None,
    back_blocks: Optional[Dict[str, List[int]]] = None,
    front_weights: Optional[Dict[str, float]] = None,
    back_weights: Optional[Dict[str, float]] = None,
    selected_front_blocks: Optional[List[str]] = None,
    selected_back_blocks: Optional[List[str]] = None,
) -> List[Dict]:
    rng = random.Random()
    rules = rules or {}
    results: List[Dict] = []

    def consecutive_pairs_count(nums: List[int]) -> int:
        cnt = 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                cnt += 1
        return cnt

    sum_range = rules.get("sum_front_range", [None, None])
    min_odd_count = rules.get("odd_even_front", [0])[0]  # 最小奇数个数
    cons_req = rules.get("consecutive_count", 0)
    cons_mode = rules.get("consecutive_mode", "exact")
    front_include = set(rules.get("front_include", []))
    front_exclude = set(rules.get("front_exclude", []))
    back_include = set(rules.get("back_include", []))
    back_exclude = set(rules.get("back_exclude", []))

    max_tries = 10000
    tries = 0

    while len(results) < count and tries < max_tries:
        tries += 1
        # ----------------- 生成前区号码 -----------------
        front_pool = []
        for block in selected_front_blocks:
            nums = [n for n in front_blocks[block] if n not in front_exclude]
            front_pool.extend(nums)
        if len(front_pool) < 5:
            continue

        # 按权重逐一选号
        f_selected = []
        pool_copy = front_pool.copy()
        block_weights = {b: front_weights[b] for b in selected_front_blocks}
        total_weight = sum(block_weights.values())
        for _ in range(5):
            # 按权重选择区块
            chosen_block = rng.choices(list(block_weights.keys()),
                                       weights=[block_weights[b]/total_weight for b in block_weights], k=1)[0]
            # 从区块剩余号码中随机取一个
            candidates = [n for n in front_blocks[chosen_block] if n not in f_selected and n not in front_exclude]
            if not candidates:
                continue
            pick = rng.choice(candidates)
            f_selected.append(pick)

        f_selected.sort()

        # ----------------- 生成后区号码 -----------------
        back_pool = []
        for block in selected_back_blocks:
            nums = [n for n in back_blocks[block] if n not in back_exclude]
            back_pool.extend(nums)
        if len(back_pool) < 2:
            continue

        b_selected = []
        pool_copy = back_pool.copy()
        block_weights = {b: back_weights[b] for b in selected_back_blocks}
        total_weight = sum(block_weights.values())
        for _ in range(2):
            chosen_block = rng.choices(list(block_weights.keys()),
                                       weights=[block_weights[b]/total_weight for b in block_weights], k=1)[0]
            candidates = [n for n in back_blocks[chosen_block] if n not in b_selected and n not in back_exclude]
            if not candidates:
                continue
            pick = rng.choice(candidates)
            b_selected.append(pick)

        b_selected.sort()

        # ----------------- 检查条件 -----------------
        ok = True
        if front_include and not front_include.issubset(f_selected):
            ok = False
        if back_include and not back_include.issubset(b_selected):
            ok = False

        s = sum(f_selected)
        smin, smax = sum_range
        if (smin is not None and s < smin) or (smax is not None and s > smax):
            ok = False

        odd_count = sum(1 for n in f_selected if n % 2 == 1)
        if odd_count < min_odd_count:
            ok = False

        cons_pairs = consecutive_pairs_count(f_selected)
        if cons_mode == "exact" and cons_pairs != cons_req:
            ok = False
        elif cons_mode == "min" and cons_pairs < cons_req:
            ok = False

        if ok:
            results.append({"front": f_selected, "back": b_selected})

    return results
