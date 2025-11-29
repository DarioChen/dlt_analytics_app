# generator.py v1.7 (随机挑选区块)
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

    # 新增参数
    selected_front_blocks = selected_front_blocks or []
    selected_back_blocks = selected_back_blocks or []
    front_block_total = len(selected_front_blocks) if selected_front_blocks else len(front_blocks or {})
    back_block_total = len(selected_back_blocks) if selected_back_blocks else len(back_blocks or {})

    top_n_blocks = rules.get("top_n_blocks", front_block_total if front_block_total > 0 else 0)
    max_per_block = rules.get("max_per_block", 2)
    random_blocks_count = rules.get("random_blocks_count", 3)  # 每期随机选择几个区块
    random_back_blocks_count = rules.get("random_back_blocks_count", back_block_total if back_block_total > 0 else 0)

    max_tries = 10000
    tries = 0

    while len(results) < count and tries < max_tries:
        tries += 1

        # ----------------- 生成前区号码 -----------------
        if not selected_front_blocks:
            continue

        # 选 top N 权重区块
        top_blocks = sorted(selected_front_blocks, key=lambda b: front_weights.get(b,1.0), reverse=True)[:top_n_blocks]

        # 每期随机挑选 random_blocks_count 个区块
        num_blocks_to_use = min(random_blocks_count, len(top_blocks))
        blocks_this_round = rng.sample(top_blocks, num_blocks_to_use)

        pool_per_block = {b: [n for n in front_blocks[b] if n not in front_exclude] for b in blocks_this_round}
        f_selected: List[int] = []

        while len(f_selected) < 5:
            for block in blocks_this_round:
                available = [n for n in pool_per_block[block] if n not in f_selected]
                already_in_block = sum(1 for x in f_selected if x in front_blocks[block])
                take = min(max_per_block - already_in_block, len(available))
                if take > 0:
                    picks = rng.sample(available, take)
                    f_selected.extend(picks)
                if len(f_selected) >= 5:
                    break

            # 如果所有区块都满了但总数不够，跳出避免死循环
            if all(sum(1 for x in f_selected if x in front_blocks[b]) >= max_per_block for b in blocks_this_round):
                break

        if len(f_selected) < 5:
            continue

        f_selected = f_selected[:5]
        f_selected.sort()

        # ----------------- 生成后区号码 -----------------
        if not selected_back_blocks:
            continue

        if random_back_blocks_count and random_back_blocks_count > 0:
            num_back_blocks = min(random_back_blocks_count, len(selected_back_blocks))
            back_blocks_this_round = rng.sample(selected_back_blocks, num_back_blocks)
        else:
            back_blocks_this_round = selected_back_blocks[:]

        back_pool = []
        for block in back_blocks_this_round:
            nums = [n for n in back_blocks[block] if n not in back_exclude]
            back_pool.extend(nums)
        if len(back_pool) < 2:
            continue

        b_selected = []
        block_weights = {b: back_weights.get(b, 1.0) for b in back_blocks_this_round}
        total_weight = sum(block_weights.values()) or len(block_weights)

        attempts = 0
        while len(b_selected) < 2 and attempts < 20:
            attempts += 1
            chosen_block = rng.choices(
                population=list(block_weights.keys()),
                weights=[block_weights[b] / total_weight for b in block_weights],
                k=1
            )[0]
            candidates = [n for n in back_blocks[chosen_block] if n not in b_selected and n not in back_exclude]
            if not candidates:
                continue
            pick = rng.choice(candidates)
            b_selected.append(pick)

        b_selected.sort()
        if len(b_selected) < 2:
            continue

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
