from __future__ import annotations
from typing import List, Dict, Optional, Set
import random

def gen_numbers(
    count: int = 5,
    rules: Optional[Dict] = None,
    rng: Optional[random.Random] = None,
    front_pool_user: Optional[List[int]] = None,  # 新增：用户可选前区池
    back_pool_user: Optional[List[int]] = None    # 新增：用户可选后区池
) -> List[Dict]:
    """
    生成候选号码列表（返回长度最多为 count 的列表，元素为 {"front": [...], "back": [...]}）
    """
    rng = rng or random.Random()
    rules = rules or {}

    def weighted_pool(pool: List[int], hot: Set[int] = set(), cold: Set[int] = set()) -> List[int]:
        out = []
        for n in pool:
            w = 3 if n in hot else (0.7 if n in cold else 1.0)
            repeat = max(1, int(w * 10))
            out.extend([n] * repeat)
        return out or pool

    def consecutive_pairs_count(front_sorted: List[int]) -> int:
        cnt = 0
        for i in range(1, len(front_sorted)):
            if front_sorted[i] == front_sorted[i - 1] + 1:
                cnt += 1
        return cnt

    results: List[Dict] = []
    tries = 0
    max_tries = max(5000, count * 2000)

    front_exclude = set(rules.get("front_exclude", []))
    front_include = set(rules.get("front_include", []))
    back_exclude = set(rules.get("back_exclude", []))
    back_include = set(rules.get("back_include", []))
    hot_front = set(rules.get("hot_front", []))
    cold_front = set(rules.get("cold_front", []))
    sum_range = rules.get("sum_front_range", [None, None])
    odd_even = rules.get("odd_even_front", None)
    cons_req = rules.get("consecutive_count", None)
    cons_mode = rules.get("consecutive_mode", "exact")

    while len(results) < count and tries < max_tries:
        tries += 1

        # 使用用户提供的 pool，如果没有就默认 1-35 / 1-12
        front_pool = front_pool_user if front_pool_user is not None else [n for n in range(1, 36)]
        back_pool = back_pool_user if back_pool_user is not None else [n for n in range(1, 13)]

        # 排除用户指定的排除号码
        front_pool = [n for n in front_pool if n not in front_exclude]
        back_pool = [n for n in back_pool if n not in back_exclude]

        if len(front_pool) < 5 or len(back_pool) < 2:
            break

        front_pool_w = weighted_pool(front_pool, hot=hot_front, cold=cold_front)
        f: List[int] = []
        attempts_inner = 0
        while len(f) < 5 and attempts_inner < 200:
            attempts_inner += 1
            candidate = rng.choice(front_pool_w)
            if candidate not in f:
                f.append(candidate)
        if len(f) < 5:
            f = sorted(rng.sample(front_pool, 5))
        else:
            f = sorted(f)

        b: List[int] = []
        attempts_b = 0
        while len(b) < 2 and attempts_b < 100:
            attempts_b += 1
            cand = rng.choice(back_pool)
            if cand not in b:
                b.append(cand)
        if len(b) < 2:
            b = sorted(rng.sample(back_pool, 2))
        else:
            b = sorted(b)

        ok = True

        # 必包含
        if front_include and not front_include.issubset(set(f)):
            ok = False
        if back_include and not back_include.issubset(set(b)):
            ok = False

        # 和值范围
        s = sum(f)
        smin, smax = sum_range if sum_range is not None else (None, None)
        if smin is not None and s < smin:
            ok = False
        if smax is not None and s > smax:
            ok = False

        # 奇偶数
        if odd_even:
            odd_need, even_need = odd_even[0], odd_even[1]
            odd_actual = sum(1 for x in f if x % 2 == 1)
            even_actual = 5 - odd_actual
            if odd_actual != odd_need or even_actual != even_need:
                ok = False

        # 连号数量
        if cons_req is not None:
            cnt = consecutive_pairs_count(f)
            if cons_mode == "exact" and cnt != cons_req:
                ok = False
            elif cons_mode == "min" and cnt < cons_req:
                ok = False

        if ok:
            results.append({"front": f, "back": b})

    return results
