from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Set
import random

def gen_numbers(
    count: int = 5,
    rules: Optional[Dict] = None,
    rng: Optional[random.Random] = None
) -> List[Dict]:
    """
    生成候选号码列表（返回长度最多为 count 的列表，元素为 {"front": [...], "back": [...]}）

    支持的 rules 字段（关键字为英文）：
      - "front_include": [ints]
      - "front_exclude": [ints]
      - "back_include": [ints]
      - "back_exclude": [ints]
      - "sum_front_range": [min, max]
      - "odd_even_front": [odd_count, even_count]
      - "hot_front": [ints]  # (可选) 权重偏好（当前未严格实现权重，只作为示例）
      - "cold_front": [ints] # (可选)
      - "consecutive_count": int  # 新加入：连号数量（相邻对数）
      - "consecutive_mode": "exact" | "min"  # "exact" 表示等于；"min" 表示至少
    """
    rng = rng or random.Random()
    rules = rules or {}

    def weighted_pool(pool: List[int], hot: Set[int] = set(), cold: Set[int] = set()) -> List[int]:
        out = []
        for n in pool:
            # 权重：hot -> 3x, cold -> 0.7x, normal -> 1x
            # 通过重复元素近似权重；注意重复会使 sample 行为变得复杂，因此后面用 choice 循环取不重复项
            w = 3 if n in hot else (0.7 if n in cold else 1.0)
            repeat = max(1, int(w * 10))
            out.extend([n] * repeat)
        return out or pool

    def consecutive_pairs_count(front_sorted: List[int]) -> int:
        """计算已排序前区中相邻相差 1 的对数（例如 01,02,05,06,10 -> 2 对）"""
        cnt = 0
        for i in range(1, len(front_sorted)):
            if front_sorted[i] == front_sorted[i - 1] + 1:
                cnt += 1
        return cnt

    results: List[Dict] = []
    tries = 0
    max_tries = max(5000, count * 2000)

    # prepare static rule values
    front_exclude = set(rules.get("front_exclude", []))
    front_include = set(rules.get("front_include", []))
    back_exclude = set(rules.get("back_exclude", []))
    back_include = set(rules.get("back_include", []))
    hot_front = set(rules.get("hot_front", []))
    cold_front = set(rules.get("cold_front", []))
    sum_range = rules.get("sum_front_range", [None, None])
    odd_even = rules.get("odd_even_front", None)
    cons_req = rules.get("consecutive_count", None)
    cons_mode = rules.get("consecutive_mode", "exact")  # 'exact' or 'min'

    while len(results) < count and tries < max_tries:
        tries += 1

        # 筛选候选池
        front_pool = [n for n in range(1, 36) if n not in front_exclude]
        back_pool = [n for n in range(1, 13) if n not in back_exclude]

        if len(front_pool) < 5 or len(back_pool) < 2:
            # 排除过多导致无法取数
            break

        front_pool_w = weighted_pool(front_pool, hot=hot_front, cold=cold_front)
        # 用 choice 循环确保取到 5 个不重复的数字（考虑权重）
        f: List[int] = []
        attempts_inner = 0
        while len(f) < 5 and attempts_inner < 200:
            attempts_inner += 1
            candidate = rng.choice(front_pool_w)
            if candidate not in f:
                f.append(candidate)
        if len(f) < 5:
            # 退回到均匀采样（最后手段）
            f = sorted(rng.sample(front_pool, 5))
        else:
            f = sorted(f)

        # 后区均匀采样，确保不重复
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

        # 规则校验
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

        # 连号数量校验（新逻辑）
        if cons_req is not None:
            cnt = consecutive_pairs_count(f)
            if cons_mode == "exact":
                if cnt != cons_req:
                    ok = False
            elif cons_mode == "min":
                if cnt < cons_req:
                    ok = False
            else:
                # 不认识的模式，默认为 exact
                if cnt != cons_req:
                    ok = False

        if ok:
            results.append({"front": f, "back": b})

    return results
