
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import random

def gen_numbers(
    count:int=5,
    rules:Optional[Dict]=None,
    rng:Optional[random.Random]=None
) -> List[Dict]:
    """
    规则示例 rules:
      {
        "front_include": [1,2],         # 前区必须包含
        "front_exclude": [3,4,5],       # 前区排除
        "back_include": [12],
        "back_exclude": [1,2],
        "sum_front_range": [60, 140],   # 前区和值
        "odd_even_front": [2,3],        # 前区奇偶数量 [奇, 偶] 之和=5
        "hot_front": [5],               # 倾向选择的前区热点（加权）
        "cold_front": [7],              # 倾向选择的前区冷号（降低权重）
      }
    """
    rng = rng or random.Random()
    rules = rules or {}

    def weighted_pool(pool:List[int], hot:Set[int]=set(), cold:Set[int]=set()) -> List[int]:
        out = []
        for n in pool:
            w = 3 if n in hot else (0.7 if n in cold else 1.0)
            # 将权重通过重复的方式近似（简单可行）
            out.extend([n]*int(w*10))
        return out or pool

    results = []
    tries = 0
    while len(results) < count and tries < count*5000:
        tries += 1
        # 前区：1-35 选5不重复，升序
        front_pool = [n for n in range(1,36) if n not in set(rules.get("front_exclude", []))]
        back_pool = [n for n in range(1,13) if n not in set(rules.get("back_exclude", []))]

        front_pool_w = weighted_pool(front_pool, hot=set(rules.get("hot_front", [])), cold=set(rules.get("cold_front", [])))
        back_pool_w = back_pool  # 后区先用均匀分布

        f = sorted(set(rng.sample(front_pool_w, 5)))
        if len(f) != 5:  # 可能因为样本中重复导致不足，再补
            f = sorted(set(rng.sample(front_pool, 5)))
        b = sorted(rng.sample(back_pool_w, 2))

        # 规则校验
        ok = True
        if set(rules.get("front_include", [])).difference(f):
            ok = False
        if set(rules.get("back_include", [])).difference(b):
            ok = False

        s = sum(f)
        smin, smax = rules.get("sum_front_range", [None,None])
        if smin is not None and s < smin: ok = False
        if smax is not None and s > smax: ok = False

        oe = rules.get("odd_even_front", None)
        if oe:
            odd = sum(1 for x in f if x%2==1)
            even = 5 - odd
            if odd != oe[0] or even != oe[1]:
                ok = False

        if ok:
            results.append({"front": f, "back": b})
    return results
