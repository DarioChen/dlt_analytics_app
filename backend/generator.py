# generator.py v2.0 (增强动态排除规则)
from typing import List, Dict, Optional, Set, Tuple
import random
import numpy as np

def gen_numbers(
    count: int = 5,
    rules: Optional[Dict] = None,
    front_blocks: Optional[Dict[str, List[int]]] = None,
    back_blocks: Optional[Dict[str, List[int]]] = None,
    front_weights: Optional[Dict[str, float]] = None,
    back_weights: Optional[Dict[str, float]] = None,
    selected_front_blocks: Optional[List[str]] = None,
    selected_back_blocks: Optional[List[str]] = None,
    historical_data: Optional[List[Dict]] = None,  # 历史数据用于动态排除
    high_prize_focus: bool = False,  # 是否优先高额奖项
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
        
    def calculate_diversity_score(numbers: List[int]) -> float:
        """计算号码组合的多样性分数"""
        if len(numbers) < 2:
            return 0.0
        
        # 计算号码范围
        range_score = max(numbers) - min(numbers)
        
        # 计算平均距离
        distances = [numbers[i] - numbers[i-1] for i in range(1, len(numbers))]
        avg_distance = sum(distances) / len(distances)
        
        # 计算奇偶分布
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        parity_balance = 1.0 - abs(odd_count - even_count) / len(numbers)
        
        # 综合评分
        diversity = (range_score * 0.3 + avg_distance * 2.0 * 0.3 + parity_balance * 0.4)
        return diversity
        
    def get_dynamic_exclusions(historical_data: Optional[List[Dict]], 
                            lookback_periods: int = 10, 
                            high_prize_focus: bool = False) -> Tuple[Set[int], Set[int]]:
        """基于历史数据生成动态排除规则"""
        front_exclude_dynamic = set()
        back_exclude_dynamic = set()
        
        if not historical_data or len(historical_data) < lookback_periods:
            return front_exclude_dynamic, back_exclude_dynamic
        
        # 分析最近几期的号码
        recent_draws = historical_data[-lookback_periods:]
        
        # 统计频率
        front_freq = {}
        back_freq = {}
        
        for draw in recent_draws:
            for num in draw.get('front', []):
                front_freq[num] = front_freq.get(num, 0) + 1
            for num in draw.get('back', []):
                back_freq[num] = back_freq.get(num, 0) + 1
        
        # 动态排除策略：
        # 1. 普通模式：排除过于频繁出现的号码（太热）
        # 2. 高额奖项模式：排除最近连续出现的号码，但保留低频号码
        threshold = 3 if not high_prize_focus else 2
        
        # 排除太热的号码
        for num, freq in front_freq.items():
            if freq >= threshold:
                front_exclude_dynamic.add(num)
        
        for num, freq in back_freq.items():
            if freq >= threshold:
                back_exclude_dynamic.add(num)
        
        # 高额奖项模式下的特殊处理：
        # 1. 检查连续出现的号码
        # 2. 分析最近出现的组合模式
        if high_prize_focus and len(recent_draws) >= 3:
            # 检查连续3期出现的号码
            for num in front_freq:
                if all(num in draw.get('front', []) for draw in recent_draws[-3:]):
                    front_exclude_dynamic.add(num)
            
            for num in back_freq:
                if all(num in draw.get('back', []) for draw in recent_draws[-3:]):
                    back_exclude_dynamic.add(num)
        
        return front_exclude_dynamic, back_exclude_dynamic
        
    def check_pattern_exclusion(numbers: List[int], historical_data: Optional[List[Dict]], 
                               pattern_type: str = 'full') -> bool:
        """检查组合是否符合模式排除规则"""
        if not historical_data or len(historical_data) < 5:
            return False
        
        numbers_set = set(numbers)
        
        # 检查最近是否出现过相似模式
        for draw in historical_data[-5:]:
            draw_nums = draw.get('front', [])
            draw_set = set(draw_nums)
            
            # 完全相同的组合
            if pattern_type == 'full' and draw_set == numbers_set:
                return True
            
            # 相似度检查
            common_nums = numbers_set.intersection(draw_set)
            # 如果有4个以上数字相同，视为需要排除的模式
            if len(common_nums) >= 4:
                return True
        
        return False

    # 基础规则配置
    sum_range = rules.get("sum_front_range", [None, None])
    min_odd_count = rules.get("odd_even_front", [0])[0]  # 最小奇数个数
    max_odd_count = rules.get("odd_even_front", [0, 5])[1]  # 最大奇数个数
    cons_req = rules.get("consecutive_count", 0)
    cons_mode = rules.get("consecutive_mode", "exact")
    
    # 固定排除规则
    front_include = set(rules.get("front_include", []))
    front_exclude_fixed = set(rules.get("front_exclude", []))
    back_include = set(rules.get("back_include", []))
    back_exclude_fixed = set(rules.get("back_exclude", []))
    
    # 获取动态排除规则
    front_exclude_dynamic, back_exclude_dynamic = set(), set()
    if historical_data and rules.get("use_dynamic_exclusions", True):
        lookback = rules.get("dynamic_lookback_periods", 10)
        front_exclude_dynamic, back_exclude_dynamic = get_dynamic_exclusions(
            historical_data, lookback, high_prize_focus
        )
    
    # 合并排除规则
    front_exclude = front_exclude_fixed.union(front_exclude_dynamic)
    back_exclude = back_exclude_fixed.union(back_exclude_dynamic)
    
    # 新增高级规则
    min_diversity = rules.get("min_diversity_score", 0.0)  # 最小多样性分数
    avoid_recent_patterns = rules.get("avoid_recent_patterns", True)  # 是否避免最近出现的模式
    max_repeats = rules.get("max_repeated_numbers", 2)  # 最大重复历史号码数

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

        # ----------------- 高级检查条件 -----------------
        ok = True
        
        # 基本包含排除检查
        if front_include and not front_include.issubset(f_selected):
            ok = False
        if back_include and not back_include.issubset(b_selected):
            ok = False
            
        # 和值范围检查
        s = sum(f_selected)
        smin, smax = sum_range
        if (smin is not None and s < smin) or (smax is not None and s > smax):
            ok = False
            
        # 奇偶个数检查（新增上限检查）
        odd_count = sum(1 for n in f_selected if n % 2 == 1)
        if odd_count < min_odd_count or (max_odd_count is not None and odd_count > max_odd_count):
            ok = False
            
        # 连号对数检查
        cons_pairs = consecutive_pairs_count(f_selected)
        if cons_mode == "exact" and cons_pairs != cons_req:
            ok = False
        elif cons_mode == "min" and cons_pairs < cons_req:
            ok = False
        elif cons_mode == "max" and cons_pairs > cons_req:
            ok = False
        
        # 多样性检查
        if min_diversity > 0:
            diversity = calculate_diversity_score(f_selected)
            if diversity < min_diversity:
                ok = False
                
        # 避免最近出现的模式
        if avoid_recent_patterns and historical_data:
            if check_pattern_exclusion(f_selected, historical_data):
                ok = False
                
        # 检查重复历史号码数
        if max_repeats < 5 and historical_data:
            recent_history = historical_data[-3:]  # 最近3期
            for draw in recent_history:
                common_count = len(set(f_selected).intersection(set(draw.get('front', []))))
                if common_count > max_repeats:
                    ok = False
                    break
        
        # 高额奖项特定检查
        if high_prize_focus and ok:
            # 高额奖项通常具有更好的分布特性
            # 1. 检查号码范围是否适中（不太小也不太大）
            num_range = max(f_selected) - min(f_selected)
            if num_range < 15 or num_range > 30:
                # 随机接受一些异常值以增加多样性
                if random.random() > 0.3:
                    ok = False
            
            # 2. 检查是否有过于集中的数字
            gaps = sorted([f_selected[i] - f_selected[i-1] for i in range(1, len(f_selected))])
            if gaps[0] < 2 and gaps[1] < 3:  # 前两个间隔都很小
                # 随机接受一些异常值
                if random.random() > 0.4:
                    ok = False
        
        if ok:
            results.append({"front": f_selected, "back": b_selected})

    return results
