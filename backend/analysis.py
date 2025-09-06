
from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

def dataframe_from_draws(rows:List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    num_cols = ["f1","f2","f3","f4","f5","b1","b2"]
    for c in num_cols:
        df[c] = df[c].astype(int)
    # 衍生指标
    df["sum_front"] = df[["f1","f2","f3","f4","f5"]].sum(axis=1)
    df["sum_back"] = df[["b1","b2"]].sum(axis=1)
    df["sum_all"] = df["sum_front"] + df["sum_back"]
    df["odd_count"] = df[["f1","f2","f3","f4","f5","b1","b2"]].apply(lambda r: sum(x%2 for x in r), axis=1)
    return df

def freq_table(df:pd.DataFrame) -> Dict[str, pd.Series]:
    front = pd.concat([df[c] for c in ["f1","f2","f3","f4","f5"]]).value_counts().sort_index()
    back = pd.concat([df[c] for c in ["b1","b2"]]).value_counts().sort_index()
    return {"front": front, "back": back}

def miss_table(df:pd.DataFrame) -> Dict[str, pd.Series]:
    # 简单遗漏：从最近一期向前数，距离上次出现的期数
    def last_seen(series:pd.Series, pool:range) -> pd.Series:
        last = {n: None for n in pool}
        miss = {n: 0 for n in pool}
        for idx, nums in enumerate(series[::-1], start=0):
            for n in pool:
                if n in nums:
                    last[n] = idx
        return pd.Series({n: (last[n] if last[n] is not None else len(series)) for n in pool}).sort_index()

    arr_front = list(zip(df["f1"],df["f2"],df["f3"],df["f4"],df["f5"]))
    arr_back = list(zip(df["b1"],df["b2"]))
    miss_front = last_seen(pd.Series(arr_front), range(1,36))
    miss_back = last_seen(pd.Series(arr_back), range(1,13))
    return {"front": miss_front, "back": miss_back}
