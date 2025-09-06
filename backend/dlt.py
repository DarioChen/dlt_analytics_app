
"""
数据源：sporttery 历史接口（gameNo=85 为大乐透）
示例：
https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=85&provinceId=0&pageSize=30&isVerify=1&pageNo=1
"""
from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import requests
from datetime import datetime

BASE = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"

def fetch_page(page_no:int=1, page_size:int=30) -> Dict:
    params = {
        "gameNo": 85,
        "provinceId": 0,
        "pageSize": page_size,
        "isVerify": 1,
        "pageNo": page_no,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://www.sporttery.cn/",
    }
    r = requests.get(BASE,  headers=headers,params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def iter_history(max_pages:int=200) -> Iterator[Dict]:
    for p in range(1, max_pages+1):
        data = fetch_page(page_no=p, page_size=30)
        result_list = (data or {}).get("value", {}).get("list", [])
        if not result_list:
            break
        for row in result_list:
            yield row

def normalize_row(row:Dict) -> Optional[Dict]:
    """
    兼容字段：
      - lotteryDrawNum: 期号（字符串）
      - lotteryDrawTime: 开奖日期（YYYY-MM-DD）
      - lotteryDrawResult: '01 02 03 04 05 06 07' （前5+后2）
      - poolBalanceAfterdraw: 奖池金额（可选）
      - totalSalesAmount: 销售额（可选）
    """
    issue = str(row.get("lotteryDrawNum") or "").strip()
    date_s = str(row.get("lotteryDrawTime") or "").strip()
    res = str(row.get("lotteryDrawResult") or "").strip()

    if not issue or not date_s or not res:
        return None
    try:
        dt = datetime.strptime(date_s, "%Y-%m-%d").date()
    except Exception:
        return None

    nums = [int(x) for x in res.split() if x.isdigit()]
    if len(nums) != 7:
        return None
    f1,f2,f3,f4,f5,b1,b2 = nums
    out = {
        "issue": issue,
        "date": dt.isoformat(),
        "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
        "b1": b1, "b2": b2,
        "sales": str(row.get("totalSalesAmount") or ""),
        "pool": str(row.get("poolBalanceAfterdraw") or ""),
    }
    return out
