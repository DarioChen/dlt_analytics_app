
"""
数据源：sporttery 历史接口（gameNo=85 为大乐透）
示例：
https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=85&provinceId=0&pageSize=30&isVerify=1&pageNo=1
"""
from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import re

BASE = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
LOTTERY_GOV_BASE = "https://www.lottery.gov.cn/kj/kjlb.html"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36"
}

def fetch_page(
    page_no:int=1,
    page_size:int=30,
    proxies:Optional[Dict[str,str]]=None,
    verify:bool=True,
    timeout:int=15
) -> Dict:
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
    r = requests.get(
        BASE,
        headers=headers,
        params=params,
        timeout=timeout,
        proxies=proxies,
        verify=verify
    )
    r.raise_for_status()
    return r.json()

def iter_history(
    max_pages:int=200,
    proxies:Optional[Dict[str,str]]=None,
    verify:bool=True,
    timeout:int=15
) -> Iterator[Dict]:
    for p in range(1, max_pages+1):
        data = fetch_page(
            page_no=p,
            page_size=30,
            proxies=proxies,
            verify=verify,
            timeout=timeout
        )
        result_list = (data or {}).get("value", {}).get("list", [])
        if not result_list:
            break
        for row in result_list:
            yield row

def _extract_numbers_from_td(td) -> List[int]:
    texts: List[str] = []
    for span in td.find_all("span"):
        txt = span.get_text(strip=True)
        if txt:
            texts.append(txt)
    if not texts:
        raw = td.get_text(" ", strip=True)
        texts = re.split(r"\s+", raw)
    nums = []
    for t in texts:
        t_clean = re.sub(r"[^\d]", "", t)
        if len(t_clean) == 0:
            continue
        nums.append(int(t_clean))
    return nums

def _normalize_lottery_gov_row(issue: str, date_str: str, front_nums: List[int], back_nums: List[int]) -> Optional[Dict]:
    if len(front_nums) < 5 or len(back_nums) < 2:
        return None
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        # some rows might only have issue (e.g. header)
        return None
    issue_clean = issue.strip().replace("第", "").replace("期", "")
    issue_clean = re.sub(r"[^\d]", "", issue_clean) or issue.strip()
    front_nums = front_nums[:5]
    back_nums = back_nums[:2]
    out = {
        "issue": issue_clean,
        "date": dt.isoformat(),
        "f1": front_nums[0],
        "f2": front_nums[1],
        "f3": front_nums[2],
        "f4": front_nums[3],
        "f5": front_nums[4],
        "b1": back_nums[0],
        "b2": back_nums[1],
        "sales": "",
        "pool": ""
    }
    return out

def iter_lottery_gov_history(
    max_pages:int=20,
    proxies:Optional[Dict[str,str]]=None,
    verify:bool=True,
    timeout:int=15
) -> Iterator[Dict]:
    """
    抓取中国体彩网（lottery.gov.cn）开奖列表页面
    """
    for page in range(1, max_pages+1):
        params = [("dlt", ""), ("page", page)]
        resp = requests.get(
            LOTTERY_GOV_BASE,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            proxies=proxies,
            verify=verify
        )
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="kj_table")
        if not table:
            break
        rows = table.find_all("tr")
        if len(rows) <= 1:
            break
        added = 0
        for tr in rows[1:]:
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            date_text = tds[0].get_text(strip=True)
            issue_text = tds[1].get_text(strip=True)
            front_nums = _extract_numbers_from_td(tds[2])
            back_nums = _extract_numbers_from_td(tds[3])
            normalized = _normalize_lottery_gov_row(issue_text, date_text, front_nums, back_nums)
            if normalized:
                yield normalized
                added += 1
        if added == 0:
            break

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
