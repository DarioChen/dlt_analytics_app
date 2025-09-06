
from __future__ import annotations
from typing import List, Dict, Optional
from .db import init_db, session_scope, Draw
from .dlt import iter_history, normalize_row
from datetime import datetime
import csv
from .db import session_scope, Draw, init_db

def upsert_from_source(progress_callback=None) -> int:
    """
    从数据源全量迭代，按 issue upsert 到数据库。
    返回：新增条数
    """
    init_db()
    added = 0
    with session_scope() as s:
        # 建立现有期号集合用于去重
        existing = {x.issue for x in s.query(Draw.issue).all()}
        for raw in iter_history(max_pages=500):
            rec = normalize_row(raw)
            if not rec: 
                continue
            if rec["issue"] in existing:
                continue
            obj = Draw(
                issue=rec["issue"],
                date=datetime.fromisoformat(rec["date"]).date(),
                f1=rec["f1"], f2=rec["f2"], f3=rec["f3"], f4=rec["f4"], f5=rec["f5"],
                b1=rec["b1"], b2=rec["b2"],
                sales=rec["sales"],
                pool=rec["pool"],
            )
            s.add(obj)
            added += 1
            if progress_callback and added % 50 == 0:
                progress_callback(added)
    return added

def import_csv(file) -> int:
    """
    从本地 CSV 导入历史开奖数据到数据库
    返回新增的记录数
    """
    init_db()
    added = 0

    if isinstance(file, str):
        f = open(file, "r", encoding="utf-8-sig")
    else:
        import io
        f = io.TextIOWrapper(file, encoding="utf-8-sig")

    with session_scope() as s:
        existing = {x.issue for x in s.query(Draw.issue).all()}
        reader = csv.DictReader(f)
        for row in reader:
            issue = row["issue"].strip()
            if issue in existing:
                continue
            try:
                dt = datetime.strptime(row["date"], "%Y-%m-%d").date()
                f1 = int(row["f1"]); f2 = int(row["f2"]); f3 = int(row["f3"]); f4 = int(row["f4"]); f5 = int(row["f5"])
                b1 = int(row["b1"]); b2 = int(row["b2"])
                sales = row.get("sales", "")
                pool = row.get("pool", "")
            except Exception:
                continue

            obj = Draw(
                issue=issue, date=dt,
                f1=f1,f2=f2,f3=f3,f4=f4,f5=f5,
                b1=b1,b2=b2,
                sales=sales, pool=pool
            )
            s.add(obj)
            added += 1
    return added

