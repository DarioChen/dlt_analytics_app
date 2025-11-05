# backend/sync.py
import pandas as pd
from backend.db import session_scope, Draw

def import_csv(file) -> dict:
    """
    导入 CSV 数据，返回详细结果。
    file: 文件对象
    返回 dict: {"new": 新增条数, "dup": 重复条数, "errors": 错误信息列表}
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return {"new":0, "dup":0, "errors":[f"文件读取失败: {e}"]}

    required_cols = ['issue','date','f1','f2','f3','f4','f5','b1','b2','sales','pool']
    for col in required_cols:
        if col not in df.columns:
            return {"new":0, "dup":0, "errors":[f"缺少列: {col}"]}

    n_new = 0
    n_dup = 0
    errors = []

    with session_scope() as s:
        for i, row in df.iterrows():
            try:
                # 判断重复
                if s.query(Draw).filter_by(issue=row['issue']).first():
                    n_dup += 1
                    continue

                d = Draw(
                    issue=str(row['issue']),
                    date=pd.to_datetime(row['date']),
                    f1=int(row['f1']), f2=int(row['f2']), f3=int(row['f3']),
                    f4=int(row['f4']), f5=int(row['f5']),
                    b1=int(row['b1']), b2=int(row['b2']),
                    sales=float(row['sales']), pool=float(row['pool'])
                )
                s.add(d)
                n_new += 1
            except Exception as e:
                errors.append(f"第{i+1}行错误: {e}")

    return {"new": n_new, "dup": n_dup, "errors": errors}
