import pandas as pd
from backend.db import init_db, session_scope, Draw
from app import dataframe_from_draws

# 初始化数据库
init_db()

# 查询数据
with session_scope() as s:
    rows = [dict(
        issue=d.issue, date=d.date.isoformat(),
        f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4, f5=d.f5,
        b1=d.b1, b2=d.b2,
        sales=d.sales, pool=d.pool
    ) for d in s.query(Draw).order_by(Draw.issue.desc()).limit(20).all()]

# 转换为DataFrame
df = dataframe_from_draws(rows)

# 检查后区数据
print("=== 后区数据检查 ===")
print("最近20期后区数据：")
print(df[['issue', 'b1', 'b2']])

print("\n后区号码统计：")
print(df[['b1', 'b2']].describe())

# 检查后区区块定义
back_bins = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)]
back_labels = [f"{lo}-{hi}" for lo, hi in back_bins]
print("\n后区区块定义：")
for label, (lo, hi) in zip(back_labels, back_bins):
    print(f"{label}: {lo}-{hi}")

# 测试热力图矩阵构建
print("\n=== 测试热力图矩阵构建 ===")
back_matrix = pd.DataFrame(0, index=df.index, columns=back_labels)
for col in ["b1", "b2"]:
    for i, (lo, hi) in enumerate(back_bins):
        mask = df[col].between(lo, hi)
        print(f"{col} in {lo}-{hi}: {mask.sum()} 次")
        back_matrix.loc[df.index[mask], back_labels[i]] += 1

print("\n热力图矩阵（转置后）：")
print(back_matrix.T)

# 检查是否有数据
print("\n矩阵中是否有非零值：", back_matrix.sum().sum() > 0)
print("每个区块的总出现次数：")
print(back_matrix.sum(axis=0))