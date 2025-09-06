
import streamlit as st
import pandas as pd
from backend.db import init_db, session_scope, Draw
from backend.sync import upsert_from_source
from backend.analysis import dataframe_from_draws, freq_table, miss_table
from backend.generator import gen_numbers
from backend.sync import import_csv


st.set_page_config(page_title="大乐透分析与选号", page_icon="🎯", layout="wide")

st.title("🎯 大乐透分析与选号（本地版）12")
st.caption("数据来源：sporttery 历史接口；开奖日通常为每周一、三、六 21:25。")

#issue,date,f1,f2,f3,f4,f5,b1,b2,sales,pool format
with st.expander("📂 CSV 导入（本地历史数据）", expanded=True):
    csv_file = st.file_uploader("选择 CSV 文件", type=["csv"])
    if csv_file:
        if st.button("导入 CSV 数据"):
            try:
                n = import_csv(csv_file)
                if n > 0:
                    st.success(f"CSV 导入完成，共新增 {n} 条记录 ✅")
                else:
                    st.info("没有新增记录（可能 CSV 数据已在数据库中）")
            except Exception as e:
                st.error(f"导入失败：{e}")


# 数据同步
with st.expander("🗃️ 数据同步 / 状态", expanded=True):
    if st.button("增量同步（抓取历史与最新）"):
        placeholder = st.empty()
        def cb(n): placeholder.info(f"已新增 {n} 条…")
        try:
            n = upsert_from_source(progress_callback=cb)
            st.success(f"同步完成，新增 {n} 条记录。")
        except Exception as e:
            st.error(f"同步失败：{e}")

# 读取数据
init_db()
with session_scope() as s:
    rows = [{
        "issue": d.issue, "date": d.date.isoformat(),
        "f1": d.f1, "f2": d.f2, "f3": d.f3, "f4": d.f4, "f5": d.f5,
        "b1": d.b1, "b2": d.b2,
        "sales": d.sales, "pool": d.pool,
    } for d in s.query(Draw).order_by(Draw.issue.desc()).all()]

if not rows:
    st.warning("数据库暂无数据，请先执行一次“增量同步”。")
    st.stop()

df = dataframe_from_draws(rows)
st.dataframe(df.head(50), use_container_width=True)

# 左侧筛选
st.sidebar.header("筛选条件")
min_issue = int(df["issue"].astype(int).min())
max_issue = int(df["issue"].astype(int).max())
issue_range = st.sidebar.slider("期号范围", min_issue, max_issue, (max_issue-100, max_issue))
mask = (df["issue"].astype(int) >= issue_range[0]) & (df["issue"].astype(int) <= issue_range[1])
dfv = df.loc[mask].copy()

st.subheader("📈 指标与图表（选定期号范围）")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("样本期数", len(dfv))
with c2:
    st.metric("最近期号", int(dfv["issue"].astype(int).max()))
with c3:
    st.metric("最早期号", int(dfv["issue"].astype(int).min()))

# 频次与遗漏
tabs = st.tabs(["出现频次", "当前遗漏", "和值/奇偶"])
with tabs[0]:
    freq = freq_table(dfv)
    st.write("前区出现次数（1-35）：")
    st.bar_chart(freq["front"])
    st.write("后区出现次数（1-12）：")
    st.bar_chart(freq["back"])

with tabs[1]:
    miss = miss_table(dfv)
    st.write("前区当前遗漏：距离最近出现的期数（越大越“冷”）")
    st.bar_chart(miss["front"])
    st.write("后区当前遗漏：")
    st.bar_chart(miss["back"])

with tabs[2]:
    st.line_chart(dfv[["sum_front","sum_back","sum_all"]])

st.divider()

# 条件选号
st.subheader("🧪 条件选号")
colA, colB, colC, colD = st.columns(4)
with colA:
    sum_min = st.number_input("前区和值最小", min_value=0, max_value=200, value=70)
    sum_max = st.number_input("前区和值最大", min_value=0, max_value=200, value=140)
    odd = st.number_input("前区奇数个数", min_value=0, max_value=5, value=3)
with colB:
    front_include = st.text_input("前区必含(逗号分隔)", value="")
    front_exclude = st.text_input("前区排除(逗号分隔)", value="")
with colC:
    back_include = st.text_input("后区必含", value="")
    back_exclude = st.text_input("后区排除", value="")
with colD:
    exclude_hot_recent = st.checkbox("排除最近N期最热号码", value=False)
    n_recent = st.number_input("最近 N 期", min_value=1, max_value=200, value=20)


def parse_nums(s):
    return [int(x) for x in s.replace("，",",").split(",") if x.strip().isdigit()] if s else []

# 选定最近 N 期
df_recent = dfv.tail(n_recent) if exclude_hot_recent else pd.DataFrame()

hot_front = []
hot_back = []
if exclude_hot_recent and not df_recent.empty:
    # 前区
    front_counts = pd.concat([df_recent[c] for c in ["f1","f2","f3","f4","f5"]]).value_counts()
    hot_front = front_counts.head(2).index.tolist()
    # 后区
    back_counts = pd.concat([df_recent[c] for c in ["b1","b2"]]).value_counts()
    hot_back = back_counts.head(1).index.tolist()

rules = {
    "sum_front_range": [sum_min, sum_max],
    "odd_even_front": [odd, 5-odd],
    "front_include": parse_nums(front_include),
    "front_exclude": parse_nums(front_exclude) + hot_front,  # 自动加上最近 N 期最热号码
    "back_include": parse_nums(back_include),
    "back_exclude": parse_nums(back_exclude) + hot_back,
}

count = st.slider("生成注数", 1, 20, 5)
if st.button("生成候选号码"):
    cands = gen_numbers(count=count, rules=rules)
    if not cands:
        st.error("未能生成满足条件的号码，请放宽条件重试。")
    else:
        st.success(f"生成 {len(cands)} 注：")
        for i, cd in enumerate(cands, 1):
            st.write(f"第{i}注：前区 {cd['front']} | 后区 {cd['back']}")
        st.info("提示：可结合“当前遗漏”与“出现频次”进行冷热混合（在 `generator.py` 可扩展 hot/cold 权重规则）。")

st.caption("© 本工具仅作学习交流，勿用于非法用途。")
