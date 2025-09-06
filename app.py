import streamlit as st
import pandas as pd
import plotly.express as px
from backend.db import init_db, session_scope, Draw
from backend.sync import import_csv
from backend.analysis import dataframe_from_draws
from backend.generator import gen_numbers
from typing import List


st.set_page_config(page_title="大乐透分析与选号", page_icon="🎯", layout="wide")
st.title("🎯 大乐透分析与选号（本地版）")

# ---------------- CSV 导入 ----------------
with st.expander("📂 CSV 导入（本地历史数据）", expanded=True):
    csv_file = st.file_uploader("选择 CSV 文件（列: issue,date,f1,f2,f3,f4,f5,b1,b2,sales,pool）", type=["csv"])
    if csv_file:
        if st.button("导入 CSV 数据"):
            try:
                n = import_csv(csv_file)
                if n > 0:
                    st.success(f"CSV 导入完成，共新增 {n} 条记录 ✅")
                else:
                    st.info("没有新增记录（可能 CSV 中的数据已在库中）")
            except Exception as e:
                st.error(f"导入失败：{e}")

# ---------------- 读取数据 ----------------
init_db()
with session_scope() as s:
    rows = [{
        "issue": d.issue, "date": d.date.isoformat(),
        "f1": d.f1, "f2": d.f2, "f3": d.f3, "f4": d.f4, "f5": d.f5,
        "b1": d.b1, "b2": d.b2,
        "sales": d.sales, "pool": d.pool,
    } for d in s.query(Draw).order_by(Draw.issue.desc()).all()]

if not rows:
    st.warning("数据库暂无数据，请先导入 CSV。")
    st.stop()

df = dataframe_from_draws(rows)
st.subheader("最近开奖（示例）")
st.dataframe(df.head(50), use_container_width=True)

def count_numbers_in_bins(df: pd.DataFrame):
    """统计每期号码落在每个格子里的次数"""
    # 前区统计
    front_counts = {label:0 for label in front_labels}
    for col in ["f1","f2","f3","f4","f5"]:
        for i,(lo,hi) in enumerate(front_bins):
            front_counts[front_labels[i]] += df[col].apply(lambda x: lo<=x<=hi).sum()
    # 后区统计
    back_counts = {label:0 for label in back_labels}
    for col in ["b1","b2"]:
        for i,(lo,hi) in enumerate(back_bins):
            back_counts[back_labels[i]] += df[col].apply(lambda x: lo<=x<=hi).sum()
    return front_counts, back_counts


# 前区格子边界
front_bins = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
front_labels = ["1-5","6-10","11-15","16-20","21-25","26-30","31-35"]
# 后区格子边界
back_bins = [(1,2),(3,4),(5,6),(7,8),(9,12)]
back_labels = ["1-2","3-4","5-6","7-8","9-12"]

# 假设 df 是你的历史开奖 DataFrame，列名 f1~f5,b1~b2
front_counts, back_counts = count_numbers_in_bins(df)

tab1, tab2 = st.tabs(["号码区间分布", "其他分析"])

with tab1:
    st.subheader("前区号码落在区间的次数")
    df_front = pd.DataFrame(list(front_counts.items()), columns=["区间","次数"])
    fig_front = px.bar(df_front, x="区间", y="次数", text="次数", color="次数", color_continuous_scale="Blues")
    st.plotly_chart(fig_front, use_container_width=True)

    st.subheader("后区号码落在区间的次数")
    df_back = pd.DataFrame(list(back_counts.items()), columns=["区间","次数"])
    fig_back = px.bar(df_back, x="区间", y="次数", text="次数", color="次数", color_continuous_scale="Reds")
    st.plotly_chart(fig_back, use_container_width=True)

# ---------------- 条件选号 UI ----------------
st.subheader("🧪 条件选号（规则）")
colA, colB, colC, colD = st.columns(4)

with colA:
    sum_min = st.number_input("前区和值最小", min_value=0, max_value=200, value=70)
    sum_max = st.number_input("前区和值最大", min_value=0, max_value=200, value=140)
    odd = st.number_input("前区奇数个数", min_value=0, max_value=5, value=3)
    # 英文参数名, 中文 label
    consecutive_count = st.number_input("前区连号数量", min_value=0, max_value=5, value=0, step=1)

with colB:
    front_include = st.text_input("前区必含(逗号分隔)", value="")
    front_exclude = st.text_input("前区排除(逗号分隔)", value="")

with colC:
    back_include = st.text_input("后区必含", value="")
    back_exclude = st.text_input("后区排除", value="")

with colD:
    exclude_hot_recent = st.checkbox("排除最近N期最热号码", value=False)
    n_recent = st.number_input("最近 N 期", min_value=1, max_value=500, value=20)
    # 连号匹配方式（中文 label，英文参数值）
    cons_mode_label = st.selectbox("连号匹配方式", options=["等于", "至少"])
    consecutive_mode = "exact" if cons_mode_label == "等于" else "min"

def parse_nums(s: str) -> List[int]:
    s = s or ""
    s = s.replace("，", ",")
    out = []
    for x in s.split(","):
        x = x.strip()
        if x.isdigit():
            out.append(int(x))
    return out

# ---------------- 计算最近 N 期最热号码（可选） ----------------
hot_front, hot_back = [], []
if exclude_hot_recent and not df.empty:
    df_recent = df.tail(n_recent)
    front_counts = pd.concat([df_recent[c] for c in ["f1","f2","f3","f4","f5"]]).value_counts()
    hot_front = front_counts.head(3).index.tolist()
    back_counts = pd.concat([df_recent[c] for c in ["b1","b2"]]).value_counts()
    hot_back = back_counts.head(2).index.tolist()
    st.info(f"排除热号：前区 {hot_front}，后区 {hot_back}")

# ---------------- 组装规则（英文 keys） ----------------
rules = {
    "sum_front_range": [sum_min, sum_max],
    "odd_even_front": [odd, 5 - odd],
    "front_include": parse_nums(front_include),
    "front_exclude": parse_nums(front_exclude) + hot_front,
    "back_include": parse_nums(back_include),
    "back_exclude": parse_nums(back_exclude) + hot_back,
    "consecutive_count": int(consecutive_count),
    "consecutive_mode": consecutive_mode,
}

# ---------------- 生成并展示 ----------------
count = st.slider("生成注数", 1, 20, 5)
if st.button("生成候选号码"):
    try:
        cands = gen_numbers(count=count, rules=rules)
        if not cands:
            st.error("未能生成满足条件的号码，请放宽条件或检查设置。")
        else:
            st.success(f"生成 {len(cands)} 注：")
            for i, cd in enumerate(cands, 1):
                st.write(f"第{i}注：前区 {cd['front']} | 后区 {cd['back']}")
    except Exception as e:
        st.error(f"生成失败：{e}")

st.caption("© 本工具仅作学习交流，勿用于非法用途。")
