# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from backend.db import init_db, session_scope, Draw
from backend.sync import import_csv
from backend.analysis import dataframe_from_draws
from backend.generator import gen_numbers
import random

st.set_page_config(page_title="大乐透分析与选号", page_icon="🎯", layout="wide")
st.title("🎯 大乐透分析与选号（本地版）")

# --------------------- 数据筛选器 ---------------------
st.sidebar.header("🔎 数据筛选器（全局）")
start_issue = st.sidebar.text_input("起始期号", value="")
end_issue = st.sidebar.text_input("结束期号", value="")
start_date = st.sidebar.date_input("起始日期", value=None)
end_date = st.sidebar.date_input("结束日期", value=None)
recent_n = st.sidebar.number_input("最近 N 期", min_value=0, max_value=500, value=0)

def filter_df(df, start_issue="", end_issue="", start_date=None, end_date=None, recent_n=0):
    df_filtered = df.copy()
    if start_issue:
        df_filtered = df_filtered[df_filtered['issue'] >= start_issue]
    if end_issue:
        df_filtered = df_filtered[df_filtered['issue'] <= end_issue]
    if start_date:
        df_filtered = df_filtered[df_filtered['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['date'] <= pd.to_datetime(end_date)]
    if recent_n > 0:
        df_filtered = df_filtered.sort_values("date", ascending=False).head(recent_n)
    return df_filtered.reset_index(drop=True)

# --------------------- 初始化数据库 ---------------------
init_db()
with session_scope() as s:
    rows = [dict(
        issue=d.issue, date=d.date.isoformat(),
        f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4, f5=d.f5,
        b1=d.b1, b2=d.b2,
        sales=d.sales, pool=d.pool
    ) for d in s.query(Draw).order_by(Draw.issue.desc()).all()]

if not rows:
    st.warning("数据库暂无数据，请先导入 CSV。")
    st.stop()

df = dataframe_from_draws(rows)
df_filtered = filter_df(df, start_issue, end_issue, start_date, end_date, recent_n)

# --------------------- Tabs ---------------------
tab_data, tab_chart, tab_generate = st.tabs(["📂 数据管理", "📊 数据图表", "🔢 号码生成"])

# --------------------- 区块配置 ---------------------
front_bins = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
front_labels = ["1-5","6-10","11-15","16-20","21-25","26-30","31-35"]
# 后区拆分为连续区块
back_bins = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12)]
back_labels = [f"{lo}-{hi}" for lo, hi in back_bins]

# --------------------- Tab1: 数据管理 ---------------------
with tab_data:
    with st.expander("CSV 导入与模板", expanded=True):
        st.markdown("**CSV 格式示例**：`issue,date,f1,f2,f3,f4,f5,b1,b2,sales,pool`")
        st.text("20251001,2025-10-01,1,2,3,4,5,1,2,1000000,5000000")
        csv_file = st.file_uploader("选择 CSV 文件", type=["csv"])
        if csv_file and st.button("导入 CSV 数据"):
            try:
                result = import_csv(csv_file)
                st.success(f"导入数据结果：{result}")
            except Exception as e:
                st.error(f"导入失败：{e}")

    st.subheader(f"数据表（共 {len(df_filtered)} 条）")
    st.dataframe(df_filtered.head(50), use_container_width=True)

# --------------------- Tab2: 数据图表 ---------------------
with tab_chart:
    st.subheader("前区落点热力图")
    front_matrix = pd.DataFrame(0, index=df_filtered.index, columns=front_labels)
    for col in ["f1","f2","f3","f4","f5"]:
        for i,(lo,hi) in enumerate(front_bins):
            mask = df_filtered[col].between(lo, hi)
            front_matrix.loc[df_filtered.index[mask], front_labels[i]] = 1
    fig_front = px.imshow(front_matrix.T, color_continuous_scale="Blues",
                          labels=dict(x="期号", y="区块", color="次数"))
    st.plotly_chart(fig_front, use_container_width=True)

    st.subheader("后区落点热力图")
    back_matrix = pd.DataFrame(0, index=df_filtered.index, columns=back_labels)
    for col in ["b1","b2"]:
        for i,(lo,hi) in enumerate(back_bins):
            mask = df_filtered[col].between(lo, hi)
            back_matrix.loc[df_filtered.index[mask], back_labels[i]] = 1
    fig_back = px.imshow(back_matrix.T, color_continuous_scale="Reds",
                         labels=dict(x="期号", y="区块", color="次数"))
    st.plotly_chart(fig_back, use_container_width=True)

# --------------------- Tab3: 号码生成 ---------------------
with tab_generate:
    st.subheader("选择号码区块")
    selected_front_blocks = st.multiselect("前区区块", front_labels, default=front_labels)
    selected_back_blocks = st.multiselect("后区区块", back_labels, default=back_labels)

    def get_numbers_from_blocks(selected_labels, all_labels, all_bins):
        numbers = []
        for label, (lo, hi) in zip(all_labels, all_bins):
            if label in selected_labels:
                numbers.extend(range(lo, hi+1))
        return numbers

    front_pool = get_numbers_from_blocks(selected_front_blocks, front_labels, front_bins)
    back_pool = get_numbers_from_blocks(selected_back_blocks, back_labels, back_bins)

    st.write(f"前区可选号码：{sorted(front_pool)}")
    st.write(f"后区可选号码：{sorted(back_pool)}")

    # --------------------- 权重滑块 ---------------------
    st.subheader("前区权重")
    cols = st.columns(len(front_labels))
    front_weights = {}
    for i, label in enumerate(front_labels):
        front_weights[label] = cols[i].slider(label, 0.0, 1.0, 0.5, 0.01)

    st.subheader("后区权重")
    cols = st.columns(len(back_labels))
    back_weights = {}
    for i, label in enumerate(back_labels):
        back_weights[label] = cols[i].slider(label, 0.0, 1.0, 0.5, 0.01)

    st.subheader("高级规则")
    colA, colB, colC = st.columns(3)
    with colA:
        sum_min = st.number_input("前区和值最小", 0, 200, 70)
        sum_max = st.number_input("前区和值最大", 0, 200, 140)
        odd_count = st.number_input("前区奇数个数", 0, 5, 3)
    with colB:
        front_include = st.text_input("前区必含(逗号分隔)", "")
        front_exclude = st.text_input("前区排除(逗号分隔)", "")
        consecutive_count = st.number_input("前区连号数量", 0, 5, 0)
        cons_mode_label = st.selectbox("连号匹配方式", ["等于", "至少"])
        consecutive_mode = "exact" if cons_mode_label=="等于" else "min"
    with colC:
        back_include = st.text_input("后区必含(逗号分隔)", "")
        back_exclude = st.text_input("后区排除(逗号分隔)", "")

    max_gen = st.number_input("生成注数上限", 1, 100, 5)
    use_block_weight = st.checkbox("使用区块权重", True)

    def parse_nums(s: str):
        s = s.replace("，", ",")
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]

    rules = {
        "sum_front_range": [sum_min, sum_max],
        "odd_even_front": [odd_count, 5 - odd_count],
        "front_include": parse_nums(front_include),
        "front_exclude": parse_nums(front_exclude),
        "back_include": parse_nums(back_include),
        "back_exclude": parse_nums(back_exclude),
        "consecutive_count": consecutive_count,
        "consecutive_mode": consecutive_mode
    }

    st.subheader("🎯 中奖号码比对")
    win_front_input = st.text_input("中奖前区号码（逗号分隔）", "")
    win_back_input = st.text_input("中奖后区号码（逗号分隔）", "")

    prize_rules = [
        ("一等奖", lambda f,b: len(f)==5 and len(b)==2, "浮动，单注最高500万"),
        ("二等奖", lambda f,b: len(f)==5 and len(b)==1, "浮动，单注最高500万"),
        ("三等奖", lambda f,b: len(f)==5, "10000元"),
        ("四等奖", lambda f,b: len(f)>=4 and len(b)==2, "3000元"),
        ("五等奖", lambda f,b: len(f)>=4 and len(b)==1, "300元"),
        ("六等奖", lambda f,b: len(f)>=3 and len(b)==2, "200元"),
        ("七等奖", lambda f,b: len(f)>=4, "100元"),
        ("八等奖", lambda f,b: (len(f)>=3 and len(b)>=1) or (len(f)>=2 and len(b)>=2), "15元"),
        ("九等奖", lambda f,b: (len(f)>=3) or (len(f)>=1 and len(b)>=2) or (len(f)>=2 and len(b)>=1) or (len(b)>=2), "5元")
    ]

    def check_prize(gen_front, gen_back, win_front, win_back):
        fc = len(set(gen_front) & set(win_front))
        bc = len(set(gen_back) & set(win_back))
        for name, cond, bonus in prize_rules:
            if cond(range(fc), range(bc)):
                return f"{name} {bonus}"
        return "未中奖"

    if st.button("生成号码并比对"):
        win_front = parse_nums(win_front_input)
        win_back = parse_nums(win_back_input)
        cands = gen_numbers(
            count=max_gen,
            rules=rules,
            front_pool_user=front_pool,
            back_pool_user=back_pool,
            front_blocks={label: list(range(lo, hi+1)) for label, (lo, hi) in zip(front_labels, front_bins)},
            back_blocks={label: list(range(lo, hi+1)) for label, (lo, hi) in zip(back_labels, back_bins)},
            front_weights=front_weights,
            back_weights=back_weights,
            use_block_weight=use_block_weight
        )
        for i, cd in enumerate(cands, 1):
            prize = check_prize(cd['front'], cd['back'], win_front, win_back)
            st.markdown(f"**第{i}注：前区 {cd['front']} | 后区 {cd['back']} => {prize}**")
