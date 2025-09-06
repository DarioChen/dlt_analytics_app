import streamlit as st
import pandas as pd
import plotly.express as px
from backend.db import init_db, session_scope, Draw
from backend.sync import import_csv
from backend.analysis import dataframe_from_draws
from typing import List, Dict
from itertools import combinations
import random

from backend.generator import gen_numbers  # 使用修改后的新版本

st.set_page_config(page_title="大乐透分析与选号", page_icon="🎯", layout="wide")
st.title("🎯 大乐透分析与选号（本地版）")

# ------------------- 模块1：数据导入 -------------------
with st.expander("📂 数据导入（本地 CSV）", expanded=True):
    csv_file = st.file_uploader(
        "选择 CSV 文件（列: issue,date,f1,f2,f3,f4,f5,b1,b2,sales,pool）", type=["csv"]
    )
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

# ------------------- 初始化数据库并读取数据 -------------------
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

# ------------------- 模块2：数据表展示 -------------------
with st.expander("📋 数据表（最近开奖示例）", expanded=True):
    st.dataframe(df.head(50), use_container_width=True)

# ------------------- 模块3：数据范围筛选 -------------------
with st.expander("📅 数据筛选", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        start_issue = st.text_input("起始期号", value="")
    with col2:
        end_issue = st.text_input("结束期号", value="")

    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("起始日期", value=None)
    with col4:
        end_date = st.date_input("结束日期", value=None)

    recent_n = st.number_input("最近 N 期", min_value=0, max_value=500, value=0)

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
            df_filtered = df_filtered.tail(recent_n)
        return df_filtered

    df_filtered = filter_df(df, start_issue, end_issue, start_date, end_date, recent_n)
    st.write(f"筛选后共 {len(df_filtered)} 条记录")

# ------------------- 模块4：数据图表 -------------------
with st.expander("📊 号码区间落点统计", expanded=True):
    front_bins = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
    front_labels = ["1-5","6-10","11-15","16-20","21-25","26-30","31-35"]
    back_bins = [(1,2),(3,4),(5,6),(7,8),(9,12)]
    back_labels = ["1-2","3-4","5-6","7-8","9-12"]

    def count_numbers_in_bins(df):
        front_counts = {label:0 for label in front_labels}
        for col in ["f1","f2","f3","f4","f5"]:
            for i,(lo,hi) in enumerate(front_bins):
                front_counts[front_labels[i]] += df[col].apply(lambda x: lo<=x<=hi).sum()
        back_counts = {label:0 for label in back_labels}
        for col in ["b1","b2"]:
            for i,(lo,hi) in enumerate(back_bins):
                back_counts[back_labels[i]] += df[col].apply(lambda x: lo<=x<=hi).sum()
        return front_counts, back_counts

    front_counts, back_counts = count_numbers_in_bins(df_filtered)

    st.subheader("前区落点统计")
    df_front = pd.DataFrame(list(front_counts.items()), columns=["区间","次数"])
    fig_front = px.bar(df_front, x="区间", y="次数", text="次数",
                       color="次数", color_continuous_scale="Blues")
    st.plotly_chart(fig_front, use_container_width=True)

    st.subheader("后区落点统计")
    df_back = pd.DataFrame(list(back_counts.items()), columns=["区间","次数"])
    fig_back = px.bar(df_back, x="区间", y="次数", text="次数",
                      color="次数", color_continuous_scale="Reds")
    st.plotly_chart(fig_back, use_container_width=True)

# ------------------- 模块5：号码生成 -------------------
with st.expander("🔢 条件选号与组合生成", expanded=True):
    st.subheader("选择号码区块（block）")
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

    st.subheader("高级选号条件")
    colA, colB, colC = st.columns(3)

    with colA:
        sum_min = st.number_input("前区和值最小", min_value=0, max_value=200, value=70)
        sum_max = st.number_input("前区和值最大", min_value=0, max_value=200, value=140)
        odd_count = st.number_input("前区奇数个数", min_value=0, max_value=5, value=3)

    with colB:
        front_include = st.text_input("前区必含(逗号分隔)", value="")
        front_exclude = st.text_input("前区排除(逗号分隔)", value="")
        consecutive_count = st.number_input("前区连号数量", min_value=0, max_value=5, value=0)
        cons_mode_label = st.selectbox("连号匹配方式", options=["等于", "至少"])
        consecutive_mode = "exact" if cons_mode_label == "等于" else "min"

    with colC:
        back_include = st.text_input("后区必含(逗号分隔)", value="")
        back_exclude = st.text_input("后区排除(逗号分隔)", value="")
        exclude_hot_recent = st.checkbox("排除最近N期最热号码", value=False)
        n_recent = st.number_input("最近 N 期", min_value=1, max_value=500, value=20)

    def parse_nums(s: str):
        s = s or ""
        s = s.replace("，", ",")
        out = []
        for x in s.split(","):
            x = x.strip()
            if x.isdigit():
                out.append(int(x))
        return out

    hot_front, hot_back = [], []
    if exclude_hot_recent and not df.empty:
        df_recent = df.tail(n_recent)
        front_counts_hot = pd.concat([df_recent[c] for c in ["f1","f2","f3","f4","f5"]]).value_counts()
        hot_front = front_counts_hot.head(3).index.tolist()
        back_counts_hot = pd.concat([df_recent[c] for c in ["b1","b2"]]).value_counts()
        hot_back = back_counts_hot.head(2).index.tolist()
        st.info(f"排除热号：前区 {hot_front}，后区 {hot_back}")

    rules = {
        "sum_front_range": [sum_min, sum_max],
        "odd_even_front": [odd_count, 5 - odd_count],
        "front_include": parse_nums(front_include),
        "front_exclude": parse_nums(front_exclude) + hot_front,
        "back_include": parse_nums(back_include),
        "back_exclude": parse_nums(back_exclude) + hot_back,
        "consecutive_count": consecutive_count,
        "consecutive_mode": consecutive_mode,
        "hot_front": hot_front
    }

    max_gen = st.number_input("生成注数上限", min_value=1, max_value=100, value=20)

    if st.button("生成号码组合"):
        with st.spinner("正在生成号码组合，请稍候..."):
            try:
                cands = gen_numbers(
                    count=max_gen,
                    rules=rules,
                    front_pool_user=front_pool,
                    back_pool_user=back_pool
                )
                if not cands:
                    st.warning("未能生成满足条件的号码，请放宽条件或检查设置。")
                else:
                    st.success(f"生成 {len(cands)} 注：")
                    for i, cd in enumerate(cands, 1):
                        st.write(f"第{i}注：前区 {cd['front']} | 后区 {cd['back']}")
            except Exception as e:
                st.error(f"生成失败：{e}")

st.caption("© 本工具仅作学习交流，勿用于非法用途。")
