# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from backend.db import init_db, session_scope, Draw
from backend.sync import import_csv
from backend.analysis import dataframe_from_draws
from backend import generator as genmod
from predictor import compute_weights_from_history, prepare_generator_inputs, compute_weights_from_history_ewma
import random
import io

st.set_page_config(page_title="大乐透分析与选号 v1.5", page_icon="🎯", layout="wide")
st.title("🎯 大乐透分析与选号（本地版） v1.5 — 含未来预测")

# --------------------- 数据筛选器（全局） ---------------------
st.sidebar.header("🔎 数据筛选器（全局）")
start_issue = st.sidebar.text_input("起始期号", value="")
end_issue = st.sidebar.text_input("结束期号", value="")
start_date = st.sidebar.date_input("起始日期", value=None)
end_date = st.sidebar.date_input("结束日期", value=None)
recent_n_global = st.sidebar.number_input("最近 N 期（全局）", min_value=0, max_value=500, value=0)

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
df_filtered = filter_df(df, start_issue, end_issue, start_date, end_date, recent_n_global)

# --------------------- Tabs ---------------------
tab_data, tab_chart, tab_generate, tab_predict = st.tabs(
    ["📂 数据管理", "📊 数据图表", "🔢 号码生成", "🔮 未来号码预测"]
)

# --------------------- 区块配置 ---------------------
front_bins = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
front_labels = ["1-5","6-10","11-15","16-20","21-25","26-30","31-35"]
back_bins = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)]
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
    st.dataframe(df_filtered.head(200), use_container_width=True)

# --------------------- Tab2: 数据图表 ---------------------
with tab_chart:
    st.subheader("前区落点热力图")
    front_matrix = pd.DataFrame(0, index=df_filtered.index, columns=front_labels)
    for col in ["f1","f2","f3","f4","f5"]:
        for i,(lo,hi) in enumerate(front_bins):
            mask = df_filtered[col].between(lo, hi)
            front_matrix.loc[df_filtered.index[mask], front_labels[i]] = 1
    fig_front = px.imshow(front_matrix.T, labels=dict(x="期号", y="区块", color="次数"))
    st.plotly_chart(fig_front, use_container_width=True)

    st.subheader("后区落点热力图")
    back_matrix = pd.DataFrame(0, index=df_filtered.index, columns=back_labels)
    for col in ["b1","b2"]:
        for i,(lo,hi) in enumerate(back_bins):
            mask = df_filtered[col].between(lo, hi)
            back_matrix.loc[df_filtered.index[mask], back_labels[i]] = 1
    fig_back = px.imshow(back_matrix.T, labels=dict(x="期号", y="区块", color="次数"))
    st.plotly_chart(fig_back, use_container_width=True)

# --------------------- Tab3: 号码生成 ---------------------
with tab_generate:
    st.subheader("选择号码区块")
    selected_front_blocks = st.multiselect("前区区块", front_labels, default=front_labels, key="gen_front")
    selected_back_blocks = st.multiselect("后区区块", back_labels, default=back_labels, key="gen_back")

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

    st.subheader("前区权重")
    cols = st.columns(len(front_labels))
    front_weights = {label:cols[i].slider(label,0.0,1.0,0.5,0.01) for i,label in enumerate(front_labels)}

    st.subheader("后区权重")
    cols = st.columns(len(back_labels))
    back_weights = {label:cols[i].slider(label,0.0,1.0,0.5,0.01) for i,label in enumerate(back_labels)}

    st.subheader("高级规则")
    colA, colB, colC = st.columns(3)
    with colA:
        sum_min = st.number_input("前区和值最小", 0, 200, 70)
        sum_max = st.number_input("前区和值最大", 0, 200, 140)
        odd_count = st.number_input("前区最小奇数个数", 0, 5, 3)
        top_n_blocks = st.number_input("前区仅使用前 N 权重区块", 1, len(front_labels), len(front_labels))
        max_per_block = st.number_input("每个区块最多选号码", 1, 5, 2)
        random_blocks_count = st.number_input("每期随机选区块数量", 1, len(front_labels), min(3, len(front_labels)))
    with colB:
        front_include = st.text_input("前区必含(逗号分隔)", "")
        front_exclude = st.text_input("前区排除(逗号分隔)", "")
        consecutive_count = st.number_input("前区连号数量(对数)", 0, 5, 0)
        cons_mode_label = st.selectbox("连号匹配方式", ["等于", "至少"])
        consecutive_mode = "exact" if cons_mode_label == "等于" else "min"
    with colC:
        back_include = st.text_input("后区必含(逗号分隔)", "")
        back_exclude = st.text_input("后区排除(逗号分隔)", "")

    max_gen = st.number_input("生成注数上限",1,100,5)

    def parse_nums(s: str):
        s = s.replace("，",",")
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]


    rules = {
        "sum_front_range": [sum_min, sum_max],
        "odd_even_front": [odd_count, 5 - odd_count],
        "front_include": parse_nums(front_include),
        "front_exclude": parse_nums(front_exclude),
        "back_include": parse_nums(back_include),
        "back_exclude": parse_nums(back_exclude),
        "consecutive_count": consecutive_count,
        "consecutive_mode": consecutive_mode,
        "top_n_blocks": top_n_blocks,
        "max_per_block": max_per_block,
        "random_blocks_count": random_blocks_count
    }

    st.subheader("🎯 中奖号码比对")
    win_front_input = st.text_input("中奖前区号码（逗号分隔）","")
    win_back_input = st.text_input("中奖后区号码（逗号分隔）","")

    prize_rules = [
        ("一等奖", lambda fc,bc: fc==5 and bc==2, "浮动，单注最高500万"),
        ("二等奖", lambda fc,bc: fc==5 and bc==1, "浮动，单注最高500万"),
        ("三等奖", lambda fc,bc: fc==5 and bc==0, "10000元"),
        ("四等奖", lambda fc,bc: fc>=4 and bc==2, "3000元"),
        ("五等奖", lambda fc,bc: fc>=4 and bc==1, "300元"),
        ("六等奖", lambda fc,bc: fc>=3 and bc==2, "200元"),
        ("七等奖", lambda fc,bc: fc>=4 and bc==0, "100元"),
        ("八等奖", lambda fc,bc: (fc>=3 and bc>=1) or (fc==2 and bc==2), "15元"),
        ("九等奖", lambda fc,bc: (fc>=3) or (fc==1 and bc==2) or (fc==2 and bc==1) or (bc==2), "5元")
    ]

    def check_prize(gen_front, gen_back, win_front, win_back):
        fc = len(set(gen_front)&set(win_front))
        bc = len(set(gen_back)&set(win_back))
        for name, cond, bonus in prize_rules:
            if cond(fc,bc):
                return f"{name} {bonus}"
        return "未中奖"

    if st.button("生成号码并比对"):
        win_front = parse_nums(win_front_input)
        win_back = parse_nums(win_back_input)
        cands = genmod.gen_numbers(
            count=max_gen,
            rules=rules,
            front_blocks={label:list(range(lo,hi+1)) for label,(lo,hi) in zip(front_labels,front_bins)},
            back_blocks={label:list(range(lo,hi+1)) for label,(lo,hi) in zip(back_labels,back_bins)},
            front_weights=front_weights,
            back_weights=back_weights,
            selected_front_blocks=selected_front_blocks,
            selected_back_blocks=selected_back_blocks
        )
        for i,cd in enumerate(cands,1):
            prize = check_prize(cd['front'],cd['back'],win_front,win_back)
            st.markdown(f"**第{i}注：前区 {cd['front']} | 后区 {cd['back']} => {prize}**")


# --------------------- Tab4: 未来号码预测 + 历史回测（优化版） ---------------------
with tab_predict:
    st.header("🔮 基于历史冷热权重的未来号码预测 & 回测")
    st.markdown(
        """
        **预测说明**：
        - 从当前筛选后的历史数据统计出现频次
        - 可设置最小奇数个数、最小连号数量
        - 可排除前 N 期出现次数最多号码
        - 支持回测历史 N 期预测结果并比对中奖情况
        """
    )

    # ----------------- 全局规则 -----------------
    predict_rules = {
        "sum_front_range": [0, 999],
        "odd_even_front": [0, 5],
        "front_include": [],
        "front_exclude": [],
        "back_include": [],
        "back_exclude": [],
        "consecutive_count": 0,
        "consecutive_mode": "min"
    }

    col1, col2 = st.columns([2,1])
    with col1:
        use_recent_n = st.number_input("用于权重计算的最近 N 期（0=全部）", min_value=0, max_value=1000, value=5, key="tab4_recent_n")
        pred_count = st.number_input("每期预测注数", min_value=1, max_value=20, value=5, key="tab4_pred_count")
        pred_selected_front = st.multiselect("预测：前区使用区块", front_labels, default=front_labels, key="tab4_front")
        pred_selected_back = st.multiselect("预测：后区使用区块", back_labels, default=back_labels, key="tab4_back")
        min_consec = st.number_input("前区最小连号数量", min_value=0, max_value=5, value=0, key="tab4_min_consec")
        min_odd = st.number_input("前区最小奇数个数", min_value=0, max_value=5, value=2, key="tab4_min_odd")

        top_n_blocks_future = st.number_input("前区仅使用前 N 权重区块", 1, len(front_labels), len(front_labels),
                                              key="tab4_top_n_blocks")
        max_per_block_future = st.number_input("每个区块最多选号码", 1, 5, 2, key="tab4_max_per_block")
        exclude_top_n = st.checkbox("排除前 N 期出现次数最多的号码", value=False, key="tab4_exclude_top_n")
        exclude_top_front_n = st.number_input("前区排除数量", min_value=0, max_value=10, value=3, key="tab4_exclude_front_n")
        exclude_top_back_n = st.number_input("后区排除数量", min_value=0, max_value=5, value=2, key="tab4_exclude_back_n")
        backtest_n = st.number_input("回测历史 N 期（0=不回测）", min_value=0, max_value=500, value=10, key="tab4_backtest_n")
        span = st.slider("移动平均 EWMA span（越小越重视近期）", min_value=1, max_value=5, value=2, key="tab4_span")
        random_blocks_count_future = st.number_input("每期随机选区块数量", 1, len(front_labels),
                                                     min(3, len(front_labels)), key="tab4_random_blocks_count")

    with col2:
        st.markdown("""
        **说明**：
        - 可生成未来预测并对历史回测进行中奖比对
        - 每期预测结果放一行，多注列显示
        """)

    # ----------------- 计算区块权重 -----------------
    front_block_weights, back_block_weights, front_freq_map, back_freq_map = compute_weights_from_history_ewma(
        df_filtered,
        front_blocks={label: list(range(lo, hi + 1)) for label, (lo, hi) in zip(front_labels, front_bins)},
        back_blocks={label: list(range(lo, hi + 1)) for label, (lo, hi) in zip(back_labels, back_bins)},
        recent_n=use_recent_n,
        out_min=0.2,
        out_max=1.5,
        span=span
    )

    # ----------------- 排除高频号码 -----------------
    exclude_front = []
    exclude_back = []
    if exclude_top_n:
        top_front = sorted(front_freq_map.items(), key=lambda x: -x[1])[:exclude_top_front_n]
        top_back = sorted(back_freq_map.items(), key=lambda x: -x[1])[:exclude_top_back_n]
        exclude_front = [n for n,_ in top_front]
        exclude_back = [n for n,_ in top_back]

    # ----------------- 准备 generator 输入 -----------------
    gen_front_blocks, gen_back_blocks, gen_front_weights, gen_back_weights = prepare_generator_inputs(
        front_blocks_labels=front_labels,
        front_bins=front_bins,
        back_blocks_labels=back_labels,
        back_bins=back_bins,
        selected_front_blocks=pred_selected_front,
        selected_back_blocks=pred_selected_back,
        block_front_weights=front_block_weights,
        block_back_weights=back_block_weights
    )

    # ----------------- 生成未来预测号码 -----------------
    if st.button("生成未来预测号码", key="tab4_gen_future"):
        rules_future = predict_rules.copy()
        rules_future["consecutive_count"] = min_consec
        rules_future["odd_even_front"] = [min_odd, 5 - min_odd]
        rules_future["front_exclude"] = exclude_front
        rules_future["back_exclude"] = exclude_back
        rules_future["top_n_blocks"] = top_n_blocks_future
        rules_future["max_per_block"] = max_per_block_future
        rules_future["random_blocks_count"] = random_blocks_count_future

        cands = genmod.gen_numbers(
            count=pred_count,
            rules=rules_future,
            front_blocks=gen_front_blocks,
            back_blocks=gen_back_blocks,
            front_weights=gen_front_weights,
            back_weights=gen_back_weights,
            selected_front_blocks=pred_selected_front,
            selected_back_blocks=pred_selected_back
        )

        if not cands:
            st.warning("未生成到符合条件的号码，请调整参数重试。")
        else:
            # 每注一列显示
            row_data = {}
            for i, c in enumerate(cands, 1):
                row_data[f"预测前区{i}"] = ",".join(map(str, c["front"]))
                row_data[f"预测后区{i}"] = ",".join(map(str, c["back"]))
                row_data[f"中奖情况{i}"] = "未比对"
            pred_df = pd.DataFrame([row_data])
            st.subheader(f"未来预测结果（共 {len(cands)} 注）")
            st.dataframe(pred_df, use_container_width=True)

    # ----------------- 历史回测 -----------------
    if backtest_n > 0:
        st.subheader(f"历史回测（最近 {backtest_n} 期，每期 {pred_count} 注）")
        history_df = df_filtered.sort_values("date", ascending=False).head(backtest_n).reset_index(drop=True)

        PRIZE_RULES = [
            ("一等奖", lambda fc,bc: fc==5 and bc==2),
            ("二等奖", lambda fc,bc: fc==5 and bc==1),
            ("三等奖", lambda fc,bc: fc==5 and bc==0),
            ("四等奖", lambda fc,bc: fc>=4 and bc==2),
            ("五等奖", lambda fc,bc: fc>=4 and bc==1),
            ("六等奖", lambda fc,bc: fc>=3 and bc==2),
            ("七等奖", lambda fc,bc: fc>=4 and bc==0),
            ("八等奖", lambda fc,bc: (fc>=3 and bc>=1) or (fc==2 and bc==2)),
            ("九等奖", lambda fc,bc: (fc>=3) or (fc==1 and bc==2) or (fc==2 and bc==1) or (bc==2))
        ]
        PRIZE_COLOR = {
            "一等奖": "background-color:#ff4d4d;color:black;",
            "二等奖": "background-color:#ff944d;color:black;",
            "三等奖": "background-color:#ffd24d;color:black;",
            "四等奖": "background-color:#ffff4d;color:black;",
            "五等奖": "background-color:#b3ff66;color:black;",
            "六等奖": "background-color:#66ffb3;color:black;",
            "七等奖": "background-color:#66b3ff;color:black;",
            "八等奖": "background-color:#b366ff;color:black;",
            "九等奖": "background-color:#ff66f2;color:black;",
            "未中奖": "background-color:#f0f0f0;color:black;"
        }

        def check_prize(fc, bc, win_fc, win_bc):
            fc_match = len(set(fc)&set(win_fc))
            bc_match = len(set(bc)&set(win_bc))
            for name, cond in PRIZE_RULES:
                if cond(fc_match, bc_match):
                    return name
            return "未中奖"

        backtest_data = []
        for _, row in history_df.iterrows():
            rules_hist = predict_rules.copy()
            rules_hist["consecutive_count"] = min_consec
            rules_hist["odd_even_front"] = [min_odd, 5-min_odd]
            rules_hist["front_exclude"] = exclude_front
            rules_hist["back_exclude"] = exclude_back

            gen = genmod.gen_numbers(
                count=pred_count,
                rules=rules_hist,
                front_blocks=gen_front_blocks,
                back_blocks=gen_back_blocks,
                front_weights=gen_front_weights,
                back_weights=gen_back_weights,
                selected_front_blocks=pred_selected_front,
                selected_back_blocks=pred_selected_back
            )

            row_data = {
                "期号": row["issue"],
                "历史前区": ",".join(map(str, row[["f1","f2","f3","f4","f5"]])),
                "历史后区": ",".join(map(str, row[["b1","b2"]]))
            }
            for i, c in enumerate(gen, 1):
                row_data[f"预测前区{i}"] = ",".join(map(str, c["front"]))
                row_data[f"预测后区{i}"] = ",".join(map(str, c["back"]))
                row_data[f"中奖情况{i}"] = check_prize(c["front"], c["back"], row[["f1","f2","f3","f4","f5"]], row[["b1","b2"]])
            backtest_data.append(row_data)

        backtest_df = pd.DataFrame(backtest_data)

        # 样式
        prize_cols = [col for col in backtest_df.columns if "中奖情况" in col]
        st.dataframe(backtest_df.style.applymap(lambda v: PRIZE_COLOR.get(v, ""), subset=prize_cols), use_container_width=True)






