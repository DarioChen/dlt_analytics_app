# app.py (v1.4)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
import math
from backend.db import init_db, session_scope, Draw
from backend.sync import import_csv, sync_remote_history
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
        df_filtered = df_filtered.tail(recent_n)
    return df_filtered

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
tab_data, tab_chart, tab_generate, tab_predict, tab_ai = st.tabs(
    ["📂 数据管理", "📊 数据图表", "🔢 号码生成", "🔮 未来号码预测", "🤖 AI优化与模型训练"]
)

# --------------------- 区块定义 ---------------------
front_bins = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
front_labels = ["1-5","6-10","11-15","16-20","21-25","26-30","31-35"]
back_bins = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)]
back_labels = [f"{lo}-{hi}" for lo, hi in back_bins]
front_blocks_full = {label: list(range(lo, hi + 1)) for label, (lo, hi) in zip(front_labels, front_bins)}
back_blocks_full = {label: list(range(lo, hi + 1)) for label, (lo, hi) in zip(back_labels, back_bins)}

def build_history_window(df_source: pd.DataFrame, recent_n: int = 0, cutoff_date=None) -> pd.DataFrame:
    """
    根据 recent_n 和可选截止日期，返回按日期升序的历史窗口
    """
    df_sorted = df_source.sort_values("date", ascending=True)
    if cutoff_date is not None:
        df_sorted = df_sorted[df_sorted["date"] <= pd.to_datetime(cutoff_date)]
    if recent_n and recent_n > 0:
        df_sorted = df_sorted.tail(recent_n)
    return df_sorted

def prepare_generation_context(
    df_window: pd.DataFrame,
    span: int,
    front_blocks_labels,
    back_blocks_labels,
    selected_front_blocks,
    selected_back_blocks,
):
    """
    基于 df_window 计算权重/频次，并生成 generator 所需的区块映射
    """
    front_block_weights, back_block_weights, front_freq_map, back_freq_map = compute_weights_from_history_ewma(
        df_window,
        front_blocks=front_blocks_full,
        back_blocks=back_blocks_full,
        recent_n=0,
        out_min=0.2,
        out_max=1.5,
        span=span
    )

    gen_front_blocks, gen_back_blocks, gen_front_weights, gen_back_weights = prepare_generator_inputs(
        front_blocks_labels=front_blocks_labels,
        front_bins=front_bins,
        back_blocks_labels=back_blocks_labels,
        back_bins=back_bins,
        selected_front_blocks=selected_front_blocks,
        selected_back_blocks=selected_back_blocks,
        block_front_weights=front_block_weights,
        block_back_weights=back_block_weights
    )

    return {
        "front_blocks": gen_front_blocks,
        "back_blocks": gen_back_blocks,
        "front_weights": gen_front_weights,
        "back_weights": gen_back_weights,
        "front_freq_map": front_freq_map,
        "back_freq_map": back_freq_map
    }

def compute_exclusions(front_freq_map, back_freq_map, exclude_top_n, exclude_front_n, exclude_back_n):
    if not exclude_top_n:
        return [], []
    top_front = sorted(front_freq_map.items(), key=lambda x: -x[1])[:exclude_front_n]
    top_back = sorted(back_freq_map.items(), key=lambda x: -x[1])[:exclude_back_n]
    return [n for n, _ in top_front], [n for n, _ in top_back]

def assemble_rules(base_rules, min_consec, min_odd, exclude_front, exclude_back,
                   top_n_blocks, max_per_block, random_blocks_count, random_back_blocks_count):
    rules = base_rules.copy()
    rules.update({
        "consecutive_count": min_consec,
        "odd_even_front": [min_odd, 5 - min_odd],
        "front_exclude": exclude_front,
        "back_exclude": exclude_back,
        "top_n_blocks": top_n_blocks,
        "max_per_block": max_per_block,
        "random_blocks_count": random_blocks_count,
        "random_back_blocks_count": random_back_blocks_count
    })
    return rules

# --------------------- Tab1: 数据管理 ---------------------
with tab_data:
    st.subheader("CSV 导入")
    st.write("模板示例：期号,日期,f1,f2,f3,f4,f5,b1,b2,sales,pool")
    st.write("示例：20251105,2025-11-05,1,3,5,7,9,1,2,1000000,5000000")
    csv_file = st.file_uploader("选择 CSV 文件", type=["csv"])
    if csv_file and st.button("导入 CSV 数据"):
        try:
            result = import_csv(csv_file)
            st.success(f"导入 {result['new']} 新增, {result['dup']} 重复, {len(result['errors'])} 错误")
            if result['errors']:
                st.write(result['errors'])
        except Exception as e:
            st.error(f"导入失败：{e}")

    total_rows = len(df_filtered)
    st.subheader(f"数据表（共 {total_rows} 条）")
    if total_rows == 0:
        st.info("暂无符合筛选条件的数据。")
    else:
        page_size = st.number_input("每页显示条数", 50, 1000, 200, step=50, key="data_page_size")
        total_pages = max(1, math.ceil(total_rows / page_size))
        page_num = st.number_input("页码", 1, total_pages, 1, key="data_page_num")
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        st.caption(f"显示第 {start_idx + 1} - {end_idx} 条")
        st.dataframe(df_filtered.iloc[start_idx:end_idx], use_container_width=True)

    st.subheader("在线同步（官方接口）")
    source_label = st.selectbox(
        "数据来源",
        options=[
            ("sporttery", "中国体彩-竞彩网接口（JSON）"),
            ("lottery", "中国体彩网官网列表（HTML）")
        ],
        format_func=lambda x: x[1],
        key="sync_source"
    )
    sync_source_value = source_label[0]
    max_pages_sync = st.number_input(
        "抓取页数",
        1,
        200 if sync_source_value == "sporttery" else 50,
        20 if sync_source_value == "sporttery" else 5,
        key="sync_max_pages"
    )
    proxy_url = st.text_input("HTTP/HTTPS 代理地址（可选）", "", key="sync_proxy")
    disable_ssl_verify = st.checkbox("忽略 SSL 证书校验", value=False, key="sync_disable_ssl")
    timeout_sync = st.number_input("请求超时时间（秒）", 5, 60, 15, key="sync_timeout")
    if st.button("从官网同步最新开奖", key="sync_remote"):
        proxies = None
        proxy_url_clean = proxy_url.strip()
        if proxy_url_clean:
            proxies = {"http": proxy_url_clean, "https": proxy_url_clean}
        verify_flag = not disable_ssl_verify
        with st.spinner("正在抓取开奖数据..."):
            try:
                result = sync_remote_history(
                    max_pages=int(max_pages_sync),
                    proxies=proxies,
                    verify=verify_flag,
                    timeout=int(timeout_sync),
                    source=sync_source_value
                )
            except Exception as e:
                st.error(f"同步失败：{e}")
            else:
                msg = f"新增 {result['new']} 条, 重复 {result['dup']} 条"
                if result["errors"]:
                    st.warning(msg + f"，有 {len(result['errors'])} 条错误")
                    st.write(result["errors"])
                else:
                    st.success(msg)
                st.rerun()

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
        for label,(lo,hi) in zip(all_labels, all_bins):
            if label in selected_labels:
                numbers.extend(range(lo,hi+1))
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
        random_back_blocks_count = st.number_input(
            "每期随机选后区区块数量",
            1,
            len(back_labels),
            min(2, len(back_labels)),
            key="gen_back_random_blocks",
        )

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
        "random_blocks_count": random_blocks_count,
        "random_back_blocks_count": random_back_blocks_count
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
    st.header("🔮 冷热权重预测 & 回测")
    st.caption("基于历史开奖动态权重生成候选号码，并可回测最近若干期命中情况。")

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

    with st.expander("🎯 预测参数", expanded=True):
        base_cols = st.columns(4)
        # 使用优化后的默认值：recent_n=3
        use_recent_n = base_cols[0].number_input("权重最近N期", 0, 1000, 2, key="tab4_recent_n")
        pred_count = base_cols[1].number_input("每期注数", 1, 20, 5, key="tab4_pred_count")
        # 使用优化后的默认值：min_consec=0
        min_consec = base_cols[2].number_input("前区最小连号", 0, 5, 0, key="tab4_min_consec")
        # 使用优化后的默认值：min_odd=2
        min_odd = base_cols[3].number_input("前区最小奇数", 0, 5, 2, key="tab4_min_odd")

        block_cols = st.columns(2)
        pred_selected_front = block_cols[0].multiselect("前区区块", front_labels, default=front_labels, key="tab4_front")
        pred_selected_back = block_cols[1].multiselect("后区区块", back_labels, default=back_labels, key="tab4_back")

        adv_cols = st.columns(3)
        # 使用优化后的默认值：top_n_blocks=5
        top_n_blocks_future = adv_cols[0].number_input(
            "前区仅用前N区块", min_value=3, max_value=7, value=4, key="tab4_top_n_blocks"
        )
        # 使用优化后的默认值：max_per_block=2
        max_per_block_future = adv_cols[1].number_input("每区块最多取", 1, 5, 2, key="tab4_max_per_block")
        # 使用优化后的默认值：random_blocks_count=4
        random_blocks_count_future = adv_cols[2].number_input(
            "每期随机区块数", 1, len(front_labels), 4, key="tab4_random_blocks_count"
        )

        adv_back_cols = st.columns(2)
        # 使用优化后的默认值：random_back_blocks_count=1
        random_back_blocks_count_future = adv_back_cols[0].number_input(
            "后区随机区块数", 1, len(back_labels), 3, key="tab4_random_back_blocks_count"
        )
        # 使用优化后的默认值：span=1
        span = adv_back_cols[1].slider("EWMA span", 1, 5, 1, key="tab4_span")

    with st.expander("🧹 排除与回测设置", expanded=False):
        # 使用优化后的默认值：exclude_top_n=False
        exclude_top_n = st.checkbox("排除近期高频号码", value=False, key="tab4_exclude_top_n")
        exclusion_cols = st.columns(2)
        # 使用优化后的默认值：exclude_front_n=3
        exclude_top_front_n = exclusion_cols[0].number_input(
            "前区排除数量", 0, 10, 3, key="tab4_exclude_front_n"
        )
        # 使用优化后的默认值：exclude_back_n=2
        exclude_top_back_n = exclusion_cols[1].number_input(
            "后区排除数量", 0, 5, 2, key="tab4_exclude_back_n"
        )
        backtest_n = st.number_input("回测历史期数（0=不回测）", 0, 500, 10, key="tab4_backtest_n")
        st.caption("建议调小 N 和区块数量以提升生成速度。")
    with st.expander("💰 成本与奖金参数", expanded=False):
        ticket_cost = st.number_input("单注投注金额（元）", 1.0, 20.0, 3.0, 0.5, key="tab4_ticket_cost")
        st.caption("大乐透普通投注为 3 元/注，根据实际玩法调整。")
        PRIZE_PAYOUT_DEFAULT = {
            "一等奖": 5000000.0,
            "二等奖": 1500000.0,
            "三等奖": 10000.0,
            "四等奖": 3000.0,
            "五等奖": 300.0,
            "六等奖": 200.0,
            "七等奖": 100.0,
            "八等奖": 15.0,
            "九等奖": 5.0
        }
        prize_amounts = {}
        prize_cols = st.columns(3)
        for idx, (name, default_val) in enumerate(PRIZE_PAYOUT_DEFAULT.items()):
            with prize_cols[idx % 3]:
                prize_amounts[name] = st.number_input(
                    f"{name}估值（元）",
                    0.0,
                    10000000.0,
                    default_val,
                    10.0,
                    key=f"prize_amount_{name}"
                )

    # ----------------- 检查是否使用AI模型 -----------------
    use_ai_model = st.checkbox("使用AI模型预测（需先训练模型）", value=False, key="tab4_use_ai")
    ml_predictor = st.session_state.get('ml_predictor', None)
    if use_ai_model and ml_predictor is None:
        st.warning("⚠️ 未检测到训练好的AI模型，请在'AI优化与模型训练'标签页先训练模型。")
        use_ai_model = False
    
    # ----------------- 基线历史窗口 & 生成上下文 -----------------
    recent_window = build_history_window(df_filtered, recent_n=use_recent_n)
    
    # 如果使用AI模型，用ML预测器计算权重
    if use_ai_model and ml_predictor is not None:
        try:
            from backend.ml_predictor import MLPredictor
            front_ml_weights, back_ml_weights = ml_predictor.predict_number_weights(
                recent_window, front_range=range(1,36), back_range=range(1,13)
            )
            # 将号码权重转换为区块权重
            front_block_weights_ai = {}
            back_block_weights_ai = {}
            for label, (lo, hi) in zip(front_labels, front_bins):
                block_nums = list(range(lo, hi+1))
                avg_weight = np.mean([front_ml_weights.get(n, 0) for n in block_nums])
                front_block_weights_ai[label] = float(avg_weight * 10 + 0.5)  # 放大并归一化
            
            for label, (lo, hi) in zip(back_labels, back_bins):
                block_nums = list(range(lo, hi+1))
                avg_weight = np.mean([back_ml_weights.get(n, 0) for n in block_nums])
                back_block_weights_ai[label] = float(avg_weight * 10 + 0.5)
            
            # 归一化到合理范围
            from predictor import normalize_block_weights
            front_block_weights_ai = normalize_block_weights(front_block_weights_ai, 0.2, 1.5)
            back_block_weights_ai = normalize_block_weights(back_block_weights_ai, 0.2, 1.5)
            
            generation_context = prepare_generation_context(
                df_window=recent_window,
                span=span,
                front_blocks_labels=front_labels,
                back_blocks_labels=back_labels,
                selected_front_blocks=pred_selected_front,
                selected_back_blocks=pred_selected_back,
            )
            # 用AI权重替换
            generation_context["front_weights"] = front_block_weights_ai
            generation_context["back_weights"] = back_block_weights_ai
            st.info(f"✅ 使用AI模型（{st.session_state.get('model_type', 'unknown')}）计算权重")
        except Exception as e:
            st.warning(f"AI模型预测失败，回退到EWMA方法: {e}")
            generation_context = prepare_generation_context(
                df_window=recent_window,
                span=span,
                front_blocks_labels=front_labels,
                back_blocks_labels=back_labels,
                selected_front_blocks=pred_selected_front,
                selected_back_blocks=pred_selected_back,
            )
    else:
        generation_context = prepare_generation_context(
            df_window=recent_window,
            span=span,
            front_blocks_labels=front_labels,
            back_blocks_labels=back_labels,
            selected_front_blocks=pred_selected_front,
            selected_back_blocks=pred_selected_back,
        )

    # ----------------- 排除高频号码（与回测共用逻辑） -----------------
    exclude_front, exclude_back = compute_exclusions(
        generation_context["front_freq_map"],
        generation_context["back_freq_map"],
        exclude_top_n,
        exclude_top_front_n,
        exclude_top_back_n
    )

    # ----------------- 生成未来预测号码 -----------------
    if st.button("生成未来预测号码", key="tab4_gen_future"):
        rules_future = assemble_rules(
            base_rules=predict_rules,
            min_consec=min_consec,
            min_odd=min_odd,
            exclude_front=exclude_front,
            exclude_back=exclude_back,
            top_n_blocks=top_n_blocks_future,
            max_per_block=max_per_block_future,
            random_blocks_count=random_blocks_count_future,
            random_back_blocks_count=random_back_blocks_count_future
        )

        cands = genmod.gen_numbers(
            count=pred_count,
            rules=rules_future,
            front_blocks=generation_context["front_blocks"],
            back_blocks=generation_context["back_blocks"],
            front_weights=generation_context["front_weights"],
            back_weights=generation_context["back_weights"],
            selected_front_blocks=pred_selected_front,
            selected_back_blocks=pred_selected_back
        )

        if not cands:
            st.warning("未生成到符合条件的号码，请调整参数重试。")
        else:
            # 每注一列显示
            rows = []
            for i, c in enumerate(cands, 1):
                rows.append({
                    "预测序号": i,
                    "预测前区": ",".join(map(str, c["front"])),
                    "预测后区": ",".join(map(str, c["back"])),
                    "中奖情况": "未比对"
                })

            pred_df = pd.DataFrame(rows)

            st.subheader(f"未来预测结果（共 {len(cands)} 注）")
            if use_ai_model:
                st.info(f"🤖 使用AI模型（{st.session_state.get('model_type', 'unknown')}）生成")
            
            # 可视化：号码分布热力图
            st.subheader("📊 预测号码分布可视化")
            
            # 前区号码分布
            front_nums_all = []
            for c in cands:
                front_nums_all.extend(c["front"])
            front_counts = pd.Series(front_nums_all).value_counts().sort_index()
            
            back_nums_all = []
            for c in cands:
                back_nums_all.extend(c["back"])
            back_counts = pd.Series(back_nums_all).value_counts().sort_index()
            
            viz_cols = st.columns(2)
            with viz_cols[0]:
                st.write("**前区号码出现频次**")
                fig_front_dist = px.bar(
                    x=front_counts.index, 
                    y=front_counts.values,
                    labels={"x": "号码", "y": "出现次数"},
                    title="前区号码分布"
                )
                st.plotly_chart(fig_front_dist, use_container_width=True)
            
            with viz_cols[1]:
                st.write("**后区号码出现频次**")
                fig_back_dist = px.bar(
                    x=back_counts.index,
                    y=back_counts.values,
                    labels={"x": "号码", "y": "出现次数"},
                    title="后区号码分布"
                )
                st.plotly_chart(fig_back_dist, use_container_width=True)
            
            # 显示详细表格
            st.subheader("📋 详细预测号码")
            st.dataframe(pred_df, use_container_width=False)

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
        total_cost_all = 0.0
        total_return_all = 0.0
        total_bets_all = 0
        for idx, row in history_df.iterrows():
            # 每一期都重新计算前N期的权重
            recent_window_dyn = build_history_window(
                df_filtered,
                recent_n=use_recent_n,
                cutoff_date=row["date"]
            )

            generation_context_dyn = prepare_generation_context(
                df_window=recent_window_dyn,
                span=span,
                front_blocks_labels=front_labels,
                back_blocks_labels=back_labels,
                selected_front_blocks=pred_selected_front,
                selected_back_blocks=pred_selected_back,
            )

            # 排除高频号码（每期动态，与预测一致）
            exclude_front_dyn, exclude_back_dyn = compute_exclusions(
                generation_context_dyn["front_freq_map"],
                generation_context_dyn["back_freq_map"],
                exclude_top_n,
                exclude_top_front_n,
                exclude_top_back_n
            )

            # 生成号码
            rules_hist_dyn = assemble_rules(
                base_rules=predict_rules,
                min_consec=min_consec,
                min_odd=min_odd,
                exclude_front=exclude_front_dyn,
                exclude_back=exclude_back_dyn,
                top_n_blocks=top_n_blocks_future,
                max_per_block=max_per_block_future,
                random_blocks_count=random_blocks_count_future,
                random_back_blocks_count=random_back_blocks_count_future
            )

            gen = genmod.gen_numbers(
                count=pred_count,
                rules=rules_hist_dyn,
                front_blocks=generation_context_dyn["front_blocks"],
                back_blocks=generation_context_dyn["back_blocks"],
                front_weights=generation_context_dyn["front_weights"],
                back_weights=generation_context_dyn["back_weights"],
                selected_front_blocks=pred_selected_front,
                selected_back_blocks=pred_selected_back
            )

            row_data = {
                "期号": row["issue"],
                "历史前区": ",".join(map(str, row[["f1", "f2", "f3", "f4", "f5"]])),
                "历史后区": ",".join(map(str, row[["b1", "b2"]]))
            }
            row_bets = len(gen)
            row_cost = row_bets * ticket_cost
            row_return = 0.0
            for i, c in enumerate(gen, 1):
                row_data[f"预测前区{i}"] = ",".join(map(str, c["front"]))
                row_data[f"预测后区{i}"] = ",".join(map(str, c["back"]))
                prize_name = check_prize(
                    c["front"],
                    c["back"],
                    row[["f1", "f2", "f3", "f4", "f5"]].tolist(),
                    row[["b1", "b2"]].tolist()
                )
                row_data[f"中奖情况{i}"] = prize_name
                row_return += prize_amounts.get(prize_name, 0.0)
            row_data["投注注数"] = row_bets
            row_data["投入(元)"] = row_cost
            row_data["回收(元)"] = row_return
            row_data["收益(元)"] = row_return - row_cost
            total_cost_all += row_cost
            total_return_all += row_return
            total_bets_all += row_bets
            backtest_data.append(row_data)

        backtest_df = pd.DataFrame(backtest_data)

        # 样式
        prize_cols = [col for col in backtest_df.columns if "中奖情况" in col]
        st.dataframe(
            backtest_df.style.applymap(lambda v: PRIZE_COLOR.get(v, ""), subset=prize_cols),
            use_container_width=False
        )

        summary_cols = st.columns(3)
        summary_cols[0].metric("累计投入", f"{total_cost_all:.0f} 元", delta=None)
        summary_cols[1].metric(
            "累计回收",
            f"{total_return_all:.0f} 元",
            delta=f"{(total_return_all - total_cost_all):.0f} 元"
        )
        roi = (total_return_all - total_cost_all) / total_cost_all if total_cost_all > 0 else 0.0
        summary_cols[2].metric(
            "ROI",
            f"{roi*100:.1f}%",
            delta=f"共 {total_bets_all} 注"
        )


# --------------------- Tab5: AI优化与模型训练 ---------------------
with tab_ai:
    st.header("🤖 AI优化与模型训练")
    st.caption("通过机器学习模型和自动参数优化，提高高金额奖项命中率并保本。")
    
    # 定义必要的辅助函数和变量（与tab_predict保持一致）
    PRIZE_RULES_AI = [
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
    
    def check_prize_ai(fc, bc, win_fc, win_bc):
        fc_match = len(set(fc)&set(win_fc))
        bc_match = len(set(bc)&set(win_bc))
        for name, cond in PRIZE_RULES_AI:
            if cond(fc_match, bc_match):
                return name
        return "未中奖"
    
    # 默认参数（可从session state或用户输入获取）
    predict_rules_ai = {
        "sum_front_range": [0, 999],
        "odd_even_front": [0, 5],
        "front_include": [],
        "front_exclude": [],
        "back_include": [],
        "back_exclude": [],
        "consecutive_count": 0,
        "consecutive_mode": "min"
    }
    
    PRIZE_PAYOUT_DEFAULT_AI = {
        "一等奖": 5000000.0,
        "二等奖": 1500000.0,
        "三等奖": 10000.0,
        "四等奖": 3000.0,
        "五等奖": 300.0,
        "六等奖": 200.0,
        "七等奖": 100.0,
        "八等奖": 15.0,
        "九等奖": 5.0
    }
    
    ai_tabs = st.tabs(["🧠 模型训练", "🔍 参数优化", "📊 策略对比"])
    
    with ai_tabs[0]:
        st.subheader("机器学习模型训练")
        st.write("训练LightGBM/XGBoost模型预测号码出现概率")
        
        model_type = st.selectbox("模型类型", ["lightgbm", "xgboost", "frequency"], key="ai_model_type")
        train_window = st.number_input("训练数据窗口（期数）", 50, 1000, 200, key="ai_train_window")
        
        if st.button("开始训练模型", key="ai_train_btn"):
            try:
                st.info("正在初始化...")
                from backend.ml_predictor import create_ml_predictor
                try:
                    from backend.ml_predictor import HAS_LIGHTGBM, HAS_XGBOOST
                except ImportError:
                    HAS_LIGHTGBM = False
                    HAS_XGBOOST = False
                from backend.features import extract_all_features
                
                # 检查依赖
                if model_type == "lightgbm" and not HAS_LIGHTGBM:
                    st.error("❌ 缺少 lightgbm 库")
                    st.code("pip install lightgbm scikit-learn", language="bash")
                    st.stop()
                elif model_type == "xgboost" and not HAS_XGBOOST:
                    st.error("❌ 缺少 xgboost 库")
                    st.code("pip install xgboost scikit-learn", language="bash")
                    st.stop()
                
                # 检查 scikit-learn（lightgbm 和 xgboost 都需要）
                try:
                    import sklearn
                except ImportError:
                    st.error("❌ 缺少 scikit-learn 库（LightGBM/XGBoost 需要）")
                    st.code("pip install scikit-learn", language="bash")
                    st.stop()
                
                # 准备训练数据
                st.info(f"准备训练数据（使用最近 {train_window} 期）...")
                if len(df_filtered) < train_window:
                    st.warning(f"可用数据只有 {len(df_filtered)} 期，少于请求的 {train_window} 期。将使用全部可用数据。")
                    train_df = df_filtered.sort_values("date", ascending=True)
                else:
                    train_df = df_filtered.sort_values("date", ascending=True).tail(train_window)
                
                st.info("提取特征中...")
                train_df = extract_all_features(train_df)
                st.info(f"特征提取完成，共 {len(train_df)} 条记录，{len(train_df.columns)} 个特征列")
                
                predictor = create_ml_predictor(model_type=model_type if model_type != "frequency" else "lightgbm")
                
                if model_type != "frequency":
                    with st.spinner(f"训练 {model_type.upper()} 模型中..."):
                        result = predictor.train_front_model(train_df, target_col="f1")
                        if result is None:
                            st.warning("训练数据不足或特征提取失败，模型未训练。将使用频率模型。")
                            predictor = None
                        else:
                            st.success(f"{model_type.upper()} 模型训练完成！")
                else:
                    st.info("使用频率模型（无需训练）")
                
                # 保存到session state
                if predictor is not None:
                    st.session_state['ml_predictor'] = predictor
                    st.session_state['model_type'] = model_type
                    st.success("模型已保存到会话状态，可在预测时使用。")
                else:
                    st.session_state['ml_predictor'] = None
                    st.session_state['model_type'] = "frequency"
                
            except ImportError as e:
                error_msg = str(e)
                if "scikit-learn" in error_msg.lower() or "sklearn" in error_msg.lower():
                    st.error("❌ 缺少 scikit-learn 库（LightGBM/XGBoost 需要）")
                    st.code("pip install scikit-learn", language="bash")
                else:
                    st.error(f"❌ 缺少依赖库：{e}")
                    st.code("pip install lightgbm xgboost scikit-learn", language="bash")
            except Exception as e:
                st.error(f"训练失败：{e}")
                import traceback
                with st.expander("查看详细错误信息"):
                    st.code(traceback.format_exc())
    
    with ai_tabs[1]:
        st.subheader("自动参数优化")
        st.write("使用遗传算法自动寻找最优策略参数，最大化ROI和高奖命中率")
        
        opt_method = st.selectbox("优化方法", ["遗传算法", "随机搜索"], key="ai_opt_method")
        opt_generations = st.number_input("迭代代数", 5, 50, 10, key="ai_opt_generations")
        opt_population = st.number_input("种群大小", 10, 50, 20, key="ai_opt_population")
        opt_backtest_n = st.number_input("优化回测期数", 10, 100, 30, key="ai_opt_backtest_n")
        
        # 目标函数选择
        opt_target = st.radio(
            "优化目标",
            ["最大化ROI", "最大化高奖命中率（一等奖+二等奖）", "平衡ROI和高奖"],
            key="ai_opt_target"
        )
        
        if st.button("开始优化", key="ai_opt_btn"):
            try:
                from backend.optimizer import genetic_algorithm_optimize, StrategyParams
                import time
                
                # 优化参数设置
                opt_ticket_cost = st.session_state.get('tab4_ticket_cost', 3.0)
                opt_pred_count = st.session_state.get('tab4_pred_count', 5)
                opt_prize_amounts = {}
                for name, default_val in PRIZE_PAYOUT_DEFAULT_AI.items():
                    opt_prize_amounts[name] = st.session_state.get(f'prize_amount_{name}', default_val)
                
                # 定义适应度函数
                def fitness_function(params: StrategyParams) -> float:
                    """计算策略参数的适应度分数"""
                    try:
                        # 使用参数进行回测
                        history_df = df_filtered.sort_values("date", ascending=False).head(opt_backtest_n).reset_index(drop=True)
                        
                        total_cost = 0.0
                        total_return = 0.0
                        high_prize_count = 0
                        total_bets = 0
                        
                        for idx, row in history_df.iterrows():
                            recent_window_dyn = build_history_window(
                                df_filtered,
                                recent_n=params.recent_n,
                                cutoff_date=row["date"]
                            )
                            
                            generation_context_dyn = prepare_generation_context(
                                df_window=recent_window_dyn,
                                span=params.span,
                                front_blocks_labels=front_labels,
                                back_blocks_labels=back_labels,
                                selected_front_blocks=front_labels,
                                selected_back_blocks=back_labels,
                            )
                            
                            exclude_front_dyn, exclude_back_dyn = compute_exclusions(
                                generation_context_dyn["front_freq_map"],
                                generation_context_dyn["back_freq_map"],
                                params.exclude_top_n,
                                params.exclude_front_n,
                                params.exclude_back_n
                            )
                            
                            rules_opt = assemble_rules(
                                base_rules=predict_rules_ai,
                                min_consec=params.min_consec,
                                min_odd=params.min_odd,
                                exclude_front=exclude_front_dyn,
                                exclude_back=exclude_back_dyn,
                                top_n_blocks=params.top_n_blocks,
                                max_per_block=params.max_per_block,
                                random_blocks_count=params.random_blocks_count,
                                random_back_blocks_count=params.random_back_blocks_count
                            )
                            
                            gen = genmod.gen_numbers(
                                count=opt_pred_count,
                                rules=rules_opt,
                                front_blocks=generation_context_dyn["front_blocks"],
                                back_blocks=generation_context_dyn["back_blocks"],
                                front_weights=generation_context_dyn["front_weights"],
                                back_weights=generation_context_dyn["back_weights"],
                                selected_front_blocks=front_labels,
                                selected_back_blocks=back_labels
                            )
                            
                            # 计算收益
                            for c in gen:
                                prize_name = check_prize_ai(
                                    c["front"], c["back"],
                                    row[["f1", "f2", "f3", "f4", "f5"]],
                                    row[["b1", "b2"]]
                                )
                                total_cost += opt_ticket_cost
                                total_return += opt_prize_amounts.get(prize_name, 0.0)
                                if prize_name in ["一等奖", "二等奖"]:
                                    high_prize_count += 1
                                total_bets += 1
                        
                        # 计算分数
                        roi = (total_return - total_cost) / total_cost if total_cost > 0 else -1.0
                        high_prize_rate = high_prize_count / total_bets if total_bets > 0 else 0.0
                        
                        if opt_target == "最大化ROI":
                            score = roi * 1000  # 放大以便优化
                        elif opt_target == "最大化高奖命中率（一等奖+二等奖）":
                            score = high_prize_rate * 10000
                        else:  # 平衡
                            score = (roi * 500) + (high_prize_rate * 5000)
                        
                        # 保本约束：如果ROI<0，大幅惩罚
                        if roi < 0:
                            score -= 10000
                        
                        return score
                    except Exception as e:
                        return float('-inf')
                
                # 参数范围
                param_ranges = {
                    "recent_n": (1, 20),
                    "span": (1, 5),
                    "top_n_blocks": (3, 7),
                    "max_per_block": (1, 3),
                    "random_blocks_count": (2, 5),
                    "random_back_blocks_count": (1, 3),
                    "min_consec": (0, 2),
                    "min_odd": (1, 4),
                }
                
                with st.spinner(f"优化中（{opt_generations}代，每代{opt_population}个个体）..."):
                    if opt_method == "遗传算法":
                        best_params, best_score = genetic_algorithm_optimize(
                            fitness_function,
                            param_ranges,
                            population_size=opt_population,
                            generations=opt_generations
                        )
                    else:
                        from backend.optimizer import bayesian_optimize
                        best_params, best_score = bayesian_optimize(
                            fitness_function,
                            param_ranges,
                            n_iterations=opt_generations * opt_population
                        )
                
                st.success("优化完成！")
                st.subheader("最优参数")
                st.json({
                    "recent_n": best_params.recent_n,
                    "span": best_params.span,
                    "top_n_blocks": best_params.top_n_blocks,
                    "max_per_block": best_params.max_per_block,
                    "random_blocks_count": best_params.random_blocks_count,
                    "random_back_blocks_count": best_params.random_back_blocks_count,
                    "min_consec": best_params.min_consec,
                    "min_odd": best_params.min_odd,
                    "exclude_top_n": best_params.exclude_top_n,
                    "exclude_front_n": best_params.exclude_front_n,
                    "exclude_back_n": best_params.exclude_back_n,
                    "适应度分数": f"{best_score:.2f}"
                })
                
                # 保存到session state
                st.session_state['optimized_params'] = best_params
                
            except Exception as e:
                st.error(f"优化失败：{e}")
                import traceback
                st.code(traceback.format_exc())
    
    with ai_tabs[2]:
        st.subheader("AI批量回测与可视化")
        st.write("使用AI模型进行批量回测，可视化回测结果和生成号码")
        
        # 检查AI模型
        ml_predictor_bt = st.session_state.get('ml_predictor', None)
        if ml_predictor_bt is None:
            st.warning("⚠️ 请先在'模型训练'标签页训练AI模型")
        else:
            st.success(f"✅ 已加载AI模型：{st.session_state.get('model_type', 'unknown')}")
        
        # 批量回测参数
        batch_backtest_cols = st.columns(3)
        batch_test_n = batch_backtest_cols[0].number_input("回测期数", 10, 200, 50, key="batch_test_n")
        batch_pred_count = batch_backtest_cols[1].number_input("每期注数", 1, 20, 5, key="batch_pred_count")
        batch_use_ai = batch_backtest_cols[2].checkbox("使用AI模型", value=True, key="batch_use_ai")
        
        # 批量回测参数设置（从session state或使用默认值）
        batch_params_cols = st.columns(4)
        batch_recent_n = batch_params_cols[0].number_input("权重最近N期", 0, 1000, 
                                                           st.session_state.get('tab4_recent_n', 3), key="batch_recent_n")
        batch_span = batch_params_cols[1].number_input("EWMA span", 1, 5, 
                                                       st.session_state.get('tab4_span', 1), key="batch_span")
        batch_min_consec = batch_params_cols[2].number_input("前区最小连号", 0, 5, 
                                                            st.session_state.get('tab4_min_consec', 0), key="batch_min_consec")
        batch_min_odd = batch_params_cols[3].number_input("前区最小奇数", 0, 5, 
                                                          st.session_state.get('tab4_min_odd', 2), key="batch_min_odd")
        
        batch_adv_cols = st.columns(3)
        batch_top_n_blocks = batch_adv_cols[0].number_input("前区仅用前N区块", 3, 7, 
                                                             st.session_state.get('tab4_top_n_blocks', 5), key="batch_top_n_blocks")
        batch_max_per_block = batch_adv_cols[1].number_input("每区块最多取", 1, 5, 
                                                             st.session_state.get('tab4_max_per_block', 2), key="batch_max_per_block")
        batch_random_blocks_count = batch_adv_cols[2].number_input("每期随机区块数", 1, 7, 
                                                                    st.session_state.get('tab4_random_blocks_count', 4), key="batch_random_blocks_count")
        
        batch_back_cols = st.columns(1)
        batch_random_back_blocks_count = batch_back_cols[0].number_input(
            "后区随机区块数", 1, len(back_labels), 
            st.session_state.get('tab4_random_back_blocks_count', 1), key="batch_random_back_blocks_count"
        )
        
        batch_exclude_cols = st.columns(3)
        batch_exclude_top_n = batch_exclude_cols[0].checkbox("排除近期高频号码", 
                                                             st.session_state.get('tab4_exclude_top_n', False), key="batch_exclude_top_n")
        batch_exclude_front_n = batch_exclude_cols[1].number_input("前区排除数量", 0, 10, 
                                                                    st.session_state.get('tab4_exclude_front_n', 3), key="batch_exclude_front_n")
        batch_exclude_back_n = batch_exclude_cols[2].number_input("后区排除数量", 0, 5, 
                                                                  st.session_state.get('tab4_exclude_back_n', 2), key="batch_exclude_back_n")
        
        # 获取奖金参数
        PRIZE_PAYOUT_DEFAULT_BT = {
            "一等奖": 5000000.0, "二等奖": 1500000.0, "三等奖": 10000.0,
            "四等奖": 3000.0, "五等奖": 300.0, "六等奖": 200.0,
            "七等奖": 100.0, "八等奖": 15.0, "九等奖": 5.0
        }
        batch_prize_amounts = {}
        for name, default_val in PRIZE_PAYOUT_DEFAULT_BT.items():
            batch_prize_amounts[name] = st.session_state.get(f'prize_amount_{name}', default_val)
        batch_ticket_cost = st.session_state.get('tab4_ticket_cost', 3.0)
        
        if st.button("开始AI批量回测", key="ai_batch_backtest_btn"):
            if ml_predictor_bt is None and batch_use_ai:
                st.error("未检测到AI模型，请先训练模型或取消'使用AI模型'选项")
            else:
                try:
                    with st.spinner("正在进行AI批量回测..."):
                        history_df_bt = df_filtered.sort_values("date", ascending=False).head(batch_test_n).reset_index(drop=True)
                        
                        backtest_results = []
                        cumulative_cost = []
                        cumulative_return = []
                        cumulative_roi = []
                        prize_counts = {name: 0 for name in ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "七等奖", "八等奖", "九等奖", "未中奖"]}
                        
                        # 定义check_prize函数
                        PRIZE_RULES_BT = [
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
                        
                        def check_prize_bt(fc, bc, win_fc, win_bc):
                            fc_match = len(set(fc)&set(win_fc))
                            bc_match = len(set(bc)&set(win_bc))
                            for name, cond in PRIZE_RULES_BT:
                                if cond(fc_match, bc_match):
                                    return name
                            return "未中奖"
                        
                        predict_rules_bt = {
                            "sum_front_range": [0, 999],
                            "odd_even_front": [0, 5],
                            "front_include": [],
                            "front_exclude": [],
                            "back_include": [],
                            "back_exclude": [],
                            "consecutive_count": 0,
                            "consecutive_mode": "min"
                        }
                        
                        for idx, row in history_df_bt.iterrows():
                            # 构建历史窗口
                            recent_window_bt = build_history_window(
                                df_filtered,
                                recent_n=batch_recent_n,
                                cutoff_date=row["date"]
                            )
                            
                            # 使用AI模型或EWMA计算权重
                            if batch_use_ai and ml_predictor_bt is not None:
                                try:
                                    front_ml_weights_bt, back_ml_weights_bt = ml_predictor_bt.predict_number_weights(
                                        recent_window_bt, front_range=range(1,36), back_range=range(1,13)
                                    )
                                    # 转换为区块权重
                                    front_block_weights_bt = {}
                                    back_block_weights_bt = {}
                                    for label, (lo, hi) in zip(front_labels, front_bins):
                                        block_nums = list(range(lo, hi+1))
                                        avg_weight = np.mean([front_ml_weights_bt.get(n, 0) for n in block_nums])
                                        front_block_weights_bt[label] = float(avg_weight * 10 + 0.5)
                                    for label, (lo, hi) in zip(back_labels, back_bins):
                                        block_nums = list(range(lo, hi+1))
                                        avg_weight = np.mean([back_ml_weights_bt.get(n, 0) for n in block_nums])
                                        back_block_weights_bt[label] = float(avg_weight * 10 + 0.5)
                                    
                                    from predictor import normalize_block_weights
                                    front_block_weights_bt = normalize_block_weights(front_block_weights_bt, 0.2, 1.5)
                                    back_block_weights_bt = normalize_block_weights(back_block_weights_bt, 0.2, 1.5)
                                    
                                    generation_context_bt = prepare_generation_context(
                                        df_window=recent_window_bt,
                                        span=batch_span,
                                        front_blocks_labels=front_labels,
                                        back_blocks_labels=back_labels,
                                        selected_front_blocks=front_labels,
                                        selected_back_blocks=back_labels,
                                    )
                                    generation_context_bt["front_weights"] = front_block_weights_bt
                                    generation_context_bt["back_weights"] = back_block_weights_bt
                                except Exception as e:
                                    st.warning(f"第{idx+1}期AI预测失败，回退到EWMA: {e}")
                                    generation_context_bt = prepare_generation_context(
                                        df_window=recent_window_bt,
                                        span=batch_span,
                                        front_blocks_labels=front_labels,
                                        back_blocks_labels=back_labels,
                                        selected_front_blocks=front_labels,
                                        selected_back_blocks=back_labels,
                                    )
                            else:
                                generation_context_bt = prepare_generation_context(
                                    df_window=recent_window_bt,
                                    span=batch_span,
                                    front_blocks_labels=front_labels,
                                    back_blocks_labels=back_labels,
                                    selected_front_blocks=front_labels,
                                    selected_back_blocks=back_labels,
                                )
                            
                            # 排除高频号码
                            exclude_front_bt, exclude_back_bt = compute_exclusions(
                                generation_context_bt["front_freq_map"],
                                generation_context_bt["back_freq_map"],
                                batch_exclude_top_n,
                                batch_exclude_front_n,
                                batch_exclude_back_n
                            )
                            
                            # 生成号码
                            rules_bt = assemble_rules(
                                base_rules=predict_rules_bt,
                                min_consec=batch_min_consec,
                                min_odd=batch_min_odd,
                                exclude_front=exclude_front_bt,
                                exclude_back=exclude_back_bt,
                                top_n_blocks=batch_top_n_blocks,
                                max_per_block=batch_max_per_block,
                                random_blocks_count=batch_random_blocks_count,
                                random_back_blocks_count=batch_random_back_blocks_count
                            )
                            
                            gen_bt = genmod.gen_numbers(
                                count=batch_pred_count,
                                rules=rules_bt,
                                front_blocks=generation_context_bt["front_blocks"],
                                back_blocks=generation_context_bt["back_blocks"],
                                front_weights=generation_context_bt["front_weights"],
                                back_weights=generation_context_bt["back_weights"],
                                selected_front_blocks=front_labels,
                                selected_back_blocks=back_labels
                            )
                            
                            # 计算收益
                            period_cost = len(gen_bt) * batch_ticket_cost
                            period_return = 0.0
                            period_prizes = []
                            
                            for c in gen_bt:
                                prize_name = check_prize_bt(
                                    c["front"], c["back"],
                                    row[["f1", "f2", "f3", "f4", "f5"]].tolist(),
                                    row[["b1", "b2"]].tolist()
                                )
                                period_return += batch_prize_amounts.get(prize_name, 0.0)
                                period_prizes.append(prize_name)
                                prize_counts[prize_name] = prize_counts.get(prize_name, 0) + 1
                            
                            period_roi = (period_return - period_cost) / period_cost if period_cost > 0 else 0.0
                            
                            cumulative_cost.append(period_cost + (cumulative_cost[-1] if cumulative_cost else 0))
                            cumulative_return.append(period_return + (cumulative_return[-1] if cumulative_return else 0))
                            current_roi = (cumulative_return[-1] - cumulative_cost[-1]) / cumulative_cost[-1] if cumulative_cost[-1] > 0 else 0.0
                            cumulative_roi.append(current_roi)
                            
                            backtest_results.append({
                                "期号": row["issue"],
                                "日期": row["date"],
                                "投入": period_cost,
                                "回收": period_return,
                                "收益": period_return - period_cost,
                                "ROI": period_roi,
                                "累计ROI": current_roi,
                                "中奖情况": ", ".join(period_prizes[:3]) + ("..." if len(period_prizes) > 3 else "")
                            })
                        
                        # 可视化结果
                        st.success(f"✅ 批量回测完成！共 {len(backtest_results)} 期")
                        
                        # 汇总统计
                        total_cost_bt = cumulative_cost[-1] if cumulative_cost else 0
                        total_return_bt = cumulative_return[-1] if cumulative_return else 0
                        final_roi_bt = cumulative_roi[-1] if cumulative_roi else 0
                        
                        summary_cols = st.columns(4)
                        summary_cols[0].metric("总投入", f"{total_cost_bt:.0f} 元")
                        summary_cols[1].metric("总回收", f"{total_return_bt:.0f} 元")
                        summary_cols[2].metric("净收益", f"{total_return_bt - total_cost_bt:.0f} 元", 
                                               delta=f"{final_roi_bt*100:.1f}%")
                        summary_cols[3].metric("总注数", f"{len(backtest_results) * batch_pred_count} 注")
                        
                        # ROI趋势图
                        st.subheader("📈 ROI趋势图")
                        roi_df = pd.DataFrame({
                            "期数": range(1, len(cumulative_roi) + 1),
                            "累计ROI": cumulative_roi
                        })
                        fig_roi = px.line(roi_df, x="期数", y="累计ROI", 
                                         title="累计ROI变化趋势",
                                         labels={"累计ROI": "累计ROI (%)"})
                        fig_roi.add_hline(y=0, line_dash="dash", line_color="red", 
                                         annotation_text="保本线")
                        st.plotly_chart(fig_roi, use_container_width=True)
                        
                        # 收益趋势图
                        st.subheader("💰 收益趋势图")
                        profit_df = pd.DataFrame({
                            "期数": range(1, len(cumulative_cost) + 1),
                            "累计投入": cumulative_cost,
                            "累计回收": cumulative_return
                        })
                        fig_profit = px.line(profit_df, x="期数", y=["累计投入", "累计回收"],
                                            title="累计投入与回收对比",
                                            labels={"value": "金额（元）", "variable": "类型"})
                        st.plotly_chart(fig_profit, use_container_width=True)
                        
                        # 中奖分布饼图
                        st.subheader("🎯 中奖分布")
                        prize_df = pd.DataFrame({
                            "奖项": list(prize_counts.keys()),
                            "次数": list(prize_counts.values())
                        })
                        prize_df = prize_df[prize_df["次数"] > 0]
                        if len(prize_df) > 0:
                            fig_prize = px.pie(prize_df, values="次数", names="奖项", 
                                             title="中奖分布统计")
                            st.plotly_chart(fig_prize, use_container_width=True)
                        
                        # 详细结果表
                        st.subheader("📊 详细回测结果")
                        results_df = pd.DataFrame(backtest_results)
                        st.dataframe(results_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"批量回测失败：{e}")
                    import traceback
                    with st.expander("查看详细错误"):
                        st.code(traceback.format_exc())

