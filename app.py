# app.py  — v1.6 (在你 v1.5 基础上增加 EMA, correlation constraint, ML models, ensemble)
import streamlit as st
import pandas as pd
import plotly.express as px
from backend.db import init_db, session_scope, Draw
from backend.sync import import_csv
from backend.analysis import dataframe_from_draws
from backend import generator as genmod
from predictor import compute_weights_from_history, prepare_generator_inputs
import random
import io
import numpy as np
from collections import defaultdict

# try optional ML libs
have_sklearn = True
have_lightgbm = True
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except Exception:
    have_sklearn = False

try:
    import lightgbm as lgb
except Exception:
    have_lightgbm = False

st.set_page_config(page_title="大乐透分析与选号 v1.6", page_icon="🎯", layout="wide")
st.title("🎯 大乐透分析与选号（本地版） v1.6 — EMA + Correlation + ML Ensemble")

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

# --------------------- Helper functions: EMA, correlation, ML dataset ---------------------
def compute_ema_freq(df, span_front=10, span_back=6, front_range=range(1,36), back_range=range(1,13), recent_n=0):
    """
    计算每个号码的 EMA（指数加权平均）出号频率——越近的期权重大
    recent_n: 若>0只取最近N期
    返回：front_ema (dict num->score), back_ema (dict)
    """
    df_use = df.sort_values("date", ascending=True)
    if recent_n and recent_n>0:
        df_use = df_use.tail(recent_n)
    # create time series per number: 1 if appears in draw, else 0
    front_series = {n: [] for n in front_range}
    back_series = {n: [] for n in back_range}
    for _, row in df_use.iterrows():
        fronts = {int(row[f'f{i}']) for i in range(1,6)}
        backs = {int(row[f'b{i}']) for i in range(1,3)}
        for n in front_range:
            front_series[n].append(1 if n in fronts else 0)
        for n in back_range:
            back_series[n].append(1 if n in backs else 0)

    # compute EMA via pandas (alpha from span)
    front_ema = {}
    back_ema = {}
    for n, seq in front_series.items():
        if len(seq) == 0:
            front_ema[n] = 0.0
        else:
            s = pd.Series(seq)
            front_ema[n] = float(s.ewm(span=span_front, adjust=False).mean().iloc[-1])
    for n, seq in back_series.items():
        if len(seq) == 0:
            back_ema[n] = 0.0
        else:
            s = pd.Series(seq)
            back_ema[n] = float(s.ewm(span=span_back, adjust=False).mean().iloc[-1])
    return front_ema, back_ema

def compute_cooccurrence_matrix(df, front_range=range(1,36)):
    """
    计算前区号码共现（关联）矩阵（对称），归一化到 [0,1]
    """
    index = list(front_range)
    mat = pd.DataFrame(0, index=index, columns=index, dtype=float)
    for _, row in df.iterrows():
        nums = [int(row[f'f{i}']) for i in range(1,6)]
        for i in nums:
            for j in nums:
                if i != j:
                    mat.loc[i,j] += 1
    if mat.values.max() > 0:
        mat = mat / mat.values.max()
    return mat

def build_ml_dataset(df, history_window=30):
    """
    为每个 draw 创建按号的训练样本：
    对于每一期 t, 对每个号码 n:
        features: recent frequency (counts) over windows, last_gap (periods since last occurrence), EMA value
        target: 1 if number n appears in draw t, else 0
    返回 X (DataFrame), y (Series), index mapping to (issue, number)
    注意：dataset 用于训练 per-number binary classifier
    """
    records = []
    idx_map = []
    df_sorted = df.sort_values("date", ascending=True).reset_index(drop=True)
    total = len(df_sorted)
    # precompute positions
    pos_history = {n: [] for n in range(1,36)}
    for idx, row in df_sorted.iterrows():
        for n in range(1,36):
            pos_history[n].append(1 if n in [int(row[f'f{i}']) for i in range(1,6)] else 0)

    for t in range(len(df_sorted)):
        # compute features based on previous `history_window` periods
        start = max(0, t - history_window)
        window_df = df_sorted.iloc[start:t]  # earlier draws only
        if window_df.empty:
            continue
        # for each number build features
        # compute EMA across window with small span
        for n in range(1,36):
            # freq in window
            freq = window_df[[f"f{i}" for i in range(1,6)]].apply(lambda row: int(n in set(row)), axis=1).sum()
            # last gap: periods since last occurrence before t
            last_idx = None
            # search backward
            for k in range(t-1, start-1, -1):
                rowk = df_sorted.iloc[k]
                if n in [int(rowk[f'f{i}']) for i in range(1,6)]:
                    last_idx = k
                    break
            last_gap = (t - last_idx) if last_idx is not None else history_window + 1
            # EMA feature: compute simple weighted freq (recent heavier)
            # use decay 0.95 per step back
            decays = [0.95 ** (t - 1 - k) for k in range(start, t)]
            presence = window_df[[f"f{i}" for i in range(1,6)]].apply(lambda row: int(n in set(row)), axis=1).values
            ema = float(np.dot(presence, decays) / (sum(decays) + 1e-9)) if len(decays)>0 else 0.0
            # target: whether n appears at t
            rowt = df_sorted.iloc[t]
            target = 1 if n in [int(rowt[f'f{i}']) for i in range(1,6)] else 0
            records.append({
                "freq_window": freq,
                "last_gap": last_gap,
                "ema": ema,
                "history_len": len(decays),
                "number": n
            })
            idx_map.append((df_sorted.iloc[t]["issue"], n, t))
    if not records:
        return None, None, None
    X = pd.DataFrame(records)
    y = []
    # recompute targets properly (we appended target conceptually but didn't store) -> recompute from idx_map
    y = []
    df_sorted = df_sorted.reset_index(drop=True)
    for (issue, n, t) in idx_map:
        rowt = df_sorted.iloc[t]
        y.append(1 if n in [int(rowt[f'f{i}']) for i in range(1,6)] else 0)
    return X, pd.Series(y), idx_map

def train_ml_model(X, y, use_lightgbm=False):
    """
    训练 ML 模型并返回预测器（fit done here）
    支持 LightGBM 或 RandomForest
    """
    if X is None or y is None or len(y)==0:
        return None
    # simple split
    if use_lightgbm and have_lightgbm:
        dtrain = lgb.Dataset(X, label=y)
        params = {"objective":"binary", "metric":"binary_logloss", "verbosity":-1}
        model = lgb.train(params, dtrain, num_boost_round=100)
        return ("lgb", model)
    if have_sklearn:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return ("rf", clf)
    return None

def ml_predict_prob(model_tuple, X_pred):
    if model_tuple is None:
        return None
    kind, model = model_tuple
    if kind == "lgb":
        prob = model.predict(X_pred)
        return prob
    elif kind == "rf":
        prob = model.predict_proba(X_pred)[:,1]
        return prob
    return None

# ----------------- Sampling / ensemble generate -----------------
def ensemble_predict_probs(df_history, recent_n_for_ema=100, ml_window=30, use_lightgbm=False):
    """
    给出每个号码的预测概率（前区1..35，后区1..12）
    组合：EMA + ML (if available)
    返回 dicts front_prob, back_prob
    """
    # EMA baseline
    front_ema, back_ema = compute_ema_freq(df_history, span_front=10, span_back=6, recent_n=recent_n_for_ema)
    # normalize to 0..1
    def normalize_map(m):
        vals = np.array(list(m.values()), dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            return {k:1.0 for k in m}
        return {k:(v - vmin) / (vmax - vmin) for k,v in m.items()}
    front_ema_norm = normalize_map(front_ema)
    back_ema_norm = normalize_map(back_ema)

    # ML model training on history (if possible)
    X, y, idx_map = build_ml_dataset(df_history, history_window=ml_window)
    ml_model = None
    ml_front_prob = None
    if X is not None and have_sklearn:
        # prepare X features aggregated by (t,number) entries -> we will predict for latest t
        # train rf on X, y
        try:
            ml_model = train_ml_model(X, y, use_lightgbm=use_lightgbm and have_lightgbm)
            # prepare latest prediction rows: take last time t = len(df_history)-1
            # we need to reconstruct X_pred for latest t (we created records in build_ml_dataset in order)
            # Simple approach: use last history_window slice to compute features for each number
            latest_t = len(df_history)-1
            start = max(0, latest_t - ml_window)
            window_df = df_history.iloc[start:latest_t]
            records = []
            for n in range(1,36):
                freq = window_df[[f"f{i}" for i in range(1,6)]].apply(lambda row: int(n in set(row)), axis=1).sum() if not window_df.empty else 0
                last_idx = None
                for k in range(latest_t-1, start-1, -1):
                    rowk = df_history.iloc[k]
                    if n in [int(rowk[f'f{i}']) for i in range(1,6)]:
                        last_idx = k
                        break
                last_gap = (latest_t - last_idx) if last_idx is not None else ml_window + 1
                decays = [0.95 ** (latest_t - 1 - k) for k in range(start, latest_t)]
                presence = window_df[[f"f{i}" for i in range(1,6)]].apply(lambda row: int(n in set(row)), axis=1).values if not window_df.empty else np.array([])
                ema = float(np.dot(presence, decays) / (sum(decays) + 1e-9)) if len(decays)>0 else 0.0
                records.append({"freq_window":freq, "last_gap":last_gap, "ema":ema, "history_len":len(decays), "number":n})
            X_pred_front = pd.DataFrame(records)
            ml_probs = ml_predict_prob(ml_model, X_pred_front)
            # normalize ml_probs
            ml_probs = np.nan_to_num(ml_probs)
            if ml_probs.max() - ml_probs.min() > 1e-9:
                ml_probs = (ml_probs - ml_probs.min()) / (ml_probs.max() - ml_probs.min())
            ml_front_prob = {n+1: float(ml_probs[n]) for n in range(len(ml_probs))}
        except Exception as e:
            st.warning(f"ML training/prediction failed: {e}")
            ml_front_prob = None
    else:
        if not have_sklearn:
            st.info("未检测到 sklearn，ML 模型将被跳过（仍会使用 EMA + 相关性约束）。")

    # back prob: we don't train ML for back in this version (could be added similarly)
    # ensemble: combine EMA and ML (if present)
    front_prob = {}
    for n in range(1,36):
        p_ema = front_ema_norm.get(n, 0.0)
        p_ml = ml_front_prob.get(n, 0.0) if ml_front_prob is not None else 0.0
        # weight between EMA and ML: give ML slightly higher if present
        w_ml = 0.6 if ml_front_prob is not None else 0.0
        w_ema = 1.0 - w_ml
        front_prob[n] = float(w_ema * p_ema + w_ml * p_ml)
    # normalize again
    s = sum(front_prob.values()) + 1e-9
    for k in front_prob:
        front_prob[k] = front_prob[k] / s

    # back prob: normalize EMA
    s2 = sum(back_ema_norm.values()) + 1e-9
    back_prob = {k: back_ema_norm[k]/s2 for k in back_ema_norm}

    return front_prob, back_prob, ml_front_prob, front_ema_norm

def sample_front_by_probs(front_prob, co_matrix, k=5, sum_range=None, min_odd=0, min_consec=0, max_tries=500, corr_penalty=0.6):
    """
    按概率从 front numbers 中采样 k 个数（不放回），并使用 cooccurrence 矩阵惩罚高相关的选择。
    corr_penalty: 0..1 参数，越大更强烈惩罚相关数对。
    sum_range: (min_sum, max_sum) 若给出，则尝试满足和值约束
    返回选出的排序列表 或 [] 表示失败
    """
    nums = list(front_prob.keys())
    probs = np.array([front_prob[n] for n in nums], dtype=float)
    if probs.sum() == 0:
        probs = np.ones_like(probs)
    probs = probs / probs.sum()
    for _ in range(max_tries):
        selected = []
        probs_curr = probs.copy()
        nums_curr = nums.copy()
        while len(selected) < k and len(nums_curr) > 0:
            # sample one according to probs_curr
            idx = np.random.choice(len(nums_curr), p=probs_curr / (probs_curr.sum()+1e-12))
            pick = nums_curr.pop(idx)
            selected.append(pick)
            # adjust remaining probabilities by penalizing those highly co-occurring with pick
            if nums_curr:
                co_scores = np.array([co_matrix.loc[pick, n] if (pick in co_matrix.index and n in co_matrix.columns) else 0.0 for n in nums_curr])
                # reduced weight = orig * (1 - corr_penalty*co_score)
                orig = probs_curr[np.arange(len(probs_curr)) != idx]
                # but we removed one element; rebuild probs_curr correctly:
                probs_new = []
                for j, n in enumerate(nums_curr):
                    orig_p = front_prob[n]
                    penalty = 1.0 - corr_penalty * co_matrix.loc[pick, n] if (pick in co_matrix.index and n in co_matrix.columns) else 1.0
                    probs_new.append(max(0.0, orig_p * penalty))
                if sum(probs_new) > 0:
                    # update probs_curr aligned with nums_curr
                    probs_curr = np.array(probs_new, dtype=float)
                else:
                    # fallback: uniform
                    probs_curr = np.ones(len(nums_curr), dtype=float)
        if len(selected) != k:
            continue
        selected_sorted = sorted(selected)
        # check odd and consec
        if sum(1 for x in selected_sorted if x%2==1) < min_odd:
            continue
        consec_pairs = sum(1 for i in range(k-1) if selected_sorted[i+1] == selected_sorted[i] + 1)
        if consec_pairs < min_consec:
            continue
        if sum_range:
            s = sum(selected_sorted)
            if not (sum_range[0] <= s <= sum_range[1]):
                continue
        return selected_sorted
    return []

def ensemble_generate(df_history, pred_count=10, recent_n_for_ema=100, ml_window=30,
                      min_odd=0, min_consec=0, sum_range=None, exclude_front=None, exclude_back=None):
    """
    使用 ensemble 概率 + correlation constraint 生成 pred_count 注号码
    exclude_front/back: list of numbers to exclude
    """
    df_hist = df_history.copy()
    front_prob, back_prob, ml_front_prob, front_ema_norm = ensemble_predict_probs(df_hist, recent_n_for_ema, ml_window)
    # remove excluded
    if exclude_front:
        for n in exclude_front:
            if n in front_prob: front_prob[n] = 0.0
    if exclude_back:
        for n in exclude_back:
            if n in back_prob: back_prob[n] = 0.0
    # normalize after exclusion
    s = sum(front_prob.values()) + 1e-9
    for k in front_prob: front_prob[k] = front_prob[k]/s
    s2 = sum(back_prob.values()) + 1e-9
    for k in back_prob: back_prob[k] = back_prob[k]/s2

    co_matrix = compute_cooccurrence_matrix(df_hist)
    results = []
    used_front_pool = set()
    used_back_pool = set()
    for _ in range(pred_count):
        # sample front with correlation penalty
        selected_front = sample_front_by_probs(front_prob, co_matrix, k=5, sum_range=sum_range,
                                               min_odd=min_odd, min_consec=min_consec, corr_penalty=0.65)
        if not selected_front:
            # fallback: random sample from remaining
            pool = [n for n in range(1,36) if n not in (exclude_front or []) and n not in used_front_pool]
            if len(pool) < 5:
                pool = [n for n in range(1,36) if n not in (exclude_front or [])]
            selected_front = sorted(random.sample(pool, 5))
        # after picking front, reduce their probabilities so future picks are diverse (from pool removal)
        for n in selected_front:
            used_front_pool.add(n)
            front_prob[n] = front_prob.get(n,0.0) * 0.1
        # sample back: choose two with highest back_prob but avoid used ones
        back_candidates = sorted([(p, n) for n, p in back_prob.items() if n not in (exclude_back or []) and n not in used_back_pool], reverse=True)
        if len(back_candidates) < 2:
            back_candidates = sorted([(p,n) for n,p in back_prob.items()], reverse=True)
        selected_back = sorted([back_candidates[0][1], back_candidates[1][1]]) if len(back_candidates)>=2 else sorted(random.sample(range(1,13),2))
        for n in selected_back:
            used_back_pool.add(n)
            back_prob[n] = back_prob.get(n,0.0) * 0.1

        results.append({"front": selected_front, "back": selected_back})
    return results

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

# --------------------- Tab3: 号码生成 (保持原样) ---------------------
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
    colA,colB,colC = st.columns(3)
    with colA:
        sum_min = st.number_input("前区和值最小",0,200,70)
        sum_max = st.number_input("前区和值最大",0,200,140)
        odd_count = st.number_input("前区最小奇数个数",0,5,3)
    with colB:
        front_include = st.text_input("前区必含(逗号分隔)","")
        front_exclude = st.text_input("前区排除(逗号分隔)","")
        consecutive_count = st.number_input("前区连号数量(对数)",0,5,0)
        cons_mode_label = st.selectbox("连号匹配方式",["等于","至少"])
        consecutive_mode = "exact" if cons_mode_label=="等于" else "min"
    with colC:
        back_include = st.text_input("后区必含(逗号分隔)","")
        back_exclude = st.text_input("后区排除(逗号分隔)","")

    max_gen = st.number_input("生成注数上限",1,100,5)

    def parse_nums(s: str):
        s = s.replace("，",",")
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]

    rules = {
        "sum_front_range":[sum_min,sum_max],
        "odd_even_front":[odd_count,5-odd_count],
        "front_include":parse_nums(front_include),
        "front_exclude":parse_nums(front_exclude),
        "back_include":parse_nums(back_include),
        "back_exclude":parse_nums(back_exclude),
        "consecutive_count":consecutive_count,
        "consecutive_mode":consecutive_mode
    }

    st.subheader("🎯 中奖号码比对")
    win_front_input = st.text_input("中奖前区号码（逗号分隔）","")
    win_back_input = st.text_input("中奖后区号码（逗号分隔)","")

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
        if not cands:
            st.warning("未生成到符合条件的号码。")
        else:
            for i,cd in enumerate(cands,1):
                prize = check_prize(cd['front'],cd['back'],win_front,win_back)
                st.markdown(f"**第{i}注：前区 {cd['front']} | 后区 {cd['back']} => {prize}**")


# --------------------- Tab4: 未来号码预测 + 历史回测（增强版） ---------------------
with tab_predict:
    st.header("🔮 基于 EMA + 相关性约束 + ML Ensemble 的未来预测 & 回测")
    st.markdown("""
    **本页功能（v1.6）**：
    - EMA（指数加权平均）作为基线概率
    - 训练 ML 模型（RandomForest / LightGBM）预测每个号码下一期出现概率（可用时）
    - ensemble：EMA 与 ML 概率融合
    - 相关性约束（co-occurrence）惩罚高度相关号码，提高组合多样性
    - 和值匹配（基于历史均值±std）用于过滤
    - 回测：每期基于其之前窗口数据计算概率并生成多注进行比对
    """)

    col1, col2 = st.columns([2,1])
    with col1:
        recent_n = st.number_input("EMA / 统计使用最近 N 期（0=全部）", min_value=0, max_value=1000, value=100, key="ens_recent_n")
        ml_window = st.number_input("ML 训练窗口（history window）", min_value=5, max_value=200, value=30, key="ens_ml_window")
        pred_count = st.number_input("预测注数（每期）", min_value=1, max_value=30, value=10, key="ens_pred_count")
        min_odd = st.number_input("前区最小奇数个数", min_value=0, max_value=5, value=2, key="ens_min_odd")
        min_consec = st.number_input("前区最小连号数量", min_value=0, max_value=5, value=0, key="ens_min_consec")
        sum_float_pct = st.slider("和值浮动比例（基于历史均值±%）", 0.0, 0.5, 0.15, key="ens_sum_float")
        exclude_top_n = st.checkbox("排除前 N 期最热号码（前区/后区）", value=False, key="ens_exclude_hot")
        exclude_front_k = st.number_input("前区排除个数(top k)", min_value=0, max_value=10, value=3, key="ens_ex_front")
        exclude_back_k = st.number_input("后区排除个数(top k)", min_value=0, max_value=5, value=2, key="ens_ex_back")
        use_lgb = st.checkbox("优先使用 LightGBM（若可用）", value=False, key="ens_use_lgb")
        backtest_n = st.number_input("回测最近 N 期（0=不回测）", min_value=0, max_value=500, value=20, key="ens_backtest_n")
    with col2:
        st.markdown("""
        说明：
        - ML 模型需要 sklearn 或 lightgbm；若不可用，系统会退回仅使用 EMA + 相关性约束方法。
        - 回测会对每期使用该期之前的历史窗口训练/计算概率并生成多注来比对中奖结果。
        """)

    # compute hot numbers if needed
    front_freq_map = df_filtered[["f1","f2","f3","f4","f5"]].stack().value_counts().to_dict()
    back_freq_map = df_filtered[["b1","b2"]].stack().value_counts().to_dict()

    exclude_front = []
    exclude_back = []
    if exclude_top_n:
        top_front = sorted(front_freq_map.items(), key=lambda x: -x[1])[:exclude_front_k]
        top_back = sorted(back_freq_map.items(), key=lambda x: -x[1])[:exclude_back_k]
        exclude_front = [n for n,_ in top_front]
        exclude_back = [n for n,_ in top_back]

    # helper: compute historical mean sum for front/back for dynamic sum_range
    def compute_recent_sum_stats(df_hist, window):
        df_sort = df_hist.sort_values("date", ascending=False)
        use = df_sort.head(window) if window>0 else df_sort
        if use.empty:
            return (70, 3)  # defaults
        front_sums = use[["f1","f2","f3","f4","f5"]].sum(axis=1)
        back_sums = use[["b1","b2"]].sum(axis=1)
        return (front_sums.mean(), back_sums.mean(), front_sums.std(), back_sums.std())

    # ----------------- 生成未来预测号码（ensemble） -----------------
    if st.button("生成未来预测号码（Ensemble）", key="ens_gen_future"):
        # compute sum target range from recent data
        mean_f, mean_b, std_f, std_b = compute_recent_sum_stats(df_filtered, recent_n)
        f_min = max(5, int(mean_f * (1 - sum_float_pct)))
        f_max = int(mean_f * (1 + sum_float_pct))
        b_min = max(2, int(mean_b * (1 - sum_float_pct)))
        b_max = int(mean_b * (1 + sum_float_pct))
        sum_range = (f_min, f_max)

        cands = ensemble_generate(
            df_filtered,
            pred_count=pred_count,
            recent_n_for_ema=recent_n,
            ml_window=ml_window,
            min_odd=min_odd,
            min_consec=min_consec,
            sum_range=sum_range,
            exclude_front=exclude_front,
            exclude_back=exclude_back
        )
        if not cands:
            st.warning("未生成符合条件的号码（可能历史窗口/约束太紧）。")
        else:
            # display as columns per predicted set (每注一列)
            row = {}
            for i, c in enumerate(cands, 1):
                row[f"预测前区{i}"] = ",".join(map(str, c["front"]))
                row[f"预测后区{i}"] = ",".join(map(str, c["back"]))
                row[f"中奖情况{i}"] = "未比对"
            st.subheader(f"未来预测结果（共 {len(cands)} 注）")
            st.dataframe(pd.DataFrame([row]), use_container_width=True)

    # ----------------- 历史回测（逐期基于其之前窗口计算概率） -----------------
    if backtest_n > 0:
        st.subheader(f"历史回测（最近 {backtest_n} 期，每期 {pred_count} 注）")
        history_df = df_filtered.sort_values("date", ascending=False).head(backtest_n).reset_index(drop=True)

        # Prize rules (same as before)
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

        def check_prize_name(gen_f, gen_b, win_f, win_b):
            fc = len(set(gen_f)&set(win_f))
            bc = len(set(gen_b)&set(win_b))
            for name, cond in PRIZE_RULES:
                if cond(fc, bc):
                    return name
            return "未中奖"

        backtest_rows = []
        # We'll iterate from newest to older (history_df is latest first)
        for idx, row in history_df.iterrows():
            # For each historical draw, compute df_before = all draws before that draw
            # We should find the draw's position in df_filtered to get prior draws
            issue = row["issue"]
            # find index of this issue in full df_sorted by date ascending
            df_sorted = df_filtered.sort_values("date", ascending=True).reset_index(drop=True)
            pos = df_sorted.index[df_sorted["issue"] == issue].tolist()
            if not pos:
                # fallback: skip
                continue
            t = pos[0]  # position in ascending list
            # prepare history window prior to t:
            start = max(0, t - max(recent_n, ml_window, 50))  # we choose max window to cover training & ema
            df_before = df_sorted.iloc[start:t]  # draws strictly before this draw
            if df_before.empty:
                # cannot build model -> fallback to simple random sample
                cands = []
                for _ in range(pred_count):
                    f = sorted(random.sample(range(1,36),5))
                    b = sorted(random.sample(range(1,13),2))
                    cands.append({"front":f,"back":b})
            else:
                # compute dynamic sum_range from df_before
                front_mean = df_before[["f1","f2","f3","f4","f5"]].sum(axis=1).mean()
                front_std = df_before[["f1","f2","f3","f4","f5"]].sum(axis=1).std()
                f_min = int(max(5, front_mean - front_std * (1 + sum_float_pct)))
                f_max = int(front_mean + front_std * (1 + sum_float_pct))
                sum_range = (f_min, f_max)
                # compute hot excludes for this window
                exclude_f = []
                exclude_b = []
                if exclude_top_n:
                    front_counts_win = df_before[["f1","f2","f3","f4","f5"]].stack().value_counts()
                    back_counts_win = df_before[["b1","b2"]].stack().value_counts()
                    exclude_f = front_counts_win.nlargest(exclude_front_k).index.tolist() if not front_counts_win.empty else []
                    exclude_b = back_counts_win.nlargest(exclude_back_k).index.tolist() if not back_counts_win.empty else []
                # generate using ensemble_generate with df_before as history
                cands = ensemble_generate(
                    df_before,
                    pred_count=pred_count,
                    recent_n_for_ema=recent_n,
                    ml_window=ml_window,
                    min_odd=min_odd,
                    min_consec=min_consec,
                    sum_range=sum_range,
                    exclude_front=exclude_f,
                    exclude_back=exclude_b
                )
            # compute prize names for each cand vs actual historical draw
            prizes = []
            for c in cands:
                name = check_prize_name(c["front"], c["back"], [row["f1"],row["f2"],row["f3"],row["f4"],row["f5"]], [row["b1"],row["b2"]])
                prizes.append(name)
            # prepare row output (one row per historical draw)
            row_out = {
                "期号": row["issue"],
                "历史前区": ",".join(map(str, [row["f1"],row["f2"],row["f3"],row["f4"],row["f5"]])),
                "历史后区": ",".join(map(str, [row["b1"],row["b2"]]))
            }
            for i, c in enumerate(cands, 1):
                row_out[f"预测前区{i}"] = ",".join(map(str, c["front"]))
                row_out[f"预测后区{i}"] = ",".join(map(str, c["back"]))
                row_out[f"中奖情况{i}"] = prizes[i-1]
            backtest_rows.append(row_out)

        if not backtest_rows:
            st.info("回测无结果（可能历史数据不足）。")
        else:
            backtest_df = pd.DataFrame(backtest_rows)
            prize_cols = [col for col in backtest_df.columns if "中奖情况" in col]
            # style mapping must be CSS 'attr: val;' strings
            def prize_style(v):
                return PRIZE_COLOR.get(v, "")
            st.dataframe(backtest_df.style.applymap(lambda v: prize_style(v), subset=prize_cols), use_container_width=True)

# ------------- END -------------
