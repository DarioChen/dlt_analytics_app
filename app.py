# app.py (v2.0)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
import math
import os
import shutil
import json
from datetime import datetime
from backend.db import init_db, session_scope, Draw
from backend.sync import import_csv, sync_remote_history
from backend.analysis import dataframe_from_draws
from backend import generator as genmod
from backend.optimizer import genetic_algorithm_optimize, bayesian_optimize, get_optimization_methods
from backend.backtest import BacktestAnalyzer
from backend.performance_test import PerformanceTester
from predictor import compute_weights_from_history, prepare_generator_inputs, compute_weights_from_history_ewma
import random
import io

st.set_page_config(page_title="大乐透分析与选号 v2.0", page_icon="🎯", layout="wide")
st.title("🎯 大乐透分析与选号（本地版） v2.0 — 增强优化与性能分析")

# 初始化回测分析器和性能测试器
backtest_analyzer = BacktestAnalyzer()
performance_tester = PerformanceTester()

# --------------------- 策略管理相关功能 ---------------------  
def ensure_strategies_dir():
    """确保策略存储目录存在"""
    strategies_dir = os.path.join(os.path.dirname(__file__), 'strategies')
    if not os.path.exists(strategies_dir):
        os.makedirs(strategies_dir)
    return strategies_dir


def get_saved_strategies():
    """获取所有保存的策略列表"""
    strategies_dir = ensure_strategies_dir()
    strategies = []
    for filename in os.listdir(strategies_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(strategies_dir, filename), 'r', encoding='utf-8') as f:
                    strategy = json.load(f)
                    strategy['filename'] = filename
                    strategies.append(strategy)
            except:
                continue
    return strategies


def save_strategy(name, params):
    """保存策略参数"""
    strategies_dir = ensure_strategies_dir()
    # 确保文件名安全
    safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
    filename = f"{safe_name}.json"
    
    # 构建策略数据
    strategy = {
        'name': name,
        'params': params,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    # 保存到文件
    with open(os.path.join(strategies_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2)
    
    return filename


def load_strategy(filename):
    """加载策略参数"""
    strategies_dir = ensure_strategies_dir()
    file_path = os.path.join(strategies_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            strategy = json.load(f)
            return strategy['params']
    return None


def delete_strategy(filename):
    """删除策略文件"""
    strategies_dir = ensure_strategies_dir()
    file_path = os.path.join(strategies_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

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
                   top_n_blocks, max_per_block, random_blocks_count, random_back_blocks_count,
                   consecutive_mode="exact", consecutive_check_type="groups"):
    rules = base_rules.copy()
    rules.update({
        "consecutive_count": min_consec,
        "consecutive_mode": consecutive_mode,
        "consecutive_check_type": consecutive_check_type,
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
    # 优化前区热力图：计算实际出现次数
    front_matrix = pd.DataFrame(0, index=df_filtered.index, columns=front_labels)
    for col in ["f1","f2","f3","f4","f5"]:
        for i,(lo,hi) in enumerate(front_bins):
            mask = df_filtered[col].between(lo, hi)
            front_matrix.loc[df_filtered.index[mask], front_labels[i]] += 1
    
    # 优化热力图样式和颜色映射
    fig_front = px.imshow(
        front_matrix.T, 
        labels=dict(x="期号", y="区块", color="出现次数"),
        color_continuous_scale="YlOrRd",  # 使用更直观的颜色映射
        title="前区号码区块出现频次热力图",
        text_auto=True,  # 显示具体数值
        aspect="auto"  # 自动调整宽高比
    )
    
    # 美化图表
    fig_front.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        coloraxis_colorbar=dict(
            title="出现次数",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside"
        )
    )
    
    # 优化坐标轴标签显示
    fig_front.update_xaxes(tickangle=45, tickfont_size=10)
    fig_front.update_yaxes(tickfont_size=12)
    
    st.plotly_chart(fig_front, use_container_width=True)

    st.subheader("后区落点热力图")
    if not df_filtered.empty:
        # 优化后区热力图：计算实际出现次数
        back_matrix = pd.DataFrame(0, index=df_filtered.index, columns=back_labels)
        for col in ["b1","b2"]:
            for i,(lo,hi) in enumerate(back_bins):
                mask = df_filtered[col].between(lo, hi)
                back_matrix.loc[df_filtered.index[mask], back_labels[i]] += 1
        
        # 优化热力图样式和颜色映射
        fig_back = px.imshow(
            back_matrix.T, 
            labels=dict(x="期号", y="区块", color="出现次数"),
            color_continuous_scale="YlGnBu",  # 使用不同的颜色映射以区分前后区
            title="后区号码区块出现频次热力图",
            text_auto=True,  # 显示具体数值
            aspect="auto"  # 自动调整宽高比
        )
        
        # 美化图表
        fig_back.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            coloraxis_colorbar=dict(
                title="出现次数",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="top", y=1,
                ticks="outside"
            )
        )
        
        # 优化坐标轴标签显示
        fig_back.update_xaxes(tickangle=45, tickfont_size=10)
        fig_back.update_yaxes(tickfont_size=12)
        
        st.plotly_chart(fig_back, use_container_width=True)
    else:
        st.warning("当前筛选条件下没有后区数据可供显示")

    # --------------------- 添加前区号码频率分布直方图 ---------------------
    st.subheader("前区号码频率分布直方图")
    # 计算前区每个号码的出现频率
    front_numbers = pd.concat([df_filtered['f1'], df_filtered['f2'], df_filtered['f3'], df_filtered['f4'], df_filtered['f5']])
    front_freq = front_numbers.value_counts().sort_index()
    
    # 创建前区号码频率直方图
    fig_front_freq = px.bar(
        x=front_freq.index, 
        y=front_freq.values,
        labels=dict(x="前区号码", y="出现次数"),
        title="前区号码出现频率分布",
        color=front_freq.values,
        color_continuous_scale="Viridis",
        text_auto=True
    )
    
    # 美化图表
    fig_front_freq.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickangle=0,
        xaxis=dict(tickmode='linear'),
        coloraxis_colorbar=dict(
            title="出现次数",
            thicknessmode="pixels", thickness=20
        )
    )
    
    st.plotly_chart(fig_front_freq, use_container_width=True)

    # --------------------- 遗漏号码分布解释 ---------------------
    with st.expander("❓ 什么是遗漏号码分布？"):
        st.markdown("""
        **遗漏号码分布**是指每个号码从上次出现到当前期号之间的间隔期数。
        
        - **遗漏值**：一个号码连续未出现的期数
        - **热号**：遗漏值小，近期频繁出现的号码
        - **冷号**：遗漏值大，长期未出现的号码
        - **温号**：遗漏值适中的号码
        
        通过分析遗漏号码分布，玩家可以了解号码的冷热程度，辅助选号决策。
        """)

    with st.expander("❓ 为什么21、32号码出现次数多？"):
        st.markdown("""
        从历史数据统计来看，21和32号出现次数较多，主要原因是：
        
        1. **随机分布的正常波动**：在大样本数据中，号码出现次数会有自然波动
        2. **统计概率的体现**：理论上每个号码出现概率相等，但实际中会有短期的集中现象
        3. **历史趋势影响**：部分号码在特定时期会形成"热号效应"
        
        需要注意的是，彩票号码是完全随机的，过去的热号并不能保证未来继续热门。
        """)

    # --------------------- 添加后区号码频率分布直方图 ---------------------
    st.subheader("后区号码频率分布直方图")
    # 计算后区每个号码的出现频率
    back_numbers = pd.concat([df_filtered['b1'], df_filtered['b2']])
    back_freq = back_numbers.value_counts().sort_index()
    
    # 创建后区号码频率直方图
    fig_back_freq = px.bar(
        x=back_freq.index, 
        y=back_freq.values,
        labels=dict(x="后区号码", y="出现次数"),
        title="后区号码出现频率分布",
        color=back_freq.values,
        color_continuous_scale="Cividis",
        text_auto=True
    )
    
    # 美化图表
    fig_back_freq.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickangle=0,
        xaxis=dict(tickmode='linear'),
        coloraxis_colorbar=dict(
            title="出现次数",
            thicknessmode="pixels", thickness=20
        )
    )
    
    st.plotly_chart(fig_back_freq, use_container_width=True)

    # --------------------- 添加前区号码遗漏值图表 ---------------------
    st.subheader("前区号码遗漏值图表")
    # 计算前区每个号码的遗漏值
    current_period = df_filtered.index.max()
    
    # 计算每个前区号码最后一次出现的期号
    front_last_occurrence = {}
    for num in range(1, 36):
        # 找出所有包含该号码的行
        mask = ((df_filtered['f1'] == num) | 
                (df_filtered['f2'] == num) | 
                (df_filtered['f3'] == num) | 
                (df_filtered['f4'] == num) | 
                (df_filtered['f5'] == num))
        
        if mask.any():
            # 获取最后一次出现的期号
            last_occurrence = df_filtered[mask].index.max()
            # 计算遗漏值 = 当前期号 - 最后一次出现的期号
            omission = current_period - last_occurrence
        else:
            # 从未出现过，遗漏值为当前期号
            omission = current_period
            
        front_last_occurrence[num] = omission
    
    # 转换为DataFrame
    front_omission_df = pd.DataFrame.from_dict(front_last_occurrence, orient='index', columns=['遗漏值'])
    front_omission_df = front_omission_df.sort_index()
    
    # 创建前区号码遗漏值图表
    fig_front_omission = px.bar(
        x=front_omission_df.index,
        y=front_omission_df['遗漏值'],
        labels=dict(x="前区号码", y="遗漏期数"),
        title="前区号码遗漏值分布",
        color=front_omission_df['遗漏值'],
        color_continuous_scale="RdYlBu_r",  # 反向颜色，遗漏值越大颜色越深
        text_auto=True
    )
    
    # 美化图表
    fig_front_omission.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickangle=0,
        xaxis=dict(tickmode='linear'),
        coloraxis_colorbar=dict(
            title="遗漏期数",
            thicknessmode="pixels", thickness=20
        )
    )
    
    st.plotly_chart(fig_front_omission, use_container_width=True)
    
    # --------------------- 添加后区号码遗漏值图表 ---------------------
    st.subheader("后区号码遗漏值图表")
    # 计算后区每个号码的遗漏值
    back_last_occurrence = {}
    for num in range(1, 13):
        # 找出所有包含该号码的行
        mask = ((df_filtered['b1'] == num) | 
                (df_filtered['b2'] == num))
        
        if mask.any():
            # 获取最后一次出现的期号
            last_occurrence = df_filtered[mask].index.max()
            # 计算遗漏值 = 当前期号 - 最后一次出现的期号
            omission = current_period - last_occurrence
        else:
            # 从未出现过，遗漏值为当前期号
            omission = current_period
            
        back_last_occurrence[num] = omission
    
    # 转换为DataFrame
    back_omission_df = pd.DataFrame.from_dict(back_last_occurrence, orient='index', columns=['遗漏值'])
    back_omission_df = back_omission_df.sort_index()
    
    # 创建后区号码遗漏值图表
    fig_back_omission = px.bar(
        x=back_omission_df.index,
        y=back_omission_df['遗漏值'],
        labels=dict(x="后区号码", y="遗漏期数"),
        title="后区号码遗漏值分布",
        color=back_omission_df['遗漏值'],
        color_continuous_scale="RdYlBu_r",  # 反向颜色，遗漏值越大颜色越深
        text_auto=True
    )
    
    # 美化图表
    fig_back_omission.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickangle=0,
        xaxis=dict(tickmode='linear'),
        coloraxis_colorbar=dict(
            title="遗漏期数",
            thicknessmode="pixels", thickness=20
        )
    )
    
    st.plotly_chart(fig_back_omission, use_container_width=True)

    # --------------------- 添加前区号码连号分析图表 ---------------------
    st.subheader("前区号码连号分析图表")
    
    # 计算每期的连号组数和最长连号长度
    def calculate_consecutive_groups(row):
        front_nums = sorted([row['f1'], row['f2'], row['f3'], row['f4'], row['f5']])
        consecutive_groups = []
        current_group = [front_nums[0]]
        
        for num in front_nums[1:]:
            if num == current_group[-1] + 1:
                current_group.append(num)
            else:
                if len(current_group) > 1:
                    consecutive_groups.append(len(current_group))
                current_group = [num]
        
        # 检查最后一组
        if len(current_group) > 1:
            consecutive_groups.append(len(current_group))
        
        return len(consecutive_groups), max(consecutive_groups) if consecutive_groups else 0
    
    # 应用函数计算连号信息
    df_filtered['连号组数'], df_filtered['最长连号长度'] = zip(*df_filtered.apply(calculate_consecutive_groups, axis=1))
    
    # 创建连号组数统计图表
    st.subheader("前区连号组数分布")
    consecutive_counts = df_filtered['连号组数'].value_counts().sort_index()
    
    fig_consecutive_groups = px.bar(
        x=consecutive_counts.index,
        y=consecutive_counts.values,
        labels=dict(x="连号组数", y="出现期数"),
        title="前区号码连号组数分布",
        color=consecutive_counts.values,
        color_continuous_scale="Plasma",
        text_auto=True
    )
    
    fig_consecutive_groups.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickangle=0,
        xaxis=dict(tickmode='linear'),
        coloraxis_colorbar=dict(title="出现期数")
    )
    
    st.plotly_chart(fig_consecutive_groups, use_container_width=True)
    
    # 创建最长连号长度统计图表
    st.subheader("前区最长连号长度分布")
    max_consecutive_counts = df_filtered['最长连号长度'].value_counts().sort_index()
    
    fig_max_consecutive = px.bar(
        x=max_consecutive_counts.index,
        y=max_consecutive_counts.values,
        labels=dict(x="最长连号长度", y="出现期数"),
        title="前区号码最长连号长度分布",
        color=max_consecutive_counts.values,
        color_continuous_scale="Jet",
        text_auto=True
    )
    
    fig_max_consecutive.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickangle=0,
        xaxis=dict(tickmode='linear'),
        coloraxis_colorbar=dict(title="出现期数")
    )
    
    st.plotly_chart(fig_max_consecutive, use_container_width=True)

    # --------------------- 添加前区和后区号码相关性分析图表 ---------------------
    st.subheader("前区和后区号码相关性分析图表")
    
    # 计算前区和后区的统计特征
    df_stats = pd.DataFrame(index=df_filtered.index)
    
    # 前区统计特征
    df_stats['前区和值'] = df_filtered['f1'] + df_filtered['f2'] + df_filtered['f3'] + df_filtered['f4'] + df_filtered['f5']
    df_stats['前区平均值'] = df_stats['前区和值'] / 5
    df_stats['前区最大值'] = df_filtered[['f1', 'f2', 'f3', 'f4', 'f5']].max(axis=1)
    df_stats['前区最小值'] = df_filtered[['f1', 'f2', 'f3', 'f4', 'f5']].min(axis=1)
    df_stats['前区极差'] = df_stats['前区最大值'] - df_stats['前区最小值']
    df_stats['前区中位数'] = df_filtered[['f1', 'f2', 'f3', 'f4', 'f5']].median(axis=1)
    
    # 后区统计特征
    df_stats['后区和值'] = df_filtered['b1'] + df_filtered['b2']
    df_stats['后区平均值'] = df_stats['后区和值'] / 2
    df_stats['后区最大值'] = df_filtered[['b1', 'b2']].max(axis=1)
    df_stats['后区最小值'] = df_filtered[['b1', 'b2']].min(axis=1)
    df_stats['后区极差'] = df_stats['后区最大值'] - df_stats['后区最小值']
    df_stats['后区中位数'] = df_filtered[['b1', 'b2']].median(axis=1)
    
    # 计算相关性矩阵
    corr_matrix = df_stats.corr()
    
    # 创建相关性热力图
    fig_corr = px.imshow(
        corr_matrix, 
        labels=dict(x="统计特征", y="统计特征", color="相关性系数"),
        title="前区和后区统计特征相关性矩阵",
        color_continuous_scale="RdBu_r",  # 红色-蓝色反向映射，红色代表正相关，蓝色代表负相关
        text_auto=True,  # 显示具体数值
        zmin=-1, zmax=1  # 设置颜色范围为-1到1
    )
    
    # 美化图表
    fig_corr.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickangle=45,
        xaxis_tickfont_size=10,
        yaxis_tickfont_size=10,
        coloraxis_colorbar=dict(
            title="相关性系数",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300
        )
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 解释相关性结果
    st.markdown("**相关性分析说明**：")
    st.markdown("- 相关性系数范围为-1到1，越接近1表示正相关越强，越接近-1表示负相关越强，越接近0表示相关性越弱")
    st.markdown("- 红色区域表示正相关，蓝色区域表示负相关，白色区域表示无相关性")
    st.markdown("- 可以观察前区统计特征（如和值、平均值）与后区统计特征之间的关联模式")

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
        consecutive_count = st.number_input("前区连号限制数量", 0, 5, 0)
        cons_mode_label = st.selectbox("连号匹配方式", ["等于", "至少", "最多"])
        if cons_mode_label == "等于":
            consecutive_mode = "exact"
        elif cons_mode_label == "至少":
            consecutive_mode = "min"
        else:  # 最多
            consecutive_mode = "max"
        
        cons_check_type_label = st.selectbox("连号检查类型", ["连号组数", "连号对数"])
        consecutive_check_type = "groups" if cons_check_type_label == "连号组数" else "pairs"
        
        # 添加说明
        if cons_check_type_label == "连号组数":
            st.caption("连号组数：连续号码段的数量。如[1,2,3,5,6]有2组连号")
        else:
            st.caption("连号对数：相邻连续数字的对数。如[1,2,3,5,6]有3对连号")
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
        "consecutive_check_type": consecutive_check_type,
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

    # 左侧边栏：参数设置
    with st.sidebar:
        st.subheader("🎯 预测设置")
        
        # 预测参数
        with st.expander("🎯 预测参数", expanded=True):
            # 基本参数 - 使用更小的列布局适配侧边栏
            use_recent_n = st.number_input("权重最近N期", 0, 1000, 2, key="tab4_recent_n")
            pred_count = st.number_input("每期注数", 1, 20, 5, key="tab4_pred_count")
            min_consec = st.number_input("前区连号限制数量", 0, 5, 0, key="tab4_min_consec")
            min_odd = st.number_input("前区最小奇数", 0, 5, 2, key="tab4_min_odd")
            
            # 连号相关参数
            consec_mode_label = st.selectbox("连号匹配方式", ["等于", "至少", "最多"], index=2, key="tab4_consec_mode")
            if consec_mode_label == "等于":
                consec_mode = "exact"
            elif consec_mode_label == "至少":
                consec_mode = "min"
            else:  # 最多
                consec_mode = "max"
            
            consec_check_type_label = st.selectbox("连号检查类型", ["连号组数", "连号对数"], key="tab4_consec_check_type")
            consec_check_type = "groups" if consec_check_type_label == "连号组数" else "pairs"

            # 多期预测参数
            multi_period_enabled = st.checkbox("启用多期预测", value=True, key="tab4_multi_period_enabled")
            future_periods = st.number_input("预测未来期数", 1, 20, 5, key="tab4_future_periods")
            backtest_gap_periods = st.number_input("回测间隔期数", 1, 20, 5, key="tab4_backtest_gap_periods")
            st.caption("回测间隔期数：回测时每期使用N期前的开奖数据进行号码生成和结果比对")

            # 区块选择
            pred_selected_front = st.multiselect("前区区块", front_labels, default=front_labels, key="tab4_front")
            pred_selected_back = st.multiselect("后区区块", back_labels, default=back_labels, key="tab4_back")

            # 高级参数
            top_n_blocks_future = st.number_input(
                "前区仅用前N区块", min_value=3, max_value=7, value=4, key="tab4_top_n_blocks"
            )
            max_per_block_future = st.number_input("每区块最多取", 1, 5, 2, key="tab4_max_per_block")
            random_blocks_count_future = st.number_input(
                "每期随机区块数", 1, len(front_labels), 4, key="tab4_random_blocks_count"
            )
            random_back_blocks_count_future = st.number_input(
                "后区随机区块数", 1, len(back_labels), 3, key="tab4_random_back_blocks_count"
            )
            span = st.slider("EWMA span", 1, 5, 1, key="tab4_span")

        # 排除与回测设置
        with st.expander("🧹 排除与回测设置", expanded=False):
            exclude_top_n = st.checkbox("排除近期高频号码", value=False, key="tab4_exclude_top_n")
            exclude_top_front_n = st.number_input(
                "前区排除数量", 0, 10, 3, key="tab4_exclude_front_n"
            )
            exclude_top_back_n = st.number_input(
                "后区排除数量", 0, 5, 2, key="tab4_exclude_back_n"
            )
            backtest_n = st.number_input("回测历史期数（0=不回测）", 0, 500, 10, key="tab4_backtest_n")
            st.caption("建议调小 N 和区块数量以提升生成速度。")
        
        # 成本与奖金参数
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
            # 侧边栏单列显示奖金设置
            for idx, (name, default_val) in enumerate(PRIZE_PAYOUT_DEFAULT.items()):
                prize_amounts[name] = st.number_input(
                    f"{name}估值（元）",
                    0.0,
                    10000000.0,
                    default_val,
                    10.0,
                    key=f"prize_amount_{name}"
                )

        # AI模型设置
        use_ai_model = st.checkbox("使用AI模型预测（需先训练模型）", value=False, key="tab4_use_ai")
        ml_predictor = st.session_state.get('ml_predictor', None)
        if use_ai_model and ml_predictor is None:
            st.warning("⚠️ 未检测到训练好的AI模型，请在'AI优化与模型训练'标签页先训练模型。")
            use_ai_model = False
        
        # 策略管理
        with st.expander("💾 策略管理", expanded=False):
            st.write("管理您的预测策略")
            
            # 保存当前设置为策略
            st.subheader("保存当前设置")
            strategy_name = st.text_input("策略名称", value="", placeholder="请输入策略名称", key="strategy_name_input")
            
            # 保存按钮
            save_col1, save_col2 = st.columns([1, 1])
            with save_col1:
                if st.button("保存策略", key="save_strategy_btn", type="primary"):
                    if not strategy_name.strip():
                        st.error("策略名称不能为空")
                    else:
                        # 收集当前设置的所有参数
                        current_params = {
                            'use_recent_n': use_recent_n,
                            'pred_count': pred_count,
                            'min_consec': min_consec,
                            'min_odd': min_odd,
                            'consec_mode': consec_mode,
                            'consec_check_type': consec_check_type,
                            'pred_selected_front': pred_selected_front,
                            'pred_selected_back': pred_selected_back,
                            'top_n_blocks_future': top_n_blocks_future,
                            'max_per_block_future': max_per_block_future,
                            'random_blocks_count_future': random_blocks_count_future,
                            'random_back_blocks_count_future': random_back_blocks_count_future,
                            'span': span,
                            'exclude_top_n': exclude_top_n,
                            'exclude_top_front_n': exclude_top_front_n,
                            'exclude_top_back_n': exclude_top_back_n,
                            'backtest_n': backtest_n,
                            'ticket_cost': ticket_cost,
                            'prize_amounts': prize_amounts,
                            'use_ai_model': use_ai_model
                        }
                        
                        # 保存策略
                        save_strategy(strategy_name, current_params)
                        st.success(f"✅ 策略 '{strategy_name}' 保存成功")
                        # 重新加载策略列表
                        st.rerun()
            
            # 删除策略
            st.subheader("删除策略")
            strategies = get_saved_strategies()
            if strategies:
                delete_options = {s['name']: s['filename'] for s in strategies}
                delete_selection = st.selectbox(
                    "选择要删除的策略",
                    list(delete_options.keys()),
                    key="delete_strategy_select"
                )
                
                with save_col2:
                    if st.button("删除策略", key="delete_strategy_btn", type="secondary"):
                        if delete_strategy(delete_options[delete_selection]):
                            st.success(f"✅ 策略 '{delete_selection}' 已删除")
                            # 重新加载策略列表
                            st.rerun()
            else:
                st.info("暂无保存的策略")
    
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

    # ----------------- 右侧主区域：策略选择、生成按钮和结果显示 -----------------
    # 初始化当前选中的策略
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = None
    
    # 生成按钮和策略选择 - 突出显示在主区域顶部
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # 获取保存的策略列表
        strategies = get_saved_strategies()
        if strategies:
            strategy_options = {s['name']: s['filename'] for s in strategies}
            strategy_names = ['自定义设置'] + list(strategy_options.keys())
            strategy_selection = st.selectbox(
                "🔧 选择生成策略", 
                strategy_names,
                index=0,
                key="tab4_strategy_select"
            )
            
            # 如果选择了非默认策略，更新session_state
            if strategy_selection != '自定义设置':
                st.session_state.selected_strategy = strategy_options[strategy_selection]
            else:
                st.session_state.selected_strategy = None
        else:
            st.selectbox(
                "🔧 选择生成策略", 
                ['自定义设置'],
                disabled=True,
                key="tab4_strategy_select"
            )
            st.caption("暂无保存的策略，请在预测设置中保存策略")
    
    with col2:
        # 添加空标签以保持与selectbox标签对齐
        st.markdown("", unsafe_allow_html=True)
        generate_button = st.button(
            "生成未来预测号码", 
            key="tab4_gen_future",
            use_container_width=True,
            type="primary"
        )
    
    if generate_button:
        # 检查是否有选中的策略
        if st.session_state.selected_strategy:
            # 加载策略参数
            strategy_params = load_strategy(st.session_state.selected_strategy)
            if strategy_params:
                st.info(f"🔧 使用策略参数生成号码")
                
                # 从策略中获取参数
                use_recent_n = strategy_params.get('use_recent_n', use_recent_n)
                pred_count = strategy_params.get('pred_count', pred_count)
                min_consec = strategy_params.get('min_consec', min_consec)
                min_odd = strategy_params.get('min_odd', min_odd)
                consec_mode = strategy_params.get('consec_mode', consec_mode)
                consec_check_type = strategy_params.get('consec_check_type', consec_check_type)
                pred_selected_front = strategy_params.get('pred_selected_front', pred_selected_front)
                pred_selected_back = strategy_params.get('pred_selected_back', pred_selected_back)
                top_n_blocks_future = strategy_params.get('top_n_blocks_future', top_n_blocks_future)
                max_per_block_future = strategy_params.get('max_per_block_future', max_per_block_future)
                random_blocks_count_future = strategy_params.get('random_blocks_count_future', random_blocks_count_future)
                random_back_blocks_count_future = strategy_params.get('random_back_blocks_count_future', random_back_blocks_count_future)
                span = strategy_params.get('span', span)
                exclude_top_n = strategy_params.get('exclude_top_n', exclude_top_n)
                exclude_top_front_n = strategy_params.get('exclude_top_front_n', exclude_top_front_n)
                exclude_top_back_n = strategy_params.get('exclude_top_back_n', exclude_top_back_n)
                use_ai_model = strategy_params.get('use_ai_model', use_ai_model)
                
                # 使用更新后的参数重新计算上下文和排除号码
                recent_window = build_history_window(df_filtered, recent_n=use_recent_n)
                generation_context = prepare_generation_context(
                    df_window=recent_window,
                    span=span,
                    front_blocks_labels=front_labels,
                    back_blocks_labels=back_labels,
                    selected_front_blocks=pred_selected_front,
                    selected_back_blocks=pred_selected_back,
                )
                exclude_front, exclude_back = compute_exclusions(
                    generation_context["front_freq_map"],
                    generation_context["back_freq_map"],
                    exclude_top_n,
                    exclude_top_front_n,
                    exclude_top_back_n
                )
        
        # 根据是否启用多期预测决定生成方式
        if multi_period_enabled:
            all_period_cands = {}
            # 生成多期预测号码
            for period_idx in range(future_periods):
                # 组装规则并生成号码
                rules_future = assemble_rules(
                    base_rules=predict_rules,
                    min_consec=min_consec,
                    min_odd=min_odd,
                    exclude_front=exclude_front,
                    exclude_back=exclude_back,
                    top_n_blocks=top_n_blocks_future,
                    max_per_block=max_per_block_future,
                    random_blocks_count=random_blocks_count_future,
                    random_back_blocks_count=random_back_blocks_count_future,
                    consecutive_mode=consec_mode,
                    consecutive_check_type=consec_check_type
                )

                # 生成当期预测号码
                period_cands = genmod.gen_numbers(
                    count=pred_count,
                    rules=rules_future,
                    front_blocks=generation_context["front_blocks"],
                    back_blocks=generation_context["back_blocks"],
                    front_weights=generation_context["front_weights"],
                    back_weights=generation_context["back_weights"],
                    selected_front_blocks=pred_selected_front,
                    selected_back_blocks=pred_selected_back
                )
                all_period_cands[f'期{period_idx + 1}'] = period_cands
            
            # 设置所有期的候选号码
            cands = []
            for period_name, period_candidates in all_period_cands.items():
                # 为每个候选号码添加期数信息
                for cand in period_candidates:
                    cand_with_period = cand.copy()
                    cand_with_period['period'] = period_name
                    cands.append(cand_with_period)
        else:
            # 单期预测
            rules_future = assemble_rules(
                base_rules=predict_rules,
                min_consec=min_consec,
                min_odd=min_odd,
                exclude_front=exclude_front,
                exclude_back=exclude_back,
                top_n_blocks=top_n_blocks_future,
                max_per_block=max_per_block_future,
                random_blocks_count=random_blocks_count_future,
                random_back_blocks_count=random_back_blocks_count_future,
                consecutive_mode=consec_mode,
                consecutive_check_type=consec_check_type
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
                # 检查是否有多期信息
                period_info = c.get('period', '')
                rows.append({
                    "预测序号": i,
                    "预测期数": period_info,
                    "预测前区": ",".join(map(str, c["front"])),
                    "预测后区": ",".join(map(str, c["back"])),
                    "中奖情况": "未比对"
                })

            pred_df = pd.DataFrame(rows)

            # 可视化：号码分布热力图 - 使用折叠面板
            with st.expander("📊 预测号码分布可视化（点击展开/折叠）"):
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

    # 使用Tabs布局显示未来预测和历史回测
    if (generate_button and len(cands) > 0) or backtest_n > 0:
        result_tabs = st.tabs(["🔮 未来预测号码", "📊 历史回测结果", "📈 未来多期回测"])
        
        # 未来预测号码Tab
        with result_tabs[0]:
            if generate_button and len(cands) > 0:
                if use_ai_model:
                    st.info(f"🤖 使用AI模型（{st.session_state.get('model_type', 'unknown')}）生成")
                
                # 使用表格形式显示
                with st.container():
                    st.dataframe(
                        pred_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "预测序号": st.column_config.NumberColumn(
                                "预测序号",
                                format="%d",
                            ),
                            "预测期数": st.column_config.Column(
                                "预测期数",
                                width="small",
                            ),
                            "预测前区": st.column_config.Column(
                                "预测前区",
                                width="medium",
                            ),
                            "预测后区": st.column_config.Column(
                                "预测后区",
                                width="small",
                            ),
                            "中奖情况": st.column_config.Column(
                                "中奖情况",
                                width="small",
                            ),
                        }
                    )
            else:
                st.info("请点击'生成未来预测号码'按钮以查看预测结果")
        
        # 历史回测结果Tab
        with result_tabs[1]:
            if backtest_n > 0:
                st.subheader(f"历史回测（最近 {backtest_n} 期，每期 {pred_count} 注）")
        
        # 未来多期回测Tab
        with result_tabs[2]:
            if multi_period_enabled and generate_button and len(cands) > 0:
                st.subheader(f"未来多期预测回测（预测 {future_periods} 期，每期 {pred_count} 注）")
                
                # 确保回测间隔期数有默认值
                if backtest_gap_periods < 0:
                    backtest_gap_periods = 0
                
                # 获取历史数据用于模拟未来回测
                if len(df_filtered) >= backtest_gap_periods + future_periods:
                    # 选择适当的历史数据作为模拟的未来开奖结果
                    # 使用历史数据中的较早期数据，跳过最近的gap期
                    future_test_data = df_filtered.sort_values("date", ascending=False).iloc[backtest_gap_periods:backtest_gap_periods+future_periods].reset_index(drop=True)
                    
                    # 多期回测数据准备
                    multi_period_backtest_data = []
                    total_cost_multi = 0.0
                    total_return_multi = 0.0
                    total_bets_multi = 0
                    
                    # 按期数分组预测号码
                    cands_by_period = {}
                    for c in cands:
                        period = c.get('period', 1)  # 默认第一期
                        if period not in cands_by_period:
                            cands_by_period[period] = []
                        cands_by_period[period].append(c)
                    
                    # 对每一期进行回测
                    for idx, row in future_test_data.iterrows():
                        period_num = idx + 1
                        period_cands = cands_by_period.get(period_num, [])
                        
                        if period_cands:
                            row_data = {
                                "预测期数": period_num,
                                "开奖期号": row["issue"],
                                "开奖前区": ",".join(map(str, row[["f1", "f2", "f3", "f4", "f5"]])),
                                "开奖后区": ",".join(map(str, row[["b1", "b2"]]))
                            }
                            row_bets = len(period_cands)
                            row_cost = row_bets * ticket_cost
                            row_return = 0.0
                            
                            for i, c in enumerate(period_cands, 1):
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
                            total_cost_multi += row_cost
                            total_return_multi += row_return
                            total_bets_multi += row_bets
                            multi_period_backtest_data.append(row_data)
                    
                    if multi_period_backtest_data:
                        multi_period_df = pd.DataFrame(multi_period_backtest_data)
                        
                        # 显示回测表格
                        prize_cols = [col for col in multi_period_df.columns if "中奖情况" in col]
                        st.dataframe(
                            multi_period_df.style.applymap(lambda v: PRIZE_COLOR.get(v, ""), subset=prize_cols),
                            use_container_width=False
                        )
                        
                        # 显示回测汇总统计
                        st.markdown("### 📊 回测汇总")
                        summary_cols = st.columns(4)
                        with summary_cols[0]:
                            st.metric("总投注注数", total_bets_multi)
                        with summary_cols[1]:
                            st.metric("总投入", f"¥{total_cost_multi:.2f}")
                        with summary_cols[2]:
                            st.metric("总回收", f"¥{total_return_multi:.2f}")
                        with summary_cols[3]:
                            profit_rate = ((total_return_multi - total_cost_multi) / total_cost_multi * 100) if total_cost_multi > 0 else 0
                            st.metric("收益率", f"{profit_rate:.2f}%")
                    else:
                        st.warning("没有足够的预测数据进行多期回测")
                else:
                    st.warning(f"历史数据不足，需要至少 {backtest_gap_periods + future_periods} 期数据进行多期回测")
            else:
                st.info("请先启用多期预测并生成未来号码以查看多期回测结果")
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
                random_back_blocks_count=random_back_blocks_count_future,
                consecutive_mode=consec_mode,
                consecutive_check_type=consec_check_type
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
    
    ai_tabs = st.tabs(["🧠 模型训练", "🔍 参数优化", "📊 批量回测", "⚡ 性能测试"])
    
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
        
        optimization_methods = get_optimization_methods()
        opt_method = st.selectbox("优化方法", list(optimization_methods.keys()), key="ai_opt_method")
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
                                random_back_blocks_count=params.random_back_blocks_count,
                                consecutive_mode=params.consecutive_mode,
                                consecutive_check_type=params.consecutive_check_type
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
                        algorithm = optimization_methods[opt_method]
                        best_params, best_score = bayesian_optimize(
                            fitness_function,
                            param_ranges,
                            n_iterations=opt_generations * opt_population,
                            algorithm=algorithm
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
                                random_back_blocks_count=batch_random_back_blocks_count,
                                consecutive_mode="max",  # 批量回测使用默认值
                                consecutive_check_type="groups"
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

    with ai_tabs[3]:
        st.subheader("⚡ 系统性能测试")
        st.write("测试各个组件的性能表现，帮助优化系统效率")
        
        # 性能测试选项
        test_options = st.multiselect(
            "选择测试组件",
            ["号码生成器", "参数优化器", "回测分析器", "全流程测试"],
            default=["号码生成器", "参数优化器", "回测分析器"]
        )
        
        # 测试参数
        test_runs = st.number_input("测试运行次数", 1, 100, 5, key="perf_test_runs")
        test_samples = st.number_input("样本数量", 10, 1000, 100, key="perf_test_samples")
        
        # 测试对比版本
        compare_versions = st.checkbox("对比优化前后性能", value=False)
        
        if st.button("开始性能测试", key="start_perf_test"):
            try:
                with st.spinner("正在执行性能测试..."):
                    # 准备测试数据
                    test_df = df_filtered.head(100)
                    
                    # 运行性能测试
                    results = performance_tester.run_all_tests(
                        test_options=test_options,
                        runs=test_runs,
                        samples=test_samples,
                        data=test_df,
                        compare_versions=compare_versions
                    )
                    
                    # 显示测试结果
                    st.success("✅ 性能测试完成！")
                    
                    # 总览统计
                    st.subheader("📊 性能测试结果总览")
                    
                    # 各组件性能指标
                    for component, metrics in results.items():
                        st.markdown(f"### {component}")
                        
                        # 如果是对比模式
                        if compare_versions and 'before' in metrics and 'after' in metrics:
                            cols = st.columns(2)
                            cols[0].markdown("**优化前**")
                            cols[0].json(metrics['before'])
                            cols[1].markdown("**优化后**")
                            cols[1].json(metrics['after'])
                            
                            # 计算改进百分比
                            st.markdown("**性能改进**")
                            improvement = {}
                            for key in metrics['before']:
                                if key != 'test_name':
                                    before_val = metrics['before'][key]
                                    after_val = metrics['after'][key]
                                    if before_val > 0:
                                        improvement[key] = f"{(before_val - after_val) / before_val * 100:.2f}%"
                                    else:
                                        improvement[key] = "∞"
                            st.json(improvement)
                        else:
                            st.json(metrics)
                    
                    # 生成性能报告
                    report = performance_tester.generate_performance_report(results)
                    st.subheader("📋 性能测试报告")
                    st.text(report)
                    
                    # 提供下载选项
                    report_bytes = report.encode('utf-8')
                    st.download_button(
                        label="下载性能测试报告",
                        data=report_bytes,
                        file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
            except Exception as e:
                st.error(f"性能测试失败：{e}")
                import traceback
                with st.expander("查看详细错误"):
                    st.code(traceback.format_exc())

# 显示系统优化信息
st.sidebar.markdown("---")
st.sidebar.subheader("💡 系统优化信息")
st.sidebar.info("✅ 贝叶斯优化已增强\n✅ 高额奖项命中分析已添加\n✅ 性能测试已集成\n✅ 多算法支持已实现")
st.sidebar.caption("大乐透分析系统 v2.0 - 智能优化版")

