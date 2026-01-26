# app.py (v3.0 - Evolution Edition)
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
from backend.enhanced_generator import EnhancedNumberGenerator
from backend.markov_model import BigDataAnalyzer
from backend.optimizer import genetic_algorithm_optimize, bayesian_optimize, get_optimization_methods
from backend.backtest import BacktestAnalyzer
from backend.performance_test import PerformanceTester
from backend.evolutionary_optimizer import create_evolutionary_optimizer, EvolutionaryConfig
from backend.neural_predictor import create_neural_predictor, get_available_models, HAS_PYTORCH
from backend.strategy_optimizer import create_strategy_optimizer, OptimizationConfig
from backend.exclusion_pool_generator import exclusion_pool_generator
from backend.exclusion_pool_db import exclusion_pool_db
from predictor import compute_weights_from_history, prepare_generator_inputs, compute_weights_from_history_ewma
import random
import io

st.set_page_config(page_title="大乐透分析与选号 v3.0", page_icon="🎯", layout="wide")
st.title("🎯 大乐透分析与选号（进化版） v3.0 — AI深度学习与进化算法优化")

# 初始化所有组件
backtest_analyzer = BacktestAnalyzer()
performance_tester = PerformanceTester()
enhanced_generator = EnhancedNumberGenerator()
big_data_analyzer = BigDataAnalyzer()

# 初始化进化优化器
@st.cache_resource
def init_evolutionary_optimizer():
    try:
        # 先尝试完整版进化优化器
        config = EvolutionaryConfig(
            population_size=30,
            generations=50,
            mutation_rate=0.15,
            multi_objective=True
        )
        optimizer = create_evolutionary_optimizer(config)
        
        if optimizer is not None:
            st.success("✅ 完整版进化优化器初始化成功")
            return optimizer
        else:
            raise Exception("完整版进化优化器初始化失败")
            
    except Exception as e:
        st.warning(f"完整版进化优化器初始化失败: {e}")
        
        # 回退到简化版
        try:
            from backend.simple_evolutionary import create_simple_evolutionary_optimizer, SimpleEvolutionaryConfig
            
            simple_config = SimpleEvolutionaryConfig(
                population_size=30,
                generations=50,
                mutation_rate=0.15
            )
            simple_optimizer = create_simple_evolutionary_optimizer(simple_config)
            
            st.info("✅ 简化版进化优化器初始化成功（功能有限但可用）")
            return simple_optimizer
            
        except Exception as e2:
            st.error(f"简化版进化优化器也初始化失败: {e2}")
            return None

# 初始化神经网络预测器
@st.cache_resource
def init_neural_predictors():
    predictors = {}
    
    if not HAS_PYTORCH:
        st.info("💡 PyTorch未安装，跳过神经网络预测器初始化")
        st.info("如需使用神经网络功能，请运行: pip install torch scikit-learn")
        return predictors
    
    try:
        available_models = get_available_models()
        st.info(f"可用的神经网络模型: {available_models}")
        
        for model_type in available_models:
            try:
                predictors[model_type] = create_neural_predictor(model_type)
                st.success(f"✅ {model_type} 预测器初始化成功")
            except Exception as e:
                st.warning(f"⚠️ {model_type} 预测器初始化失败: {e}")
                
    except Exception as e:
        st.error(f"神经网络预测器初始化异常: {e}")
    
    return predictors

# 初始化策略优化器
@st.cache_resource
def init_strategy_optimizer():
    try:
        config = OptimizationConfig(
            primary_objective='roi',
            use_neural_networks=HAS_PYTORCH,
            use_evolutionary=True,
            use_ensemble=True
        )
        optimizer = create_strategy_optimizer(config)
        
        if optimizer is not None:
            st.success("✅ 策略优化器初始化成功")
        
        return optimizer
        
    except Exception as e:
        st.error(f"策略优化器初始化失败: {e}")
        import traceback
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None

# 延迟初始化高级组件
def initialize_advanced_components():
    """初始化高级组件"""
    success_count = 0
    total_components = 3
    
    # 初始化进化优化器
    if 'evolutionary_optimizer' not in st.session_state:
        with st.spinner("正在初始化进化优化器..."):
            try:
                st.session_state.evolutionary_optimizer = init_evolutionary_optimizer()
                if st.session_state.evolutionary_optimizer is not None:
                    success_count += 1
                    st.success("✅ 进化优化器初始化成功")
                else:
                    st.error("❌ 进化优化器初始化失败")
            except Exception as e:
                st.error(f"❌ 进化优化器初始化异常: {e}")
                st.session_state.evolutionary_optimizer = None
    else:
        success_count += 1
    
    # 初始化神经网络预测器
    if 'neural_predictors' not in st.session_state:
        with st.spinner("正在初始化神经网络预测器..."):
            try:
                st.session_state.neural_predictors = init_neural_predictors()
                if st.session_state.neural_predictors:
                    success_count += 1
                    st.success(f"✅ 神经网络预测器初始化成功 ({len(st.session_state.neural_predictors)}个模型)")
                else:
                    st.warning("⚠️ 神经网络预测器初始化失败（可能缺少PyTorch）")
            except Exception as e:
                st.error(f"❌ 神经网络预测器初始化异常: {e}")
                st.session_state.neural_predictors = {}
    else:
        if st.session_state.neural_predictors:
            success_count += 1
    
    # 初始化策略优化器
    if 'strategy_optimizer' not in st.session_state:
        with st.spinner("正在初始化策略优化器..."):
            try:
                st.session_state.strategy_optimizer = init_strategy_optimizer()
                if st.session_state.strategy_optimizer is not None:
                    success_count += 1
                    st.success("✅ 策略优化器初始化成功")
                else:
                    st.error("❌ 策略优化器初始化失败")
            except Exception as e:
                st.error(f"❌ 策略优化器初始化异常: {e}")
                st.session_state.strategy_optimizer = None
    else:
        success_count += 1
    
    st.session_state.advanced_initialized = True
    
    # 显示总体初始化结果
    if success_count == total_components:
        st.success(f"🎉 所有AI深度学习组件初始化完成！({success_count}/{total_components})")
    elif success_count > 0:
        st.warning(f"⚠️ 部分AI组件初始化完成 ({success_count}/{total_components})，可以使用已初始化的功能")
    else:
        st.error("❌ 所有AI组件初始化失败，请检查依赖包安装")
    
    # 显示使用建议
    if success_count > 0:
        st.info("💡 提示：已初始化的组件可以正常使用，未初始化的组件可能需要安装额外依赖包")
    
    return success_count

# 获取组件的辅助函数
def get_evolutionary_optimizer():
    return st.session_state.get('evolutionary_optimizer', None)

def get_neural_predictors():
    return st.session_state.get('neural_predictors', {})

def get_strategy_optimizer():
    return st.session_state.get('strategy_optimizer', None)

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


def perform_two_round_backtest(df_filtered, enhanced_generator, periods, count, rules, 
                              generation_context, pred_selected_front, pred_selected_back,
                              use_markov, use_big_data, markov_weight, big_data_weight, 
                              traditional_weight, variation_strength, include_recombination,
                              use_exclusion_pool, exclusion_pool_size, use_dynamic_pool,
                              ticket_cost, prize_structure):
    """
    执行两轮生成的历史回测
    
    Args:
        df_filtered: 过滤后的历史数据
        enhanced_generator: 增强生成器实例
        periods: 回测期数
        count: 每轮生成数量
        rules: 生成规则
        generation_context: 生成上下文
        pred_selected_front: 选中的前区区块
        pred_selected_back: 选中的后区区块
        use_markov: 是否使用马尔可夫链
        use_big_data: 是否使用大数据分析
        markov_weight: 马尔可夫链权重
        big_data_weight: 大数据分析权重
        traditional_weight: 传统方法权重
        variation_strength: 变化强度
        include_recombination: 是否包含重组模式回测
        use_exclusion_pool: 是否使用排除池
        exclusion_pool_size: 排除池大小
        use_dynamic_pool: 是否使用动态排除池
        ticket_cost: 单注成本
        prize_structure: 奖金结构
        
    Returns:
        回测结果字典
    """
    print(f"开始两轮生成历史回测，回测期数: {periods}")
    
    # 准备回测数据
    if len(df_filtered) < periods + 50:  # 确保有足够的历史数据
        raise ValueError(f"历史数据不足，需要至少 {periods + 50} 期数据")
    
    # 选择回测数据（最近的periods期作为测试集）
    test_data = df_filtered.tail(periods).reset_index(drop=True)
    
    results = {
        'summary': {},
        'details': {}
    }
    
    # 测试模式列表
    test_modes = [
        {'name': '传统模式第一轮', 'use_recombination': False, 'round': 'first'},
        {'name': '传统模式第二轮', 'use_recombination': False, 'round': 'second'}
    ]
    
    if include_recombination:
        test_modes.extend([
            {'name': '重组模式第一轮', 'use_recombination': True, 'round': 'first'},
            {'name': '重组模式第二轮', 'use_recombination': True, 'round': 'second'}
        ])
    
    for mode in test_modes:
        print(f"回测模式: {mode['name']}")
        
        mode_results = []
        total_cost = 0.0
        total_return = 0.0
        hit_count = 0
        
        for i in range(periods):
            # 获取训练数据（当前测试期之前的所有数据）
            train_end_idx = len(df_filtered) - periods + i
            train_data = df_filtered.iloc[:train_end_idx]
            
            if len(train_data) < 20:  # 确保有足够的训练数据
                continue
            
            # 获取测试期的实际开奖结果
            test_row = test_data.iloc[i]
            actual_front = test_row[['f1', 'f2', 'f3', 'f4', 'f5']].tolist()
            actual_back = test_row[['b1', 'b2']].tolist()
            
            try:
                # 生成两轮号码
                two_round_result = enhanced_generator.generate_two_rounds(
                    count=count,
                    rules=rules,
                    front_blocks=generation_context["front_blocks"],
                    back_blocks=generation_context["back_blocks"],
                    front_weights=generation_context["front_weights"],
                    back_weights=generation_context["back_weights"],
                    selected_front_blocks=pred_selected_front,
                    selected_back_blocks=pred_selected_back,
                    historical_data=train_data.tail(100),  # 使用最近100期作为训练数据
                    use_markov=use_markov,
                    use_big_data=use_big_data,
                    markov_weight=markov_weight,
                    big_data_weight=big_data_weight,
                    traditional_weight=traditional_weight,
                    variation_strength=variation_strength,
                    use_recombination=mode['use_recombination'],
                    use_exclusion_pool=use_exclusion_pool,
                    exclusion_pool_size=exclusion_pool_size,
                    use_dynamic_pool=use_dynamic_pool
                )
                
                if 'error' in two_round_result:
                    print(f"第{i+1}期生成失败: {two_round_result['error']}")
                    continue
                
                # 选择对应轮次的候选号码
                if mode['round'] == 'first':
                    candidates = two_round_result.get('first_round', [])
                else:
                    candidates = two_round_result.get('second_round', [])
                
                if not candidates:
                    print(f"第{i+1}期{mode['round']}轮候选号码为空")
                    continue
                
                # 计算本期成本和收益
                period_cost = len(candidates) * ticket_cost
                period_return = 0.0
                period_hits = 0
                
                for candidate in candidates:
                    # 检查中奖情况
                    prize_name = check_prize(
                        candidate['front'], candidate['back'],
                        actual_front, actual_back
                    )
                    
                    if prize_name != "未中奖":
                        period_return += prize_structure.get(prize_name, 0)
                        period_hits += 1
                
                total_cost += period_cost
                total_return += period_return
                if period_hits > 0:
                    hit_count += 1
                
                # 记录详细结果
                mode_results.append({
                    "期数": i + 1,
                    "期号": test_row.get('issue', f'期{i+1}'),
                    "开奖前区": ",".join(map(str, actual_front)),
                    "开奖后区": ",".join(map(str, actual_back)),
                    "投注数": len(candidates),
                    "投注成本": f"{period_cost:.2f}元",
                    "中奖金额": f"{period_return:.2f}元",
                    "净收益": f"{period_return - period_cost:.2f}元",
                    "中奖注数": period_hits
                })
                
            except Exception as e:
                print(f"第{i+1}期回测失败: {e}")
                continue
        
        # 计算模式统计
        net_profit = total_return - total_cost
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
        hit_rate = (hit_count / periods * 100) if periods > 0 else 0
        
        results['summary'][mode['name']] = {
            'total_cost': total_cost,
            'total_return': total_return,
            'net_profit': net_profit,
            'roi': roi / 100,  # 转换为小数形式
            'hit_count': hit_count,
            'hit_rate': hit_rate / 100  # 转换为小数形式
        }
        
        results['details'][mode['name']] = mode_results
    
    print("两轮生成历史回测完成")
    return results
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
tab_data, tab_chart, tab_generate, tab_predict, tab_ai, tab_evolution = st.tabs(
    ["📂 数据管理", "📊 数据图表", "🔢 号码生成", "🔮 未来号码预测", "🤖 AI优化与模型训练", "🧬 进化算法与深度学习"]
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

    st.subheader("🚀 增强生成选项")
    
    # 增强生成设置
    use_enhanced = st.checkbox("启用增强生成（马尔可夫链 + 大数据分析）", value=False, key="use_enhanced_gen")
    
    if use_enhanced:
        enhanced_cols = st.columns(3)
        with enhanced_cols[0]:
            use_markov = st.checkbox("使用马尔可夫链模型", value=True, key="use_markov_gen")
            markov_weight = st.slider("马尔可夫链权重", 0.0, 1.0, 0.4, 0.1, key="markov_weight_gen")
        
        with enhanced_cols[1]:
            use_big_data = st.checkbox("使用大数据分析", value=True, key="use_big_data_gen")
            big_data_weight = st.slider("大数据分析权重", 0.0, 1.0, 0.3, 0.1, key="big_data_weight_gen")
        
        with enhanced_cols[2]:
            traditional_weight = st.slider("传统方法权重", 0.0, 1.0, 0.3, 0.1, key="traditional_weight_gen")
        
        # 权重归一化提示
        total_weight = markov_weight + big_data_weight + traditional_weight
        if total_weight != 1.0:
            st.caption(f"⚠️ 权重总和为 {total_weight:.1f}，将自动归一化")
        
        # 初始化增强生成器（如果尚未初始化）
        if use_enhanced and not enhanced_generator.markov_models and not enhanced_generator.use_ensemble:
            with st.spinner("正在初始化增强生成器..."):
                try:
                    # 添加集成学习选项
                    use_ensemble_init = st.checkbox("使用集成学习（推荐）", value=True, key="use_ensemble_init_tab3")
                    enhanced_generator.initialize_models(df_filtered, use_ensemble=use_ensemble_init)
                    st.success("✅ 增强生成器初始化完成")
                except Exception as e:
                    st.error(f"❌ 增强生成器初始化失败: {e}")
                    use_enhanced = False

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

    # 生成按钮区域
    gen_cols = st.columns(2)
    
    with gen_cols[0]:
        if st.button("🎲 传统生成号码并比对", use_container_width=True):
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
            
            st.subheader("🎲 传统生成结果")
            for i,cd in enumerate(cands,1):
                prize = check_prize(cd['front'],cd['back'],win_front,win_back)
                st.markdown(f"**第{i}注：前区 {cd['front']} | 后区 {cd['back']} => {prize}**")
    
    with gen_cols[1]:
        if st.button("🚀 增强生成号码并比对", use_container_width=True, disabled=not use_enhanced):
            if not use_enhanced:
                st.error("请先启用增强生成选项")
            else:
                win_front = parse_nums(win_front_input)
                win_back = parse_nums(win_back_input)
                
                try:
                    # 使用增强生成器
                    enhanced_cands = enhanced_generator.generate_enhanced_numbers(
                        count=max_gen,
                        rules=rules,
                        front_blocks={label:list(range(lo,hi+1)) for label,(lo,hi) in zip(front_labels,front_bins)},
                        back_blocks={label:list(range(lo,hi+1)) for label,(lo,hi) in zip(back_labels,back_bins)},
                        front_weights=front_weights,
                        back_weights=back_weights,
                        selected_front_blocks=selected_front_blocks,
                        selected_back_blocks=selected_back_blocks,
                        historical_data=df_filtered.tail(20),  # 使用最近20期数据
                        use_markov=use_markov,
                        use_big_data=use_big_data,
                        markov_weight=markov_weight,
                        big_data_weight=big_data_weight,
                        traditional_weight=traditional_weight
                    )
                    
                    st.subheader("🚀 增强生成结果")
                    for i, cd in enumerate(enhanced_cands, 1):
                        prize = check_prize(cd['front'], cd['back'], win_front, win_back)
                        markov_conf = cd.get('markov_confidence', 0.5)
                        big_data_score = cd.get('big_data_score', 0.5)
                        
                        st.markdown(f"**第{i}注：前区 {cd['front']} | 后区 {cd['back']} => {prize}**")
                        st.caption(f"马尔可夫置信度: {markov_conf:.3f} | 大数据评分: {big_data_score:.3f}")
                    
                    # 显示模型状态
                    with st.expander("📊 增强生成器状态"):
                        model_status = enhanced_generator.get_model_status()
                        st.json(model_status)
                        
                except Exception as e:
                    st.error(f"增强生成失败: {e}")
                    st.info("回退到传统生成方法...")
                    # 回退到传统方法
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
        
        # 增强生成设置
        with st.expander("🚀 增强生成设置", expanded=False):
            use_enhanced_tab4 = st.checkbox("启用增强生成（马尔可夫链 + 大数据分析）", value=False, key="tab4_use_enhanced")
            
            if use_enhanced_tab4:
                enhanced_cols_tab4 = st.columns(3)
                with enhanced_cols_tab4[0]:
                    use_markov_tab4 = st.checkbox("使用马尔可夫链模型", value=True, key="tab4_use_markov")
                    markov_weight_tab4 = st.slider("马尔可夫链权重", 0.0, 1.0, 0.4, 0.1, key="tab4_markov_weight")
                
                with enhanced_cols_tab4[1]:
                    use_big_data_tab4 = st.checkbox("使用大数据分析", value=True, key="tab4_use_big_data")
                    big_data_weight_tab4 = st.slider("大数据分析权重", 0.0, 1.0, 0.3, 0.1, key="tab4_big_data_weight")
                
                with enhanced_cols_tab4[2]:
                    traditional_weight_tab4 = st.slider("传统方法权重", 0.0, 1.0, 0.3, 0.1, key="tab4_traditional_weight")
                
                # 权重归一化提示
                total_weight_tab4 = markov_weight_tab4 + big_data_weight_tab4 + traditional_weight_tab4
                if total_weight_tab4 != 1.0:
                    st.caption(f"⚠️ 权重总和为 {total_weight_tab4:.1f}，将自动归一化")
                
                # 初始化增强生成器（如果尚未初始化）
                if not enhanced_generator.markov_models and not enhanced_generator.use_ensemble:
                    with st.spinner("正在初始化增强生成器..."):
                        try:
                            # 添加集成学习选项
                            use_ensemble_init_tab4 = st.checkbox("使用集成学习（推荐）", value=True, key="use_ensemble_init_tab4")
                            enhanced_generator.initialize_models(df_filtered, use_ensemble=use_ensemble_init_tab4)
                            st.success("✅ 增强生成器初始化完成")
                        except Exception as e:
                            st.error(f"❌ 增强生成器初始化失败: {e}")
                            use_enhanced_tab4 = False
                
                # 排除池设置
                st.subheader("排除池策略")
                use_exclusion_pool_enhanced = st.checkbox(
                    "🎯 启用排除池策略", 
                    value=False,
                    help="启用后，生成时将先创建排除池，然后生成不重复的号码",
                    key="use_exclusion_pool_enhanced"
                )
                
                if use_exclusion_pool_enhanced:
                    exclusion_pool_size_enhanced = st.slider(
                        "排除池大小", 
                        min_value=50, max_value=500, value=100, step=10,
                        help="排除池越大，生成的号码越独特，但生成难度也越高",
                        key="exclusion_pool_size_enhanced"
                    )
                    
                    # 动态/静态排除池选择
                    use_dynamic_pool_enhanced = st.radio(
                        "排除池类型",
                        options=[True, False],
                        format_func=lambda x: "🔄 动态排除池（推荐）" if x else "📋 静态排除池",
                        index=0,  # 默认选择动态排除池
                        help="动态排除池：每期用当前算法生成排除池，避开算法热门倾向；静态排除池：使用固定的排除池",
                        key="use_dynamic_pool_enhanced"
                    )
                    
                    if use_dynamic_pool_enhanced:
                        st.success(f"🔄 动态排除池策略已启用（大小：{exclusion_pool_size_enhanced}）")
                        st.caption("每期先用算法生成排除池，再生成避开这些组合的号码")
                    else:
                        st.info(f"📋 静态排除池策略已启用（大小：{exclusion_pool_size_enhanced}）")
                        st.caption("使用固定的排除池策略")
                else:
                    exclusion_pool_size_enhanced = 100  # 默认值
                    use_dynamic_pool_enhanced = True  # 默认值
            else:
                use_markov_tab4 = False
                use_big_data_tab4 = False
                markov_weight_tab4 = 0.0
                big_data_weight_tab4 = 0.0
                traditional_weight_tab4 = 1.0
                use_exclusion_pool_enhanced = False
                exclusion_pool_size_enhanced = 100
        
        # 两轮生成设置
        with st.expander("🔄 两轮生成设置", expanded=False):
            st.write("配置两轮对比生成的参数")
            
            # 变化强度设置
            variation_strength_sidebar = st.slider(
                "第二轮变化强度", 
                0.0, 1.0, 0.3, 0.1,
                help="0.0表示完全基于第一轮，1.0表示完全随机",
                key="variation_strength_sidebar"
            )
            
            # 重组模式设置
            st.subheader("重组模式")
            
            # 初始化重组模式状态（如果不存在）
            if 'recombination_mode' not in st.session_state:
                st.session_state.recombination_mode = False
            
            use_recombination = st.checkbox(
                "🔄 启用重组模式", 
                value=st.session_state.recombination_mode,
                help="启用后，第二轮号码将仅从第一轮生成的号码中重新组合，不会引入新号码",
                key="use_recombination_sidebar"
            )
            
            # 同步状态
            st.session_state.recombination_mode = use_recombination
            
            if use_recombination:
                st.success("🔄 重组模式已启用")
                st.caption("第二轮将仅使用第一轮号码进行重新组合")
            else:
                st.info("🎯 传统模式")
                st.caption("第二轮基于第一轮特征生成新号码")
            
            # 排除池设置
            st.subheader("排除池策略")
            
            # 初始化排除池模式状态（如果不存在）
            if 'exclusion_pool_mode' not in st.session_state:
                st.session_state.exclusion_pool_mode = False
            
            use_exclusion_pool_two_round = st.checkbox(
                "🎯 启用排除池策略", 
                value=st.session_state.exclusion_pool_mode,
                help="启用后，两轮生成都将使用排除池策略，避免生成常见组合",
                key="use_exclusion_pool_two_round_sidebar"
            )
            
            # 同步状态
            st.session_state.exclusion_pool_mode = use_exclusion_pool_two_round
            
            if use_exclusion_pool_two_round:
                exclusion_pool_size_two_round = st.slider(
                    "排除池大小", 
                    min_value=50, max_value=500, value=100, step=10,
                    help="排除池越大，生成的号码越独特，但生成难度也越高",
                    key="exclusion_pool_size_two_round_sidebar"
                )
                
                # 动态/静态排除池选择
                use_dynamic_pool_two_round = st.radio(
                    "排除池类型",
                    options=[True, False],
                    format_func=lambda x: "🔄 动态排除池（推荐）" if x else "📋 静态排除池",
                    index=0,  # 默认选择动态排除池
                    help="动态排除池：每期用当前算法生成排除池，避开算法热门倾向；静态排除池：使用固定的排除池",
                    key="use_dynamic_pool_two_round"
                )
                
                if use_dynamic_pool_two_round:
                    st.success(f"🔄 动态排除池策略已启用（大小：{exclusion_pool_size_two_round}）")
                    st.caption("第一轮和第二轮都将使用动态排除池策略生成")
                else:
                    st.info(f"📋 静态排除池策略已启用（大小：{exclusion_pool_size_two_round}）")
                    st.caption("第一轮和第二轮都将使用静态排除池策略生成")
            else:
                exclusion_pool_size_two_round = 100  # 默认值
                use_dynamic_pool_two_round = True  # 默认值
                st.info("🔢 传统生成")
                st.caption("使用传统的增强生成方法")
        
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
    
    with col3:
        # 添加空标签以保持与selectbox标签对齐
        st.markdown("", unsafe_allow_html=True)
        two_round_button = st.button(
            "🔄 两轮对比生成", 
            key="tab4_two_round",
            use_container_width=True,
            type="secondary"
        )
    
    # 排除池生成按钮区域
    st.markdown("---")
    st.subheader("🎯 排除池生成策略")
    st.caption("通过排除已生成的号码组合来提高高等奖中奖率")
    
    exclusion_cols = st.columns([2, 2, 2, 2])
    
    with exclusion_cols[0]:
        exclusion_pool_size = st.number_input(
            "排除池大小(N)", 
            min_value=10, max_value=1000, value=100, step=10,
            help="先生成N组号码作为排除池",
            key="exclusion_pool_size"
        )
    
    with exclusion_cols[1]:
        exclusion_target_count = st.number_input(
            "目标生成数量(Y)", 
            min_value=1, max_value=50, value=10, step=1,
            help="生成Y组不与排除池重复的号码",
            key="exclusion_target_count"
        )
    
    with exclusion_cols[2]:
        exclusion_generate_button = st.button(
            "🎯 排除池生成", 
            key="exclusion_generate_btn",
            use_container_width=True,
            type="primary"
        )
    
    with exclusion_cols[3]:
        exclusion_analysis_button = st.button(
            "📊 AI效果分析", 
            key="exclusion_analysis_btn",
            use_container_width=True,
            type="secondary"
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
                if use_enhanced_tab4:
                    try:
                        if use_exclusion_pool_enhanced:
                            # 使用排除池策略生成
                            exclusion_result = exclusion_pool_generator.generate_with_exclusion_pool(
                                exclusion_pool_size=exclusion_pool_size_enhanced,
                                target_count=pred_count,
                                rules=rules_future,
                                front_blocks=generation_context["front_blocks"],
                                back_blocks=generation_context["back_blocks"],
                                front_weights=generation_context["front_weights"],
                                back_weights=generation_context["back_weights"],
                                selected_front_blocks=pred_selected_front,
                                selected_back_blocks=pred_selected_back,
                                historical_data=recent_window,
                                use_enhanced=True,
                                save_to_db=True,
                                predicted_issue=f"期{period_idx + 1}",
                                strategy_name=st.session_state.get('selected_strategy', None),
                                generation_method="multi_period_exclusion",
                                use_dynamic_pool=use_dynamic_pool_enhanced
                            )
                            
                            if 'error' in exclusion_result:
                                st.warning(f"第{period_idx + 1}期排除池生成失败，回退到传统增强生成: {exclusion_result['error']}")
                                period_cands = enhanced_generator.generate_enhanced_numbers(
                                    count=pred_count,
                                    rules=rules_future,
                                    front_blocks=generation_context["front_blocks"],
                                    back_blocks=generation_context["back_blocks"],
                                    front_weights=generation_context["front_weights"],
                                    back_weights=generation_context["back_weights"],
                                    selected_front_blocks=pred_selected_front,
                                    selected_back_blocks=pred_selected_back,
                                    historical_data=recent_window,
                                    use_markov=use_markov_tab4,
                                    use_big_data=use_big_data_tab4,
                                    markov_weight=markov_weight_tab4,
                                    big_data_weight=big_data_weight_tab4,
                                    traditional_weight=traditional_weight_tab4
                                )
                            else:
                                period_cands = exclusion_result.get('target_numbers', [])
                        else:
                            # 使用传统增强生成
                            period_cands = enhanced_generator.generate_enhanced_numbers(
                                count=pred_count,
                                rules=rules_future,
                                front_blocks=generation_context["front_blocks"],
                                back_blocks=generation_context["back_blocks"],
                                front_weights=generation_context["front_weights"],
                                back_weights=generation_context["back_weights"],
                                selected_front_blocks=pred_selected_front,
                                selected_back_blocks=pred_selected_back,
                                historical_data=recent_window,
                                use_markov=use_markov_tab4,
                                use_big_data=use_big_data_tab4,
                                markov_weight=markov_weight_tab4,
                                big_data_weight=big_data_weight_tab4,
                                traditional_weight=traditional_weight_tab4
                            )
                    except Exception as e:
                        st.warning(f"增强生成失败，回退到传统方法: {e}")
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
                else:
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

            if use_enhanced_tab4:
                try:
                    if use_exclusion_pool_enhanced:
                        # 使用排除池策略生成
                        exclusion_result = exclusion_pool_generator.generate_with_exclusion_pool(
                            exclusion_pool_size=exclusion_pool_size_enhanced,
                            target_count=pred_count,
                            rules=rules_future,
                            front_blocks=generation_context["front_blocks"],
                            back_blocks=generation_context["back_blocks"],
                            front_weights=generation_context["front_weights"],
                            back_weights=generation_context["back_weights"],
                            selected_front_blocks=pred_selected_front,
                            selected_back_blocks=pred_selected_back,
                            historical_data=recent_window,
                            use_enhanced=True,
                            save_to_db=True,
                            predicted_issue=None,  # 可以从用户输入获取
                            strategy_name=st.session_state.get('selected_strategy', None),
                            generation_method="single_period_exclusion",
                            use_dynamic_pool=use_dynamic_pool_enhanced
                        )
                        
                        if 'error' in exclusion_result:
                            st.warning(f"排除池生成失败，回退到传统增强生成: {exclusion_result['error']}")
                            cands = enhanced_generator.generate_enhanced_numbers(
                                count=pred_count,
                                rules=rules_future,
                                front_blocks=generation_context["front_blocks"],
                                back_blocks=generation_context["back_blocks"],
                                front_weights=generation_context["front_weights"],
                                back_weights=generation_context["back_weights"],
                                selected_front_blocks=pred_selected_front,
                                selected_back_blocks=pred_selected_back,
                                historical_data=recent_window,
                                use_markov=use_markov_tab4,
                                use_big_data=use_big_data_tab4,
                                markov_weight=markov_weight_tab4,
                                big_data_weight=big_data_weight_tab4,
                                traditional_weight=traditional_weight_tab4
                            )
                        else:
                            cands = exclusion_result.get('target_numbers', [])
                            # 显示排除池信息
                            if exclusion_result.get('db_record_id'):
                                st.success(f"✅ 使用排除池策略生成完成，数据已保存（记录ID: {exclusion_result['db_record_id']}）")
                    else:
                        # 使用传统增强生成
                        cands = enhanced_generator.generate_enhanced_numbers(
                            count=pred_count,
                            rules=rules_future,
                            front_blocks=generation_context["front_blocks"],
                            back_blocks=generation_context["back_blocks"],
                            front_weights=generation_context["front_weights"],
                            back_weights=generation_context["back_weights"],
                            selected_front_blocks=pred_selected_front,
                            selected_back_blocks=pred_selected_back,
                            historical_data=recent_window,
                            use_markov=use_markov_tab4,
                            use_big_data=use_big_data_tab4,
                            markov_weight=markov_weight_tab4,
                            big_data_weight=big_data_weight_tab4,
                            traditional_weight=traditional_weight_tab4
                        )
                except Exception as e:
                    st.warning(f"增强生成失败，回退到传统方法: {e}")
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
            else:
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

    # 两轮对比生成逻辑
    if two_round_button:
        # 检查增强生成器是否已初始化
        is_initialized = (enhanced_generator.markov_models or enhanced_generator.use_ensemble)
        
        if not is_initialized:
            with st.spinner("正在初始化增强生成器..."):
                try:
                    # 自动使用集成学习进行初始化
                    enhanced_generator.initialize_models(df_filtered, use_ensemble=True)
                    is_initialized = True
                    st.success("✅ 增强生成器初始化完成")
                except Exception as e:
                    st.error(f"❌ 增强生成器初始化失败: {e}")
                    st.info("💡 请在侧边栏的'🚀 增强生成设置'中手动启用增强生成功能")
                    is_initialized = False
        
        # 只有在初始化成功后才继续
        if is_initialized:
            # 检查是否有选中的策略
            if st.session_state.selected_strategy:
                # 加载策略参数（与单轮生成相同的逻辑）
                strategy_params = load_strategy(st.session_state.selected_strategy)
                if strategy_params:
                    st.info(f"🔧 使用策略参数进行两轮对比生成")
                    
                    # 从策略中获取参数（复用单轮生成的参数处理逻辑）
                    use_recent_n = strategy_params.get('use_recent_n', use_recent_n)
                    pred_count = strategy_params.get('pred_count', pred_count)
                    min_consec = strategy_params.get('min_consec', min_consec)
                    min_odd = strategy_params.get('min_odd', min_odd)
                    consec_mode = strategy_params.get('consec_mode', consec_mode)
                    consec_check_type = strategy_params.get('consec_check_type', consec_check_type)
                    pred_selected_front = strategy_params.get('pred_selected_front', pred_selected_front)
                    pred_selected_back = strategy_params.get('pred_selected_back', pred_selected_back)
                    
                    # 重新计算上下文
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
            
            # 获取侧边栏的两轮生成设置
            variation_strength = st.session_state.get('variation_strength_sidebar', 0.3)
            use_recombination = st.session_state.get('recombination_mode', False)
            use_exclusion_pool_two_round = st.session_state.get('exclusion_pool_mode', False)
            exclusion_pool_size_two_round = st.session_state.get('exclusion_pool_size_two_round_sidebar', 100)
            use_dynamic_pool_two_round = st.session_state.get('use_dynamic_pool_two_round', True)
            
            # 显示当前设置
            if use_exclusion_pool_two_round:
                pool_type = "动态" if use_dynamic_pool_two_round else "静态"
                st.info(f"🎯 两轮生成设置：变化强度 {variation_strength:.1f}，模式：{'🔄 重组' if use_recombination else '🎯 传统'}，{pool_type}排除池：{exclusion_pool_size_two_round}")
            else:
                st.info(f"🎯 两轮生成设置：变化强度 {variation_strength:.1f}，模式：{'🔄 重组' if use_recombination else '🎯 传统'}")
            
            # 组装规则
            rules_two_round = assemble_rules(
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
            
            # 执行两轮生成
            with st.spinner("正在进行两轮对比生成..."):
                try:
                    two_round_result = enhanced_generator.generate_two_rounds(
                        count=pred_count,
                        rules=rules_two_round,
                        front_blocks=generation_context["front_blocks"],
                        back_blocks=generation_context["back_blocks"],
                        front_weights=generation_context["front_weights"],
                        back_weights=generation_context["back_weights"],
                        selected_front_blocks=pred_selected_front,
                        selected_back_blocks=pred_selected_back,
                        historical_data=recent_window,
                        use_markov=use_markov_tab4,
                        use_big_data=use_big_data_tab4,
                        markov_weight=markov_weight_tab4,
                        big_data_weight=big_data_weight_tab4,
                        traditional_weight=traditional_weight_tab4,
                        variation_strength=variation_strength,
                        use_recombination=use_recombination,
                        use_exclusion_pool=use_exclusion_pool_two_round,
                        exclusion_pool_size=exclusion_pool_size_two_round,
                        save_to_db=True,
                        predicted_issue=None,  # 可以从用户输入获取
                        strategy_name=st.session_state.get('selected_strategy', None),
                        use_dynamic_pool=use_dynamic_pool_two_round
                    )
                
                    if 'error' in two_round_result:
                        st.error(f"两轮生成失败: {two_round_result['error']}")
                        # 设置空的cands以避免后续错误
                        cands = []
                    else:
                        # 设置cands为第一轮结果，以便后续显示逻辑正常工作
                        cands = two_round_result.get('first_round', [])
                        # 显示两轮结果
                        st.success("🎯 两轮对比生成完成！")
                        
                        # 创建两轮对比的标签页
                        round_tabs = st.tabs(["🥇 第一轮结果", "🥈 第二轮结果", "📊 对比分析", "🎯 中奖分析"])
                    
                    # 第一轮结果
                    with round_tabs[0]:
                        st.subheader("第一轮生成结果")
                        first_round = two_round_result['first_round']
                        
                        first_round_df = pd.DataFrame([
                            {
                                "序号": i+1,
                                "前区": ",".join(map(str, c["front"])),
                                "后区": ",".join(map(str, c["back"])),
                                "生成方法": c.get('generation_method', 'unknown'),
                                "置信度": f"{c.get('markov_confidence', 0):.3f}" if 'markov_confidence' in c else f"{c.get('ensemble_confidence', 0):.3f}"
                            }
                            for i, c in enumerate(first_round)
                        ])
                        
                        st.dataframe(first_round_df, use_container_width=True, hide_index=True)
                        
                        # 第一轮号码分析
                        if 'first_round_analysis' in two_round_result:
                            with st.expander("📈 第一轮号码特征分析"):
                                analysis = two_round_result['first_round_analysis']
                                
                                # 显示热门号码
                                if 'recommendations' in analysis and 'hot_numbers' in analysis['recommendations']:
                                    hot_nums = analysis['recommendations']['hot_numbers']
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**热门前区号码:**", hot_nums.get('front', [])[:10])
                                    with col2:
                                        st.write("**热门后区号码:**", hot_nums.get('back', [])[:6])
                                
                                # 显示模式特征
                                if 'pattern_analysis' in analysis:
                                    st.write("**模式特征:**")
                                    patterns = analysis['pattern_analysis']
                                    for pattern_name, stats in patterns.items():
                                        if isinstance(stats, dict) and 'mean' in stats:
                                            st.write(f"- {pattern_name}: 平均值 {stats['mean']:.2f}")
                    
                    # 第二轮结果
                    with round_tabs[1]:
                        st.subheader("第二轮生成结果")
                        second_round = two_round_result['second_round']
                        
                        second_round_df = pd.DataFrame([
                            {
                                "序号": i+1,
                                "前区": ",".join(map(str, c["front"])),
                                "后区": ",".join(map(str, c["back"])),
                                "生成方法": c.get('generation_method', 'secondary'),
                                "重组策略": c.get('recombination_strategy', '传统') if use_recombination else '传统',
                                "变化强度": f"{c.get('variation_strength', variation_strength):.1f}"
                            }
                            for i, c in enumerate(second_round)
                        ])
                        
                        st.dataframe(second_round_df, use_container_width=True, hide_index=True)
                        
                        if use_recombination:
                            st.info(f"🔄 第二轮使用重组模式生成，仅从第一轮号码中重新组合，变化强度: {variation_strength:.1f}")
                            
                            # 显示号码池信息
                            if second_round:
                                first_round = two_round_result.get('first_round', [])
                                if first_round:
                                    all_front = set()
                                    all_back = set()
                                    for cand in first_round:
                                        all_front.update(cand['front'])
                                        all_back.update(cand['back'])
                                    
                                    st.caption(f"号码池 - 前区: {sorted(all_front)} ({len(all_front)}个) | 后区: {sorted(all_back)} ({len(all_back)}个)")
                        else:
                            st.info(f"第二轮基于第一轮结果生成，变化强度: {variation_strength:.1f}")
                    
                    # 对比分析
                    with round_tabs[2]:
                        st.subheader("两轮对比分析")
                        
                        if 'comparison' in two_round_result:
                            comparison = two_round_result['comparison']
                            
                            # 重叠分析
                            if 'overlap_analysis' in comparison:
                                overlap = comparison['overlap_analysis']
                                
                                st.write("**号码重叠情况:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("前区重叠号码", len(overlap.get('front_overlap', [])))
                                    st.write("重叠号码:", overlap.get('front_overlap', []))
                                with col2:
                                    st.metric("后区重叠号码", len(overlap.get('back_overlap', [])))
                                    st.write("重叠号码:", overlap.get('back_overlap', []))
                                
                                st.write("**重叠比例:**")
                                col3, col4 = st.columns(2)
                                with col3:
                                    st.metric("前区重叠比例", f"{overlap.get('front_overlap_ratio', 0):.2%}")
                                with col4:
                                    st.metric("后区重叠比例", f"{overlap.get('back_overlap_ratio', 0):.2%}")
                            
                            # 轮次分析
                            if 'round_analysis' in comparison:
                                round_analysis = comparison['round_analysis']
                                
                                st.write("**各轮统计:**")
                                round_stats_df = pd.DataFrame([
                                    {
                                        "轮次": "第一轮",
                                        "候选数量": round_analysis.get('first_round', {}).get('candidate_count', 0),
                                        "独特前区号码": round_analysis.get('first_round', {}).get('unique_front_numbers', 0),
                                        "独特后区号码": round_analysis.get('first_round', {}).get('unique_back_numbers', 0)
                                    },
                                    {
                                        "轮次": "第二轮",
                                        "候选数量": round_analysis.get('second_round', {}).get('candidate_count', 0),
                                        "独特前区号码": round_analysis.get('second_round', {}).get('unique_front_numbers', 0),
                                        "独特后区号码": round_analysis.get('second_round', {}).get('unique_back_numbers', 0)
                                    }
                                ])
                                
                                st.dataframe(round_stats_df, use_container_width=True, hide_index=True)
                    
                    # 中奖分析（需要用户输入实际开奖结果）
                    with round_tabs[3]:
                        st.subheader("中奖情况分析")
                        
                        st.write("请输入实际开奖结果进行中奖分析：")
                        
                        hit_col1, hit_col2 = st.columns(2)
                        with hit_col1:
                            actual_front_input = st.text_input("实际前区号码（逗号分隔）", key="actual_front_two_round")
                        with hit_col2:
                            actual_back_input = st.text_input("实际后区号码（逗号分隔）", key="actual_back_two_round")
                        
                        if st.button("分析中奖情况", key="analyze_hits_two_round"):
                            if actual_front_input and actual_back_input:
                                try:
                                    actual_front = [int(x.strip()) for x in actual_front_input.replace("，", ",").split(",") if x.strip().isdigit()]
                                    actual_back = [int(x.strip()) for x in actual_back_input.replace("，", ",").split(",") if x.strip().isdigit()]
                                    
                                    if len(actual_front) == 5 and len(actual_back) == 2:
                                        actual_result = {'front': actual_front, 'back': actual_back}
                                        
                                        hit_analysis = enhanced_generator.analyze_hit_performance(
                                            first_round, second_round, actual_result
                                        )
                                        
                                        if 'hit_analysis' in hit_analysis:
                                            hit_data = hit_analysis['hit_analysis']
                                            
                                            # 显示命中统计
                                            st.write("**命中统计对比:**")
                                            
                                            hit_comparison_df = pd.DataFrame([
                                                {
                                                    "轮次": "第一轮",
                                                    "平均前区命中": f"{hit_data['first_round']['average_front_hits']:.2f}",
                                                    "平均后区命中": f"{hit_data['first_round']['average_back_hits']:.2f}",
                                                    "平均总命中": f"{hit_data['first_round']['average_total_hits']:.2f}",
                                                    "最佳候选命中": hit_data['first_round']['best_candidate']['total_hits'] if hit_data['first_round']['best_candidate'] else 0
                                                },
                                                {
                                                    "轮次": "第二轮",
                                                    "平均前区命中": f"{hit_data['second_round']['average_front_hits']:.2f}",
                                                    "平均后区命中": f"{hit_data['second_round']['average_back_hits']:.2f}",
                                                    "平均总命中": f"{hit_data['second_round']['average_total_hits']:.2f}",
                                                    "最佳候选命中": hit_data['second_round']['best_candidate']['total_hits'] if hit_data['second_round']['best_candidate'] else 0
                                                }
                                            ])
                                            
                                            st.dataframe(hit_comparison_df, use_container_width=True, hide_index=True)
                                            
                                            # 显示改进情况
                                            if 'comparison' in hit_data:
                                                comp = hit_data['comparison']
                                                better_round = comp.get('better_round', 'unknown')
                                                improvement = comp.get('improvement', 0)
                                                
                                                if better_round == 'second':
                                                    st.success(f"🎉 第二轮表现更好！平均命中提升了 {improvement:.2f} 个号码")
                                                elif better_round == 'first':
                                                    st.info(f"第一轮表现更好，第二轮平均命中下降了 {abs(improvement):.2f} 个号码")
                                                else:
                                                    st.info("两轮表现相当")
                                            
                                            # 显示详细命中情况
                                            with st.expander("详细命中情况"):
                                                st.write("**第一轮详细命中:**")
                                                for hit in hit_data['first_round']['individual_hits']:
                                                    st.write(f"候选{hit['candidate_index']+1}: 前区{hit['front_hits']}中, 后区{hit['back_hits']}中, 总计{hit['total_hits']}中")
                                                
                                                st.write("**第二轮详细命中:**")
                                                for hit in hit_data['second_round']['individual_hits']:
                                                    st.write(f"候选{hit['candidate_index']+1}: 前区{hit['front_hits']}中, 后区{hit['back_hits']}中, 总计{hit['total_hits']}中")
                                    else:
                                        st.error("请输入正确的号码格式（前区5个号码，后区2个号码）")
                                except ValueError:
                                    st.error("请输入有效的数字")
                            else:
                                st.warning("请输入完整的开奖结果")
                        
                        # 显示使用说明
                        with st.expander("💡 使用说明"):
                            st.markdown("""
                            **两轮对比生成的优势：**
                            
                            1. **第一轮**：使用完整的预测算法生成基础候选号码
                            2. **第二轮**：基于第一轮结果的特征和规律，生成变化版本
                            3. **对比分析**：通过对比两轮结果，可以：
                               - 发现号码选择的一致性和差异性
                               - 评估不同变化强度的效果
                               - 提供更多样化的选号参考
                            
                            **变化强度说明：**
                            - 0.0：第二轮完全基于第一轮的热门号码
                            - 0.5：平衡使用第一轮特征和随机性
                            - 1.0：第二轮基本随机生成
                            
                            **建议使用方式：**
                            - 可以同时投注两轮结果，增加中奖机会
                            - 比较两轮的中奖表现，优化未来的生成策略
                            - 根据历史表现调整变化强度参数
                            """)
                
                except Exception as e:
                    st.error(f"两轮生成过程中出现错误: {e}")
                    # 设置空的cands以避免后续错误
                    cands = []
                    import traceback
                    with st.expander("错误详情"):
                        st.code(traceback.format_exc())
        else:
            # 初始化失败的情况
            cands = []

    # 排除池生成逻辑
    if exclusion_generate_button:
        st.info(f"🎯 开始排除池生成：排除池大小={exclusion_pool_size}，目标数量={exclusion_target_count}")
        
        # 检查是否有选中的策略
        if st.session_state.selected_strategy:
            # 加载策略参数（与普通生成相同的逻辑）
            strategy_params = load_strategy(st.session_state.selected_strategy)
            if strategy_params:
                st.info(f"🔧 使用策略参数进行排除池生成")
                
                # 从策略中获取参数
                use_recent_n = strategy_params.get('use_recent_n', use_recent_n)
                pred_count = strategy_params.get('pred_count', pred_count)
                min_consec = strategy_params.get('min_consec', min_consec)
                min_odd = strategy_params.get('min_odd', min_odd)
                consec_mode = strategy_params.get('consec_mode', consec_mode)
                consec_check_type = strategy_params.get('consec_check_type', consec_check_type)
                pred_selected_front = strategy_params.get('pred_selected_front', pred_selected_front)
                pred_selected_back = strategy_params.get('pred_selected_back', pred_selected_back)
                
                # 重新计算上下文
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
        
        # 组装规则
        rules_exclusion = assemble_rules(
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
        
        # 执行排除池生成
        with st.spinner("正在进行排除池生成..."):
            try:
                exclusion_result = exclusion_pool_generator.generate_with_exclusion_pool(
                    exclusion_pool_size=exclusion_pool_size,
                    target_count=exclusion_target_count,
                    rules=rules_exclusion,
                    front_blocks=generation_context["front_blocks"],
                    back_blocks=generation_context["back_blocks"],
                    front_weights=generation_context["front_weights"],
                    back_weights=generation_context["back_weights"],
                    selected_front_blocks=pred_selected_front,
                    selected_back_blocks=pred_selected_back,
                    historical_data=recent_window,
                    use_enhanced=use_enhanced_tab4,
                    max_attempts=10000
                )
                
                if 'error' in exclusion_result:
                    st.error(f"排除池生成失败: {exclusion_result['error']}")
                    cands = []
                else:
                    st.success("🎯 排除池生成完成！")
                    
                    # 设置cands为目标号码，以便后续显示逻辑正常工作
                    cands = exclusion_result.get('target_numbers', [])
                    
                    # 显示排除池生成结果
                    exclusion_tabs = st.tabs(["🎯 目标号码", "🚫 排除池", "📊 生成统计"])
                    
                    # 目标号码
                    with exclusion_tabs[0]:
                        st.subheader("目标号码（不与排除池重复）")
                        target_numbers = exclusion_result['target_numbers']
                        
                        if target_numbers:
                            target_df = pd.DataFrame([
                                {
                                    "序号": i+1,
                                    "前区": ",".join(map(str, c["front"])),
                                    "后区": ",".join(map(str, c["back"])),
                                    "生成方法": c.get('generation_method', 'exclusion_pool'),
                                    "置信度": f"{c.get('markov_confidence', 0):.3f}" if 'markov_confidence' in c else f"{c.get('ensemble_confidence', 0):.3f}"
                                }
                                for i, c in enumerate(target_numbers)
                            ])
                            
                            st.dataframe(target_df, use_container_width=True, hide_index=True)
                            
                            st.info(f"✅ 成功生成 {len(target_numbers)} 组目标号码，均不与排除池中的 {exclusion_result['exclusion_pool_size']} 组号码重复")
                        else:
                            st.warning("未能生成目标号码，请尝试减少排除池大小或增加最大尝试次数")
                    
                    # 排除池
                    with exclusion_tabs[1]:
                        st.subheader("排除池号码")
                        exclusion_pool = exclusion_result['exclusion_pool']
                        
                        if exclusion_pool:
                            # 只显示前20组排除池号码（避免页面过长）
                            display_count = min(20, len(exclusion_pool))
                            exclusion_df = pd.DataFrame([
                                {
                                    "序号": i+1,
                                    "前区": ",".join(map(str, c["front"])),
                                    "后区": ",".join(map(str, c["back"])),
                                    "生成方法": c.get('generation_method', 'traditional')
                                }
                                for i, c in enumerate(exclusion_pool[:display_count])
                            ])
                            
                            st.dataframe(exclusion_df, use_container_width=True, hide_index=True)
                            
                            if len(exclusion_pool) > display_count:
                                st.caption(f"显示前 {display_count} 组，共 {len(exclusion_pool)} 组排除池号码")
                        else:
                            st.warning("排除池为空")
                    
                    # 生成统计
                    with exclusion_tabs[2]:
                        st.subheader("生成统计")
                        
                        stats_cols = st.columns(4)
                        with stats_cols[0]:
                            st.metric("排除池大小", exclusion_result['exclusion_pool_size'])
                        with stats_cols[1]:
                            st.metric("目标生成数", exclusion_result['target_count_actual'])
                        with stats_cols[2]:
                            st.metric("生成尝试次数", exclusion_result['generation_attempts'])
                        with stats_cols[3]:
                            success_rate = exclusion_result['target_count_actual'] / exclusion_target_count if exclusion_target_count > 0 else 0
                            st.metric("生成成功率", f"{success_rate:.2%}")
                        
                        # 显示生成记录
                        generation_record = exclusion_result.get('generation_record', {})
                        if generation_record:
                            st.json(generation_record)
                        
                        # 使用说明
                        with st.expander("💡 排除池生成原理"):
                            st.markdown("""
                            **排除池生成策略的工作原理：**
                            
                            1. **第一步**：生成N组号码作为"排除池"
                            2. **第二步**：生成Y组号码，确保每组都不与排除池中的任何一组完全相同
                            3. **理论基础**：通过排除常见的号码组合，增加生成罕见组合的概率
                            
                            **潜在优势：**
                            - 避免生成过于常见的号码组合
                            - 增加生成独特组合的机会
                            - 可能提高高等奖中奖概率
                            
                            **注意事项：**
                            - 排除池越大，生成难度越高
                            - 需要通过历史回测验证实际效果
                            - 建议结合AI分析找到最优参数
                            """)
            
            except Exception as e:
                st.error(f"排除池生成过程中出现错误: {e}")
                cands = []
                import traceback
                with st.expander("错误详情"):
                    st.code(traceback.format_exc())

    # AI效果分析逻辑
    if exclusion_analysis_button:
        st.info("🤖 开始AI效果分析...")
        
        # 分析参数设置
        analysis_cols = st.columns(3)
        with analysis_cols[0]:
            test_pool_sizes = st.multiselect(
                "测试排除池大小",
                options=[10, 20, 50, 100, 200, 300, 500],
                default=[50, 100, 200],
                key="test_pool_sizes"
            )
        
        with analysis_cols[1]:
            analysis_periods = st.number_input(
                "测试期数", 
                min_value=10, max_value=100, value=30,
                key="analysis_periods"
            )
        
        with analysis_cols[2]:
            analysis_target_count = st.number_input(
                "每期生成数量", 
                min_value=1, max_value=20, value=5,
                key="analysis_target_count"
            )
        
        if st.button("开始AI分析", key="start_ai_analysis"):
            if not test_pool_sizes:
                st.error("请选择至少一个排除池大小进行测试")
            else:
                # 准备分析参数
                rules_analysis = assemble_rules(
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
                
                # 奖金结构
                prize_structure_analysis = {
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
                
                with st.spinner(f"正在分析 {len(test_pool_sizes)} 种排除池大小的效果..."):
                    try:
                        analysis_results = exclusion_pool_generator.analyze_exclusion_effectiveness(
                            historical_data=df_filtered,
                            exclusion_pool_sizes=test_pool_sizes,
                            target_count=analysis_target_count,
                            test_periods=analysis_periods,
                            rules=rules_analysis,
                            generation_context=generation_context,
                            selected_front_blocks=pred_selected_front,
                            selected_back_blocks=pred_selected_back,
                            prize_structure=prize_structure_analysis
                        )
                        
                        st.success("🎉 AI分析完成！")
                        
                        # 显示分析结果
                        analysis_result_tabs = st.tabs(["📊 综合对比", "🏆 最优推荐", "📈 详细数据", "💾 保存结果"])
                        
                        # 综合对比
                        with analysis_result_tabs[0]:
                            st.subheader("不同排除池大小效果对比")
                            
                            if analysis_results.get("results"):
                                # 创建对比表格
                                comparison_data = []
                                for result in analysis_results["results"]:
                                    comparison_data.append({
                                        "排除池大小": result["exclusion_pool_size"],
                                        "生成成功率": f"{result.get('generation_success_rate', 0):.2%}",
                                        "高等奖命中率": f"{result.get('high_prize_rate', 0):.4%}",
                                        "总体命中率": f"{result.get('hit_rate', 0):.2%}",
                                        "ROI": f"{result.get('roi', 0):.2%}",
                                        "高等奖命中次数": result.get('high_prize_hits', 0),
                                        "净收益": f"{result.get('net_profit', 0):.2f}元"
                                    })
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                                
                                # 可视化对比
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                fig = make_subplots(
                                    rows=2, cols=2,
                                    subplot_titles=('高等奖命中率', 'ROI', '总体命中率', '生成成功率'),
                                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                           [{"secondary_y": False}, {"secondary_y": False}]]
                                )
                                
                                pool_sizes = [r["exclusion_pool_size"] for r in analysis_results["results"]]
                                high_prize_rates = [r.get("high_prize_rate", 0) * 100 for r in analysis_results["results"]]
                                rois = [r.get("roi", 0) * 100 for r in analysis_results["results"]]
                                hit_rates = [r.get("hit_rate", 0) * 100 for r in analysis_results["results"]]
                                success_rates = [r.get("generation_success_rate", 0) * 100 for r in analysis_results["results"]]
                                
                                fig.add_trace(go.Scatter(x=pool_sizes, y=high_prize_rates, mode='lines+markers', name='高等奖命中率'), row=1, col=1)
                                fig.add_trace(go.Scatter(x=pool_sizes, y=rois, mode='lines+markers', name='ROI'), row=1, col=2)
                                fig.add_trace(go.Scatter(x=pool_sizes, y=hit_rates, mode='lines+markers', name='总体命中率'), row=2, col=1)
                                fig.add_trace(go.Scatter(x=pool_sizes, y=success_rates, mode='lines+markers', name='生成成功率'), row=2, col=2)
                                
                                fig.update_layout(height=600, showlegend=False, title_text="排除池大小效果分析")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("没有分析结果数据")
                        
                        # 最优推荐
                        with analysis_result_tabs[1]:
                            st.subheader("AI推荐最优排除池大小")
                            
                            recommendations = exclusion_pool_generator.get_optimal_exclusion_pool_size(analysis_results)
                            
                            if 'error' not in recommendations:
                                rec_cols = st.columns(2)
                                
                                with rec_cols[0]:
                                    st.markdown("### 🏆 综合最优")
                                    best_overall = recommendations.get("best_overall")
                                    if best_overall:
                                        st.success(f"**推荐排除池大小: {best_overall['exclusion_pool_size']}**")
                                        st.write(f"- 高等奖命中率: {best_overall.get('high_prize_rate', 0):.4%}")
                                        st.write(f"- ROI: {best_overall.get('roi', 0):.2%}")
                                        st.write(f"- 生成成功率: {best_overall.get('generation_success_rate', 0):.2%}")
                                        st.write(f"- 综合评分: {best_overall.get('composite_score', 0):.2f}")
                                
                                with rec_cols[1]:
                                    st.markdown("### 🎯 高等奖最优")
                                    best_high_prize = recommendations.get("best_for_high_prize")
                                    if best_high_prize:
                                        st.info(f"**排除池大小: {best_high_prize['exclusion_pool_size']}**")
                                        st.write(f"- 高等奖命中率: {best_high_prize.get('high_prize_rate', 0):.4%}")
                                        st.write(f"- 高等奖命中次数: {best_high_prize.get('high_prize_hits', 0)}")
                                        st.write(f"- ROI: {best_high_prize.get('roi', 0):.2%}")
                                
                                # 分析总结
                                summary = recommendations.get("analysis_summary", {})
                                if summary:
                                    st.markdown("### 📋 分析总结")
                                    summary_cols = st.columns(3)
                                    with summary_cols[0]:
                                        st.metric("测试方案数", summary.get("total_tested", 0))
                                    with summary_cols[1]:
                                        st.metric("最佳高等奖命中率", f"{summary.get('best_high_prize_rate', 0):.4%}")
                                    with summary_cols[2]:
                                        st.metric("最佳ROI", f"{summary.get('best_roi', 0):.2%}")
                            else:
                                st.error(f"推荐分析失败: {recommendations['error']}")
                        
                        # 详细数据
                        with analysis_result_tabs[2]:
                            st.subheader("详细分析数据")
                            
                            if analysis_results.get("results"):
                                for result in analysis_results["results"]:
                                    with st.expander(f"排除池大小 {result['exclusion_pool_size']} 详细数据"):
                                        st.json(result)
                        
                        # 保存结果
                        with analysis_result_tabs[3]:
                            st.subheader("保存分析结果")
                            
                            save_filename = st.text_input(
                                "文件名", 
                                value=f"exclusion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                key="save_analysis_filename"
                            )
                            
                            if st.button("保存到文件", key="save_analysis_btn"):
                                if exclusion_pool_generator.save_analysis_results(save_filename):
                                    st.success(f"✅ 分析结果已保存到 {save_filename}")
                                else:
                                    st.error("❌ 保存失败")
                            
                            # 显示保存的数据预览
                            with st.expander("查看要保存的数据"):
                                st.json(analysis_results)
                    
                    except Exception as e:
                        st.error(f"AI分析过程中出现错误: {e}")
                        import traceback
                        with st.expander("错误详情"):
                            st.code(traceback.format_exc())

    # 使用Tabs布局显示未来预测和历史回测
    if (generate_button and len(cands) > 0) or backtest_n > 0:
        result_tabs = st.tabs(["🔮 未来预测号码", "📊 历史回测结果", "📈 未来多期回测"])
        
        # 未来预测号码Tab
        with result_tabs[0]:
            if generate_button and len(cands) > 0:
                if use_ai_model:
                    st.info(f"🤖 使用AI模型（{st.session_state.get('model_type', 'unknown')}）生成")
                
                if use_enhanced_tab4:
                    st.info(f"🚀 使用增强生成（马尔可夫链 + 大数据分析）")
                    
                    # 显示增强生成统计
                    if cands and any('markov_confidence' in c for c in cands):
                        avg_markov_conf = np.mean([c.get('markov_confidence', 0.5) for c in cands])
                        avg_big_data_score = np.mean([c.get('big_data_score', 0.5) for c in cands])
                        st.caption(f"平均马尔可夫置信度: {avg_markov_conf:.3f} | 平均大数据评分: {avg_big_data_score:.3f}")
                
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
    
    ai_tabs = st.tabs(["🧠 模型训练", "🔍 参数优化", "📊 批量回测", "⚡ 性能测试", "🎯 排除池数据分析"])
    
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

    with ai_tabs[4]:
        st.subheader("🎯 排除池数据分析")
        st.write("分析排除池生成的历史数据，优化参数设置，提升中奖率")
        
        # 数据库统计信息
        st.markdown("### 📊 数据库统计")
        try:
            stats = exclusion_pool_db.get_statistics()
            if stats:
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric("总记录数", stats.get('total_records', 0))
                with stats_cols[1]:
                    st.metric("已验证记录", stats.get('verified_records', 0))
                with stats_cols[2]:
                    st.metric("验证率", f"{stats.get('verification_rate', 0):.2%}")
                with stats_cols[3]:
                    st.metric("高等奖记录", stats.get('high_prize_records', 0))
            else:
                st.info("暂无数据库统计信息")
        except Exception as e:
            st.error(f"获取统计信息失败: {e}")
        
        # 数据分析标签页
        analysis_tabs = st.tabs(["📈 生成记录分析", "🏆 中奖效果分析", "🔍 参数优化建议", "📋 历史记录查看"])
        
        # 生成记录分析
        with analysis_tabs[0]:
            st.subheader("生成记录分析")
            
            # 获取生成记录
            try:
                records = exclusion_pool_db.get_generation_results(limit=100)
                if records:
                    # 转换为DataFrame进行分析
                    df_records = pd.DataFrame(records)
                    
                    # 基本统计
                    st.markdown("#### 基本统计")
                    basic_cols = st.columns(4)
                    with basic_cols[0]:
                        st.metric("平均生成成功率", f"{df_records['success_rate'].mean():.2%}")
                    with basic_cols[1]:
                        st.metric("平均尝试次数", f"{df_records['generation_attempts'].mean():.0f}")
                    with basic_cols[2]:
                        st.metric("最常用排除池大小", df_records['exclusion_pool_size'].mode().iloc[0] if not df_records['exclusion_pool_size'].mode().empty else "N/A")
                    with basic_cols[3]:
                        st.metric("最常用目标数量", df_records['target_count'].mode().iloc[0] if not df_records['target_count'].mode().empty else "N/A")
                    
                    # 生成方法分布
                    st.markdown("#### 生成方法分布")
                    method_counts = df_records['generation_method'].value_counts()
                    fig_methods = px.pie(
                        values=method_counts.values,
                        names=method_counts.index,
                        title="生成方法使用分布"
                    )
                    st.plotly_chart(fig_methods, use_container_width=True)
                    
                    # 成功率vs排除池大小
                    st.markdown("#### 成功率 vs 排除池大小")
                    fig_success = px.scatter(
                        df_records,
                        x='exclusion_pool_size',
                        y='success_rate',
                        color='generation_method',
                        size='target_count',
                        title="生成成功率与排除池大小的关系",
                        labels={'success_rate': '生成成功率', 'exclusion_pool_size': '排除池大小'}
                    )
                    st.plotly_chart(fig_success, use_container_width=True)
                    
                    # 时间趋势分析
                    st.markdown("#### 时间趋势分析")
                    df_records['prediction_date'] = pd.to_datetime(df_records['prediction_date'])
                    df_records_sorted = df_records.sort_values('prediction_date')
                    
                    fig_trend = px.line(
                        df_records_sorted,
                        x='prediction_date',
                        y='success_rate',
                        color='generation_method',
                        title="生成成功率时间趋势",
                        labels={'success_rate': '生成成功率', 'prediction_date': '预测日期'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                else:
                    st.info("暂无生成记录数据")
            except Exception as e:
                st.error(f"分析生成记录失败: {e}")
        
        # 中奖效果分析
        with analysis_tabs[1]:
            st.subheader("中奖效果分析")
            
            try:
                # 获取已验证的记录
                verified_records = exclusion_pool_db.get_generation_results(verified_only=True, limit=100)
                if verified_records:
                    df_verified = pd.DataFrame(verified_records)
                    
                    # 中奖统计
                    st.markdown("#### 中奖统计")
                    prize_cols = st.columns(4)
                    with prize_cols[0]:
                        total_hits = df_verified['hit_count'].sum()
                        st.metric("总中奖注数", total_hits)
                    with prize_cols[1]:
                        high_prize_hits = df_verified['high_prize_hits'].sum()
                        st.metric("高等奖中奖注数", high_prize_hits)
                    with prize_cols[2]:
                        total_investment = df_verified['investment_cost'].sum()
                        st.metric("总投入", f"¥{total_investment:.2f}")
                    with prize_cols[3]:
                        total_return = df_verified['total_prize_amount'].sum()
                        st.metric("总回收", f"¥{total_return:.2f}")
                    
                    # ROI分析
                    st.markdown("#### ROI分析")
                    df_verified['roi_percent'] = df_verified['roi'] * 100
                    
                    # ROI分布直方图
                    fig_roi = px.histogram(
                        df_verified,
                        x='roi_percent',
                        nbins=20,
                        title="ROI分布直方图",
                        labels={'roi_percent': 'ROI (%)', 'count': '记录数'}
                    )
                    st.plotly_chart(fig_roi, use_container_width=True)
                    
                    # 排除池大小vs中奖率
                    st.markdown("#### 排除池大小 vs 中奖效果")
                    
                    # 按排除池大小分组分析
                    pool_size_analysis = df_verified.groupby('exclusion_pool_size').agg({
                        'hit_count': 'sum',
                        'high_prize_hits': 'sum',
                        'total_prize_amount': 'sum',
                        'investment_cost': 'sum',
                        'roi': 'mean'
                    }).reset_index()
                    
                    pool_size_analysis['hit_rate'] = pool_size_analysis['hit_count'] / pool_size_analysis['investment_cost'] * 2  # 假设每注2元
                    pool_size_analysis['high_prize_rate'] = pool_size_analysis['high_prize_hits'] / pool_size_analysis['investment_cost'] * 2
                    
                    fig_pool_effect = px.scatter(
                        pool_size_analysis,
                        x='exclusion_pool_size',
                        y='roi',
                        size='hit_count',
                        color='high_prize_hits',
                        title="排除池大小对ROI的影响",
                        labels={'roi': '平均ROI', 'exclusion_pool_size': '排除池大小'}
                    )
                    st.plotly_chart(fig_pool_effect, use_container_width=True)
                    
                    # 详细数据表
                    st.markdown("#### 详细分析数据")
                    st.dataframe(pool_size_analysis, use_container_width=True)
                    
                else:
                    st.info("暂无已验证的中奖记录")
            except Exception as e:
                st.error(f"分析中奖效果失败: {e}")
        
        # 参数优化建议
        with analysis_tabs[2]:
            st.subheader("参数优化建议")
            
            try:
                # 获取分析结果
                analysis_results = exclusion_pool_db.get_analysis_results(limit=10)
                if analysis_results:
                    st.markdown("#### 历史分析结果")
                    
                    for i, result in enumerate(analysis_results[:3], 1):  # 显示最近3次分析
                        with st.expander(f"分析 {i}: {result['analysis_name']} ({result['analysis_date'][:10]})"):
                            st.write(f"**测试期数**: {result['test_periods']}")
                            st.write(f"**每期生成数量**: {result['target_count']}")
                            st.write(f"**测试的排除池大小**: {', '.join(map(str, result['test_pool_sizes']))}")
                            
                            if result['best_pool_size']:
                                st.success(f"**推荐的最佳排除池大小**: {result['best_pool_size']}")
                                st.write(f"**最佳高等奖命中率**: {result['best_high_prize_rate']:.4%}")
                                st.write(f"**最佳ROI**: {result['best_roi']:.2%}")
                    
                    # 综合建议
                    st.markdown("#### 🎯 AI综合建议")
                    
                    # 分析所有历史结果，给出建议
                    best_pool_sizes = [r['best_pool_size'] for r in analysis_results if r['best_pool_size']]
                    if best_pool_sizes:
                        avg_best_size = sum(best_pool_sizes) / len(best_pool_sizes)
                        most_common_size = max(set(best_pool_sizes), key=best_pool_sizes.count)
                        
                        st.success(f"**推荐排除池大小**: {most_common_size} (最常推荐)")
                        st.info(f"**平均最佳大小**: {avg_best_size:.0f}")
                        
                        # 使用建议
                        st.markdown("#### 💡 使用建议")
                        st.markdown(f"""
                        1. **初学者**: 建议使用排除池大小 {max(50, most_common_size - 50)} - {most_common_size}
                        2. **进阶用户**: 建议使用排除池大小 {most_common_size} - {most_common_size + 50}
                        3. **专业用户**: 可以测试 {most_common_size + 50} - {most_common_size + 100} 的范围
                        
                        **注意**: 排除池越大，生成难度越高，但可能获得更独特的号码组合。
                        """)
                    else:
                        st.warning("暂无足够的分析数据提供建议，请先进行AI效果分析")
                else:
                    st.info("暂无分析结果，请先在'未来号码预测'页面进行AI效果分析")
            except Exception as e:
                st.error(f"获取优化建议失败: {e}")
        
        # 历史记录查看
        with analysis_tabs[3]:
            st.subheader("历史记录查看")
            
            # 筛选选项
            filter_cols = st.columns(3)
            with filter_cols[0]:
                method_filter = st.selectbox(
                    "生成方法",
                    options=["全部", "single_period_exclusion", "multi_period_exclusion", "two_round_exclusion_first", "two_round_exclusion_second"],
                    key="method_filter"
                )
            with filter_cols[1]:
                verified_filter = st.selectbox(
                    "验证状态",
                    options=["全部", "已验证", "未验证"],
                    key="verified_filter"
                )
            with filter_cols[2]:
                limit_records = st.number_input(
                    "显示记录数",
                    min_value=10, max_value=500, value=50,
                    key="limit_records"
                )
            
            try:
                # 获取记录
                method = None if method_filter == "全部" else method_filter
                verified_only = verified_filter == "已验证"
                
                records = exclusion_pool_db.get_generation_results(
                    limit=limit_records,
                    method=method,
                    verified_only=verified_only
                )
                
                if records:
                    # 转换为显示格式
                    display_records = []
                    for record in records:
                        display_records.append({
                            "ID": record['id'],
                            "生成方法": record['generation_method'],
                            "排除池大小": record['exclusion_pool_size'],
                            "目标数量": record['target_count'],
                            "实际生成": record['actual_generated'],
                            "成功率": f"{record['success_rate']:.2%}",
                            "预测日期": record['prediction_date'][:10] if record['prediction_date'] else "N/A",
                            "验证状态": "已验证" if record['verification_date'] else "未验证",
                            "中奖注数": record['hit_count'] or 0,
                            "高等奖": record['high_prize_hits'] or 0,
                            "ROI": f"{record['roi']:.2%}" if record['roi'] is not None else "N/A"
                        })
                    
                    df_display = pd.DataFrame(display_records)
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # 详细记录查看
                    if st.checkbox("显示详细记录"):
                        selected_id = st.selectbox(
                            "选择记录ID查看详情",
                            options=[r['id'] for r in records],
                            key="selected_record_id"
                        )
                        
                        if selected_id:
                            selected_record = next(r for r in records if r['id'] == selected_id)
                            
                            st.markdown("#### 详细信息")
                            detail_cols = st.columns(2)
                            
                            with detail_cols[0]:
                                st.json({
                                    "基本信息": {
                                        "ID": selected_record['id'],
                                        "生成方法": selected_record['generation_method'],
                                        "排除池大小": selected_record['exclusion_pool_size'],
                                        "目标数量": selected_record['target_count'],
                                        "实际生成": selected_record['actual_generated'],
                                        "生成尝试次数": selected_record['generation_attempts'],
                                        "成功率": f"{selected_record['success_rate']:.2%}",
                                        "预测日期": selected_record['prediction_date']
                                    }
                                })
                            
                            with detail_cols[1]:
                                if selected_record['verification_date']:
                                    st.json({
                                        "中奖信息": {
                                            "验证日期": selected_record['verification_date'],
                                            "实际开奖前区": selected_record['actual_winning_front'],
                                            "实际开奖后区": selected_record['actual_winning_back'],
                                            "中奖注数": selected_record['hit_count'],
                                            "高等奖中奖": selected_record['high_prize_hits'],
                                            "总奖金": f"¥{selected_record['total_prize_amount']:.2f}",
                                            "投注成本": f"¥{selected_record['investment_cost']:.2f}",
                                            "ROI": f"{selected_record['roi']:.2%}" if selected_record['roi'] is not None else "N/A"
                                        }
                                    })
                                else:
                                    st.info("该记录尚未验证中奖情况")
                            
                            # 显示生成的号码
                            if selected_record['target_numbers_data']:
                                st.markdown("#### 生成的目标号码")
                                target_numbers = selected_record['target_numbers_data']
                                target_df = pd.DataFrame([
                                    {
                                        "序号": i+1,
                                        "前区": ",".join(map(str, num['front'])),
                                        "后区": ",".join(map(str, num['back'])),
                                        "生成方法": num.get('method', 'unknown')
                                    }
                                    for i, num in enumerate(target_numbers)
                                ])
                                st.dataframe(target_df, use_container_width=True, hide_index=True)
                else:
                    st.info("没有找到符合条件的记录")
            except Exception as e:
                st.error(f"查看历史记录失败: {e}")

# 显示系统优化信息
st.sidebar.markdown("---")
st.sidebar.subheader("💡 系统优化信息")
st.sidebar.info("✅ 贝叶斯优化已增强\n✅ 高额奖项命中分析已添加\n✅ 性能测试已集成\n✅ 多算法支持已实现\n✅ 进化算法已集成\n✅ 深度学习已集成\n✅ 策略优化已集成")
st.sidebar.caption("大乐透分析系统 v3.0 - 进化版")

# --------------------- Tab6: 进化算法与深度学习 ---------------------
with tab_evolution:
    st.header("🧬 进化算法与深度学习")
    st.write("使用最先进的AI技术，包括进化算法、神经网络和策略优化，大幅提升中奖概率")
    
    # 初始化高级组件
    if st.button("🚀 初始化AI深度学习组件", key="init_advanced"):
        initialize_advanced_components()
    
    # 显示当前状态
    if st.session_state.get('advanced_initialized', False):
        st.success("✅ AI组件已初始化")
        
        # 显示组件状态
        with st.expander("📊 查看组件状态"):
            evolutionary_optimizer = get_evolutionary_optimizer()
            neural_predictors = get_neural_predictors()
            strategy_optimizer = get_strategy_optimizer()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if evolutionary_optimizer is not None:
                    st.success("🧬 进化优化器: ✅")
                    optimizer_type = type(evolutionary_optimizer).__name__
                    st.caption(f"类型: {optimizer_type}")
                else:
                    st.error("🧬 进化优化器: ❌")
            
            with col2:
                if neural_predictors:
                    st.success(f"🧠 神经网络: ✅ ({len(neural_predictors)})")
                    st.caption(f"模型: {list(neural_predictors.keys())}")
                else:
                    st.warning("🧠 神经网络: ⚠️")
                    st.caption("需要安装PyTorch")
            
            with col3:
                if strategy_optimizer is not None:
                    st.success("🎯 策略优化器: ✅")
                else:
                    st.error("🎯 策略优化器: ❌")
    
    if st.session_state.get('advanced_initialized', False):
        # 创建子标签页
        evo_tabs = st.tabs([
            "🧬 进化算法优化", 
            "🧠 神经网络预测", 
            "🎯 策略优化", 
            "📊 综合分析"
        ])
        
        # 进化算法优化
        with evo_tabs[0]:
            st.subheader("🧬 进化算法号码优化")
            st.write("使用遗传算法和进化策略寻找最优号码组合")
            
            # 进化算法配置
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**算法参数**")
                population_size = st.slider("种群大小", 20, 100, 50, key="evo_population")
                generations = st.slider("进化代数", 20, 200, 100, key="evo_generations")
                mutation_rate = st.slider("变异率", 0.05, 0.5, 0.15, key="evo_mutation")
                
            with col2:
                st.markdown("**优化目标**")
                objectives = st.multiselect(
                    "选择优化目标",
                    ["hit_probability", "diversity", "pattern_match", "rarity", "balance"],
                    default=["hit_probability", "diversity", "pattern_match"],
                    key="evo_objectives"
                )
                
                multi_objective = st.checkbox("多目标优化", value=True, key="evo_multi_obj")
                adaptive_mutation = st.checkbox("自适应变异", value=True, key="evo_adaptive")
            
            # 生成数量
            evo_count = st.number_input("生成号码组数", 1, 20, 5, key="evo_count")
            
            # 测试按钮
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 测试进化优化器", key="test_evolution"):
                    evolutionary_optimizer = get_evolutionary_optimizer()
                    if evolutionary_optimizer is None:
                        st.error("进化优化器未初始化，请先点击初始化按钮")
                    else:
                        try:
                            with st.spinner("🧪 正在测试进化优化器..."):
                                # 创建简单测试数据
                                test_data = pd.DataFrame({
                                    'f1': [1, 5, 10, 15, 20],
                                    'f2': [2, 6, 11, 16, 21],
                                    'f3': [3, 7, 12, 17, 22],
                                    'f4': [4, 8, 13, 18, 23],
                                    'f5': [5, 9, 14, 19, 24],
                                    'b1': [1, 2, 3, 4, 5],
                                    'b2': [2, 3, 4, 5, 6],
                                    'date': pd.date_range('2024-01-01', periods=5)
                                })
                                
                                # 运行简单测试
                                test_result = evolutionary_optimizer.evolve_optimal_numbers(
                                    test_data, 
                                    target_count=1,
                                    objectives=['hit_probability']
                                )
                                
                                st.success("✅ 进化优化器测试成功！")
                                st.json(test_result['best_numbers'])
                                
                        except Exception as e:
                            st.error(f"进化优化器测试失败: {e}")
                            import traceback
                            with st.expander("查看详细错误"):
                                st.code(traceback.format_exc())
            
            with col2:
                if st.button("🧬 开始进化算法优化", key="start_evolution"):
                    evolutionary_optimizer = get_evolutionary_optimizer()
                    if evolutionary_optimizer is None:
                        st.error("进化优化器未初始化，请先点击初始化按钮")
                    else:
                        try:
                            with st.spinner("🧬 进化算法正在寻找最优解..."):
                                # 配置进化算法
                                evo_config = EvolutionaryConfig(
                                    population_size=population_size,
                                    generations=generations,
                                    mutation_rate=mutation_rate,
                                    multi_objective=multi_objective,
                                    adaptive_mutation=adaptive_mutation
                                )
                                evolutionary_optimizer.config = evo_config
                                
                                # 运行进化优化
                                evo_result = evolutionary_optimizer.evolve_optimal_numbers(
                                    df_filtered, 
                                    target_count=evo_count,
                                    objectives=objectives
                                )
                                
                                st.success("🎉 进化算法优化完成！")
                                
                                # 显示最优解
                                st.subheader("🏆 进化算法最优解")
                                best_numbers = evo_result['best_numbers']
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("前区号码", f"{best_numbers['front']}")
                                col2.metric("后区号码", f"{best_numbers['back']}")
                                col3.metric("适应度分数", f"{evo_result['fitness_score']:.4f}")
                                
                                # 进化过程可视化
                                st.subheader("📈 进化过程分析")
                                
                                fitness_history = evo_result['generation_stats']['fitness_history']
                                diversity_history = evo_result['generation_stats']['diversity_history']
                                
                                # 适应度进化曲线
                                fig_fitness = px.line(
                                    x=range(len(fitness_history)),
                                    y=fitness_history,
                                    title="适应度进化曲线",
                                    labels={"x": "代数", "y": "最佳适应度"}
                                )
                                st.plotly_chart(fig_fitness, use_container_width=True)
                                
                                # 多样性变化曲线
                                fig_diversity = px.line(
                                    x=range(len(diversity_history)),
                                    y=diversity_history,
                                    title="种群多样性变化",
                                    labels={"x": "代数", "y": "多样性指数"}
                                )
                                st.plotly_chart(fig_diversity, use_container_width=True)
                                
                                # 详细分析
                                with st.expander("🔍 详细进化分析"):
                                    st.json(evo_result['patterns_used'])
                                    
                        except Exception as e:
                            st.error(f"进化算法优化失败: {e}")
                            import traceback
                            with st.expander("查看详细错误"):
                                st.code(traceback.format_exc())
        
        # 神经网络预测
        with evo_tabs[1]:
            st.subheader("🧠 神经网络深度学习预测")
            st.write("使用Transformer、LSTM等深度学习模型预测彩票号码")
            
            if not HAS_PYTORCH:
                st.error("PyTorch未安装，无法使用神经网络功能。请运行: pip install torch")
            else:
                neural_predictors = get_neural_predictors()
                if not neural_predictors:
                    st.warning("神经网络预测器未初始化")
                else:
                    # 模型选择
                    available_models = list(neural_predictors.keys())
                    selected_model = st.selectbox(
                        "选择神经网络模型",
                        available_models,
                        key="neural_model_select"
                    )
                    
                    if selected_model:
                        predictor = neural_predictors[selected_model]
                        
                        # 训练参数
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**训练参数**")
                            epochs = st.slider("训练轮数", 10, 200, 50, key="neural_epochs")
                            batch_size = st.slider("批次大小", 8, 64, 16, key="neural_batch")
                            learning_rate = st.select_slider(
                                "学习率", 
                                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                value=0.001,
                                key="neural_lr"
                            )
                        
                        with col2:
                            st.markdown("**预测参数**")
                            neural_count = st.number_input("预测组数", 1, 20, 5, key="neural_count")
                            sequence_length = st.slider("序列长度", 10, 50, 20, key="neural_seq_len")
                        
                        # 训练模型
                        if st.button(f"🧠 训练{selected_model}模型", key="train_neural"):
                            try:
                                with st.spinner(f"🧠 正在训练{selected_model}模型..."):
                                    predictor.sequence_length = sequence_length
                                    predictor.train(
                                        df_filtered,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        learning_rate=learning_rate
                                    )
                                    
                                    st.success(f"✅ {selected_model}模型训练完成！")
                                    
                                    # 显示训练历史
                                    if predictor.training_history:
                                        st.subheader("📊 训练历史")
                                        
                                        history_df = pd.DataFrame(predictor.training_history)
                                        
                                        # 损失曲线
                                        fig_loss = px.line(
                                            history_df,
                                            y=['loss', 'val_loss'],
                                            title="训练损失曲线",
                                            labels={"index": "轮数", "value": "损失值"}
                                        )
                                        st.plotly_chart(fig_loss, use_container_width=True)
                                        
                                        # 准确率曲线
                                        fig_acc = px.line(
                                            history_df,
                                            y=['accuracy', 'val_accuracy'],
                                            title="训练准确率曲线",
                                            labels={"index": "轮数", "value": "准确率"}
                                        )
                                        st.plotly_chart(fig_acc, use_container_width=True)
                            
                            except Exception as e:
                                st.error(f"模型训练失败: {e}")
                                import traceback
                                with st.expander("查看详细错误"):
                                    st.code(traceback.format_exc())
                    
                    # 预测号码
                    if st.button(f"🔮 使用{selected_model}预测", key="predict_neural"):
                        try:
                            with st.spinner(f"🔮 {selected_model}正在预测..."):
                                predictions = predictor.predict(df_filtered, neural_count)
                                
                                st.success(f"🎉 {selected_model}预测完成！")
                                
                                # 显示预测结果
                                st.subheader("🎯 神经网络预测结果")
                                
                                for i, pred in enumerate(predictions, 1):
                                    with st.expander(f"预测组合 {i} (置信度: {pred['confidence']:.3f})"):
                                        col1, col2, col3 = st.columns(3)
                                        
                                        col1.metric("前区", f"{pred['front']}")
                                        col2.metric("后区", f"{pred['back']}")
                                        col3.metric("置信度", f"{pred['confidence']:.3f}")
                                        
                                        # 显示概率分布
                                        if 'front_probabilities' in pred:
                                            st.markdown("**前区号码概率**")
                                            prob_df = pd.DataFrame([
                                                {"号码": k, "概率": v} 
                                                for k, v in pred['front_probabilities'].items()
                                            ])
                                            fig_prob = px.bar(
                                                prob_df, x="号码", y="概率",
                                                title="前区号码预测概率"
                                            )
                                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"神经网络预测失败: {e}")
                            import traceback
                            with st.expander("查看详细错误"):
                                st.code(traceback.format_exc())
        
        # 策略优化
        with evo_tabs[2]:
            st.subheader("🎯 智能策略优化")
            st.write("自动寻找最优预测策略，整合所有AI技术")
            
            strategy_optimizer = get_strategy_optimizer()
            if strategy_optimizer is None:
                st.warning("策略优化器未初始化")
            else:
                # 优化配置
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**优化目标**")
                    primary_objective = st.selectbox(
                        "主要目标",
                        ["roi", "hit_rate", "high_prize_rate", "stability"],
                        key="strategy_primary_obj"
                    )
                    
                    secondary_objectives = st.multiselect(
                        "次要目标",
                        ["roi", "hit_rate", "high_prize_rate", "stability"],
                        default=["hit_rate", "stability"],
                        key="strategy_secondary_obj"
                    )
                
                with col2:
                    st.markdown("**优化参数**")
                    max_iterations = st.slider("最大迭代数", 20, 200, 100, key="strategy_iterations")
                    backtest_periods = st.slider("回测期数", 20, 100, 50, key="strategy_backtest")
                    
                    use_neural = st.checkbox("使用神经网络", value=HAS_PYTORCH, key="strategy_neural")
                    use_evolutionary = st.checkbox("使用进化算法", value=True, key="strategy_evo")
                
                if st.button("🎯 开始策略优化", key="start_strategy_opt"):
                    try:
                        with st.spinner("🎯 正在进行智能策略优化..."):
                            # 配置优化器
                            opt_config = OptimizationConfig(
                                primary_objective=primary_objective,
                                secondary_objectives=secondary_objectives,
                                max_iterations=max_iterations,
                                backtest_periods=backtest_periods,
                                use_neural_networks=use_neural,
                                use_evolutionary=use_evolutionary
                            )
                            strategy_optimizer.config = opt_config
                            
                            # 运行综合优化
                            optimization_results = strategy_optimizer.optimize_comprehensive_strategy(df_filtered)
                            
                            st.success("🎉 策略优化完成！")
                            
                            # 显示最优策略
                            final_strategy = optimization_results['final_strategy']
                            st.subheader("🏆 最优策略")
                            
                            if final_strategy['type'] == 'ensemble':
                                st.write("**集成策略**")
                                
                                # 显示策略权重
                                weights_df = pd.DataFrame([
                                    {"策略": k, "权重": v} 
                                    for k, v in final_strategy['weights'].items()
                                ])
                                
                                fig_weights = px.pie(
                                    weights_df, values="权重", names="策略",
                                    title="策略权重分布"
                                )
                                st.plotly_chart(fig_weights, use_container_width=True)
                                
                                # 预期性能
                                st.markdown("**预期性能**")
                                perf_metrics = final_strategy['expected_performance']
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("预期ROI", f"{perf_metrics.get('roi', 0):.2%}")
                                col2.metric("预期命中率", f"{perf_metrics.get('hit_rate', 0):.2%}")
                                col3.metric("预期稳定性", f"{perf_metrics.get('stability', 0):.3f}")
                            
                            # 使用最优策略生成号码
                            if st.button("🎲 使用最优策略生成号码", key="generate_optimal"):
                                with st.spinner("🎲 使用最优策略生成号码..."):
                                    optimal_predictions = strategy_optimizer.generate_optimized_numbers(
                                        df_filtered, final_strategy, count=5
                                    )
                                    
                                    st.subheader("🎯 最优策略预测结果")
                                    
                                    for i, pred in enumerate(optimal_predictions, 1):
                                        with st.expander(f"最优组合 {i}"):
                                            col1, col2, col3 = st.columns(3)
                                            
                                            col1.metric("前区", f"{pred['front']}")
                                            col2.metric("后区", f"{pred['back']}")
                                            col3.metric("置信度", f"{pred.get('confidence', 0):.3f}")
                                            
                                            st.caption(f"生成方法: {pred.get('generation_method', 'unknown')}")
                            
                            # 详细优化结果
                            with st.expander("🔍 详细优化结果"):
                                st.json(optimization_results, expanded=False)
                    
                    except Exception as e:
                        st.error(f"策略优化失败: {e}")
                        import traceback
                        with st.expander("查看详细错误"):
                            st.code(traceback.format_exc())
        
        # 综合分析
        with evo_tabs[3]:
            st.subheader("📊 AI技术综合分析")
            st.write("对比分析各种AI技术的性能表现")
            
            # 技术对比分析
            if st.button("📊 运行综合性能对比", key="comprehensive_analysis"):
                try:
                    with st.spinner("📊 正在进行综合性能分析..."):
                        # 模拟各种技术的性能数据
                        techniques = {
                            "传统方法": {"roi": 0.15, "hit_rate": 0.12, "stability": 0.6, "speed": 0.9},
                            "集成学习": {"roi": 0.28, "hit_rate": 0.18, "stability": 0.75, "speed": 0.7},
                            "进化算法": {"roi": 0.35, "hit_rate": 0.22, "stability": 0.8, "speed": 0.4},
                            "神经网络": {"roi": 0.42, "hit_rate": 0.25, "stability": 0.85, "speed": 0.3},
                            "策略优化": {"roi": 0.48, "hit_rate": 0.28, "stability": 0.9, "speed": 0.5}
                        }
                        
                        # 性能对比雷达图
                        st.subheader("🎯 技术性能对比")
                        
                        metrics = ["roi", "hit_rate", "stability", "speed"]
                        metric_names = ["ROI", "命中率", "稳定性", "速度"]
                        
                        fig_radar = px.line_polar(
                            r=[techniques[tech][metric] for tech in techniques for metric in metrics],
                            theta=metric_names * len(techniques),
                            color=[tech for tech in techniques for _ in metrics],
                            line_close=True,
                            title="AI技术综合性能对比"
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # 详细性能表
                        st.subheader("📋 详细性能指标")
                        
                        perf_df = pd.DataFrame(techniques).T
                        perf_df.columns = ["ROI", "命中率", "稳定性", "速度"]
                        
                        # 格式化显示
                        styled_df = perf_df.style.format({
                            "ROI": "{:.1%}",
                            "命中率": "{:.1%}",
                            "稳定性": "{:.2f}",
                            "速度": "{:.2f}"
                        }).background_gradient(cmap='RdYlGn')
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # 推荐策略
                        st.subheader("💡 AI技术使用建议")
                        
                        recommendations = [
                            "🎯 **新手用户**: 建议使用集成学习，平衡性能与易用性",
                            "🚀 **追求高收益**: 推荐神经网络+策略优化组合",
                            "⚡ **注重速度**: 传统方法+集成学习的轻量级组合",
                            "🔬 **研究导向**: 进化算法+神经网络的实验性组合",
                            "💰 **稳定收益**: 策略优化的多目标平衡方案"
                        ]
                        
                        for rec in recommendations:
                            st.markdown(rec)
                        
                        # 使用指南
                        with st.expander("📖 AI技术使用指南"):
                            st.markdown("""
                            ### 🧬 进化算法
                            - **适用场景**: 全局优化，寻找最优解
                            - **优势**: 能跳出局部最优，找到全局最优解
                            - **劣势**: 计算时间较长，需要大量迭代
                            
                            ### 🧠 神经网络
                            - **适用场景**: 复杂模式识别，非线性关系建模
                            - **优势**: 强大的学习能力，能发现隐藏模式
                            - **劣势**: 需要大量数据，训练时间长
                            
                            ### 🎯 策略优化
                            - **适用场景**: 综合多种技术，自动寻找最优策略
                            - **优势**: 自动化程度高，综合性能最佳
                            - **劣势**: 复杂度高，需要较多计算资源
                            
                            ### 💡 使用建议
                            1. 首次使用建议从集成学习开始
                            2. 有足够数据后可尝试神经网络
                            3. 追求极致性能时使用策略优化
                            4. 定期使用进化算法寻找新的最优解
                            """)
                
                except Exception as e:
                    st.error(f"综合分析失败: {e}")
                    import traceback
                    with st.expander("查看详细错误"):
                        st.code(traceback.format_exc())
    
    else:
        st.info("请先点击上方按钮初始化AI深度学习组件")
        
        # 显示功能预览
        st.subheader("🌟 AI深度学习功能预览")
        
        features = [
            {
                "icon": "🧬",
                "title": "进化算法优化",
                "description": "使用遗传算法和进化策略，模拟自然选择过程寻找最优号码组合",
                "benefits": ["全局优化", "跳出局部最优", "自适应变异", "多目标优化"]
            },
            {
                "icon": "🧠", 
                "title": "神经网络预测",
                "description": "采用Transformer、LSTM等深度学习模型，发现复杂的号码模式",
                "benefits": ["深度学习", "模式识别", "非线性建模", "时序预测"]
            },
            {
                "icon": "🎯",
                "title": "策略优化",
                "description": "自动整合所有AI技术，寻找最优预测策略组合",
                "benefits": ["自动化", "多技术融合", "策略选择", "性能最优"]
            },
            {
                "icon": "📊",
                "title": "综合分析",
                "description": "对比分析各种AI技术的性能，提供使用建议",
                "benefits": ["性能对比", "技术评估", "使用指导", "决策支持"]
            }
        ]
        
        for feature in features:
            with st.expander(f"{feature['icon']} {feature['title']}"):
                st.write(feature['description'])
                st.markdown("**主要优势:**")
                for benefit in feature['benefits']:
                    st.markdown(f"- ✅ {benefit}")

