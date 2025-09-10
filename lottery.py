"""
Food Wheel — Streamlit 轮盘抽奖小程序
功能：
- 用户可配置餐馆/美食店铺清单（新增、删除、导入）
- 点击“旋转轮盘”随机选择一家并记录历史（带日期时间）
- 历史记录可导出/导入/重置
- 可视化：按出现次数生成占比图（饼图/柱状图）和历史表格

运行：
1) 安装依赖：pip install streamlit pandas matplotlib
2) 运行：streamlit run food_wheel_streamlit.py

文件行为：在当前目录读写 food_list.csv 和 food_wheel_history.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import io

# 文件路径
FOOD_LIST_CSV = 'food_list.csv'
HISTORY_CSV = 'food_wheel_history.csv'

# 辅助：加载/保存清单

def load_food_list(path=FOOD_LIST_CSV):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if 'name' in df.columns:
                return list(df['name'].dropna().astype(str))
        except Exception:
            pass
    # 默认示例
    return ['附近小吃A', '火锅店B', '日式料理C', '披萨店D']


def save_food_list(lst, path=FOOD_LIST_CSV):
    df = pd.DataFrame({'name': lst})
    df.to_csv(path, index=False)


# 辅助：加载/保存历史

def load_history(path=HISTORY_CSV):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=['datetime','choice'])


def save_history(df, path=HISTORY_CSV):
    df.to_csv(path, index=False)


def append_history(choice, path=HISTORY_CSV):
    df = load_history(path)
    row = {'datetime': datetime.now().isoformat(), 'choice': choice}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_history(df, path)


# 轮盘抽取逻辑

def spin_wheel(food_list):
    # 根据当前清单随机选择
    if not food_list:
        raise ValueError('清单为空')
    choice = np.random.choice(food_list)
    return choice


# 可视化：饼图/柱状图

def plot_distribution(history_df):
    if history_df.empty:
        st.info('历史记录为空，无法生成占比图')
        return
    counts = history_df['choice'].value_counts()
    fig1, ax1 = plt.subplots()
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, counterclock=False, ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title('美食选择占比（饼图）')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    counts.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('店铺')
    ax2.set_ylabel('次数')
    ax2.set_title('美食选择次数（柱状图）')
    fig2.autofmt_xdate(rotation=45)
    st.pyplot(fig2)


# Streamlit 界面

def main():
    st.set_page_config(page_title='Food Wheel - 轮盘抽奖决定吃什么', layout='wide')
    st.title('🎡 Food Wheel — 决定今天吃什么')

    # 读取数据
    food_list = load_food_list()
    history_df = load_history()

    # 左侧：清单管理与操作
    with st.sidebar:
        st.header('清单管理')
        st.write('当前清单：')
        for i, name in enumerate(food_list):
            st.write(f"{i+1}. {name}")

        new_name = st.text_input('新增店铺（输入名称后点添加）')
        if st.button('添加店铺'):
            if new_name.strip():
                food_list.append(new_name.strip())
                save_food_list(food_list)
                st.success(f'已添加: {new_name.strip()}')
            else:
                st.warning('请输入有效店铺名称')

        remove_name = st.selectbox('从清单中删除（选择后点删除）', [''] + food_list)
        if st.button('删除选中店铺'):
            if remove_name and remove_name in food_list:
                food_list.remove(remove_name)
                save_food_list(food_list)
                st.success(f'已删除: {remove_name}')
            else:
                st.warning('请选择有效店铺')

        st.markdown('---')
        st.header('历史记录操作')
        if st.button('导出历史 CSV'):
            if history_df.empty:
                st.warning('历史为空，无法导出')
            else:
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button('点击下载历史 CSV', data=csv, file_name='food_wheel_history.csv', mime='text/csv')

        if st.button('重置历史（清空）'):
            if os.path.exists(HISTORY_CSV):
                os.remove(HISTORY_CSV)
            history_df = load_history()
            st.success('历史已清空')

        uploaded = st.file_uploader('从 CSV 导入历史（包含 datetime,choice 列）', type=['csv'])
        if uploaded is not None:
            try:
                udf = pd.read_csv(uploaded)
                if 'choice' in udf.columns and 'datetime' in udf.columns:
                    udf['datetime'] = pd.to_datetime(udf['datetime'])
                    save_history(udf)
                    st.success('已导入历史')
                else:
                    st.error('CSV 需要包含 datetime 和 choice 两列')
            except Exception as e:
                st.error(f'导入失败: {e}')

    # 中间：轮盘与结果
    st.subheader('轮盘抽取')
    st.write('清单（可在侧栏管理）')
    st.write(', '.join(food_list) if food_list else '（清单为空）')

    col1, col2 = st.columns([2,1])
    with col1:
        if st.button('🎯 旋转轮盘，帮我决定（随机选择）'):
            try:
                choice = spin_wheel(food_list)
                append_history(choice)
                history_df = load_history()
                st.success(f'推荐：**{choice}** — 已记录到历史')
                # 高亮显示模拟：饼图并突出选中项
                counts = history_df['choice'].value_counts()
                labels = counts.index.tolist()
                sizes = counts.values.tolist()
                explode = [0.15 if lbl==choice else 0 for lbl in labels]
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, counterclock=False)
                ax.set_title(f'当前选择分布（高亮: {choice}）')
                st.pyplot(fig)
            except Exception as e:
                st.error(f'抽取失败: {e}')

        if st.button('🧾 显示最近 20 条历史'):
            history_df = load_history()
            st.table(history_df.sort_values('datetime', ascending=False).head(20))

    with col2:
        st.subheader('快速操作')
        if st.button('随机一项但不保存历史'):
            try:
                choice = np.random.choice(food_list)
                st.info(f'随机结果（未保存）：{choice}')
            except Exception as e:
                st.error(f'失败: {e}')

        st.markdown('---')
        st.write('统计/可视化')
        if st.button('生成占比图（饼图+柱状）'):
            history_df = load_history()
            plot_distribution(history_df)

    # 右侧：实时分布和历史表格
    st.subheader('统计与历史')
    history_df = load_history()
    if not history_df.empty:
        st.write('总抽取次数：', len(history_df))
        st.dataframe(history_df.sort_values('datetime', ascending=False).reset_index(drop=True))
        # 显示占比图（小）
        counts = history_df['choice'].value_counts()
        fig, ax = plt.subplots(figsize=(6,3))
        counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('店铺')
        ax.set_ylabel('次数')
        fig.autofmt_xdate(rotation=45)
        st.pyplot(fig)
    else:
        st.info('历史记录为空，先旋转轮盘一次吧')


if __name__ == '__main__':
    main()
