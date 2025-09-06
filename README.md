# 大乐透分析与选号系统（本地版）

一个用 Python 构建的“超级大乐透”数据采集、存储、可视化与条件选号项目。

## 功能概览
- **数据采集与入库**：从「中国体彩网/中国竞彩网接口」抓取历史与最新开奖结果，保存至 SQLite。
- **可视化界面（Streamlit）**：
  - 查看往期开奖明细、筛选期数范围。
  - 常见分析：冷热号、遗漏值、和值、奇偶比、区间比等。
  - 自定义筛选条件，多条件组合过滤历史数据。
  - 条件选号：按规则（例如和值范围、奇偶比、冷热混合、排除号码等）生成若干候选号码。
- **一键同步**：UI 内可点击按钮增量拉取最近开奖。

> 免责声明：彩票属随机事件，任何分析与选号均不保证中奖，本项目仅用于数据处理与编程学习。

## 数据来源
- 中国竞彩网（sporttery）历史接口（gameNo=85 为大乐透）：
  `https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry`
- 各省体彩网/中彩网等公开页面（备用）。

## 快速开始
1. 安装依赖（建议 Python 3.9+）：
   ```bash
   pip install -r requirements.txt
   ```
2. 初始化与启动：
   ```bash
   # 第一次运行会自动建库；建议先同步历史数据（可在 UI 点“增量同步”）
   streamlit run app.py
   ```
3. 浏览器打开 `http://localhost:8501`。

## 目录结构
```
dlt_analytics_app/
├─ app.py                # Streamlit 前端与操作入口
├─ requirements.txt
├─ backend/
│  ├─ db.py             # SQLite/SQLAlchemy 数据模型与会话
│  ├─ dlt.py            # 数据源适配（抓取/解析）
│  ├─ analysis.py       # 指标计算（频次、遗漏、和值、奇偶等）
│  ├─ generator.py      # 条件选号与候选集生成
│  └─ sync.py           # 同步历史/增量数据的服务
└─ data/
   └─ dlt.sqlite        # 运行后生成的数据库文件
```

## 常见问题
- **无法抓取**：网络环境可能限制，稍后再试或自行配置代理。
- **接口变更**：如官方接口字段改动，请到 `backend/dlt.py` 调整解析逻辑。
