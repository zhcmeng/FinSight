# FinSight 工程分析：步骤与能力边界

本文档基于对代码库的递归分析，总结了 FinSight 项目的核心工作流、主要能力及系统边界。

## 1. 核心工作流 (Core Workflow)

项目采用**分层级、异步并行**的多智能体（Multi-Agent）协作模式，由 [run_report.py](file:///c:\Study\FinSight\run_report.py) 进行全局编排。主要执行步骤如下：

### 1.1 初始化与任务规划
*   **系统初始化**：加载全局配置 ([Config](file:///c:\Study\FinSight\src\config\config.py))、环境变量，并初始化共享内存系统 ([Memory](file:///c:\Study\FinSight\src\memory\variable_memory.py)) 和日志记录器。
*   **智能任务拆解**：系统根据用户输入的研究目标（如“分析某上市公司”），利用 LLM 自动将宏大的目标拆解为具体的子任务（如“收集财务数据”、“分析竞争对手”）。

### 1.2 分层异步执行 (Layered Execution)
系统将所有 Agent 分为三个优先级队列，按顺序依次执行，同一优先级的 Agent 并发运行：

1.  **第一阶段：全网数据收集 (Data Collection)**
    *   **执行者**：[DataCollector](file:///c:\Study\FinSight\src\agents\data_collector\data_collector.py)
    *   **行为**：
        *   调用 [DeepSearchAgent](file:///c:\Study\FinSight\src\agents\search_agent\search_agent.py) 进行全网深度搜索，获取新闻、公告等非结构化文本。
        *   调用 `Financial Tools` (集成 AkShare, eFinance) 获取 A 股/港股的行情、K线、财务报表等结构化数据。
    *   **产出**：原始数据存入共享 Memory。

2.  **第二阶段：深度数据分析 (Data Analysis)**
    *   **执行者**：[DataAnalyzer](file:///c:\Study\FinSight\src\agents\data_analyzer\data_analyzer.py)
    *   **行为**：
        *   从 Memory 读取原始数据。
        *   利用内置的 **Code Interpreter (代码解释器)** 编写并执行 Python 代码进行数据清洗和计算。
        *   **多模态闭环绘图**：生成绘图代码 -> 执行绘图 -> **VLM (视觉模型) 评审** -> 自动修正代码 -> 输出最终图表。
    *   **产出**：分析结论文本和统计图表。

3.  **第三阶段：报告生成与渲染 (Report Generation)**
    *   **执行者**：[ReportGenerator](file:///c:\Study\FinSight\src\agents\report_generator\report_generator.py)
    *   **行为**：
        *   基于收集的数据和分析结果，生成详细的报告大纲。
        *   逐章节撰写内容，自动引用数据来源。
        *   进行后期处理：替换图片占位符、生成摘要/封面/参考文献。
    *   **产出**：最终渲染为 Markdown、Word (`.docx`) 和 PDF 格式的研报。

### 1.3 状态管理与持久化
*   系统支持**断点续传**机制。每个 Agent 在关键步骤后都会保存运行快照 (`.pkl`)，确保长流程任务在意外中断后可恢复。

---

## 2. 核心能力 (Key Capabilities)

### 2.1 全流程自动化
能够实现从“输入一个公司名”到“输出一份完整研报”的端到端自动化，无需人工干预中间环节。

### 2.2 混合数据源处理
*   **结构化数据**：深度集成 `AkShare` 和 `eFinance`，原生支持中国市场（A股、港股）的财务、估值、行情数据获取。
*   **非结构化信息**：具备强大的网络搜索和爬虫能力，能够处理新闻、研报、公告等文本信息。

### 2.3 自我修正的代码执行
所有核心 Agent 均具备 Python 代码执行能力，能够处理复杂的数据计算。特别是 `DataAnalyzer` 引入了 VLM 视觉反馈机制，能够自我检查生成的图表质量并修正代码错误。

### 2.4 专业级文档输出
支持生成格式严谨的文档，包含封面、目录、图文混排、引用标注和参考文献页，可以直接作为初稿使用。

---

## 3. 能力边界与限制 (Boundaries & Limitations)

### 3.1 市场覆盖范围
*   **主要支持**：中国市场（A股、港股）。
*   **限制**：代码库目前缺乏针对美股或其他海外市场的专用结构化数据 API 封装（如 Yahoo Finance 等未在核心路径中），海外市场研究主要依赖通用网络搜索，数据深度不如国内市场。

### 3.2 实时性与频度
*   **适用场景**：深度、静态的投资研究报告（运行耗时在分钟级到小时级）。
*   **限制**：不适用于高频交易、毫秒级实时监控或需要亚秒级响应的场景。

### 3.3 数据源交互
*   **限制**：系统设计主要面向公开网络数据。目前没有内置标准接口来对接本地私有数据库（如 SQL Server, Oracle）或大规模本地文件系统，处理私有数据需要二次开发。

### 3.4 可视化形式
*   **限制**：生成的图表为静态图片 (`.png`)，虽然质量高但不支持交互（如缩放、悬停显示数值）。不支持生成 Plotly/Echarts 等交互式 HTML 图表。

---

## 4. 工具列表 (Tool List)

本项目共包含 **53** 个工具，分为金融、行业、宏观和网络四大类。

### 4.1 金融工具 (Financial Tools)
主要用于查询公司财报、市场指数和个股数据。

**公司报表 (Company Statements)** - [src/tools/financial/company_statements.py](file:///c:\Study\FinSight\src\tools\financial\company_statements.py)
- `BalanceSheet`: 资产负债表查询工具 (A股/港股)
- `IncomeStatement`: 利润表查询工具 (A股/港股)
- `CashFlowStatement`: 现金流量表查询工具 (A股/港股)

**市场数据 (Market Data)** - [src/tools/financial/market.py](file:///c:\Study\FinSight\src\tools\financial\market.py)
- `HuShen_Index`: 沪深300指数日线数据
- `HengSheng_Index`: 恒生指数 (HSI) 日线数据
- `ShangZheng_Index`: 上证指数日线数据
- `NSDK_Index`: 纳斯达克综合指数日线数据

**个股数据 (Stock Data)** - [src/tools/financial/stock.py](file:///c:\Study\FinSight\src\tools\financial\stock.py)
- `StockBasicInfo`: 股票基础信息 (行业、主营业务等)
- `ShareHoldingStructure`: 股东结构查询 (十大股东、持股比例)
- `StockBaseInfo`: 权益估值指标 (PE, PB, ROE 等)
- `StockPrice`: 股票历史行情 (K线数据)

### 4.2 行业工具 (Industry Tools)
关注特定的行业经济指标，数据多来自东方财富等平台。
对应文件：[src/tools/industry/industry.py](file:///c:\Study\FinSight\src\tools\industry\industry.py)

- `Industry_gyzjz`: 工业增加值增长数据
- `Industry_production_yoy`: 规模以上工业生产同比增长
- `Industry_China_PMI`: 中国官方制造业 PMI
- `Industry_China_CX_services_PMI`: 中国财新服务业 PMI
- `Industry_China_CPI`: 消费者物价指数 (CPI) 月度数据
- `Industry_China_GDP`: 国内生产总值 (GDP) 统计数据
- `Industry_China_PPI`: 生产者物价指数 (PPI) 月度数据
- `Industry_China_xfzxx`: 消费者信心指数
- `Industry_China_consumer_goods_retail`: 社会消费品零售总额
- `Industry_China_retail_price_index`: 零售价格指数
- `Industry_China_qyspjg`: 企业商品价格指数

### 4.3 宏观经济工具 (Macro Economic Tools)
提供宏观层面的经济指标，涵盖货币、贸易、就业等领域。
对应文件：[src/tools/macro/macro.py](file:///c:\Study\FinSight\src\tools\macro\macro.py)

- `Macro_China_Leverage_Ratio`: 中国宏观杠杆率 (居民/企业/政府)
- `Macro_China_qyspjg`: 企业商品价格指数 (宏观版)
- `Macro_China_LPR`: 中国 LPR 利率 (贷款市场报价利率)
- `Macro_China_urban_unemployment`: 城镇调查失业率
- `Macro_China_shrzgm`: 社会融资规模增量
- `Macro_China_GDP_yearly`: 中国 GDP 年率
- `Macro_China_CPI_yearly`: 中国 CPI 年率
- `Macro_China_PPI_yearly`: 中国 PPI 年率
- `Macro_USA_CPI_yearly`: 美国 CPI 年率
- `Macro_China_exports_yearly`: 中国出口年率
- `Macro_China_imports_yearly`: 中国进口年率
- `Macro_China_trade_balance`: 中国贸易帐
- `Macro_China_czsr`: 财政收入
- `Macro_China_whxd`: 外汇贷款数据
- `Macro_China_bond_public`: 新债发行数据
- `Macro_China_bank_balance`: 央行资产负债表
- `Macro_China_supply_of_money`: 货币供应量 (M0/M1/M2)
- `Macro_China_reserve_requirement_ratio`: 存款准备金率
- `Macro_China_fx_gold`: 外汇和黄金储备
- `Macro_China_stock_market_cap`: 全国股票交易统计
- `Macro_China_epu_index`: 中国经济政策不确定性指数

### 4.4 网络工具 (Web Tools)
提供网页抓取和搜索引擎功能，支持 Playwright 和 Requests 两种后端。

**网页抓取 (Crawler)** - [src/tools/web/web_crawler.py](file:///c:\Study\FinSight\src\tools\web\web_crawler.py)
- `Click`: 网页/PDF 内容抓取工具

**搜索引擎 (Search Engines)** - [src/tools/web/search_engine_playwright.py](file:///c:\Study\FinSight\src\tools\web\search_engine_playwright.py) & [src/tools/web/search_engine_requests.py](file:///c:\Study\FinSight\src\tools\web\search_engine_requests.py)
- `PlaywrightSearch`: Bing 网页搜索 (基于 Playwright，支持动态页面)
- `InDomainSearch_Playwright`: 金融垂直领域搜索 (基于 Playwright)
- `BingSearch`: Bing 网页搜索 (基于 Requests)
- `BochaSearch`: 博查 (Bocha) 网页搜索
- `SerperSearch`: Google 搜索 (基于 Serper API)
- `DuckDuckGoSearch`: DuckDuckGo 网页搜索
- `SogouSearch`: 搜狗网页搜索
- `InDomainSearch_Request`: 金融垂直领域搜索 (基于 Requests)
- `BingImageSearch`: Bing 图片搜索
