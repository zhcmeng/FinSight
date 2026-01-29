# FinSight 工程学习线路图

欢迎来到 **FinSight** 项目！这是一个基于 **CAVM (Code Agent with Variable Memory)** 架构的自动化金融深度研究系统。为了帮助你快速掌握本项目，我们设计了以下学习线路图。

---

## 1. 项目概览与核心理念

在开始编码前，请先理解 FinSight 的设计哲学。
- **目标**: 实现从原始数据到专业金融报告的自动化生产。
- **核心架构**: **CAVM**。不同于传统的固定工作流，我们的智能体在统一的“变量空间”内通过执行 Python 代码来操作工具、处理数据和管理记忆。
- **关键文档**:
    - [README.md](file:///c:/Study/FinSight/README.md): 项目背景、特性与快速启动。

- **核心技术栈**:
    - **开发语言**: Python 3.10+
    - **大模型能力**: OpenAI API (及其兼容接口), Pydantic
    - **数据获取**: AkShare (金融数据), Playwright/Crawl4AI (网页爬取)
    - **数据处理**: Pandas, NumPy, Dill, JSON_Repair
    - **可视化**: Matplotlib, Seaborn
    - **文档生成**: Python-Docx, PDFPlumber, Pandoc (外部依赖)
    - **交互界面**: FastAPI (后端), Vite + React / Streamlit (前端)

---

## 2. 第一阶段：环境搭建与跑通 Demo

先让项目在本地跑起来。
- **基础环境**: Python 3.10+, Pandoc, Node.js。
- **配置**: 参考 `.env.example` 配置 LLM API Key。
- **快速开始**:
    - 运行 `run_report.py` 生成一份测试报告。
    - 进入 `demo/` 目录体验 Web UI 界面。

---

## 3. 第二阶段：基础工具层 (Tools)

了解系统如何获取外部数据。
- **代码路径**: `src/tools/`
- **学习重点**:
    - [financial/](file:///c:/Study/FinSight/src/tools/financial/): 学习如何使用 AkShare 获取股票、财报等数据。
    - [web/](file:///c:/Study/FinSight/src/tools/web/): 了解网页搜索与爬取逻辑。
    - **练习**: 尝试在 `tests/basic_components/test_tool.py` 中调用一个新的金融接口。

---

## 4. 第三阶段：内存管理层 (Variable Memory)

理解智能体如何“记住”信息。
- **代码路径**: `src/memory/`
- **学习重点**:
    - [variable_memory.py](file:///c:/Study/FinSight/src/memory/variable_memory.py): 核心组件。了解它是如何通过向量检索 (RAG) 和变量存储解决长上下文问题的。
    - **练习**: 阅读 `tests/basic_components/test_embedding.py`。

---

## 5. 第四阶段：智能体层 (Agents)

这是系统的大脑。
- **代码路径**: `src/agents/`
- **学习重点**:
    - [base_agent.py](file:///c:/Study/FinSight/src/agents/base_agent.py): 所有智能体的基类，定义了思考与执行循环。
    - [data_collector.py](file:///c:/Study/FinSight/src/agents/data_collector/data_collector.py): 负责数据收集的 Agent。
    - [data_analyzer.py](file:///c:/Study/FinSight/src/agents/data_analyzer/data_analyzer.py): 负责编写 Python 代码进行统计分析的 Agent。
- **进阶**: 查看 `src/agents/*/prompts/`，了解如何通过 YAML 配置复杂的提示词。

---

## 6. 第五阶段：工作流与报告生成

了解多智能体是如何协作的。
- **核心逻辑**: [report_generator.py](file:///c:/Study/FinSight/src/agents/report_generator/report_generator.py)。
- **流程**:
    1. **收集器 (Collector)** 获取原始数据。
    2. **分析器 (Analyzer)** 生成图表与深度洞察。
    3. **报告员 (Report Generator)** 汇总并调用 Pandoc 生成最终文档。

---

## 7. 第六阶段：AI 原生演进 (MCP & Skills)

结合你最近的学习，思考如何将本项目重构为更现代的 AI 原生架构（如 Trae Agents）。
- **重点**: 
    - **MCP (Model Context Protocol)**: 将 `src/tools/` 下的工具封装为标准化服务，提升跨平台连接性。
    - **Skills**: 将 `src/agents/` 中的复杂处理逻辑（如绘图、报告编译）封装为可复用的技能包。
- **思考**: 如何利用“渐进式披露”机制来优化当前系统的 Token 消耗？

---

## 💡 学习建议
1. **先看 Test**: `tests/` 目录下的文件是最好的“使用说明”。
2. **多打日志**: 使用 `src/utils/logger.py` 观察智能体的思考过程。
3. **关注 Prompt**: 金融深度分析的核心往往隐藏在 `prompts.yaml` 的约束条件中。
