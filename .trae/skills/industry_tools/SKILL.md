---
name: 行业工具
description: 中国宏观经济行业指标的终端查询工作流
---

# 行业工具

## 描述
本技能专注于获取中国宏观经济中的行业指标，数据涵盖工业生产、PMI、物价水平及消费信心等。

## 何时使用
- 需要用 PMI、工业增加值等数据判断景气度与行业周期
- 需要用 CPI、PPI、社零等数据判断需求与通胀传导

## 指令

**严禁**自行编写 Python 脚本。你 **必须** 通过终端（Terminal）调用提供的 CLI 适配器。

### 1. 工业与生产
- **工业增加值增长**:
  ```bash
  python src/trae_tool_adapter.py industry --type gyzjz
  ```
- **规模以上工业生产同比增长**:
  ```bash
  python src/trae_tool_adapter.py industry --type production_yoy
  ```

### 2. 采购经理指数 (PMI)
- **官方制造业 PMI**:
  ```bash
  python src/trae_tool_adapter.py industry --type pmi
  ```
- **财新服务业 PMI**:
  ```bash
  python src/trae_tool_adapter.py industry --type cx_pmi
  ```

### 3. 物价与消费
- **消费者物价指数 (CPI)**:
  ```bash
  python src/trae_tool_adapter.py industry --type cpi
  ```
- **生产者物价指数 (PPI)**:
  ```bash
  python src/trae_tool_adapter.py industry --type ppi
  ```
- **国内生产总值 (GDP)**:
  ```bash
  python src/trae_tool_adapter.py industry --type gdp
  ```
- **消费者信心指数**:
  ```bash
  python src/trae_tool_adapter.py industry --type xfzxx
  ```
- **社会消费品零售总额**:
  ```bash
  python src/trae_tool_adapter.py industry --type retail
  ```

## 最佳实践
1. **宏观与行业结合**: 分析特定行业（如制造业）时，务必参考 PMI 和工业增加值数据。
2. **趋势分析**: 行业数据通常具有季节性，建议对比历史同期数据（YoY）。
3. **通胀监测**: 结合 CPI 和 PPI 观察行业成本端和需求端的价格传导。
