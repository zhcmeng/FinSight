---
name: 金融工具
description: 公司财报、市场指数与个股数据的终端查询工作流
---

# 金融工具

## 描述
本技能提供了一系列用于查询公司财报、市场指数和个股数据的工具。

## 何时使用
- 需要获取某家公司的报表、估值、股东结构或历史行情
- 需要获取核心市场指数作为宏观背景或基准对比

## 指令

**重要提示**: 
- **严禁**自行编写 Python 脚本。
- 你 **必须** 通过终端（Terminal）调用提供的 CLI 适配器。
- 下列命令模板已通过验证，**无需**通过 `--help` 再次确认参数。
- **强制落地**: 请务必使用 `--output` 参数将结果保存到任务指定的 `data/` 目录下。

### 1. 公司报表查询 (Company Statements)
- **资产负债表**:
  ```bash
  python src/trae_tool_adapter.py statement --type balance --code <代码> --market <HK|A> --output <路径>
  ```
- **利润表**:
  ```bash
  python src/trae_tool_adapter.py statement --type income --code <代码> --market <HK|A> --output <路径>
  ```
- **现金流量表**:
  ```bash
  python src/trae_tool_adapter.py statement --type cashflow --code <代码> --market <HK|A> --output <路径>
  ```

### 2. 市场指数查询 (Market Data)
- **指数查询**:
  ```bash
  python src/trae_tool_adapter.py market --index <hushen|hengsheng|shangzheng|nsdk> --output <路径>
  ```

### 3. 个股数据查询 (Stock Data)
- **基础信息**: `python src/trae_tool_adapter.py stock --type basic --code <代码> --market <HK|A> --output <路径>`
- **分红历史**: `python src/trae_tool_adapter.py stock --type dividend --code <代码> --market <HK|A> --output <路径>`
- **股东结构**: `python src/trae_tool_adapter.py stock --type holders --code <代码> --market <HK|A> --output <路径>`
- **估值指标**: `python src/trae_tool_adapter.py stock --type valuation --code <代码> --market <HK|A> --output <路径>`
- **历史行情**: `python src/trae_tool_adapter.py stock --type price --code <代码> --market <HK|A> --output <路径>`

## 最佳实践
1. **优先获取基础信息**: 在进行任何深度分析前，先调用 `stock --type basic` 了解公司所属行业和主营业务。
2. **多维度对比**: 结合报表数据和估值指标进行综合研判。
3. **数据清洗**: 适配器返回的数据已进行初步清洗，单位通常为百万人民币（财报项）。
