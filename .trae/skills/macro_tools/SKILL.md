---
name: 宏观经济工具
description: 宏观经济核心指标的终端查询工作流
---

# 宏观经济工具

## 描述
本技能提供宏观层面的经济指标，涵盖杠杆率、利率、就业、贸易、财政及货币供应等核心领域。

## 何时使用
- 需要评估流动性环境、信用扩张与政策取向
- 需要将个股/行业分析放入宏观周期背景中

## 指令

**重要提示**: 
- **严禁**自行编写 Python 脚本。你 **必须** 通过终端（Terminal）调用提供的 CLI 适配器。
- 下列命令模板已通过验证，**无需**通过 `--help` 确认参数。
- **强制落地**: 请务必使用 `--output <路径>` 将结果保存到任务指定的 `data/` 目录下。

### 1. 债务与利率
- **宏观杠杆率**: `python src/trae_tool_adapter.py macro --type leverage --output <路径>`
- **LPR 利率**: `python src/trae_tool_adapter.py macro --type lpr --output <路径>`

### 2. 就业与融资
- **失业率**: `python src/trae_tool_adapter.py macro --type unemployment --output <路径>`
- **社融规模**: `python src/trae_tool_adapter.py macro --type shrzgm --output <路径>`

### 3. 贸易与财政
- **贸易帐**: `python src/trae_tool_adapter.py macro --type trade --output <路径>`
- **财政收入**: `python src/trae_tool_adapter.py macro --type czsr --output <路径>`

### 4. 货币与金融
- **货币供应 (M0/M1/M2)**: `python src/trae_tool_adapter.py macro --type money_supply --output <路径>`
- **外汇和黄金储备**: `python src/trae_tool_adapter.py macro --type reserve --output <路径>`
- **央行资产负债表**: `python src/trae_tool_adapter.py macro --type bank_balance --output <路径>`

## 最佳实践
1. **资金面判断**: 通过社会融资规模 (shrzgm) 和货币供应量 (M2) 判断市场流动性。
2. **基本面分析**: GDP、CPI 和 PPI 的年率数据是判断宏观周期阶段的核心指标。
3. **政策敏感度**: 密切关注 LPR 和准备金率的变化，这些通常预示着货币政策的转向。
