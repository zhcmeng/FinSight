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

**严禁**自行编写 Python 脚本。你 **必须** 通过终端（Terminal）调用提供的 CLI 适配器。

### 1. 债务与利率
- **宏观杠杆率 (居民/企业/政府)**:
  ```bash
  python src/trae_tool_adapter.py macro --type leverage
  ```
- **LPR 利率 (贷款市场报价利率)**:
  ```bash
  python src/trae_tool_adapter.py macro --type lpr
  ```

### 2. 就业与融资
- **城镇调查失业率**:
  ```bash
  python src/trae_tool_adapter.py macro --type unemployment
  ```
- **社会融资规模增量**:
  ```bash
  python src/trae_tool_adapter.py macro --type shrzgm
  ```

### 3. 贸易与财政
- **贸易帐 (出口/进口/差额)**:
  ```bash
  python src/trae_tool_adapter.py macro --type trade
  ```
- **财政收入**:
  ```bash
  python src/trae_tool_adapter.py macro --type czsr
  ```

### 4. 货币与金融
- **货币供应量 (M0/M1/M2)**:
  ```bash
  python src/trae_tool_adapter.py macro --type money_supply
  ```
- **外汇和黄金储备**:
  ```bash
  python src/trae_tool_adapter.py macro --type reserve
  ```
- **央行资产负债表**:
  ```bash
  python src/trae_tool_adapter.py macro --type bank_balance
  ```

## 最佳实践
1. **资金面判断**: 通过社会融资规模 (shrzgm) 和货币供应量 (M2) 判断市场流动性。
2. **基本面分析**: GDP、CPI 和 PPI 的年率数据是判断宏观周期阶段的核心指标。
3. **政策敏感度**: 密切关注 LPR 和准备金率的变化，这些通常预示着货币政策的转向。
