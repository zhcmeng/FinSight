
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.append(root)

from src.tools.financial.company_statements import BalanceSheet, IncomeStatement, CashFlowStatement

async def test_fix():
    print("--- Testing BalanceSheet ---")
    tool_bs = BalanceSheet()
    results_bs = await tool_bs.api_function(stock_code="600519", market="A")
    if results_bs and results_bs[0].data is not None:
        print("BS Success!")
    else:
        print("BS Failed!")

    print("\n--- Testing IncomeStatement ---")
    tool_is = IncomeStatement()
    results_is = await tool_is.api_function(stock_code="600519", market="A")
    if results_is and results_is[0].data is not None:
        print("IS Success!")
    else:
        print("IS Failed!")

    print("\n--- Testing CashFlowStatement ---")
    tool_cs = CashFlowStatement()
    results_cs = await tool_cs.api_function(stock_code="600519", market="A")
    if results_cs and results_cs[0].data is not None:
        print("CS Success!")
    else:
        print("CS Failed!")

if __name__ == "__main__":
    asyncio.run(test_fix())
