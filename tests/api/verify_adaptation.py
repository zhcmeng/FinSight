
import asyncio
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.append(root)

from src.tools.financial.company_statements import BalanceSheet, IncomeStatement, CashFlowStatement

async def verify_all_statements():
    stock_code = "600519" # Kweichow Moutai
    market = "A"
    
    statements = [
        ("Balance Sheet", BalanceSheet()),
        ("Income Statement", IncomeStatement()),
        ("Cash Flow Statement", CashFlowStatement())
    ]
    
    for name, tool in statements:
        print(f"\n--- Verifying {name} for {stock_code} ({market}) ---")
        try:
            results = await tool.api_function(stock_code=stock_code, market=market)
            if results and results[0].data is not None:
                df = results[0].data
                print(f"Successfully fetched {name}!")
                print(f"Columns: {df.columns.tolist()}")
                print(f"First column name: {df.columns[0]}")
                print("First 5 rows:\n", df.head())
                
                # Check if format matches HK expectations
                if df.columns[0] == '会计年度 (人民币百万)':
                    print("✅ Format matches: Column name is correct.")
                else:
                    print(f"❌ Format mismatch: Expected '会计年度 (人民币百万)', got '{df.columns[0]}'")
                
                if isinstance(df.iloc[0, 1], (float, int)):
                    print("✅ Format matches: Values are numbers.")
                else:
                    print(f"❌ Format mismatch: Expected numbers, got {type(df.iloc[0, 1])}")
            else:
                print(f"❌ Failed to fetch {name}: Data is None")
        except Exception as e:
            print(f"❌ Error verifying {name}: {e}")

if __name__ == "__main__":
    asyncio.run(verify_all_statements())
