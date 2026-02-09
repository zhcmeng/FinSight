
import akshare as ak
import pandas as pd

def check_ths_format():
    print("Checking Balance Sheet (ths)...")
    df_bs = ak.stock_financial_debt_ths(symbol="600519", indicator="按年度")
    print(df_bs.iloc[0, :10])
    
    print("\nChecking Income Statement (ths)...")
    df_is = ak.stock_financial_benefit_ths(symbol="600519", indicator="按年度")
    print(df_is.iloc[0, :10])

    print("\nChecking Cash Flow (ths)...")
    df_cf = ak.stock_financial_cash_ths(symbol="600519", indicator="按年度")
    print(df_cf.iloc[0, :10])

if __name__ == "__main__":
    check_ths_format()
