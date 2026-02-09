
import akshare as ak
try:
    print("Trying stock_balance_sheet_by_yearly_em...")
    data = ak.stock_balance_sheet_by_yearly_em(symbol="600519")
    print(data.head())
except Exception as e:
    print(f"stock_balance_sheet_by_yearly_em failed: {e}")

try:
    print("\nTrying stock_financial_report_sina...")
    data = ak.stock_financial_report_sina(stock="600519", symbol="资产负债表")
    print(data.head())
    print(f"Columns: {data.columns.tolist()}")
except Exception as e:
    print(f"stock_financial_report_sina failed: {e}")

try:
    print("\nTrying stock_financial_benefit_ths...")
    data = ak.stock_financial_benefit_ths(symbol="600519", indicator="按年度")
    print(data.head())
except Exception as e:
    print(f"stock_financial_benefit_ths failed: {e}")
