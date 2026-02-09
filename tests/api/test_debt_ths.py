
import akshare as ak
try:
    data = ak.stock_financial_debt_ths(symbol="600519", indicator="按年度")
    print(data.head())
    print(f"Columns: {data.columns.tolist()}")
except Exception as e:
    print(f"Error: {e}")
