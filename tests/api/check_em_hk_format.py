
import akshare as ak
import pandas as pd

def check_em_hk_format():
    print("Checking HK Balance Sheet (em)...")
    try:
        # 00700 is Tencent
        df = ak.stock_financial_hk_report_em(stock="00700", symbol="资产负债表", indicator="年度")
        print("Raw columns:", df.columns.tolist())
        print("Raw head:\n", df.head())
        
        # Now let's see how _preprocess_data processes it
        # (Copying logic from company_statements.py)
        data = df.copy()
        data.drop(['SECUCODE','SECURITY_CODE','SECURITY_NAME_ABBR','ORG_CODE', 'DATE_TYPE_CODE', 'FISCAL_YEAR','STD_ITEM_CODE','REPORT_DATE'], axis=1, inplace=True)
        data['YEAR'] = data['STD_REPORT_DATE'].apply(lambda x: pd.to_datetime(x).year)
        data.drop(['STD_REPORT_DATE'], axis=1, inplace=True)
        data['AMOUNT'] = data['AMOUNT'].apply(lambda x: float(x)//1000000)
        
        pivot_df = data.pivot_table(
            index='STD_ITEM_NAME',
            columns='YEAR',
            values='AMOUNT',
            aggfunc='sum'
        ).reset_index()
        
        # item_order logic...
        item_order = {item: idx for idx, item in enumerate(data['STD_ITEM_NAME'].unique())}
        pivot_df['sort_key'] = pivot_df['STD_ITEM_NAME'].map(item_order)
        filtered_df = pivot_df.sort_values('sort_key').drop('sort_key', axis=1)
        filtered_df = filtered_df.rename(columns={'STD_ITEM_NAME': '会计年度 (人民币百万)'})
        
        print("\nProcessed columns:", filtered_df.columns.tolist())
        print("Processed head:\n", filtered_df.head())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_em_hk_format()
