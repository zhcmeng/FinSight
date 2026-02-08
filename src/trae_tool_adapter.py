import argparse
import asyncio
import json
import sys
import os

# Add the project root to sys.path to allow absolute imports from 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from typing import List

# Import Financial Tools
from src.tools.financial.stock import StockBasicInfo, StockPrice, StockBaseInfo, ShareHoldingStructure
from src.tools.financial.company_statements import BalanceSheet, IncomeStatement, CashFlowStatement
from src.tools.financial.market import HuShen_Index, HengSheng_Index, ShangZheng_Index, NSDK_Index

# Import Industry Tools
from src.tools.industry.industry import (
    Industry_gyzjz, Industry_production_yoy, Industry_China_PMI, 
    Industry_China_CX_services_PMI, Industry_China_CPI, Industry_China_GDP,
    Industry_China_PPI, Industry_China_xfzxx, Industry_China_consumer_goods_retail
)

# Import Macro Tools
from src.tools.macro.macro import (
    Macro_China_Leverage_Ratio, Macro_China_LPR, Macro_China_urban_unemployment,
    Macro_China_shrzgm, Macro_China_trade_balance, Macro_China_czsr,
    Macro_China_supply_of_money, Macro_China_fx_gold, Macro_China_bank_balance
)

# Import Web Tools
from src.tools.web.web_crawler import Click
from src.tools.web.search_engine_playwright import PlaywrightSearch
from src.tools.web.search_engine_requests import BochaSearch, SerperSearch, InDomainSearch_Request

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super().default(obj)

def convert_to_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj

async def run_tool(tool_instance, **kwargs):
    # Some tools use api_function with positional or keyword args
    # We try to adapt based on common patterns in the codebase
    try:
        if hasattr(tool_instance, 'api_function'):
            # Check if it's a financial stock tool which has specific signature
            if isinstance(tool_instance, (StockBasicInfo, StockPrice, StockBaseInfo, ShareHoldingStructure)):
                return await tool_instance.api_function(stock_code=kwargs.get('code'), market=kwargs.get('market', 'HK'))
            
            # Check if it's a statement tool
            if isinstance(tool_instance, (BalanceSheet, IncomeStatement, CashFlowStatement)):
                return await tool_instance.api_function(stock_code=kwargs.get('code'), market=kwargs.get('market', 'HK'))

            # Check if it's a web tool
            if isinstance(tool_instance, Click):
                urls = kwargs.get('urls', '').split(',')
                return await tool_instance.api_function(urls=urls, task=kwargs.get('task', ''))
            
            if isinstance(tool_instance, (PlaywrightSearch, BochaSearch, SerperSearch, InDomainSearch_Request)):
                return await tool_instance.api_function(query=kwargs.get('query', ''))

            # Macro/Industry/Market tools usually take no params in api_function
            return await tool_instance.api_function()
    except Exception as e:
        return [{"error": str(e)}]

async def main():
    parser = argparse.ArgumentParser(description='FinSight Trae Tool Adapter')
    subparsers = parser.add_subparsers(dest='category', help='Tool category')

    # Stock Category
    stock_parser = subparsers.add_parser('stock', help='Individual stock tools')
    stock_parser.add_argument('--type', choices=['basic', 'price', 'valuation', 'holders'], required=True)
    stock_parser.add_argument('--code', required=True)
    stock_parser.add_argument('--market', default='HK', choices=['HK', 'A'])

    # Statement Category
    stmt_parser = subparsers.add_parser('statement', help='Financial statement tools')
    stmt_parser.add_argument('--type', choices=['balance', 'income', 'cashflow'], required=True)
    stmt_parser.add_argument('--code', required=True)
    stmt_parser.add_argument('--market', default='HK', choices=['HK', 'A'])

    # Market Category
    mkt_parser = subparsers.add_parser('market', help='Market index tools')
    mkt_parser.add_argument('--index', choices=['hushen', 'hengsheng', 'shangzheng', 'nsdk'], required=True)

    # Industry Category
    ind_parser = subparsers.add_parser('industry', help='Industry indicators')
    ind_parser.add_argument('--type', choices=['gyzjz', 'production_yoy', 'pmi', 'cx_pmi', 'cpi', 'ppi', 'gdp', 'xfzxx', 'retail'], required=True)

    # Macro Category
    macro_parser = subparsers.add_parser('macro', help='Macro indicators')
    macro_parser.add_argument('--type', choices=['leverage', 'lpr', 'unemployment', 'shrzgm', 'trade', 'czsr', 'money_supply', 'reserve', 'bank_balance'], required=True)

    # Web Category
    web_parser = subparsers.add_parser('web', help='Web and search tools')
    web_parser.add_argument('--type', choices=['click', 'search_bing', 'search_bocha', 'search_google', 'search_finance'], required=True)
    web_parser.add_argument('--urls', help='Comma separated URLs for click')
    web_parser.add_argument('--task', help='Task description for click')
    web_parser.add_argument('--query', help='Search query')

    args = parser.parse_args()
    if not args.category:
        parser.print_help()
        return

    # Map types to classes
    tool_instance = None
    
    if args.category == 'stock':
        tool_map = {'basic': StockBasicInfo, 'price': StockPrice, 'valuation': StockBaseInfo, 'holders': ShareHoldingStructure}
        tool_instance = tool_map[args.type]()
    elif args.category == 'statement':
        tool_map = {'balance': BalanceSheet, 'income': IncomeStatement, 'cashflow': CashFlowStatement}
        tool_instance = tool_map[args.type]()
    elif args.category == 'market':
        tool_map = {'hushen': HuShen_Index, 'hengsheng': HengSheng_Index, 'shangzheng': ShangZheng_Index, 'nsdk': NSDK_Index}
        tool_instance = tool_map[args.index]()
    elif args.category == 'industry':
        tool_map = {
            'gyzjz': Industry_gyzjz, 'production_yoy': Industry_production_yoy, 'pmi': Industry_China_PMI,
            'cx_pmi': Industry_China_CX_services_PMI, 'cpi': Industry_China_CPI, 'ppi': Industry_China_PPI,
            'gdp': Industry_China_GDP, 'xfzxx': Industry_China_xfzxx, 'retail': Industry_China_consumer_goods_retail
        }
        tool_instance = tool_map[args.type]()
    elif args.category == 'macro':
        tool_map = {
            'leverage': Macro_China_Leverage_Ratio, 'lpr': Macro_China_LPR, 'unemployment': Macro_China_urban_unemployment,
            'shrzgm': Macro_China_shrzgm, 'trade': Macro_China_trade_balance, 'czsr': Macro_China_czsr,
            'money_supply': Macro_China_supply_of_money, 'reserve': Macro_China_fx_gold, 'bank_balance': Macro_China_bank_balance
        }
        tool_instance = tool_map[args.type]()
    elif args.category == 'web':
        tool_map = {
            'click': Click, 'search_bing': PlaywrightSearch, 'search_bocha': BochaSearch,
            'search_google': SerperSearch, 'search_finance': InDomainSearch_Request
        }
        tool_instance = tool_map[args.type]()

    if tool_instance:
        results = await run_tool(tool_instance, **vars(args))
        
        output_data = []
        if isinstance(results, list):
            for res in results:
                if hasattr(res, 'data'):
                    output_data.append(convert_to_serializable(res.data))
                else:
                    output_data.append(convert_to_serializable(res))
        else:
            output_data = convert_to_serializable(results)
                    
        print(json.dumps(output_data, ensure_ascii=False, indent=2, cls=DateEncoder))
    else:
        print(json.dumps({"error": "Unknown tool"}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
