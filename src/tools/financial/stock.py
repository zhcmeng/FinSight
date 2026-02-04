"""
该模块提供了一系列工具，用于获取个股的基础信息、股东结构、估值指标以及历史价格数据。
支持 A 股和港股市场，集成了 AkShare、eFinance 和自定义爬虫逻辑。
"""

import requests
import json
import akshare as ak
import pandas as pd
import efinance as ef
from bs4 import BeautifulSoup

from ..base import Tool, ToolResult

# TODO: 添加更细粒度的雪球接口（提前区分沪/深）。
class StockBasicInfo(Tool):
    """
    股票基础信息查询工具。
    获取上市公司的企业概况，如所属行业、主营业务等。
    支持 A 股（同花顺接口）和港股（东方财富接口）。
    """
    def __init__(self):
        super().__init__(
            name="Stock profile",
            description="Return the basic corporate profile for a given ticker. Confirm the exchange-specific ticker before calling.",
            parameters=[
                {"name": "stock_code", "type": "str", "description": "Ticker, e.g., 000001", "required": True},
                {"name": "market", "type": "str", "description": "Market flag: HK for Hong Kong, A for A-share", "required": True},
            ],
        )

    def prepare_params(self, task) -> dict:
        """
        从路由任务中构建 API 调用参数。
        """
        if task.stock_code is None:
            # 应该在流转前已验证
            assert False, "Stock code cannot be empty"
        else:
            return {"stock_code": task.stock_code, "market": task.market}

    async def api_function(self, stock_code: str, market: str = "HK"):
        """
        根据市场标识调用对应的接口获取股票概况。
        """
        try:
            if market == "A":
                # 获取 A 股主营介绍
                data = ak.stock_zyjs_ths(symbol=stock_code)
            elif market == "HK":
                # 获取港股公司概况
                data = ak.stock_hk_company_profile_em(symbol=stock_code)
            else:
                raise ValueError(f"Unsupported market flag: {market}. Use 'HK' or 'A'.")
        except Exception as e:
            print("Failed to fetch basic stock info", e)
            data = None
        return [
            ToolResult(
                name = f"{self.name}: {stock_code}",
                description = f"Corporate profile for {stock_code}.",
                data = data,
                source="Xueqiu: Stock basic information. https://xueqiu.com/S"
            )
        ]


class ShareHoldingStructure(Tool):
    """
    股东结构查询工具。
    返回主要股东名称、持股数量、持股比例以及股份性质等。
    """
    def __init__(self):
        super().__init__(
            name="Shareholding structure",
            description="Return shareholder composition, including holder names, share counts, percentages, and equity type.",
            parameters=[
                {"name": "stock_code", "type": "str", "description": "Ticker, e.g., 000001", "required": True},
                {"name": "market", "type": "str", "description": "Market flag: HK for Hong Kong, A for A-share", "required": True},
            ],
        )

    def prepare_params(self, task) -> dict:
        """
        构建参数。
        """
        if task.stock_code is None:
            assert False, "Stock code cannot be empty"
        else:
            return {"stock_code": task.stock_code, "market": task.market}

    async def api_function(self, stock_code: str, market: str = "HK"):
        """
        获取指定股票的股东名单。
        A 股使用 AkShare 接口，港股通过东方财富 API 爬取。
        """
        try:
            if market == "A":
                # 获取 A 股主要股东
                data = ak.stock_main_stock_holder(stock=stock_code)
            elif market == "HK":
                # 从东方财富数据中心爬取港股持股数据
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                # 注意：此处的 API 链接包含特定的报告日期和 SecuCode 格式
                output = requests.get(
                    f"https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=RPT_HKF10_EQUITYCHG_HOLDER&columns=SECURITY_CODE%2CSECUCODE%2CORG_CODE%2CNOTICE_DATE%2CREPORT_DATE%2CHOLDER_NAME%2CTOTAL_SHARES%2CTOTAL_SHARES_RATIO%2CDIRECT_SHARES%2CSHARES_CHG_RATIO%2CSHARES_TYPE%2CEQUITY_TYPE%2CHOLD_IDENTITY%2CIS_ZJ&quoteColumns=&filter=(SECUCODE%3D%22{stock_code}.HK%22)(REPORT_DATE%3D%272024-12-31%27)&pageNumber=1&pageSize=&sortTypes=-1%2C-1&sortColumns=EQUITY_TYPE%2CTOTAL_SHARES&source=F10&client=PC&v=032666133943694553",
                    headers = headers,
                )
                try:
                    html = output.text
                    output = json.loads(html)
                    data = output["result"]["data"]
                    data = pd.DataFrame(data)
                    # 重命名列名以增强可读性
                    data = data.rename(columns={
                        'HOLDER_NAME': 'holder_name',
                        'TOTAL_SHARES': 'shares',
                        'TOTAL_SHARES_RATIO': 'ownership_pct',
                        'DIRECT_SHARES': 'direct_shares',
                        'HOLD_IDENTITY': 'ownership_type',
                        'IS_ZJ': 'is_direct'
                    })
                    # 仅保留核心字段
                    data = data.loc[:, ['holder_name', 'shares', 'ownership_pct', 'ownership_type', 'is_direct']]
                    data['is_direct'] = data['is_direct'].map({'1': 'Yes', '0': 'No'})
                    # 按持股比例降序排列
                    data.sort_values(by='ownership_pct', ascending=False, inplace=True)
                    data.reset_index(drop=True, inplace=True)
                except Exception as e:
                    print("Failed to parse Hong Kong shareholding structure", e)
                    data = None
            else:
                raise ValueError(f"Unsupported market flag: {market}. Use 'HK' or 'A'.")
        except Exception as e:
            print("Failed to fetch shareholding structure", e)
            data = None
        return [
            ToolResult(
                name=f"{self.name} (ticker: {stock_code})",
                description=self.description,
                data=data,
                source="Sina Finance: Shareholder structure. https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockHolder/stockid/600004.phtml"
            )
        ]

class StockBaseInfo(Tool):
    """
    权益估值指标工具。
    获取市盈率 (PE)、市净率 (PB)、净资产收益率 (ROE) 和毛利率等财务与估值数据。
    """
    def __init__(self):
        super().__init__(
            name="Equity valuation metrics",
            description="Return valuation and profitability metrics such as PE, PB, ROE, and gross margin.",
            parameters=[
                {"name": "stock_code", "type": "str", "description": "Ticker, e.g., 000001", "required": True},
            ],
        )

    def prepare_params(self, task) -> dict:
        """构建参数"""
        return {"stock_code": task.stock_code}

    async def api_function(self, stock_code: str, market: str = "HK"):
        """
        通过 efinance 获取个股的基础估值信息。
        """
        try:
            data = ef.stock.get_base_info(stock_code)
        except Exception as e:
            print("Failed to fetch stock valuation info", e)
            data = None
        return [
            ToolResult(
                name=f"{self.name} (ticker: {stock_code})",
                description=self.description,
                data=data,
                source="Exchange filings: Equity valuation metrics."
            )
        ]


class StockPrice(Tool):
    """
    股票历史行情工具。
    获取日线级别的 K 线数据，包括开盘价、收盘价、成交量以及涨跌幅等。
    """
    def __init__(self):
        super().__init__(
            name="Stock candlestick data",
            description="Daily OHLCV data including turnover and rate-of-change metrics.",
            parameters=[
                {"name": "stock_code", "type": "str", "description": "Ticker/Stock Code (support A-share and HK-share), e.g., 000001", "required": True},
            ],
        )

    def prepare_params(self, task) -> dict:
        """构建参数"""
        return {"stock_code": task.stock_code}

    async def api_function(self, stock_code: str, market: str = "HK"):
        """
        获取请求股票的历史行情数据。
        """
        try:
            # 使用 efinance 获取历史 K 线数据
            data = ef.stock.get_quote_history(stock_code)
        except Exception as e:
            print("Failed to fetch stock price history", e)
            data = None
        return [
            ToolResult(
                name=f"{self.name} (ticker: {stock_code})",
                description=self.description,
                data=data,
                source="Exchange trading data: OHLCV history."
            )
        ]
