"""
该模块提供了获取全球主要股票市场指数日线数据的工具。
通过 AkShare 接口获取包括沪深300、恒生指数、上证指数和纳斯达克指数在内的行情数据。
"""

import akshare as ak
import pandas as pd
from ..base import Tool, ToolResult


class HuShen_Index(Tool):
    """
    沪深300指数日线数据获取工具。
    返回包括开盘价、最高价、最低价、收盘价、成交量、成交额等信息。
    """
    def __init__(self):
        super().__init__(
            name="CSI 300 daily data",
            description="Daily CSI 300 index data, including OHLC, volume, turnover, returns, and turnover ratio.",
            parameters=[]
        )

    def prepare_params(self, task) -> dict:
        """
        从路由任务构建参数（此工具不需要额外参数）。
        """
        return {}

    async def api_function(self):
        """
        获取沪深300指数的历史时间序列数据。
        使用新浪财经接口，代码为 sh000300。
        """
        try:
            # 获取沪深300指数日线数据
            data = ak.stock_zh_index_daily(symbol="sh000300")
        except Exception as e:
            print("Failed to fetch CSI 300 data", e)
            data = None
        if data is not None:
            return [
                ToolResult(
                    name = f"{self.name}",
                    description = self.description,
                    data = data,
                    source="Sina Finance: CSI 300 daily data. https://finance.sina.com.cn/realstock/company/sz000300/nc.shtml"
                )
            ]
        else:
            return []


class HengSheng_Index(Tool):
    """
    恒生指数 (HSI) 日线数据获取工具。
    提供港股市场的基准指数行情。
    """
    def __init__(self):
        super().__init__(
            name="Hang Seng Index daily data",
            description="Daily Hang Seng Index data including OHLC, volume, turnover, returns, and turnover ratio.",
            parameters=[]
        )

    def prepare_params(self, task) -> dict:
        """
        构建参数（不需要额外参数）。
        """
        return {}

    async def api_function(self):
        """
        获取恒生指数的历史时间序列。
        使用新浪财经的港股指数接口，代码为 HSI。
        """
        try:
            # 获取恒生指数日线数据
            data = ak.stock_hk_index_daily_sina(symbol="HSI")
        except Exception as e:
            print("Failed to fetch Hang Seng data", e)
            data = None
        if data is not None:
            return [
                ToolResult(
                    name = f"{self.name}",
                    description = self.description,
                    data = data,
                    source="Sina Finance: Hang Seng Index daily data. https://stock.finance.sina.com.cn/hkstock/quotes/HSI.html."
                )
            ]
        else:
            return []
        
class ShangZheng_Index(Tool):
    """
    上证指数日线数据获取工具。
    反映上海证券交易所挂牌股票的总体走势。
    """
    def __init__(self):
        super().__init__(
            name="SSE Composite daily data",
            description="Daily Shanghai Composite index data with OHLC, volume, turnover, returns, and turnover ratio.",
            parameters=[]
        )

    def prepare_params(self, task) -> dict:
        """
        构建参数。
        """
        return {}

    async def api_function(self):
        """
        获取上证指数的时间序列数据。
        代码为 sh000001。
        """
        try:
            # 获取上证指数日线数据
            data = ak.stock_zh_index_daily(symbol="sh000001")
        except Exception as e:
            print("Failed to fetch SSE Composite data", e)
            data = None
        if data is not None:
            return [
                ToolResult(
                    name = f"{self.name}",
                    description = self.description,
                    data = data,
                    source="Sina Finance: SSE Composite daily data. https://finance.sina.com.cn/realstock/company/sz000001/nc.shtml"
                )
            ]
        else:
            return []


class NSDK_Index(Tool):
    """
    纳斯达克综合指数 (.IXIC) 日线数据获取工具。
    反映美国纳斯达克市场的整体表现。
    """
    def __init__(self):
        super().__init__(
            name="Nasdaq Composite daily data",
            description="Daily Nasdaq Composite data covering OHLC, volume, turnover, returns, and turnover ratio.",
            parameters=[]
        )

    def prepare_params(self, task) -> dict:
        """
        构建参数。
        """
        return {}

    async def api_function(self):
        """
        获取纳斯达克指数的时间序列。
        使用美股行情接口，代码为 .IXIC。
        """
        try:
            # 获取纳斯达克指数日线数据
            data = ak.index_us_stock_sina(symbol=".IXIC")
        except Exception as e:
            print("Failed to fetch Nasdaq data", e)
            data = None
        if data is not None:
            return [
                ToolResult(
                    name = f"{self.name}",
                    description = self.description,
                    data = data,
                    source="Sina Finance: Nasdaq Composite daily data. https://stock.finance.sina.com.cn/usstock/quotes/.IXIC.html"
                )
            ]
        else:
            return []
