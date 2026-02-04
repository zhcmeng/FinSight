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
    
    该指数由沪深市场中规模大、流动性好的最具代表性的300只股票组成，综合反映沪深A股市场整体表现。
    返回数据包括：日期、开盘、收盘、最高、最低、成交量、成交额等。
    """
    def __init__(self):
        super().__init__(
            name="CSI 300 daily data",
            description="Daily CSI 300 index data, including OHLC (Open, High, Low, Close), volume, turnover, returns, and turnover ratio.",
            parameters=[]
        )

    def prepare_params(self, task) -> dict:
        """
        从路由任务构建参数。此工具为全量指数查询，不需要额外参数。
        """
        return {}

    async def api_function(self):
        """
        获取沪深300指数的历史时间序列数据。
        
        接口说明：
            - 数据源：新浪财经 (Sina Finance)
            - 指数代码：sh000300 (上海交易所代码前缀为 sh)
            - 返回字段含义：
                - date: 交易日期
                - open: 当日开盘价
                - high: 当日最高价
                - low: 当日最低价
                - close: 当日收盘价
                - volume: 成交量（股）
        """
        try:
            # ak.stock_zh_index_daily 是 AkShare 提供的标准指数历史数据接口
            data = ak.stock_zh_index_daily(symbol="sh000300")
        except Exception as e:
            print(f"Failed to fetch CSI 300 data from Sina: {e}")
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
    
    恒生指数是香港蓝筹股变化的指标，由香港最大且最具流动性的公司组成。
    """
    def __init__(self):
        super().__init__(
            name="Hang Seng Index daily data",
            description="Daily Hang Seng Index data including OHLC, volume, turnover, returns, and turnover ratio.",
            parameters=[]
        )

    def prepare_params(self, task) -> dict:
        """
        构建参数。
        """
        return {}

    async def api_function(self):
        """
        获取恒生指数的历史时间序列。
        
        接口说明：
            - 数据源：新浪财经港股接口
            - 指数代码：HSI
            - 返回数据包含：日期、开盘、最高、最低、收盘、成交量、成交额。
        """
        try:
            # ak.stock_hk_index_daily_sina 专门用于获取港股指数的历史日线
            data = ak.stock_hk_index_daily_sina(symbol="HSI")
        except Exception as e:
            print(f"Failed to fetch Hang Seng data: {e}")
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
    
    上证指数（上证综指）反映了上海证券交易所全部上市股票的价格变动情况，是 A 股市场最重要的风向标。
    """
    def __init__(self):
        super().__init__(
            name="SSE Composite Index daily data",
            description="Daily SSE Composite Index (sh000001) data including OHLC and volume.",
            parameters=[]
        )

    def prepare_params(self, task) -> dict:
        """
        构建参数。
        """
        return {}

    async def api_function(self):
        """
        获取上证指数的历史时间序列。
        
        代码说明：
            - sh000001: 上证指数在上海交易所的标准代码。
        """
        try:
            # 获取上证指数日线数据
            data = ak.stock_zh_index_daily(symbol="sh000001")
        except Exception as e:
            print(f"Failed to fetch SSE Composite Index data: {e}")
            data = None
        if data is not None:
            return [
                ToolResult(
                    name = f"{self.name}",
                    description = self.description,
                    data = data,
                    source="Sina Finance: SSE Composite Index daily data. https://finance.sina.com.cn/realstock/company/sz000001/nc.shtml"
                )
            ]
        else:
            return []


class NSDK_Index(Tool):
    """
    纳斯达克综合指数 (.IXIC) 日线数据获取工具。
    
    纳斯达克综合指数反映了纳斯达克市场所有上市公司的股价走势，包含大量高科技企业。
    """
    def __init__(self):
        super().__init__(
            name="Nasdaq Composite Index daily data",
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
        
        代码说明：
            - .IXIC: 纳斯达克综合指数的代码。
        """
        try:
            # 获取纳斯达克指数日线数据
            data = ak.index_us_stock_sina(symbol=".IXIC")
        except Exception as e:
            print(f"Failed to fetch Nasdaq data: {e}")
            data = None
        if data is not None:
            return [
                ToolResult(
                    name = f"{self.name}",
                    description = self.description,
                    data = data,
                    source="Sina Finance: Nasdaq Composite Index daily data. https://stock.finance.sina.com.cn/usstock/quotes/.IXIC.html"
                )
            ]
        else:
            return []
