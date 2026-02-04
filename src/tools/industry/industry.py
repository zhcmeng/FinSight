"""
该模块提供了一系列工具，用于获取中国宏观经济和行业相关的核心指标。
数据涵盖工业增加值、PMI、CPI、GDP、PPI、消费信心及社会零售总额等关键宏观经济数据。
所有数据均通过 AkShare 接口从东方财富、新浪财经等主流金融门户获取。
"""

import akshare as ak
import pandas as pd
from ..base import Tool, ToolResult


class Industry_gyzjz(Tool):
    """
    工业增加值增长工具。
    获取 2008 年至今中国工业增加值的增长数据（来自东方财富）。
    """
    def __init__(self):
        super().__init__(
            name = "Industrial value-added growth",
            description = "China industrial value-added growth from 2008 onward (Eastmoney).",
            parameters = [],
        ) 
        
    async def api_function(self):
        """
        调用 AkShare 接口获取工业增加值增长数据。
        """
        data = ak.macro_china_gyzjz()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Industrial value-added growth. https://data.eastmoney.com/cjsj/gyzjz.html",
            )
        ]
        
class Industry_production_yoy(Tool):
    """
    规模以上工业生产同比增长工具。
    获取 1990 年至今中国规模以上工业增加值的同比增长数据。
    """
    def __init__(self):
        super().__init__(
            name = "Above-scale industrial production YoY",
            description = "China's YoY industrial production growth for enterprises above designated size, from 1990 onward.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取规模以上工业生产同比增长数据。
        """
        data = ak.macro_china_industrial_production_yoy()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Above-scale industrial production YoY. https://datacenter.jin10.com/reportType/dc_chinese_industrial_production_yoy",
            )
        ]
        
class Industry_China_PMI(Tool):
    """
    官方制造业 PMI 工具。
    获取 2005 年至今的中国官方制造业采购经理指数 (PMI) 序列。
    """
    def __init__(self):
        super().__init__(
            name = "Official manufacturing PMI",
            description = "China's official manufacturing PMI series from 2005 onward.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取中国官方制造业 PMI 年度数据。
        """
        data = ak.macro_china_pmi_yearly()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Official manufacturing PMI. https://datacenter.jin10.com/reportType/dc_chinese_manufacturing_pmi",
            )
        ]
        
class Industry_China_CX_services_PMI(Tool):
    """
    财新服务业 PMI 工具。
    获取 2012 年至今的中国财新服务业采购经理指数 (PMI) 报告。
    """
    def __init__(self):
        super().__init__(
            name = "Caixin services PMI",
            description = "China's Caixin services PMI report from 2012 onward.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取财新服务业 PMI 年度数据。
        """
        data = ak.macro_china_cx_services_pmi_yearly()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Caixin services PMI. https://datacenter.jin10.com/reportType/dc_chinese_caixin_services_pmi",
            )
        ]
        
class Industry_China_CPI(Tool):
    """
    消费者物价指数 (CPI) 工具。
    获取 2008 年至今中国月度 CPI 数据。
    """
    def __init__(self):
        super().__init__(
            name = "Consumer price index",
            description = "Monthly CPI data for China from 2008 onward.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取中国月度 CPI 数据。
        """
        data = ak.macro_china_cpi()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Consumer price index. http://data.eastmoney.com/cjsj/cpi.html",
            )
        ]
        
class Industry_China_GDP(Tool):
    """
    国内生产总值 (GDP) 工具。
    获取 2006 年至今中国 GDP 相关的季度或月度统计数据。
    """
    def __init__(self):
        super().__init__(
            name = "Gross domestic product",
            description = "Monthly GDP-related statistics for China from 2006 onward.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取中国 GDP 统计数据。
        """
        data = ak.macro_china_gdp()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Gross domestic product. http://data.eastmoney.com/cjsj/gdp.html",
            )
        ]
        
class Industry_China_PPI(Tool):
    """
    生产者物价指数 (PPI) 工具。
    获取 2006 年至今中国月度工业品出厂价格指数 (PPI)。
    """
    def __init__(self):
        super().__init__(
            name = "Producer price index",
            description = "Monthly producer price index (ex-factory) for China from 2006 onward.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取中国月度 PPI 数据。
        """
        data = ak.macro_china_ppi()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Producer price index. http://data.eastmoney.com/cjsj/ppi.html",
            )
        ]
        
class Industry_China_xfzxx(Tool):
    """
    消费者信心指数工具。
    获取历史消费者信心指数及其同比和环比变化数据。
    """
    def __init__(self):
        super().__init__(
            name = "Consumer confidence index",
            description = "Historical consumer confidence index with YoY and MoM changes (Eastmoney).",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取消费者信心指数数据。
        """
        data = ak.macro_china_xfzxx()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Consumer confidence index. https://data.eastmoney.com/cjsj/xfzxx.html",
            )
        ]
        
class Industry_China_consumer_goods_retail(Tool):
    """
    社会消费品零售总额工具。
    获取社会消费品零售总额的历史统计数据及其变化趋势。
    """
    def __init__(self):
        super().__init__(
            name = "Total retail sales of consumer goods",
            description = "Historical stats for total retail sales of consumer goods with YoY and MoM changes.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取社会消费品零售总额数据。
        """
        data = ak.macro_china_consumer_goods_retail()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Total retail sales of consumer goods. http://data.eastmoney.com/cjsj/xfp.html",
            )
        ]
        
class Industry_China_retail_price_index(Tool):
    """
    零售价格指数工具。
    获取国家统计局发布的历史零售价格指数数据。
    """
    def __init__(self):
        super().__init__(
            name = "Retail price index",
            description = "Historical retail price index from the National Bureau of Statistics.",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取零售价格指数数据。
        """
        data = ak.macro_china_retail_price_index()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Sina Finance: Retail price index. http://finance.sina.com.cn/mac/#price-12-0-31-1",
            )
        ]
        
class Industry_China_qyspjg(Tool):
    """
    企业商品价格指数工具。
    获取 2005 年至今的企业商品价格指数序列。
    """
    def __init__(self):
        super().__init__(
            name = "Enterprise commodity price index",
            description = "Enterprise commodity price index series from 2005 onward (Eastmoney).",
            parameters = [],
        )
        
    async def api_function(self):
        """
        获取企业商品价格指数数据。
        """
        data = ak.macro_china_qyspjg()
        return [
            ToolResult(
                name=self.name,
                description=self.description,
                data=data,
                source="Eastmoney: Enterprise commodity price index. http://data.eastmoney.com/cjsj/qyspjg.html",
            )
        ]
