"""
该模块提供了一系列工具，用于从 AkShare 获取和预处理上市公司的财务报表数据。
主要包括资产负债表、利润表和现金流量表的查询与清洗逻辑。
"""

import akshare as ak
import pandas as pd
from ..base import Tool, ToolResult

def parse_ths_amount(val):
    """解析同花顺金额字符串，转换为百万单位的浮点数"""
    if pd.isna(val) or val == "" or val == "--":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val) / 1000000
    val = str(val)
    unit = 1.0
    if '亿' in val:
        unit = 100.0  # 1亿 = 100百万
        val = val.replace('亿', '')
    elif '万' in val:
        unit = 0.01  # 1万 = 0.01百万
        val = val.replace('万', '')
    
    try:
        val = val.replace(',', '')
        return float(val) * unit
    except:
        return 0.0

def preprocess_ths_data(df):
    """将同花顺 (ths) 的宽表格式转换为与东方财富 (em) 一致的格式"""
    if df is None or df.empty:
        return df
    
    df = df.copy()
    # 报告期作为年份列
    years = df['报告期'].tolist()
    
    # 转置
    df = df.set_index('报告期')
    df = df.transpose()
    
    # 重置索引
    df = df.reset_index()
    item_col = '会计年度 (人民币百万)'
    df = df.rename(columns={'index': item_col})
    
    # 转换数值
    for year in years:
        df[year] = df[year].apply(parse_ths_amount)
    
    # 移除导航/分类行
    blacklist = ['报表核心指标', '报表全部指标', '一、经营活动产生的现金流量', 
                 '二、投资活动产生的现金流量', '三、筹资活动产生的现金流量',
                 '一、营业总收入', '二、营业总成本', '三、营业利润', '四、利润总额', '五、净利润']
    df = df[~df[item_col].isin(blacklist)]
    
    # 视觉增强
    df[item_col] = df[item_col].apply(lambda x: f"**{x}**" if '合计' in x or '总计' in x or '净利润' in x or '总收入' in x else x)
    
    # 仅保留最近 5 年
    year_cols = sorted([col for col in df.columns if col != item_col])[-5:]
    df = df[[item_col] + year_cols]
    
    return df

def preprocess_balance_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    预处理原始资产负债表数据，将其转换为整洁的透视表格式。
    
    参数:
        data: 从 akshare 获取的原始 DataFrame。
        
    返回:
        处理后的 DataFrame，包含最近 5 年的数据，金额单位为百万人民币。
    """
    # 移除不需要的元数据列（多为东财内部使用的机构代码、证券代码等非财务指标列）
    data.drop(['SECUCODE','SECURITY_CODE','SECURITY_NAME_ABBR','ORG_CODE', 'DATE_TYPE_CODE', 'FISCAL_YEAR','STD_ITEM_CODE','REPORT_DATE'], axis=1, inplace=True)
    
    # 从报告日期提取年份，以便后续按年度进行数据透视
    data['YEAR'] = data['STD_REPORT_DATE'].apply(lambda x: pd.to_datetime(x).year)
    data.drop(['STD_REPORT_DATE'], axis=1, inplace=True)
    
    # 统一数值显示格式，并将原始单位（元）转换为百万人民币，方便在 UI 中简洁展示
    pd.set_option('display.float_format', '{:.2f}'.format) 
    data['AMOUNT'] = data['AMOUNT'].apply(lambda x: float(x)//1000000)

    # 记录原始科目出现的顺序。由于 pivot 操作会破坏行序，我们需要在透视后按此顺序恢复，
    # 确保报表符合会计准则的常规展示顺序（如：流动资产 -> 非流动资产）。
    item_order = {item: idx for idx, item in enumerate(data['STD_ITEM_NAME'].unique())}

    # 执行数据透视：将纵向排列的科目金额转换为以年份为列的横向对比格式
    pivot_df = data.pivot_table(
        index='STD_ITEM_NAME',
        columns='YEAR',
        values='AMOUNT',
        aggfunc='sum'
    ).reset_index()

    pivot_df.columns.name = None
    # 恢复会计科目的原始展示顺序
    pivot_df['original_order'] = pivot_df['STD_ITEM_NAME'].map(item_order)
    pivot_df = pivot_df.sort_values('original_order').drop(columns='original_order')

    # 确保年份列按从旧到新的顺序排列
    year_cols = sorted([col for col in pivot_df.columns if col != 'STD_ITEM_NAME'])
    pivot_df = pivot_df[['STD_ITEM_NAME'] + year_cols]

    # 质量过滤：过滤掉在查询周期内缺失值过多的科目。
    # 这里允许最多 3 年的数据缺失（在通常的 5-10 年观察期内），以保留可能有意义的历史科目。
    pivot_df['nan_count'] = pivot_df.iloc[:, 1:].isna().sum(axis=1)
    filtered_df = pivot_df[pivot_df['nan_count'] <= 3].copy()
    filtered_df.drop(columns='nan_count', inplace=True)

    filtered_df.reset_index(drop=True, inplace=True)

    # 格式化输出：仅展示最近 5 年的数据，并将科目列重命名为中文。
    filtered_df = filtered_df.rename(columns={'STD_ITEM_NAME': '类目'})
    use_columns = ['类目'] + [col for col in filtered_df.columns if col != '类目'][-5:]
    filtered_df = filtered_df.loc[:, use_columns]
    
    # 增强可读性：利用 Markdown 语法加粗“总计”或“合计”类科目，方便视觉识别
    filtered_df['类目'] = filtered_df['类目'].apply(lambda x: f"**{x}**" if x.startswith('总') else x)
    return filtered_df


def preprocess_income_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    预处理原始利润表数据，将其转换为整洁的透视表格式。
    
    参数:
        data: 从 akshare 获取的原始 DataFrame。
        
    返回:
        处理后的 DataFrame，包含最近 5 年的数据，金额单位为百万人民币。
    """
    # 移除冗余的元数据列
    data.drop(['SECUCODE','SECURITY_CODE','SECURITY_NAME_ABBR','ORG_CODE', 'DATE_TYPE_CODE', 'FISCAL_YEAR','STD_ITEM_CODE','REPORT_DATE'], axis=1, inplace=True)
    
    # 利润表数据通常带有开始日期和结束日期，此处提取年份用于年度对比
    data['YEAR'] = data['START_DATE'].apply(lambda x: pd.to_datetime(x).year)
    data.drop(['START_DATE'], axis=1, inplace=True)
    
    # 转换金额单位至百万人民币
    pd.set_option('display.float_format', '{:.2f}'.format) 
    data['AMOUNT'] = data['AMOUNT'].apply(lambda x: float(x)//1000000)

    # 维护会计科目原始顺序（营业收入 -> 营业成本 -> ... -> 净利润）
    item_order = {item: idx for idx, item in enumerate(data['STD_ITEM_NAME'].unique())}

    # 数据透视处理
    pivot_df = data.pivot_table(
        index='STD_ITEM_NAME',
        columns='YEAR',
        values='AMOUNT',
        aggfunc='sum'
    ).reset_index()

    pivot_df.columns.name = None
    pivot_df['original_order'] = pivot_df['STD_ITEM_NAME'].map(item_order)
    pivot_df = pivot_df.sort_values('original_order').drop(columns='original_order')

    # 年份升序排列
    year_cols = sorted([col for col in pivot_df.columns if col != 'STD_ITEM_NAME'])
    pivot_df = pivot_df[['STD_ITEM_NAME'] + year_cols]

    # 过滤空值超过 3 年的科目
    pivot_df['nan_count'] = pivot_df.iloc[:, 1:].isna().sum(axis=1)
    filtered_df = pivot_df[pivot_df['nan_count'] <= 3].copy()
    filtered_df.drop(columns='nan_count', inplace=True)

    filtered_df.reset_index(drop=True, inplace=True)

    # 取最近 5 年数据并加粗总计项
    filtered_df = filtered_df.rename(columns={'STD_ITEM_NAME': '类目'})
    use_columns = ['类目'] + [col for col in filtered_df.columns if col != '类目'][-5:]
    filtered_df = filtered_df.loc[:, use_columns]
    filtered_df['类目'] = filtered_df['类目'].apply(lambda x: f"**{x}**" if x.startswith('总') else x)
    return filtered_df

class BalanceSheet(Tool):
    """
    资产负债表查询工具。
    提供指定股票的资产、负债和股东权益数据。
    支持港股 (HK) 和 A 股市场。数据来源主要为东方财富。
    """
    def __init__(self):
        super().__init__(
            name = "Balance sheet",
            description = "Returns the balance sheet covering assets, liabilities, and shareholders' equity for a given ticker.",
            parameters = [
                {"name": "stock_code", "type": "str", "description": "Ticker, e.g., 000001", "required": True},
                {"name": "market", "type": "str", "description": "Market flag: HK or A", "required": True},
                {"name": "period", "type": "str", "description": "Reporting period (defaults to annual)", "required": False},
            ],
        )

    def prepare_params(self, task) -> dict:
        """
        从任务对象中构建 API 调用参数。
        """
        if task.stock_code is None:
            # 内部校验，防止空代码进入 api_function
            assert False, "Stock code cannot be empty"
        else:
            # 默认返回年度数据，因为年度报表最能反映企业的长期财务状况
            return {"stock_code": task.stock_code, "market": task.market, "period": "annual"}
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        类内部使用的预处理逻辑，与全局函数类似，但主要针对 BalanceSheet 的展示需求。
        """
        # 移除技术性元数据列
        data.drop(['SECUCODE','SECURITY_CODE','SECURITY_NAME_ABBR','ORG_CODE', 'DATE_TYPE_CODE', 'FISCAL_YEAR','STD_ITEM_CODE','REPORT_DATE'], axis=1, inplace=True)
        # 提取报告年份
        data['YEAR'] = data['STD_REPORT_DATE'].apply(lambda x: pd.to_datetime(x).year)
        data.drop(['STD_REPORT_DATE'], axis=1, inplace=True)
        # 换算金额单位：元 -> 百万人民币
        pd.set_option('display.float_format', '{:.2f}'.format) 
        data['AMOUNT'] = data['AMOUNT'].apply(lambda x: float(x)//1000000)

        # 保持会计科目的逻辑顺序
        item_order = {item: idx for idx, item in enumerate(data['STD_ITEM_NAME'].unique())}

        # 数据透视
        pivot_df = data.pivot_table(
            index='STD_ITEM_NAME',
            columns='YEAR',
            values='AMOUNT',
            aggfunc='sum'
        ).reset_index()

        pivot_df.columns.name = None
        pivot_df['original_order'] = pivot_df['STD_ITEM_NAME'].map(item_order)
        pivot_df = pivot_df.sort_values('original_order').drop(columns='original_order')

        # 排序年份
        year_cols = sorted([col for col in pivot_df.columns if col != 'STD_ITEM_NAME'])
        pivot_df = pivot_df[['STD_ITEM_NAME'] + year_cols]

        # 过滤低频科目
        pivot_df['nan_count'] = pivot_df.iloc[:, 1:].isna().sum(axis=1)
        filtered_df = pivot_df[pivot_df['nan_count'] <= 3].copy()
        filtered_df.drop(columns='nan_count', inplace=True)

        filtered_df.reset_index(drop=True, inplace=True)

        # 设置列名并保留最近 5 年
        filtered_df = filtered_df.rename(columns={'STD_ITEM_NAME': '会计年度 (人民币百万)'})
        use_columns = ['会计年度 (人民币百万)'] + [col for col in filtered_df.columns if col != '会计年度 (人民币百万)'][-5:]
        filtered_df = filtered_df.loc[:, use_columns]
        # 视觉增强
        filtered_df['会计年度 (人民币百万)'] = filtered_df['会计年度 (人民币百万)'].apply(lambda x: f"**{x}**" if x.startswith('总') else x)
        return filtered_df

    async def api_function(self, stock_code: str, market: str = "HK", period: str = "年度"):
        """
        异步调用 AkShare 获取并预处理资产负债表数据。
        """
        period = "年度"
        try:
            if market == "HK":
                # 获取港股资产负债表，使用东方财富提供的香港财报接口
                data = ak.stock_financial_hk_report_em(
                    stock = stock_code,
                    symbol = "资产负债表",
                    indicator = period,
                )
                try:
                    data = self._preprocess_data(data)
                except Exception as e:
                    print(f"Failed to preprocess balance-sheet data for {stock_code}", e)
            elif market == "A":
                # 获取 A 股年度资产负债表，使用同花顺接口，该接口比东财接口更稳定
                data = ak.stock_financial_debt_ths(
                    symbol = stock_code,
                    indicator = "按年度",
                )
                try:
                    data = preprocess_ths_data(data)
                except Exception as e:
                    print(f"Failed to preprocess A-share balance-sheet data for {stock_code}", e)
            else:
                raise ValueError(f"Unsupported market flag: {market}. Use 'HK' or 'A'.")
        except Exception as e:
            print(f"Failed to fetch balance sheet for {stock_code}", e)
            data = None
        return [
            ToolResult(
                name = f"{self.name} (ticker: {stock_code})",
                description = f"Balance sheet for ticker {stock_code}.",
                data = data,
                source=f"Eastmoney financials: balance sheet for {stock_code}. https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index?type=web&code={stock_code}#lrb-0."
            )
        ]

class IncomeStatement(Tool):
    """
    利润表查询工具。
    提供收入、成本、费用和盈余等详细数据，帮助用户分析企业的盈利能力。
    """
    def __init__(self):
        super().__init__(
            name = "Income statement",
            description = "Returns the income statement detailing revenue, costs, expenses, and earnings for a given ticker.",
            parameters = [
                {"name": "stock_code", "type": "str", "description": "Ticker, e.g., 000001", "required": True},
                {"name": "market", "type": "str", "description": "Market flag: HK or A", "required": True},
            ],
        )

    def prepare_params(self, task) -> dict:
        """构建参数"""
        if task.stock_code is None:
            assert False, "Stock code cannot be empty"
        else:
            return {"stock_code": task.stock_code, "market": task.market, "period": "annual"}

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """利润表数据清洗逻辑"""
        data.drop(['SECUCODE','SECURITY_CODE','SECURITY_NAME_ABBR','ORG_CODE', 'DATE_TYPE_CODE', 'FISCAL_YEAR','STD_ITEM_CODE','REPORT_DATE'], axis=1, inplace=True)
        data['YEAR'] = data['START_DATE'].apply(lambda x: pd.to_datetime(x).year)
        data.drop(['START_DATE'], axis=1, inplace=True)
        pd.set_option('display.float_format', '{:.2f}'.format) 
        data['AMOUNT'] = data['AMOUNT'].apply(lambda x: float(x)//1000000)

        item_order = {item: idx for idx, item in enumerate(data['STD_ITEM_NAME'].unique())}

        pivot_df = data.pivot_table(
            index='STD_ITEM_NAME',
            columns='YEAR',
            values='AMOUNT',
            aggfunc='sum'
        ).reset_index()

        pivot_df.columns.name = None
        pivot_df['original_order'] = pivot_df['STD_ITEM_NAME'].map(item_order)
        pivot_df = pivot_df.sort_values('original_order').drop(columns='original_order')

        year_cols = sorted([col for col in pivot_df.columns if col != 'STD_ITEM_NAME'])
        pivot_df = pivot_df[['STD_ITEM_NAME'] + year_cols]

        pivot_df['nan_count'] = pivot_df.iloc[:, 1:].isna().sum(axis=1)
        filtered_df = pivot_df[pivot_df['nan_count'] <= 3].copy()
        filtered_df.drop(columns='nan_count', inplace=True)

        filtered_df.reset_index(drop=True, inplace=True)

        filtered_df = filtered_df.rename(columns={'STD_ITEM_NAME': '会计年度 (人民币百万)'})
        use_columns = ['会计年度 (人民币百万)'] + [col for col in filtered_df.columns if col != '会计年度 (人民币百万)'][-5:]
        filtered_df = filtered_df.loc[:, use_columns]
        filtered_df['会计年度 (人民币百万)'] = filtered_df['会计年度 (人民币百万)'].apply(lambda x: f"**{x}**" if x.startswith('总') else x)
        return filtered_df
        

    async def api_function(self, stock_code: str, market: str = "HK", period: str = "年度"):
        """获取请求股票的利润表"""
        period = "年度"
        try:
            if market == "HK":
                # 获取港股利润表
                data = ak.stock_financial_hk_report_em(stock=stock_code, symbol="利润表", indicator=period)
                try:
                    data = self._preprocess_data(data)
                except Exception as e:
                    print(f"Failed to preprocess income-statement data for {stock_code}", e)
            elif market == "A":
                # 获取 A 股利润表，通过同花顺 (iFinD) 接口获取，该接口数据较为全面
                data = ak.stock_financial_benefit_ths(symbol=stock_code, indicator='按年度')
                try:
                    data = preprocess_ths_data(data)
                except Exception as e:
                    print(f"Failed to preprocess A-share income-statement data for {stock_code}", e)
            else:
                raise ValueError(f"Unsupported market flag: {market}. Use 'HK' or 'A'.")
        except Exception as e:
            print(f"Failed to fetch income statement for {stock_code}", e)
            data = None
        
        return [
            ToolResult(
                name = f"{self.name} (ticker: {stock_code})",
                description = f"Income statement for ticker {stock_code}.",
                data = data,
                source=f"iFinD/10jqka financials: income statement for {stock_code}. https://basic.10jqka.com.cn/new/{stock_code}/finance.html."
            )
        ]


class CashFlowStatement(Tool):
    """
    现金流量表查询工具。
    展示企业在经营、投资和筹资活动中产生的现金流入与流出，衡量企业的现金管理能力。
    """
    def __init__(self):
        super().__init__(
            name="Cash-flow statement",
            description="Returns cash-flow statements showing operating, investing, and financing cash movements for a given ticker.",
            parameters=[
                {"name": "stock_code", "type": "str", "description": "Ticker, e.g., 000001", "required": True},
                {"name": "market", "type": "str", "description": "Market flag: HK or A", "required": True},
            ],
        )

    def prepare_params(self, task) -> dict:
        """构建参数"""
        if task.stock_code is None:
            assert False, "Stock code cannot be empty"
        else:
            return {"stock_code": task.stock_code, "market": task.market, "period": "年度"}

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """现金流量表数据清洗"""
        data.drop(['SECUCODE','SECURITY_CODE','SECURITY_NAME_ABBR','ORG_CODE', 'DATE_TYPE_CODE', 'FISCAL_YEAR','STD_ITEM_CODE','REPORT_DATE'], axis=1, inplace=True)
        data['YEAR'] = data['START_DATE'].apply(lambda x: pd.to_datetime(x).year)
        data.drop(['START_DATE'], axis=1, inplace=True)
        pd.set_option('display.float_format', '{:.2f}'.format) 
        data['AMOUNT'] = data['AMOUNT'].apply(lambda x: float(x)//1000000)

        item_order = {item: idx for idx, item in enumerate(data['STD_ITEM_NAME'].unique())}

        pivot_df = data.pivot_table(
            index='STD_ITEM_NAME',
            columns='YEAR',
            values='AMOUNT',
            aggfunc='sum'
        ).reset_index()

        pivot_df.columns.name = None
        pivot_df['original_order'] = pivot_df['STD_ITEM_NAME'].map(item_order)
        pivot_df = pivot_df.sort_values('original_order').drop(columns='original_order')

        year_cols = sorted([col for col in pivot_df.columns if col != 'STD_ITEM_NAME'])
        pivot_df = pivot_df[['STD_ITEM_NAME'] + year_cols]

        pivot_df['nan_count'] = pivot_df.iloc[:, 1:].isna().sum(axis=1)
        filtered_df = pivot_df[pivot_df['nan_count'] <= 3].copy()
        filtered_df.drop(columns='nan_count', inplace=True)

        filtered_df.reset_index(drop=True, inplace=True)

        filtered_df = filtered_df.rename(columns={'STD_ITEM_NAME': '会计年度 (人民币百万)'})
        use_columns = ['会计年度 (人民币百万)'] + [col for col in filtered_df.columns if col != '会计年度 (人民币百万)'][-5:]
        filtered_df = filtered_df.loc[:, use_columns]
        filtered_df['会计年度 (人民币百万)'] = filtered_df['会计年度 (人民币百万)'].apply(lambda x: f"**{x}**" if x.startswith('总') else x)
        return filtered_df


    async def api_function(self, stock_code: str, market: str = "HK", period: str = "年度"):
        """获取请求股票的现金流量表"""
        period = "年度"
        try:
            if market == "HK":
                # 获取港股现金流量表
                data = ak.stock_financial_hk_report_em(stock=stock_code, symbol="现金流量表", indicator=period)
                try:
                    data = self._preprocess_data(data)
                except Exception as e:
                    print(f"Failed to preprocess cash-flow data for {stock_code}", e)
            elif market == "A":
                # 获取 A 股现金流量表，同样使用同花顺接口
                data = ak.stock_financial_cash_ths(symbol=stock_code, indicator='按年度')
                try:
                    data = preprocess_ths_data(data)
                except Exception as e:
                    print(f"Failed to preprocess A-share cash-flow data for {stock_code}", e)
            else:
                raise ValueError(f"Unsupported market flag: {market}. Use 'HK' or 'A'.")
        except Exception as e:
            print(f"Failed to fetch cash-flow statement for {stock_code}", e)
            data = None
        return [
            ToolResult(
                name = f"{self.name} (ticker: {stock_code})",
                description = f"Cash-flow statement for ticker {stock_code}.",
                data = data,
                source=f"iFinD/10jqka financials: cash-flow statement for {stock_code}. https://basic.10jqka.com.cn/new/{stock_code}/finance.html."
            )
        ]
