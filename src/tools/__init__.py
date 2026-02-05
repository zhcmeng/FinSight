"""
FinSight 工具模块 (Tools Module)

该模块是整个系统的“工具箱”核心，负责统一管理、自动发现和分发所有数据采集工具。
它通过动态导入机制，将分散在各个子目录（financial, web, macro, industry）中的工具类
自动注册到全局注册表中，供 Agent 调用。
"""

import importlib
import inspect
from typing import Dict, List, Type, Any, Optional
from .base import Tool, ToolResult

# 以下导入用于确保所有子模块被加载，从而触发工具注册
from .web.web_crawler import *
from .web.search_engine_requests import *
from .web.search_engine_playwright import *
from .web.base_search import *
from .macro.macro import *
from .financial.company_statements import *
from .financial.stock import *
from .financial.market import *
from .industry.industry import *

# 全局工具注册表：存储 工具名称 -> 工具类对象 的映射
_REGISTERED_TOOLS: Dict[str, Type[Tool]] = {}

# 工具分类表：存储 类别名称 -> [工具名称列表] 的映射
# 预设了四个核心财务研究类别
_TOOL_CATEGORIES: Dict[str, List[str]] = {
    'financial': [], # 财务报表、个股数据等
    'macro': [],     # 宏观经济指标
    'industry': [],  # 行业研究数据
    'web': []        # 通用网页搜索与爬虫
}

def register_tool(tool_class: Type[Tool], category: str = 'general') -> Type[Tool]:
    """
    手动注册一个工具类到全局注册表中。
    
    参数:
        tool_class: 继承自 Tool 的类对象
        category: 工具所属类别，默认为 'general'
    """
    try:
        # 实例化工具以获取其定义的名称
        tool_name = tool_class().name
        _REGISTERED_TOOLS[tool_name] = tool_class
        
        # 如果类别不存在，则新建该类别
        if category not in _TOOL_CATEGORIES:
            _TOOL_CATEGORIES[category] = []
        
        # 将工具名称加入对应的类别列表
        _TOOL_CATEGORIES[category].append(tool_name)
        
    except Exception as e:
        print(f"Warning: Failed to register tool {tool_class.__name__}: {e}")
    
    return tool_class

def get_avail_tools(category: Optional[str] = None) -> Dict[str, Type[Tool]]:
    """
    获取可用的工具列表，可选按类别过滤。
    
    参数:
        category: 类别名称（如 'financial'）。如果为 None，则返回所有已注册工具。
        
    返回:
        工具名称到工具类的字典映射
    """
    if category is None:
        return _REGISTERED_TOOLS.copy()
    
    if category not in _TOOL_CATEGORIES:
        return {}
    
    return {
        tool_name: _REGISTERED_TOOLS[tool_name] 
        for tool_name in _TOOL_CATEGORIES[category]
        if tool_name in _REGISTERED_TOOLS
    }

def get_tool_by_name(tool_name: str) -> Optional[Type[Tool]]:
    """
    根据名称获取特定的工具类。
    
    参数:
        tool_name: 工具的唯一名称（如 "Balance sheet"）
        
    返回:
        找到的工具类，若未找到则返回 None
    """
    return _REGISTERED_TOOLS.get(tool_name)

def get_tool_categories() -> Dict[str, List[str]]:
    """
    获取所有工具类别及其包含的工具名称。
    
    返回:
        类别名称映射到工具名称列表的字典
    """
    return _TOOL_CATEGORIES.copy()

def list_tools() -> List[str]:
    """
    列出所有已注册工具的名称。
    
    返回:
        包含所有工具名称的列表
    """
    return list(_REGISTERED_TOOLS.keys())

def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    获取特定工具的详细信息（名称、描述、参数说明）。
    
    参数:
        tool_name: 工具名称
        
    返回:
        包含工具元数据的字典，若未找到则返回 None
    """
    tool_class = get_tool_by_name(tool_name)
    if tool_class is None:
        return None

    return {
            'name': tool_class.name,
            'description': tool_class.description,
            'parameters': tool_class.parameters,
        }

def _auto_register_tools():
    """
    自动发现并注册子模块中的工具。
    该函数会扫描当前目录下的所有 Python 文件，识别所有继承自 Tool 的类并自动分类。
    """
    import os
    import pkgutil
    
    # 获取当前目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    submodules = []
    
    # 递归遍历所有子目录，寻找 Python 模块
    for importer, modname, ispkg in pkgutil.walk_packages([current_dir], prefix=f"{__name__}."):
        if not ispkg:  # 只导入模块文件，不导入包目录
            submodules.append(modname)
    
    for submodule in submodules:
        try:
            # 将模块名转换为相对导入格式
            relative_name = submodule.replace(f"{__name__}.", "")
            module = importlib.import_module(f'.{relative_name}', package=__name__)
            
            # 遍历模块中的所有成员，寻找继承自 Tool 的类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, Tool) and 
                    obj != Tool and 
                    obj.__module__ == module.__name__):
                    
                    # 根据子模块路径确定其类别
                    category = 'general'  # 默认类别
                    if '.' in relative_name:
                        # 从路径提取类别 (例如: 'financial.stock' -> 'financial')
                        category = relative_name.split('.')[0]
                    
                    # 执行自动注册
                    register_tool(obj, category)
                    
        except Exception as e:
            print(f"Warning: Failed to import submodule {submodule}: {e}")

# 在模块加载时自动执行工具扫描
_auto_register_tools()

# 导出核心接口，方便外部导入
__all__ = [
    'Tool',
    'ToolResult', 
    'register_tool',
    'get_avail_tools',
    'get_tool_by_name',
    'get_tool_categories',
    'list_tools',
    'get_tool_info'
]

