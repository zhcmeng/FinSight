import asyncio
import io
import sys
import os
import dill  # Use dill instead of pickle for more robust serialization
import traceback
import uuid
import inspect
import importlib
import types
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Tuple
import pandas as pd

class AsyncCodeExecutor:
    """
    轻量级 Python 沙箱，能够执行 LLM 生成的代码。
    支持状态持久化、变量注入以及异步执行。
    """
    def __init__(self, working_dir: str):
        """
        初始化执行器。
        
        Args:
            working_dir (str): 执行器的工作目录，用于存储缓存和临时文件。
        """
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        # 初始化干净的全局命名空间
        self.globals: Dict[str, Any] = self.create_clean_globals()

    def create_clean_globals(self) -> Dict[str, Any]:
        """
        创建一个填充了内置函数和预导入库的干净全局命名空间。
        这确保了 LLM 生成的代码可以直接访问常用库，而无需重复编写 import 语句。
        """
        context = {'__builtins__': __builtins__}

        # 预导入标准库
        import os
        import json
        import math
        import re
        import random
        import datetime
        import asyncio
        import io
        import sys
        
        context.update({
            'os': os,
            'json': json,
            'math': math,
            're': re,
            'random': random,
            'datetime': datetime,
            'asyncio': asyncio,
            'io': io,
            'sys': sys
        })

        # 尝试预导入数据分析常用第三方库
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib
            import matplotlib.font_manager as fm
            
            # 设置 Matplotlib 支持中文显示（使用黑体）
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'KaiTi', 'sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            context.update({
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'plt': plt,
                'matplotlib': matplotlib,
            })
        except ImportError as e:
            print(f"Warning: Failed to pre-import data libraries: {e}")

        return context


    def set_variable(self, name: str, value: Any):
        """
        向执行器的全局作用域注入外部变量或函数。
        """
        self.globals[name] = value

    def get_variable(self, name: str) -> Any:
        """
        从执行器的全局变量中获取指定变量。
        """
        return self.globals.get(name)

    def save_state(self) -> bytes:
        """
        捕获一个最小化的、可重建的执行状态，包括：
        - imports: 恢复时需要导入的模块名列表
        - definitions: 用户定义的函数/类（存储其源代码）
        - variables: 简单的可序列化变量（尽可能跳过复杂对象以减小体积）
        """
        state: Dict[str, Any] = {
            'imports': [],
            'definitions': [],  # 字典列表：{name, kind, source}
            'variables': {},    # 名称 -> dill 序列化后的字节
        }

        # 1) 追踪已导入的模块
        module_names: List[str] = []
        for name, value in list(self.globals.items()):
            if isinstance(value, types.ModuleType):
                if value.__name__ not in ('__builtins__',):
                    module_names.append(value.__name__)
        # 去重并排序，保证结果的确定性
        state['imports'] = sorted(set(module_names))

        # 2) 收集用户定义的函数和类（带源码）
        def try_collect_definition(obj_name: str, obj: Any, kind: str):
            try:
                # 仅捕获在 exec 中定义的对象（其源码可获取）
                source = inspect.getsource(obj)
                state['definitions'].append({'name': obj_name, 'kind': kind, 'source': source})
            except Exception:
                # 如果源码获取失败（如内置函数），则跳过
                pass

        for name, value in list(self.globals.items()):
            # 跳过魔法方法/变量
            if name.startswith('__') and name.endswith('__'):
                continue
            if inspect.isfunction(value):
                try_collect_definition(name, value, 'function')
            elif inspect.isclass(value):
                try_collect_definition(name, value, 'class')

        # 3) 收集简单变量（优先使用简单的基础类型）
        SIMPLE_ALLOWED_TYPES = (int, float, str, bool)
        CONTAINER_TYPES = (list, dict, tuple, set)

        def is_simple(obj: Any, depth: int = 0) -> bool:
            """判断对象是否为简单可序列化类型。"""
            if isinstance(obj, SIMPLE_ALLOWED_TYPES):
                return True
            if isinstance(obj, (pd.DataFrame, )):
                return False
            if isinstance(obj, CONTAINER_TYPES) and depth < 2:
                try:
                    if isinstance(obj, dict):
                        return all(isinstance(k, (str, int)) and is_simple(v, depth + 1) for k, v in obj.items())
                    else:
                        return all(is_simple(v, depth + 1) for v in obj)
                except Exception:
                    return False
            return False

        for name, value in list(self.globals.items()):
            if name in ('__builtins__',):
                continue
            if inspect.isfunction(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
                continue
            if name.startswith('_'):
                continue
            
            # 尝试序列化变量
            to_store = None
            if is_simple(value):
                try:
                    to_store = dill.dumps(value)
                except Exception:
                    to_store = None
            else:
                # 对于复杂对象，谨慎尝试使用 dill 序列化；失败则跳过
                try:
                    to_store = dill.dumps(value)
                except Exception:
                    to_store = None
            if to_store is not None:
                state['variables'][name] = to_store

        try:
            return dill.dumps(state)
        except Exception as e:
            print(f"[{self.session_id}] Warning: failed to save lightweight state: {e}")
            return dill.dumps({'imports': [], 'definitions': [], 'variables': {}})

    def load_state(self, state: bytes):
        """
        通过重新导入模块、重新执行定义和加载变量来恢复状态。
        """
        try:
            payload = dill.loads(state)
        except Exception as e:
            print(f"[{self.session_id}] Error: failed to load state: {e}. Resetting to a clean environment.")
            self.globals = self.create_clean_globals()
            return

        self.globals = self.create_clean_globals()

        # 1) 恢复模块导入
        for mod_name in payload.get('imports', []) or []:
            try:
                mod = importlib.import_module(mod_name)
                self.globals[mod_name.split('.')[-1]] = mod
            except Exception:
                continue

        # 2) 重新执行函数/类定义
        for item in payload.get('definitions', []) or []:
            source = item.get('source')
            if not source:
                continue
            try:
                exec(source, self.globals)
            except Exception:
                continue

        # 3) 恢复变量值
        for name, raw in (payload.get('variables', {}) or {}).items():
            try:
                self.globals[name] = dill.loads(raw)
            except Exception:
                continue
    

    def get_environment_info(self) -> str:
        """
        总结当前执行环境，用于辅助 LLM 生成后续代码（提供上下文）。
        """
        info_parts = []
        
        # 捕获重要的数据变量描述
        important_vars = {}
        for var_name, var_value in self.globals.items():
            # 忽略私有变量和 Jupyter/系统内置变量
            if not var_name.startswith('_') and var_name not in ['In', 'Out', 'get_ipython', 'exit', 'quit']:
                try:
                    if hasattr(var_value, 'shape'):  # 处理 DataFrame, numpy 数组
                        important_vars[var_name] = f"{type(var_value).__name__} with shape {var_value.shape}"
                    elif var_name in ['session_output_dir']:
                        important_vars[var_name] = str(var_value)
                    elif isinstance(var_value, (int, float, str, bool)) and len(str(var_value)) < 100:
                        important_vars[var_name] = f"{type(var_value).__name__}: {var_value}"
                    elif hasattr(var_value, '__module__') and var_value.__module__ in ['pandas', 'numpy', 'matplotlib.pyplot']:
                        important_vars[var_name] = f"Imported module: {var_value.__module__}"
                    
                    # 针对 DataFrame 提供更详细的列信息
                    if isinstance(var_value, pd.DataFrame):
                        important_vars[var_name] += ", and dtypes: " + str(var_value.dtypes)
                except:
                    continue
        
        if important_vars:
            info_parts.append("Current environment variables:")
            for var_name, var_info in important_vars.items():
                info_parts.append(f"- {var_name}: {var_info}")
        else:
            info_parts.append("Environment preloads pandas, numpy, matplotlib, and related libraries.")
        
        if 'session_output_dir' in self.globals:
            info_parts.append(f"Image output directory: session_output_dir = '{self.globals['session_output_dir']}'")
        
        return "\n".join(info_parts)

    async def execute(self, code: str) -> dict:
        """
        异步执行代码。
        通过将 exec 委派给线程池来保持事件循环的响应。
        如果代码中定义了 `async def async_main():`，则在执行完初始代码后自动调用它。

        Returns:
            dict: 包含 {stdout: str, stderr: str, error: bool}。
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        has_error = False
        
        # 为每一段执行的代码自动注入 Matplotlib 的中文配置头
        header = "import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False"       
        code = header + '\n' + code

        # 定义同步执行包装器，以便在线程中运行
        def sync_exec():
            nonlocal has_error
            try:
                # 重定向标准输出和错误输出
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # 在自定义全局作用域中执行代码
                    exec(code, self.globals)
            except Exception:
                has_error = True
                stderr_capture.write(traceback.format_exc())
                # 打印出错的代码，方便调试
                print("error code: code = \n", code)

        # 在线程池中运行同步代码，避免阻塞异步主线程
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, sync_exec)
        
        # 如果用户定义了异步入口点 async_main，则执行它
        if 'async_main' in self.globals and \
           asyncio.iscoroutinefunction(self.globals['async_main']):
            
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # 等待异步任务完成
                    await self.globals['async_main']()
            except Exception:
                has_error = True
                stderr_capture.write(traceback.format_exc())
            finally:
                # 运行完后删除该协程，防止下次执行时被误触发
                del self.globals['async_main']
        
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        if stdout == "":
            stdout = 'Run completed with no output.'
        return {
            'stdout': stdout,
            'stderr': stderr,
            'error': has_error
        }

