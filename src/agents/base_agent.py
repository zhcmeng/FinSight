"""
基础智能体模块，提供保存/加载快照、ReAct 运行循环以及工具调用能力。
这是所有具体智能体（如 DataCollector, DataAnalyzer）的基类。
"""
from typing import Dict, Any, Union, Optional, Type
import sys
import os
import pickle
import dill
import uuid
import re
import asyncio
from datetime import datetime
from src.config import Config
from src.tools import list_tools, get_tool_by_name
from src.utils import AsyncCodeExecutor, get_logger
from src.tools.base import Tool


# 全局智能体注册表，用于从快照中通过名称恢复具体的智能体类
_AGENT_REGISTRY: Dict[str, Type['BaseAgent']] = {}

def register_agent_class(agent_class: Type['BaseAgent']):
    """将智能体类注册到全局注册表中。"""
    _AGENT_REGISTRY[agent_class.AGENT_NAME] = agent_class
    return agent_class

class BaseAgent:
    """
    智能体基类，定义了通用的生命周期管理和执行逻辑。
    支持状态持久化、断点续传、代码执行和工具调用。
    """
    AGENT_NAME = 'base'
    AGENT_DESCRIPTION = 'base agent'
    NECESSARY_KEYS = ['task'] # 运行任务时必须包含的键
    
    def __init_subclass__(cls, **kwargs):
        """自动注册子类到智能体注册表中。"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'AGENT_NAME') and cls.AGENT_NAME != 'base':
            register_agent_class(cls)
    
    def __init__(
        self, 
        config: Config, 
        tools: list[Union[Tool, 'BaseAgent']],
        use_llm_name: str = "deepseek-chat",
        enable_code = True,
        memory = None,
        agent_id: str = None,
    ):
        """
        初始化智能体。
        
        Args:
            config: 系统全局配置。
            tools: 该智能体可以调用的工具或子智能体列表。
            use_llm_name: 指定使用的 LLM 模型名称。
            enable_code: 是否启用 Python 代码执行功能。
            memory: 统一变量内存系统实例。
            agent_id: 智能体唯一标识，若为 None 则自动生成。
        """
        self.config = config
        self.name = self.AGENT_NAME
        self.type = f'agent_{self.AGENT_NAME}'
        
        # 确定智能体 ID
        if agent_id is None:
            self.id = f'agent_{self.AGENT_NAME}_{uuid.uuid4().hex[:8]}'
        else:
            self.id = agent_id
            
        self.state = None
        # 设置智能体的工作目录和快照缓存目录
        self.working_dir = os.path.join(self.config.working_dir, 'agent_working', self.id)
        os.makedirs(self.working_dir, exist_ok=True)
        self.cache_dir = os.path.join(self.working_dir, '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 初始化代码执行器
        self.enable_code = enable_code
        if self.enable_code:
            self.executor_path = os.path.join(self.working_dir, '.executor_cache')
            os.makedirs(self.executor_path, exist_ok=True)
            self.code_executor = AsyncCodeExecutor(self.executor_path)
            self.executor_state_path = os.path.join(self.executor_path, 'state.dill')
        
        self.use_llm_name = use_llm_name
        self.llm = self.config.llm_dict[use_llm_name]
        self.memory = memory
        
        # 如果未提供工具，则调用子类的默认工具设置
        if tools is None or tools == []:
            self._set_default_tools()
        else:
            self.tools = tools
            
        # 在内存系统中记录工具依赖关系
        for tool in self.tools:
            self.memory.add_dependency(tool.id, self.id)
            
        self.current_task_data = {}
        self.current_checkpoint = {}
        self._resume_state: Dict[str, Any] | None = None
        self.current_round = 0
        
        # 初始化日志并设置上下文（以便日志输出能带上 agent_id）
        self.logger = get_logger()
        self.logger.set_agent_context(self.id, self.AGENT_NAME)
    
    def _set_default_tools(self):
        """子类可重写此方法以设置默认工具。"""
        return []

    def _get_persist_extra_state(self) -> Dict[str, Any]:
        """子类钩子：用于持久化额外的状态。"""
        return {}

    def _load_persist_extra_state(self, state: Dict[str, Any]):
        """子类钩子：用于恢复额外的状态。"""
        return
    
    @classmethod
    async def from_checkpoint(
        cls,
        config: Config,
        memory,
        agent_id: str,
        checkpoint_name: str = 'latest.pkl',
        tools: Optional[list] = None,
        restored_agents: Optional[Dict[str, 'BaseAgent']] = None,
        **kwargs
    ) -> Optional['BaseAgent']:
        """
        从磁盘快照中恢复智能体实例。
        
        此方法会自动识别智能体类型、恢复依赖工具，并重新实例化。
        """
        if restored_agents is None:
            restored_agents = {}
        
        # 如果已经恢复过，直接返回缓存的实例
        if agent_id in restored_agents:
            return restored_agents[agent_id]
        
        # 构建快照路径
        working_dir = os.path.join(config.working_dir, 'agent_working', agent_id)
        cache_dir = os.path.join(working_dir, '.cache')
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)
        
        logger = get_logger()
        
        if not os.path.exists(checkpoint_path):
            # 尝试查找备选快照
            if not os.path.exists(cache_dir):
                logger.warning(
                    f"Cache directory not found for agent {agent_id}: {cache_dir}"
                )
                return None
            
            other_checkpoints = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
            if other_checkpoints:
                checkpoint_path = os.path.join(cache_dir, other_checkpoints[0])
                logger.info(
                    f"Using alternative checkpoint for agent {agent_id}: "
                    f"{checkpoint_path} (requested: {checkpoint_name})"
                )
            else:
                logger.warning(
                    f"No checkpoint file found for agent {agent_id}: "
                    f"requested={checkpoint_name}, cache_dir={cache_dir}, "
                    f"available_files={os.listdir(cache_dir) if os.path.exists(cache_dir) else 'N/A'}"
                )
                return None
        
        # 加载快照数据
        try:
            with open(checkpoint_path, 'rb') as f:
                state = dill.load(f)
        except Exception as e1:
            try:
                with open(checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
            except Exception as e2:
                logger.error(
                    f"Failed to load checkpoint for agent {agent_id}: "
                    f"path={checkpoint_path}, "
                    f"dill_error={type(e1).__name__}: {e1}, "
                    f"pickle_error={type(e2).__name__}: {e2}"
                )
                return None
        
        agent_name = state.get('agent_name')
        if not agent_name:
            logger.warning(
                f"Checkpoint for agent {agent_id} has no agent_name: "
                f"path={checkpoint_path}"
            )
            return None
        
        # 从注册表中查找对应的智能体类
        agent_class = _AGENT_REGISTRY.get(agent_name)
        if not agent_class:
            logger.warning(
                f"Agent class not found in registry for agent {agent_id}: "
                f"agent_name={agent_name}, "
                f"available_agents={list(_AGENT_REGISTRY.keys())}"
            )
            return None
        
        # 恢复依赖工具
        if tools is None:
            tools = await cls._restore_tools_from_checkpoint(
                config, memory, state, checkpoint_name, restored_agents, **kwargs
            )
        
        # 恢复初始参数
        checkpoint_use_llm_name = state.get('use_llm_name', 'deepseek-chat')
        use_llm_name = kwargs.get('use_llm_name', checkpoint_use_llm_name)
        
        # 恢复额外的初始化参数
        init_params = state.get('init_params', {})
        for key, value in init_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # 实例化智能体
        agent = agent_class(
            config=config,
            memory=memory,
            agent_id=agent_id,
            use_llm_name=use_llm_name,
            tools=tools,
            **{k: v for k, v in kwargs.items() if k != 'use_llm_name'}
        )
        
        # 恢复运行状态
        agent._resume_state = state
        agent.current_task_data = state.get('current_task_data', {})
        
        # 设置日志上下文
        agent.logger.set_agent_context(agent.id, agent.AGENT_NAME)
        
        # 恢复代码执行器状态（变量等）
        if agent.enable_code and os.path.exists(agent.executor_state_path):
            try:
                with open(agent.executor_state_path, 'rb') as ef:
                    exec_state = ef.read()
                agent.code_executor.load_state(exec_state)
            except Exception as e:
                agent.logger.error(f"Failed to load code-executor state: {e}", exc_info=True)
        
        # 恢复子类特有的状态
        agent._load_persist_extra_state(state)
        
        restored_agents[agent_id] = agent
        return agent
    
    @classmethod
    async def _restore_tools_from_checkpoint(
        cls,
        config: Config,
        memory,
        state: Dict[str, Any],
        checkpoint_name: str,
        restored_agents: Dict[str, 'BaseAgent'],
        **kwargs
    ) -> list:
        """从快照状态中恢复工具列表。"""
        from src.tools import get_tool_by_name
        
        tool_dependencies = state.get('tool_dependencies', [])
        restored_tools = []
        logger = get_logger()
        
        for dep in tool_dependencies:
            if dep['type'] == 'agent':
                # 递归恢复依赖的智能体
                dep_agent_id = dep['agent_id']
                dep_working_dir = os.path.join(config.working_dir, 'agent_working', dep_agent_id)
                dep_cache_dir = os.path.join(dep_working_dir, '.cache')
                dep_checkpoint_path = os.path.join(dep_cache_dir, checkpoint_name)
                
                # 如果指定快照不存在，尝试查找备选
                if not os.path.exists(dep_checkpoint_path):
                    if os.path.exists(dep_cache_dir):
                        other_checkpoints = [f for f in os.listdir(dep_cache_dir) if f.endswith('.pkl')]
                        if other_checkpoints:
                            dep_checkpoint_path = os.path.join(dep_cache_dir, other_checkpoints[0])
                
                # 读取依赖智能体的快照以获取其初始化参数
                dep_kwargs = {}
                if os.path.exists(dep_checkpoint_path):
                    try:
                        with open(dep_checkpoint_path, 'rb') as f:
                            dep_state = dill.load(f)
                        dep_init_params = dep_state.get('init_params', {})
                        dep_kwargs['use_llm_name'] = kwargs.get('use_llm_name', dep_init_params.get('use_llm_name'))
                        for key in ['use_vlm_name', 'use_embedding_name', 'enable_code']:
                            if key in dep_init_params:
                                dep_kwargs[key] = dep_init_params[key]
                    except Exception as e:
                        logger.warning(f"Failed to load dependency agent checkpoint for {dep_agent_id}: {e}")
                        dep_kwargs['use_llm_name'] = kwargs.get('use_llm_name')
                else:
                    dep_kwargs['use_llm_name'] = kwargs.get('use_llm_name')
                
                dep_agent = await cls.from_checkpoint(
                    config=config,
                    memory=memory,
                    agent_id=dep_agent_id,
                    checkpoint_name=checkpoint_name,
                    restored_agents=restored_agents,
                    **dep_kwargs
                )
                if dep_agent:
                    restored_tools.append(dep_agent)
            elif dep['type'] == 'tool':
                # 重新创建普通工具实例
                tool_instance = get_tool_by_name(dep['tool_name'])()
                if tool_instance.id != dep['tool_id']:
                    tool_instance.id = dep['tool_id']
                restored_tools.append(tool_instance)
        
        return restored_tools

    async def save(self, state: Dict[str, Any] | None = None, checkpoint_name: str = 'latest.pkl'):
        """将当前智能体状态持久化到磁盘快照。"""
        # 记录工具依赖（ID 和 名称）
        tool_dependencies = []
        for tool in self.tools:
            if isinstance(tool, BaseAgent):
                tool_dependencies.append({
                    'type': 'agent',
                    'agent_name': tool.AGENT_NAME,
                    'agent_id': tool.id,
                })
            elif isinstance(tool, Tool):
                tool_dependencies.append({
                    'type': 'tool',
                    'tool_name': tool.name,
                    'tool_id': tool.id,
                })
        
        # 记录初始化参数
        init_params = {}
        for key in ['use_llm_name', 'enable_code', 'use_vlm_name', 'use_embedding_name']:
            if hasattr(self, key):
                init_params[key] = getattr(self, key)
        
        checkpoint = {
            'agent_name': self.AGENT_NAME,
            'agent_id': self.id,
            'use_llm_name': self.use_llm_name,
            'current_time': self.current_time,
            'current_task_data': getattr(self, 'current_task_data', {}),
            'tool_dependencies': tool_dependencies,
            'init_params': init_params,
        }

        if state:
            self.current_checkpoint.update(state)
        checkpoint.update(self.current_checkpoint)
        
        # 原子性写入快照文件
        target_path = os.path.join(self.cache_dir, checkpoint_name)
        tmp_path = target_path + '.tmp'
        try:
            with open(tmp_path, 'wb') as f:
                dill.dump(checkpoint, f)
            os.replace(tmp_path, target_path)
        except Exception:
            with open(tmp_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            os.replace(tmp_path, target_path)

        # 持久化代码执行器的状态（变量等）
        if self.enable_code and hasattr(self, 'code_executor'):
            try:
                state_bytes = self.code_executor.save_state()
                with open(self.executor_state_path, 'wb') as ef:
                    ef.write(state_bytes)
            except Exception as e:
                self.logger.error(f"Failed to save code-executor state: {e}", exc_info=True)

    async def load(self, checkpoint_name: str = 'latest.pkl') -> Dict[str, Any] | None:
        """从磁盘加载状态。"""
        target_path = os.path.join(self.cache_dir, checkpoint_name)
        if not os.path.exists(target_path):
            return None
        try:
            with open(target_path, 'rb') as f:
                state = dill.load(f)
        except Exception:
            with open(target_path, 'rb') as f:
                state = pickle.load(f)
        self.state = state
        # 恢复基础字段
        self.current_task_data = state.get('current_task_data', {})
        # 恢复代码执行器状态
        if self.enable_code and hasattr(self, 'code_executor') and os.path.exists(self.executor_state_path):
            try:
                with open(self.executor_state_path, 'rb') as ef:
                    exec_state = ef.read()
                self.code_executor.load_state(exec_state)
                # 重新注入工具调用函数到代码环境
                self.code_executor.set_variable("call_tool", self._agent_tool_function)
            except Exception as e:
                self.logger.error(f"Failed to load code-executor state: {e}", exc_info=True)
        return state

    async def _prepare_executor(self):
        """准备代码执行环境，注入工具调用函数。"""
        if self.enable_code:
            self.code_executor.set_variable("call_tool", self._agent_tool_function)

    async def _prepare_init_prompt(self, input_data: dict) -> list[dict]:
        """构造初始提示词，子类必须实现。"""
        raise NotImplementedError
    

    def _agent_tool_function(self, tool_name: str = None, **kwargs):
        """
        供 LLM 在代码块中调用的工具函数：`call_tool(tool_name='xxx', ...)`。
        
        支持调用原子工具 (Tool) 或其他智能体 (BaseAgent)。
        """
        if tool_name is None:
            raise ValueError("tool_name is required")
        target_tool = None
        # 查找匹配的工具
        for tool in self.tools:
            if isinstance(tool, Tool):
                if tool.name == tool_name:
                    target_tool = tool
                    break
            elif isinstance(tool, BaseAgent):
                if tool.AGENT_NAME == tool_name:
                    target_tool = tool
                    break
        if target_tool is None:
            self.logger.warning(f"No available tools for tool_name: {tool_name}")
            self.memory.add_log(self.id, None, kwargs, [], error=True, note=f"No available tools for tool_name: {tool_name}")
            return []

        try:
            # 场景 A: 调用子智能体
            if issubclass(type(target_tool), BaseAgent):
                if 'task' not in kwargs:
                    kwargs['task'] = self.current_task_data['task']
                # 同步运行子智能体（在代码执行器的异步环境下）
                response = asyncio.run(target_tool.async_run(input_data=kwargs))
                response = response['final_result']
                self.memory.add_log(target_tool.id, target_tool.type, kwargs, response, error=False, note=f"Tool {target_tool.name} executed successfully")
                return response
            # 场景 B: 调用原子工具
            elif issubclass(type(target_tool), Tool):
                response = asyncio.run(target_tool.api_function(**kwargs))
                data_list = [item.data for item in response]
                
                # 在终端输出工具执行概览，方便调试
                display_note = f"[Tool Result Overview] Gather {len(response)} Tool Results.\n"
                for i, item in enumerate(response):
                    display_note += f"-{i}. Name: {item.name}\nSource: {item.source}\n"
                print(f"\n\n{display_note}\n\n", file=sys.stdout, flush=True)

                self.memory.add_log(target_tool.id, target_tool.type, kwargs, response, error=False, note=f"Tool {target_tool.name} executed successfully")
                return data_list
            else:
                self.logger.warning(f"Unknown tools: {tool_name}")
                self.memory.add_log(self.id, target_tool.type, kwargs, [], error=True, note=f"Unknown tools: {tool_name}")
                return []
        except Exception as e:
            self.logger.error(f"Tool {tool_name} execution failed: {e}", exc_info=True)
            self.memory.add_log(self.id, target_tool.type, kwargs, [], error=True, note=f"Tool {tool_name} executed failed: {e}")
            return []
    
    def _get_api_descriptions(self) -> str:
        """构造工具描述字符串，供 LLM 了解如何调用工具。"""
        desc = 'The usage of tool calling: `tool_result = call_tool(tool_name=\'tool_name\', **kwargs)`. (you can use custom variable names for the tool result)\n\n'
        desc += 'Below are the tools and their descriptions:\n\n'
        for tool in self.tools:
            if issubclass(type(tool), Tool):
                desc += f"- Tool: {tool.name}\nDescription: {tool.description}\nParameters: {tool.parameters}\n\nOutput: "
            elif issubclass(type(tool), BaseAgent):
                desc += f"- Tool: {tool.AGENT_NAME}\nDescription: {tool.AGENT_DESCRIPTION}\n\n"
        desc += 'The result of each tool is a varaible, please use `print` to print the result.'
        return desc
    
    def _check_necessary_data(self, input_data):
        """检查输入数据是否包含必要的键。"""
        check_keys = self.NECESSARY_KEYS
        for key in check_keys:
            if key not in input_data:
                self.logger.warning(f"{key} not in input_data")

    async def async_run(
        self, 
        input_data: dict, 
        max_iterations: int = 10,
        stop_words: list[str] = ["</execute>", "</final_result>"],
        echo=False,
        resume: bool = False,
        checkpoint_name: str = 'latest.pkl',
        prompt_function=None,
    ) -> dict:
        """
        Agent 的主运行循环 (Think-Act 循环)。
        
        该方法实现了典型的 ReAct 模式：
        1.  **断点恢复**：如果设置了 resume=True，会尝试从本地 checkpoint 加载历史对话状态。
        2.  **初始化**：调用 prompt_function（默认为 _prepare_init_prompt）构建初始的 System Prompt 和 User Task。
        3.  **迭代循环**：
            -   **Think**：调用 LLM (self.llm.generate) 根据当前对话历史生成思考和行动指令。
            -   **Act**：解析 LLM 响应中的 Action（如调用工具、执行代码），并调用 _execute_action 执行。
            -   **Observe**：将执行结果作为新的 Observation 存入对话历史，作为下一轮思考的上下文。
        4.  **状态持久化**：每一轮迭代后都会将当前状态（对话历史、当前轮次等）保存到磁盘，以便异常中断后恢复。
        5.  **终止条件**：当 LLM 给出最终结论 (action_result['continue'] 为 False) 或达到最大迭代次数时退出。

        Args:
            input_data (dict): 包含任务相关信息的输入字典，如 {'task': '...'}。
            max_iterations (int): 最大迭代轮次，防止 Agent 进入无限循环。
            stop_words (list[str]): LLM 生成的停止词，用于截断响应。
            echo (bool): 是否在日志中打印详细的中间过程。
            resume (bool): 是否从上次中断的地方恢复运行。
            checkpoint_name (str): 持久化状态的文件名。
            prompt_function (callable): 用于初始化对话历史的函数。

        Returns:
            dict: 包含最终结果 (final_result) 和完整对话历史 (conversation_history) 的结果字典。
        """
        # 确保日志上下文已设置（对于异步执行中的日志归属非常重要）
        self.logger.set_agent_context(self.id, self.AGENT_NAME)
        
        self._check_necessary_data(input_data)
        self.current_task_data = input_data
        await self._prepare_executor()

        # 恢复或初始化对话历史
        conversation_history: list[dict]
        current_round: int
        if prompt_function is None:
            prompt_function = self._prepare_init_prompt
            
        if resume:
            state = await self.load(checkpoint_name=checkpoint_name)
            if state is not None:
                conversation_history = state.get('conversation_history', [])
                current_round = int(state.get('current_round', 0))
                # 如果已经有了最终返回结果，直接返回
                if 'return_dict' in state:
                    return state['return_dict']
            else:
                conversation_history = await prompt_function(input_data)
                current_round = 0
        else:
            conversation_history = await prompt_function(input_data)
            current_round = 0
    
        # 进入 Think-Act 循环
        while current_round < max_iterations+1:
            self.logger.info(f"Iteration {current_round + 1}")
            current_round += 1
            self.current_round = current_round
            
            # 步骤 1: LLM 思考 (Think)
            response = await self.llm.generate(messages = conversation_history, stop=stop_words)
            
            # 步骤 2: 解析行动 (Act)
            action_type, action_content = self._parse_llm_response(response)
            
            if echo:
                self.logger.info(f"LLM response this step: {response}")
                self.logger.info("--------")
                
            # 步骤 3: 执行行动 (Execute)
            action_result = await self._execute_action(action_type, action_content)
            action_result['llm_response'] = response
            
            if echo:
                self.logger.info(f"Action result this step: {action_result['result']}")
                self.logger.info("--------")
                
            # 步骤 4: 更新历史 (Observe)
            conversation_history.append({"role": "assistant", "content": action_result['llm_response']})
            conversation_history.append({"role": "user", "content": action_result['result']})
            
            self.logger.debug("--Begin of Execution Result--")
            self.logger.debug(action_result['result'])
            self.logger.debug("--End of Execution Result--")

            # 步骤 5: 保存当前迭代状态 (Checkpointing)
            current_state = {
                'conversation_history': conversation_history,
                'current_round': current_round,
                'input_data': input_data,
                'stop_words': stop_words,
            }
            current_state.update(self._get_persist_extra_state())
            self.state = current_state
            await self.save(
                state=current_state,
                checkpoint_name=checkpoint_name,
            )
            
            # 如果行动标记为停止（如输出了 final_result），则跳出循环
            if not action_result['continue']:
                break
        
        # 处理最终返回
        return_dict = {}
        if current_round >= max_iterations and action_result['continue']:
            # 达到最大迭代次数但 LLM 未给出终结信号，触发强制汇总
            return_dict = await self._handle_max_round(conversation_history)
        else:
            return_dict = {
                'conversation_history': conversation_history,
                'final_result': action_result['result'],
            }
        return_dict['input_data'] = input_data
        return_dict['working_dir'] = self.working_dir
        
        # 退出前保存最终状态
        current_state = {
            'conversation_history': conversation_history,
            'current_round': current_round,
            'input_data': input_data,
            'stop_words': stop_words,
            'return_dict': return_dict,
        }
        current_state.update(self._get_persist_extra_state())
        await self.save(
            state=current_state,
            checkpoint_name=checkpoint_name,
        )
        self.memory.save()
        
        return return_dict

    async def _handle_max_round(self, conversation_history):
        """当达到最大迭代次数时的兜底处理：取最后一条回复。"""
        return {'coversation_history': conversation_history, 'final_result': conversation_history[-1]['content']}

    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """解析 LLM 响应，提取 XML 标签中的内容（如 <execute> 或 <final_result>）。"""
        # 移除可能的思考标签干扰
        response = response.replace("<thinking>", "\n").replace("</thinking>", "\n")
        response = response.replace("<think>", "\n").replace("</think>", "\n")
        
        # 匹配 <tag>content</tag> 格式
        pattern = re.compile(r"<([\w_]+)>(.*?)</\1>", re.DOTALL)
        matches = list(pattern.finditer(response))
        
        # 如果没找到任何标签，默认为最终结论
        if not matches:
            return "final", response
            
        match = matches[-1] # 取最后一个标签（通常是最关键的动作）

        tag_name = match.group(1)
        # 标签重映射，增强鲁棒性
        if tag_name == 'execute':
            tag_name = 'code'
        if tag_name == 'final_result':
            tag_name = 'final'
        content_string = match.group(2).strip()

        return tag_name, content_string

    
    async def _execute_action(self, action_type: str, action_content: str):
        """根据行动类型路由到对应的处理函数。"""
        handler_method_name = f"_handle_{action_type}_action"
        handler = getattr(self, handler_method_name, None)

        if handler and callable(handler):
            return await handler(action_content)
        else:
            return await self._handle_default_action(action_type, action_content)
    

    async def _handle_code_action(self, action_content: str):
        """处理代码执行请求。"""
        code_result = await self.code_executor.execute(code=action_content)
        code_result = self._format_execution_result(code_result)
        return {
            "action": "generate_code",
            "action_content": action_content,
            "result": code_result,
            "continue": True, # 执行代码后需继续循环
        }

    async def _handle_final_action(self, action_content: str):
        """处理最终结果输出。"""
        return {
            "action": "final_result",
            "action_content": action_content,
            "result": action_content,
            "continue": False, # 给出结论后停止循环
        }

    async def _handle_default_action(self, action_type: str, action_content: str):
        """处理无效标签或默认情况。"""
        return {
            "action": "invalid_response",
            "action_content": action_content,
            "result": f"Unknown action_type '{action_type}'. Please respond using the required XML tags.",
            "continue": True,
        }
     
        
    def _format_execution_result(self, result: Dict[str, Any]) -> str:
        """将代码执行结果（stdout, stderr, variables）格式化为字符串。"""
        feedback = []

        if result["error"] is False:
            feedback.append("Code execution: success\n")

            if result["stdout"]:
                feedback.append(f"Console output:\n{result['stdout']}\n\n")

            if result.get("variables"):
                feedback.append("New variables:")
                for var_name, var_info in result["variables"].items():
                    feedback.append(f"  - {var_name}: {var_info}")
            if result.get("additional_notes"):
                feedback.append(f"Additional notes: {result['additional_notes']}\n")
        else:
            feedback.append("Code execution: failed\n")
            if result["stderr"]:
                feedback.append(f"Error message: {result['stderr']}\n")
            if result["stdout"]:
                feedback.append(f"Partial output: {result['stdout']}\n")
        return "\n".join(feedback)
        
        
        
