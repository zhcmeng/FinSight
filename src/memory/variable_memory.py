import os
import re
import json
import dill
import asyncio
import datetime
import json_repair
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Type
import numpy as np

from src.config import Config
from src.tools import ToolResult
from src.agents import AnalysisResult
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.utils.prompt_loader import get_prompt_loader
from src.tools.web.base_search import SearchResult
from src.tools.web.web_crawler import ClickResult
from src.agents.search_agent.search_agent import DeepSearchResult


class Memory:
    """
    统一变量内存系统 (Variable Memory)
    
    核心功能：
    1. 状态持久化：保存和加载整个研究过程的日志、数据、依赖关系和智能体状态。
    2. 智能体管理：支持断点恢复 (Resume)，管理智能体实例的生命周期，实现 Agent 级别的“热启动”。
    3. 任务调度支持：记录任务映射 (Task Mapping) 和任务优先级，支持多阶段复杂任务流。
    4. 变量检索 (RAG)：存储数据 Embedding，支持基于语义的相关数据检索，减少 LLM 的 Token 消耗。
    """
    def __init__(
        self,
        config: Config,
    ):
        """
        初始化内存系统。
        
        Args:
            config (Config): 系统配置对象，包含工作目录、LLM 配置等。
        """
        self.config = config
        self.save_dir = os.path.join(config.working_dir, "memory")
        os.makedirs(self.save_dir, exist_ok=True)

        self.log = [] # 运行日志列表，记录每个 Agent 的调用详情
        self.data = [] # 统一变量空间：存储收集到的 ToolResult 或 AnalysisResult
        self.dependency: Dict[str, List[str]] = {} # 智能体依赖树：parent_agent_id -> [child_agent_id]
        self.task_mapping = [] # 任务持久化映射：[{task_key, agent_class_name, task_input, agent_id, agent_kwargs}, ...]
        self.data2embedding = {} # 语义检索缓存：数据特征(name+description) -> embedding 向量
        self.generated_analysis_tasks = [] # 动态生成的分析任务列表（由 Planning 模块产生）
        self.generated_collect_tasks = [] # 动态生成的数据收集任务列表（由 Planning 模块产生）
        
        # 运行时缓存（不持久化）
        self._agents: Dict[str, BaseAgent] = {}  # 当前活跃的 agent_id -> 智能体实例
        self._restored_agents: Dict[str, BaseAgent] = {}  # 从快照恢复的智能体共享池，用于复用实例
        
        self.logger = get_logger()

        # 根据配置的 target_type 加载对应的 Prompt 模板
        target_type = config.config.get('target_type', 'general')
        report_type = 'financial' if 'financial' in target_type else 'general'
        self.prompt_loader = get_prompt_loader('memory', report_type=report_type)

    
    def save(self, checkpoint_name: str = 'memory.pkl'):
        """
        将内存状态序列化并保存到磁盘。
        
        该方法会保存日志、数据空间、任务映射和 Embedding 缓存。
        注意：不直接保存 BaseAgent 实例对象，而是保存元数据，以便在 Resume 时按需重新实例化。
        
        Args:
            checkpoint_name (str): 保存的文件名，默认为 'memory.pkl'。
        """
        # Note: agent instances themselves are not saved—only metadata.
        # Agents are reloaded on demand from their checkpoints.
        memory_state = {
            'log': self.log,
            'data': self.data,
            'dependency': self.dependency,
            'task_mapping': self.task_mapping,
            'data2embedding': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in self.data2embedding.items()},
            'generated_analysis_tasks': self.generated_analysis_tasks,
            'generated_collect_tasks': self.generated_collect_tasks,
        }
        target_path = os.path.join(self.save_dir, checkpoint_name)
        tmp_path = target_path + '.tmp'
        try:
            self.logger.info(f"Memory save start: path={target_path}, log={len(self.log)}, data={len(self.data)}, tasks={len(self.task_mapping)}")
        except Exception:
            pass
        
        try:
            # 先写入临时文件再替换，确保原子性，防止保存过程中断导致数据损坏
            with open(tmp_path, 'wb') as f:
                dill.dump(memory_state, f)
            os.replace(tmp_path, target_path)
            try:
                file_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
                self.logger.info(f"Memory saved: path={target_path}, size={file_size} bytes")
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f"Failed to save memory state: {e}", exc_info=True)
            raise
    
    def load(self, checkpoint_name: str = 'memory.pkl'):
        """
        从磁盘加载内存状态快照。
        
        用于支持系统崩溃后的恢复或多阶段任务的接力执行。
        
        Args:
            checkpoint_name (str): 加载的文件名。
            
        Returns:
            bool: 是否成功加载。
        """
        target_path = os.path.join(self.save_dir, checkpoint_name)
        if not os.path.exists(target_path):
            return False
        
        try:
            try:
                self.logger.info(f"Memory load start: path={target_path}")
            except Exception:
                pass
            with open(target_path, 'rb') as f:
                memory_state = dill.load(f)
            
            self.log = memory_state.get('log', [])
            self.data = memory_state.get('data', [])
            self.dependency = memory_state.get('dependency', {})
            self.task_mapping = memory_state.get('task_mapping', [])
            # 恢复 Embedding：将列表转换回 numpy 数组以支持后续的向量计算
            data2embedding_raw = memory_state.get('data2embedding', {})
            self.data2embedding = {k: np.array(v) if isinstance(v, list) else v 
                                  for k, v in data2embedding_raw.items()}
            self.generated_analysis_tasks = memory_state.get('generated_analysis_tasks', [])
            self.generated_collect_tasks = memory_state.get('generated_collect_tasks', [])
            
            # 清空运行时缓存，迫使 Agent 在被访问时从快照重新实例化
            self._agents = {}
            self._restored_agents = {}
            
            try:
                self.logger.info(f"Memory loaded: log={len(self.log)}, data={len(self.data)}, tasks={len(self.task_mapping)}")
            except Exception:
                pass
            return True
        except Exception as e:
            self.logger.error(f"Failed to load memory state: {e}", exc_info=True)
            return False

    def _get_task_key(self, agent_class: Type[BaseAgent], task_input: dict) -> str:
        """
        为“智能体+任务输入”组合生成唯一标识符。
        用于在内存中查找是否已经执行过相同的任务，实现幂等性和断点恢复。
        """
        input_data = task_input.get('input_data', {})
        
        # 针对特定 Agent 类型的优化识别逻辑
        if hasattr(agent_class, 'AGENT_NAME'):
            agent_name = agent_class.AGENT_NAME
            if agent_name == 'data_collector':
                return input_data.get('task', '')
            elif agent_name == 'data_analyzer':
                return input_data.get('analysis_task', '')
        
        # 通用降级逻辑：对输入字典排序后序列化，确保只要内容一致 key 就一致
        return str(sorted(input_data.items()))
    
    async def get_or_create_agent(
        self,
        agent_class: Type[BaseAgent],
        task_input: dict,
        resume: bool = True,
        checkpoint_name: str = 'latest.pkl',
        priority: int = 0,
        **agent_kwargs
    ) -> BaseAgent:
        """
        获取现有的智能体实例（从快照恢复）或创建一个全新的智能体。
        
        这是实现 Agent 级“热启动”的核心逻辑：
        1. 计算 Task Key：识别当前任务的唯一性。
        2. 查找历史：如果开启 resume，检查 task_mapping 是否记录过此任务。
        3. 状态重构：如果找到历史记录，通过 BaseAgent.from_checkpoint 恢复该 Agent 的变量空间和状态机进度。
        4. 实例化：若无历史或恢复失败，则按常规方式新建 Agent 并注册到任务映射中。
        
        Args:
            agent_class: Agent 的类对象。
            task_input: 任务输入参数字典。
            resume: 是否尝试从断点恢复。
            checkpoint_name: 指定加载的快照文件名。
            priority: 任务优先级。
            **agent_kwargs: 传递给 Agent 构造函数的其他参数。
            
        Returns:
            BaseAgent: 准备就绪的智能体实例。
        """
        task_key = self._get_task_key(agent_class, task_input)
        
        # 尝试在任务映射中反向搜索最近的匹配记录
        agent_id = None
        saved_task_info = None
        
        if resume:
            for task_info in self.task_mapping[::-1]:
                if (task_info.get('task_key') == task_key and 
                    task_info.get('agent_class_name') == agent_class.AGENT_NAME):
                    agent_id = task_info.get('agent_id')
                    saved_task_info = task_info
                    self.logger.info(f"Find {agent_id} in task_mapping")
                    break
        
        # 恢复逻辑
        agent = None
        if resume and agent_id:
            self.logger.info(f"Restoring agent: agent_id={agent_id}, task_key={task_key}, agent_class_name={agent_class.AGENT_NAME}, priority={priority}")  
            # 优先使用保存的参数，但也允许被当前传入的参数覆盖
            saved_kwargs = saved_task_info.get('agent_kwargs', {}) if saved_task_info else {}
            final_kwargs = {**saved_kwargs, **agent_kwargs}
            
            # 校验快照文件是否存在
            working_dir = os.path.join(self.config.working_dir, 'agent_working', agent_id)
            cache_dir = os.path.join(working_dir, '.cache')
            checkpoint_path = os.path.join(cache_dir, checkpoint_name)
            
            if not os.path.exists(checkpoint_path):
                # 如果指定快照丢失，尝试列出可用快照辅助调试
                other_checkpoints = []
                if os.path.exists(cache_dir):
                    other_checkpoints = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                self.logger.warning(
                    f"Checkpoint file not found: agent_id={agent_id}, "
                    f"checkpoint_path={checkpoint_path}, "
                    f"other_checkpoints={other_checkpoints}"
                )
            else:
                self.logger.info(f"Checkpoint file found: {checkpoint_path}")
            
            try:
                # 调用基类的恢复工厂方法
                agent = await BaseAgent.from_checkpoint(
                    config=self.config,
                    memory=self,
                    agent_id=agent_id,
                    checkpoint_name=checkpoint_name,
                    restored_agents=self._restored_agents,
                    **final_kwargs
                )
                if agent is None:
                    self.logger.warning(
                        f"Failed to restore agent: agent_id={agent_id}, "
                        f"checkpoint_name={checkpoint_name}, "
                        f"will create new agent instead"
                    )
                else:
                    self.logger.info(f"Successfully restored agent: agent_id={agent_id}")
            except Exception as e:
                self.logger.error(
                    f"Exception while restoring agent: agent_id={agent_id}, "
                    f"error={type(e).__name__}: {e}, "
                    f"will create new agent instead"
                )
                agent = None
        
        # 如果恢复失败或未找到历史，则创建新实例
        if agent is None:
            self.logger.info(f"Creating new agent: task_key={task_key}, agent_class_name={agent_class.AGENT_NAME}, priority={priority}")
            agent = agent_class(
                config=self.config,
                memory=self,
                **agent_kwargs
            )
            # 将新 Agent 及其元数据注册到内存中，以便未来持久化
            task_info = {
                'task_key': task_key,
                'agent_class_name': agent_class.AGENT_NAME,
                'task_input': task_input,
                'agent_id': agent.id,
                'agent_kwargs': agent_kwargs,
                'priority': priority,  # 持久化的优先级
            }
            self.task_mapping.append(task_info)
        else:
            # 恢复期间，同步更新优先级
            if saved_task_info:
                saved_priority = saved_task_info.get('priority')
                if saved_priority is not None:
                    saved_task_info['priority'] = saved_priority
                else:
                    saved_task_info['priority'] = priority
        
        # 更新运行时缓存
        self._agents[agent.id] = agent
        self._restored_agents[agent.id] = agent
        
        return agent
    
    def get_tasks_by_priority(self) -> List[Dict[str, Any]]:
        """获取所有任务，并按优先级（值越小越优先）排序。"""
        return sorted(self.task_mapping, key=lambda x: x.get('priority', 0))
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """根据 ID 检索运行时 Agent 实例。"""
        return self._agents.get(agent_id)
    
    def is_agent_finished(self, agent_id: str, checkpoint_name: str = 'latest.pkl') -> bool:
        """
        通过检查磁盘快照的状态，判断该 Agent 是否已经完成了执行任务。
        用于工作流控制器跳过已完成的任务。
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        
        checkpoint_path = os.path.join(agent.cache_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                state = dill.load(f)
            return state.get('finished', False)
        except Exception:
            return False
        
        
    def add_data(self, data: Any):
        """向统一变量空间添加新的数据块（ToolResult 或 AnalysisResult）。"""
        self.data.append(data)
        return True

    def add_dependency(self, child_id: str, parent_id: str):
        """记录 Agent 之间的调用依赖关系。"""
        if parent_id not in self.dependency:
            self.dependency[parent_id] = []
        if child_id not in self.dependency[parent_id]:
            self.dependency[parent_id].append(child_id)
        return True

    def add_log(self, id: str, type: str, input_data: dict, output_data: dict, error: bool = False, note: str = ''):
        """向内存日志中添加一条运行记录。"""
        self.log.append({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'id': id,
            'type': type,
            'input_data': input_data,
            'output_data': output_data,
            'error': error,
            'note': note
        })
        return True
    
    def get_log(self, parent_id: str, key: str=None):
        """获取特定父 Agent 下属所有子 Agent 的运行日志。"""
        child_list = self.dependency.get(parent_id, [])
        return_log = []
        for child_id in child_list:
            if key is not None:
                if key not in child_id:
                    continue
            child_log = [item for item in self.log if item['id'] == child_id]
            return_log.extend(child_log)
        return return_log
    
    def get_log_by_type(self, input_type: str):
        """根据类型（如 'search', 'click'）筛选日志。"""
        return [item for item in self.log if input_type in item['type']]

    def get_url_title(self, url: str):
        """根据 URL 在已有的搜索结果中查找对应的网页标题。"""
        # select search result in data
        search_result = [item for item in self.data if isinstance(item, SearchResult)]
        url2title = {}
        for item in search_result:
            url2title[item.link] = item.name
        return url2title.get(url, '')

    def get_collect_data(self, exclude_type: List[str] = []):
        """从变量空间中获取所有原始数据（ToolResult），支持类型过滤。"""
        collected_data = [item for item in self.data if isinstance(item, ToolResult)]
        collected_data = [item for item in collected_data if not isinstance(item, DeepSearchResult)]
        
        if exclude_type != []:
            for exclude_type_item in exclude_type:
                if exclude_type_item == 'search':
                    collected_data = [item for item in collected_data if not isinstance(item, SearchResult)]
                elif exclude_type_item == 'click':
                    collected_data = [item for item in collected_data if not isinstance(item, ClickResult)]
        return collected_data
    
    def get_analysis_result(self):
        """获取所有已完成的分析报告（AnalysisResult）。"""
        return [item for item in self.data if isinstance(item, AnalysisResult)]

    def get_formatted_analysis_result(self, analysis_result_list: List[AnalysisResult] = None):
        """将分析结果列表格式化为适合 LLM 阅读的文本字符串。"""
        if analysis_result_list is None:
            analysis_result_list = self.get_analysis_result()
        formatted_analysis_result = ""
        for idx, item in enumerate(analysis_result_list):
            formatted_analysis_result += f"Analysis report {idx+1}:\n"
            formatted_analysis_result += str(item)
            formatted_analysis_result += "\n\n"
        return formatted_analysis_result
    
    def get_formatted_data_description(self, data_list: List[ToolResult] = None):
        """将原始数据描述格式化为适合 LLM 阅读的文本字符串，默认排除基础搜索结果以节省 Token。"""
        if data_list is None:
            data_list = self.get_collect_data()
        # exclude naive search results to shorten contexts
        data_list = [item for item in data_list if not isinstance(item, SearchResult)]

        formatted_data_description = ""
        for idx, item in enumerate(data_list):
            formatted_data_description += str(item)
            formatted_data_description += "\n\n"
        return formatted_data_description
    
    async def select_data_by_llm(self, query: str, max_k: int = -1, model_name: str = "deepseek/deepseek-chat-v3.1"):
        """
        利用 LLM 从变量空间中筛选出与当前查询最相关的数据。
        
        这是一个“语义选择”过程，比简单的向量检索更精准，但消耗更多 Token。
        
        Returns:
            tuple(list, str): (选中的数据列表, 格式化后的描述)
        """
        model = self.config.llm_dict[model_name]
        prompt = self.prompt_loader.get_prompt('select_data',
            data_description = self.get_formatted_data_description(),
            section_description = query,
        )
        output = await model.generate(messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        
        if output is not None:
            match = re.search(r'```json([\s\S]*?)```', output)
            if match:
                output = match.group(1).strip()
            output = json_repair.loads(output)['selected_data_list']
            output = output[:max_k]
        else:
            return [], ""
        selected_data_list = [item for item in self.get_collect_data() if item.name in output]
        return selected_data_list, self.get_formatted_data_description(selected_data_list)
    
    async def select_analysis_result_by_llm(self, query: str, max_k: int = -1, model_name: str = "deepseek/deepseek-chat-v3.1"):
        """利用 LLM 筛选相关的分析报告结果。"""
        # return:  tuple(list, str)
        model = self.config.llm_dict[model_name]
        prompt = self.prompt_loader.get_prompt('select_analysis',
            analysis_description = self.get_formatted_analysis_result(),
            section_description = query,
        )
        output = await model.generate(messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        if output is not None:
            match = re.search(r'```json([\s\S]*?)```', output)
            if match:
                output = match.group(1).strip()
            output = json_repair.loads(output)['selected_analysis_list']
        else:
            return [], ""
        
        selected_analysis_result_list = [item for item in self.get_analysis_result() if item.title in output]
        return selected_analysis_result_list, self.get_formatted_analysis_result(selected_analysis_result_list)


    async def retrieve_relevant_data(self, query: str, top_k: int = 10, embedding_model: str = "deepseek/deepseek-chat-v3.1"):
        """
        基于向量相似度的语义检索 (RAG 核心逻辑)。
        
        1. 检查数据空间中哪些条目还没有 Embedding。
        2. 调用 Embedding 模型补全缺失的向量并存入缓存。
        3. 计算查询向量与数据向量的点积（相似度）。
        4. 返回相关性最高的 top_k 个数据条目。
        """
        self.embedding_model = self.config.llm_dict[embedding_model]
        collect_data_list = self.get_collect_data()
        if len(collect_data_list) <= top_k:
            return collect_data_list

        # 对缺少向量表示的条目进行 Embedding 处理
        need_to_embed_data = []
        for item in collect_data_list:
            key = item.name + item.description
            if key not in self.data2embedding:
                need_to_embed_data.append(item)
        if len(need_to_embed_data) > 0:
            embedding_list = await self.embedding_model.generate_embeddings([item.brief_str() for item in need_to_embed_data])
            for i, item in enumerate(need_to_embed_data):
                key = item.name + item.description
                self.data2embedding[key] = np.array(embedding_list[i])
        
        # 执行语义搜索：计算余弦相似度（由于是点积，建议向量先归一化，此处默认点积衡量相关性）
        query_embedding = await self.embedding_model.generate_embeddings([query])
        query_embedding = query_embedding[0]
        query_embedding = np.array(query_embedding)

        data_embeddings = [self.data2embedding[item.name + item.description] for item in collect_data_list]
        distances = np.dot(data_embeddings, query_embedding)
        top_k_indices = np.argsort(distances)[::-1][:top_k]
        top_k_data = [collect_data_list[i] for i in top_k_indices]
        return top_k_data
    
    async def generate_analyze_tasks(self, query: str, use_llm_name: str, max_num=10, existing_tasks: List[str] = None) -> List[str]:
        """
        利用 LLM 自动生成针对特定研究目标的分析子任务。
        
        Args:
            query (str): 核心研究查询。
            use_llm_name (str): 指定使用的模型名称。
            max_num (int): 最多生成的任务数量。
            existing_tasks (List[str], optional): 已有的任务列表，避免重复生成。
            
        Returns:
            List[str]: 生成的分析子任务描述列表。
        """
        llm = self.config.llm_dict[use_llm_name]
        
        # 格式化已有任务，用于 Prompt 提示
        if existing_tasks is None:
            existing_tasks = []
        existing_tasks_str = "\n".join([f"- {task}" for task in existing_tasks]) if existing_tasks else "None"
        
        prompt = self.prompt_loader.get_prompt('generate_task',
            query=query,
            existing_tasks=existing_tasks_str,
            max_num=max_num,
        )
        output = await llm.generate(messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        output = json_repair.loads(output)
        
        # 兼容不同模型的 JSON 返回格式
        if isinstance(output, dict):
            output = output.get('tasks', output.get('analysis_tasks', []))
        
        output = output[:max_num]
        self.generated_analysis_tasks = output
        self.save()
        return output
    
    async def generate_collect_tasks(self, query: str, use_llm_name: str, max_num=10, existing_tasks: List[str] = None) -> List[str]:
        """
        利用大语言模型（LLM）根据研究目标自动生成数据采集任务。

        该方法会结合当前的研究查询（query）和已有的任务列表（existing_tasks），
        调用指定的 LLM 模型来拆解并生成一系列具体的数据收集指令。生成的任务
        会自动更新到内存状态中并持久化到磁盘，以支持后续的断点恢复。

        Args:
            query (str): 研究目标描述，例如 "研究目标: 中国移动 (ticker: 00941)"。
            use_llm_name (str): 配置文件中指定的 LLM 模型名称。
            max_num (int, optional): 允许生成的最大任务数量。默认为 10。
            existing_tasks (List[str], optional): 已存在的任务列表，用于引导 LLM 避免生成重复任务。

        Returns:
            List[str]: 生成的具体数据采集任务描述列表。
        """
        llm = self.config.llm_dict[use_llm_name]
        
        # 格式化已有任务，用于 Prompt 提示
        if existing_tasks is None:
            existing_tasks = []
        existing_tasks_str = "\n".join([f"- {task}" for task in existing_tasks]) if existing_tasks else "None"
        
        prompt = self.prompt_loader.get_prompt('generate_collect_task',
            query=query,
            existing_tasks=existing_tasks_str,
            max_num=max_num,
        )
        output = await llm.generate(messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        output = json_repair.loads(output)
        
        # 兼容不同模型的 JSON 返回格式
        if isinstance(output, dict):
            output = output.get('tasks', output.get('collect_tasks', output.get('collection_tasks', [])))
        
        output = output[:max_num]
        self.generated_collect_tasks = output
        self.save()
        return output
