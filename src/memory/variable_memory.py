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
    2. 智能体管理：支持断点恢复 (Resume)，管理智能体实例的生命周期。
    3. 任务调度支持：记录任务映射 (Task Mapping) 和任务优先级。
    4. 变量检索 (RAG)：存储数据 Embedding，支持基于语义的相关数据检索。
    """
    def __init__(
        self,
        config: Config,
    ):

        self.config = config
        self.save_dir = os.path.join(config.working_dir, "memory")
        os.makedirs(self.save_dir, exist_ok=True)

        self.log = [] # 运行日志列表
        self.data = [] # 收集到的 ToolResult 或 AnalysisResult 列表
        self.dependency: Dict[str, List[str]] = {} # 智能体依赖关系：parent_agent_id -> [child_agent_id]
        self.task_mapping = [] # 任务映射：[{task_key, agent_class_name, task_input, agent_id, agent_kwargs}, ...]
        self.data2embedding = {} # 语义检索缓存：name+description -> embedding 向量
        self.generated_analysis_tasks = [] # LLM 自动生成的分析任务列表
        self.generated_collect_tasks = [] # LLM 自动生成的数据收集任务列表
        
        # 智能体实例缓存
        self._agents: Dict[str, BaseAgent] = {}  # agent_id -> 智能体实例
        self._restored_agents: Dict[str, BaseAgent] = {}  # 已恢复的智能体共享池
        
        # 日志句柄
        self.logger = get_logger()

        target_type = config.config.get('target_type', 'general')
        report_type = 'financial' if 'financial' in target_type else 'general'
        self.prompt_loader = get_prompt_loader('memory', report_type=report_type)

    
    def save(self, checkpoint_name: str = 'memory.pkl'):
        """
        将内存状态持久化到磁盘。
        注意：不直接保存智能体实例，而是保存其元数据，实例将在需要时通过断点加载。
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
        从磁盘加载内存状态，支持研究流程的中断恢复。
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
            # Restore embeddings (convert lists back to numpy arrays)
            data2embedding_raw = memory_state.get('data2embedding', {})
            self.data2embedding = {k: np.array(v) if isinstance(v, list) else v 
                                  for k, v in data2embedding_raw.items()}
            self.generated_analysis_tasks = memory_state.get('generated_analysis_tasks', [])
            self.generated_collect_tasks = memory_state.get('generated_collect_tasks', [])
            # Reset agent caches; they will be reloaded on demand
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
        """Generate a unique identifier for the agent/task combination."""
        input_data = task_input.get('input_data', {})
        
        # Agent-specific key logic
        if hasattr(agent_class, 'AGENT_NAME'):
            agent_name = agent_class.AGENT_NAME
            if agent_name == 'data_collector':
                return input_data.get('task', '')
            elif agent_name == 'data_analyzer':
                return input_data.get('analysis_task', '')
        
        # Fallback: stringified sorted items
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
        获取一个现有的智能体或创建一个新智能体。
        核心逻辑：
        1. 任务识别：根据 agent_class 和 task_input 生成唯一标识 task_key。
        2. 断点查找：如果开启 resume，在 task_mapping 中查找是否存在相同的 task_key。
        3. 状态恢复：如果找到匹配的 agent_id，尝试从磁盘加载该智能体的历史状态（包括变量空间、执行进度等）。
        4. 全新创建：如果未找到或恢复失败，则实例化一个新的智能体，并记录其元数据。
        """
        task_key = self._get_task_key(agent_class, task_input)
        
        # Check whether the task already exists in task_mapping
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
        
        # Attempt to restore an agent if possible
        agent = None
        if resume and agent_id:
            self.logger.info(f"Restoring agent: agent_id={agent_id}, task_key={task_key}, agent_class_name={agent_class.AGENT_NAME}, priority={priority}")  
            # Load saved kwargs if available; otherwise use supplied kwargs
            saved_kwargs = saved_task_info.get('agent_kwargs', {}) if saved_task_info else {}
            # Merge kwargs (saved values take precedence unless overridden)
            final_kwargs = {**saved_kwargs, **agent_kwargs}
            
            # Verify checkpoint existence
            working_dir = os.path.join(self.config.working_dir, 'agent_working', agent_id)
            cache_dir = os.path.join(working_dir, '.cache')
            checkpoint_path = os.path.join(cache_dir, checkpoint_name)
            
            if not os.path.exists(checkpoint_path):
                # Inspect other checkpoints if the target is missing
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
        
        # Instantiate a fresh agent if restore fails
        if agent is None:
            self.logger.info(f"Creating new agent: task_key={task_key}, agent_class_name={agent_class.AGENT_NAME}, priority={priority}")
            agent = agent_class(
                config=self.config,
                memory=self,
                **agent_kwargs
            )
            # Record task metadata
            task_info = {
                'task_key': task_key,
                'agent_class_name': agent_class.AGENT_NAME,
                'task_input': task_input,
                'agent_id': agent.id,
                'agent_kwargs': agent_kwargs,
                'priority': priority,  # persisted priority
            }
            self.task_mapping.append(task_info)
        else:
            # During resume: prefer saved priority; fallback to incoming value
            if saved_task_info:
                saved_priority = saved_task_info.get('priority')
                if saved_priority is not None:
                    saved_task_info['priority'] = saved_priority
                else:
                    saved_task_info['priority'] = priority
        
        # Cache the agent instance
        self._agents[agent.id] = agent
        self._restored_agents[agent.id] = agent
        
        return agent
    
    def get_tasks_by_priority(self) -> List[Dict[str, Any]]:
        """Return task metadata sorted by priority (lower value first)."""
        return sorted(self.task_mapping, key=lambda x: x.get('priority', 0))
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve an agent instance by id."""
        return self._agents.get(agent_id)
    
    def is_agent_finished(self, agent_id: str, checkpoint_name: str = 'latest.pkl') -> bool:
        """Check whether an agent completed execution (based on checkpoint state)."""
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
        self.data.append(data)
        return True

    def add_dependency(self, child_id: str, parent_id: str):
        if parent_id not in self.dependency:
            self.dependency[parent_id] = []
        if child_id not in self.dependency[parent_id]:
            self.dependency[parent_id].append(child_id)
        return True

    def add_log(self, id: str, type: str, input_data: dict, output_data: dict, error: bool = False, note: str = ''):
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
        return [item for item in self.log if input_type in item['type']]

    def get_url_title(self, url: str):
        # select search result in data
        search_result = [item for item in self.data if isinstance(item, SearchResult)]
        url2title = {}
        for item in search_result:
            url2title[item.link] = item.name
        return url2title.get(url, '')

    def get_collect_data(self, exclude_type: List[str] = []):
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
        return [item for item in self.data if isinstance(item, AnalysisResult)]

    def get_formatted_analysis_result(self, analysis_result_list: List[AnalysisResult] = None):
        if analysis_result_list is None:
            analysis_result_list = self.get_analysis_result()
        formatted_analysis_result = ""
        for idx, item in enumerate(analysis_result_list):
            formatted_analysis_result += f"Analysis report {idx+1}:\n"
            formatted_analysis_result += str(item)
            formatted_analysis_result += "\n\n"
        return formatted_analysis_result
    
    def get_formatted_data_description(self, data_list: List[ToolResult] = None):
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
        # return: tuple(list, str), list: selected data list, str: formatted data description
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
        self.embedding_model = self.config.llm_dict[embedding_model]
        collect_data_list = self.get_collect_data()
        if len(collect_data_list) <= top_k:
            return collect_data_list

        # Embed entries that lack vector representations
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
        
        # Perform semantic search
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
        Generate analysis tasks using LLM.
        
        Args:
            query: Research query describing the target and requirements
            use_llm_name: Name of LLM to use
            max_num: Maximum number of tasks to generate
            existing_tasks: List of existing tasks to avoid duplication
            
        Returns:
            List of generated analysis task strings
        """
        llm = self.config.llm_dict[use_llm_name]
        
        # Format existing tasks for prompt
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
        
        # Handle both list and dict responses
        if isinstance(output, dict):
            output = output.get('tasks', output.get('analysis_tasks', []))
        
        output = output[:max_num]
        self.generated_analysis_tasks = output
        self.save()
        return output
    
    async def generate_collect_tasks(self, query: str, use_llm_name: str, max_num=10, existing_tasks: List[str] = None) -> List[str]:
        """
        Generate data collection tasks using LLM.
        
        Args:
            query: Research query describing the target and requirements
            use_llm_name: Name of LLM to use
            max_num: Maximum number of tasks to generate
            existing_tasks: List of existing tasks to avoid duplication
            
        Returns:
            List of generated collection task strings
        """
        llm = self.config.llm_dict[use_llm_name]
        
        # Format existing tasks for prompt
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
        
        # Handle both list and dict responses
        if isinstance(output, dict):
            output = output.get('tasks', output.get('collect_tasks', output.get('collection_tasks', [])))
        
        output = output[:max_num]
        self.generated_collect_tasks = output
        self.save()
        return output
