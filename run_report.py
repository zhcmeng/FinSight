import argparse
import os
import sys
from pathlib import Path
import asyncio
import traceback
from collections import defaultdict
import logging
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

from src.config import Config
from src.agents import DataCollector, DataAnalyzer, ReportGenerator
from src.memory import Memory
from src.utils import setup_logger
from src.utils import get_logger

# 设置日志上下文
get_logger().set_agent_context('runner', 'main')


async def run_report(resume: bool = True):
    """
    运行报告生成的全流程
    :param resume: 是否从断点恢复执行
    """
    # 获取模型名称
    use_llm_name = os.getenv("DS_MODEL_NAME")
    use_vlm_name = os.getenv("VLM_MODEL_NAME")
    use_embedding_name = os.getenv("EMBEDDING_MODEL_NAME")
    
    # 初始化配置
    config = Config(
        config_file_path='my_config.yaml',
        config_dict={}
    )
    collect_tasks = config.config.get('custom_collect_tasks') or []
    analysis_tasks = config.config.get('custom_analysis_tasks') or []
    
    # 步骤 1: 初始化内存系统
    memory = Memory(config=config)
    
    # 步骤 2: 初始化日志
    log_dir = os.path.join(config.working_dir, 'logs')
    logger = setup_logger(log_dir=log_dir, log_level=logging.INFO)
    
    if resume:
        memory.load()
        logger.info("Memory state loaded")
    
    # 步骤 3: 如果任务尚未生成，利用 LLM 自动拆解研究任务
    research_query = f"Research target: {config.config['target_name']} (ticker: {config.config['stock_code']}), target type: {config.config.get('target_type', 'company')}"
    
    # 生成数据收集任务
    if not memory.generated_collect_tasks:
        logger.info("Generating collect tasks using LLM...")
        generated_collect_tasks = await memory.generate_collect_tasks(
            query=research_query,
            use_llm_name=use_llm_name,
            max_num=5,
            existing_tasks=collect_tasks  # 传入现有任务以避免重复
        )
        logger.info(f"Generated {len(generated_collect_tasks)} collect tasks")
    else:
        generated_collect_tasks = memory.generated_collect_tasks
        logger.info(f"Using {len(generated_collect_tasks)} previously generated collect tasks")
    
    # 生成分析任务
    if not memory.generated_analysis_tasks:
        logger.info("Generating analysis tasks using LLM...")
        generated_analysis_tasks = await memory.generate_analyze_tasks(
            query=research_query,
            use_llm_name=use_llm_name,
            max_num=5,
            existing_tasks=analysis_tasks  # 传入现有任务以避免重复
        )
        logger.info(f"Generated {len(generated_analysis_tasks)} analysis tasks")
    else:
        generated_analysis_tasks = memory.generated_analysis_tasks
        logger.info(f"Using {len(generated_analysis_tasks)} previously generated analysis tasks")
    
    # 合并自定义任务与生成的任务
    all_collect_tasks = list(collect_tasks) + [task for task in generated_collect_tasks if task not in collect_tasks]
    all_analysis_tasks = list(analysis_tasks) + [task for task in generated_analysis_tasks if task not in analysis_tasks]
    
    logger.info(f"Total collect tasks: {len(all_collect_tasks)} (custom: {len(collect_tasks)}, generated: {len(generated_collect_tasks)})")
    logger.info(f"Total analysis tasks: {len(all_analysis_tasks)} (custom: {len(analysis_tasks)}, generated: {len(generated_analysis_tasks)})")
    
    # 更新待执行任务列表
    collect_tasks = all_collect_tasks
    analysis_tasks = all_analysis_tasks
    
    # 步骤 4: 准备带优先级的任务队列 (值越小优先级越高)
    tasks_to_run = []
    
    # 4.1 数据收集阶段
    for task in collect_tasks:
        tasks_to_run.append({
            'agent_class': DataCollector,
            'task_input': {
                'input_data': {'task': f'Research target: {config.config["target_name"]} (ticker: {config.config["stock_code"]}), task: {task}'},
                'echo': True,
                'max_iterations': 20,
                'resume': resume,
            },
            'agent_kwargs': {
                'use_llm_name': use_llm_name,
            },
            'priority': 1,
        })
    
    # 4.2 数据分析阶段 (在收集之后运行)
    for task in analysis_tasks:
        tasks_to_run.append({
            'agent_class': DataAnalyzer,
            'task_input': {
                'input_data': {
                    'task': f'Research target: {config.config["target_name"]} (ticker: {config.config["stock_code"]})',
                    'analysis_task': task
                },
                'echo': True,
                'max_iterations': 20,
                'resume': resume,
            },
            'agent_kwargs': {
                'use_llm_name': use_llm_name,
                'use_vlm_name': use_vlm_name,
                'use_embedding_name': use_embedding_name,
            },
            'priority': 2,
        })
    
    # 4.3 报告生成阶段
    tasks_to_run.append({
        'agent_class': ReportGenerator,
        'task_input': {
            'input_data': {
                'task': f'Research target: {config.config["target_name"]} (ticker: {config.config["stock_code"]})',
                'task_type': 'company',
            },
            'echo': True,
            'max_iterations': 20,
            'resume': True,
        },
        'agent_kwargs': {
            'use_llm_name': use_llm_name,
            'use_embedding_name': use_embedding_name,
        },
        'priority': 3,
    })

    # 步骤 5: 通过 memory 获取或创建所需的 Agent (内部会记录任务状态)
    agents_info = []
    for task_info in tasks_to_run:
        agent = await memory.get_or_create_agent(
            agent_class=task_info['agent_class'],
            task_input=task_info['task_input'],
            resume=resume,
            priority=task_info['priority'],
            **task_info['agent_kwargs']
        )
        # 获取持久化的优先级 (如果是断点恢复，可能与初始设定不同)
        actual_priority = task_info['priority']
        for saved_task in memory.task_mapping:
            if saved_task.get('agent_id') == agent.id:
                actual_priority = saved_task.get('priority', task_info['priority'])
                break
        
        agents_info.append({
            'agent': agent,
            'task_input': task_info['task_input'],
            'priority': actual_priority,
        })
    
    # 保存初始状态
    memory.save()
    
    
    # 步骤 6: 按优先级分层执行任务 (同一层级内并行执行)
    agents_info.sort(key=lambda x: x['priority'])
    
    # 按优先级分组
    priority_groups = defaultdict(list)
    for agent_info in agents_info:
        priority_groups[agent_info['priority']].append(agent_info)
    
    # 顺序执行每个优先级层级
    sorted_priorities = sorted(priority_groups.keys())
    for priority in sorted_priorities:
        group = priority_groups[priority]
        logger.info(f"\nExecuting priority {priority} group ({len(group)} task(s))")
        
        # 步骤 6.1: 过滤掉已完成的任务
        tasks_to_run = []
        for agent_info in group:
            agent = agent_info['agent']
            if resume and memory.is_agent_finished(agent.id):
                logger.info(f"Agent {agent.id} already completed; skip")
                continue
            tasks_to_run.append(agent_info)
        
        if not tasks_to_run:
            logger.info(f"All tasks with priority {priority} are complete")
            continue
        
        # 步骤 6.2: 在当前层级内并发启动 Agent
        async_tasks = []
        for agent_info in tasks_to_run:
            agent = agent_info['agent']
            logger.info(f"  Starting agent {agent.id}")
            async_tasks.append(asyncio.create_task(
                agent.async_run(**agent_info['task_input'])
            ))
            
        
        # 步骤 6.3: 等待当前层级所有任务完成
        if async_tasks:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            for agent_info, result in zip(tasks_to_run, results):
                agent = agent_info['agent']
                if isinstance(result, Exception):
                    # 格式化异常信息用于调试
                    tb_str = ''.join(traceback.format_exception(type(result), result, result.__traceback__))
                    logger.error(f"  Task failed: Agent {agent.id}, error: {result}\n{tb_str}")
                else:
                    logger.info(f"  Task finished: Agent {agent.id}")
        
        logger.info(f"Priority {priority} group finished\n")
    
    # 步骤 7: 持久化最终状态
    memory.save()
    logger.info("All tasks completed")


if __name__ == '__main__':
    # 默认开启断点恢复模式
    resume = False
    asyncio.run(run_report(resume=resume))
