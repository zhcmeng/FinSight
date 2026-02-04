from typing import List, Dict, Any, Tuple
import asyncio
import os
import re
import copy
import subprocess
import numpy as np
import docx2pdf
from src.agents.base_agent import BaseAgent
from src.agents import DeepSearchAgent
from src.tools.web.web_crawler import ClickResult
from src.tools import ToolResult, get_tool_categories, get_tool_by_name
from src.agents.report_generator.report_class import Report, Section
from src.utils.helper import extract_markdown, get_md_img
from src.utils.index_builder import IndexBuilder
from src.utils.figure_helper import draw_kline_chart

class ReportGenerator(BaseAgent):
    """
    报告生成智能体 (ReportGenerator Agent)
    
    该智能体负责从原始数据和分析结果中生成完整的结构化报告。它采用了多阶段的 Agent 工作流，
    包括大纲生成、章节撰写、以及后期处理（图片替换、摘要生成、封面制作、参考文献引用和最终渲染）。
    
    主要环节：
    1. _prepare_executor: 注入数据和分析结果的读取接口，支持代码化的内容编排。
    2. _phase 流程管理：
       - 'outline': 生成并评审报告大纲。
       - 'sections': 逐章节撰写，调用 LLM 根据数据和分析结论填充内容。
       - 'post_process': 后期处理，包括图片路径替换、摘要/标题生成、封面制作及最终渲染。
    3. _final_polish: 对章节内容进行润色，并确保图表引用正确。
    4. async_run: 管理多阶段的报告生产流水线。
    """
    
    AGENT_NAME = 'report_generator'
    AGENT_DESCRIPTION = 'a agent that can generate report from the data'
    NECESSARY_KEYS = ['task']

    def __init__(
        self,
        config,
        tools = [],
        use_llm_name: str = "deepseek-chat",
        use_embedding_name: str = "qwen3-embedding",
        enable_code = True,
        memory = None,
        agent_id: str = None
    ):
        """
        初始化报告生成智能体。

        Args:
            config: 配置对象，包含运行路径、语言设置等。
            tools: 工具列表，初始通常为空。
            use_llm_name: 使用的语言模型名称。
            use_embedding_name: 使用的向量模型名称。
            enable_code: 是否启用代码执行能力。
            memory: 共享记忆组件。
            agent_id: 智能体唯一标识符。
        """
        super().__init__(
            config=config,
            tools=tools,
            use_llm_name=use_llm_name,
            enable_code=enable_code,
            memory=memory,
            agent_id=agent_id
        )
        # 根据语言和目标类型设置加载 Prompts
        from src.utils.prompt_loader import get_prompt_loader
        
        target_language = self.config.config.get('language', 'zh')
        language_mapping = {
            'zh': 'Chinese (中文)',
            'en': 'English'
        }
        target_language_name = language_mapping.get(target_language, target_language)
        self.target_language_name = target_language_name
        target_type = self.config.config.get('target_type', 'general')
        
        # 使用基于 YAML 的加载器加载 Prompt
        self.prompt_loader = get_prompt_loader('report_generator', report_type=target_type)
        
        # 存储 Prompt 为实例属性以便访问
        self.SECTION_WRITING_PROMPT = self.prompt_loader.get_prompt('section_writing')
        self.SECTION_WRITING_WO_CHART_PROMPT = self.prompt_loader.get_prompt('section_writing_wo_chart')
        self.FINAL_POLISH_PROMPT = self.prompt_loader.get_prompt('final_polish')
        
        # 对于通用报告和金融报告，都使用 outline_draft 键
        self.DRAFT_GENERATOR_PROMPT = self.prompt_loader.get_prompt('outline_draft')
        
        self.CRITIQUE_PROMPT = self.prompt_loader.get_prompt('outline_critique')
        self.REFINEMENT_PROMPT = self.prompt_loader.get_prompt('outline_refinement')
        
        # 用于添加摘要和标题的 Prompt
        self.TITLE_PROMPT = self.prompt_loader.get_prompt('title_generation')
        self.ABSTRACT_PROMPT = self.prompt_loader.get_prompt('abstract')

        # 用于美化封面的表格 Prompt
        self.TABLE_BEAUTIFY_PROMPT = self.prompt_loader.get_prompt('table_beautify')
        
        self.use_embedding_name = use_embedding_name
        # 阶段检查点：outline → sections → post_process
        self._phase: str = 'outline'
        # 章节级别的进度计数器
        self._section_index_done: int = 0
        # 后期处理子阶段：0-图片替换, 1-摘要/标题, 2-封面, 3-参考文献, 4-渲染
        self._post_stage: int = 0

    def _set_default_tools(self):
        """
        附加报告生成器所需的默认工具/智能体。
        目前主要是 DeepSearchAgent，用于在生成过程中进行深度搜索。
        """
        tool_list = []
        # 附加深度搜索智能体（共享相同的记忆）
        tool_list.append(DeepSearchAgent(config=self.config, use_llm_name=self.use_llm_name, memory=self.memory))
        for tool in tool_list:
            self.memory.add_dependency(tool.id, self.id)
        self.tools = tool_list
    
    async def _prepare_executor(self):
        """
        准备代码执行器，注入用于章节撰写的数据访问函数。
        
        注入的函数包括：
        - get_data(data_id): 根据索引获取数据集。
        - get_analysis_result(data_id): 获取匹配查询的分析结果。
        - get_data_from_deep_search(query): 调用深度搜索智能体获取信息。
        """
        current_task_data = self.current_task_data
        tool_list = self.tools
        collect_data_list = self.memory.get_collect_data(exclude_type=['search', 'click'])
        analysis_result_list = self.memory.get_analysis_result()
        
        def _get_data(data_id: int):
            """根据索引获取数据集"""
            if 0 <= data_id < len(collect_data_list):
                return collect_data_list[data_id].data
            else:
                raise ValueError(f"Invalid data_id: {data_id}. Available range: 0-{len(collect_data_list)-1}")
        
        def _get_analysis_result(data_id: int):
            """获取分析结果"""
            if 0 <= data_id < len(analysis_result_list):
                return str(analysis_result_list[data_id])[:3000]
            else:
                raise ValueError(f"Invalid data_id: {data_id}. Available range: 0-{len(analysis_result_list)-1}")
        
        def _get_deepsearch_result(query: str):
            """同步运行深度搜索智能体"""
            ds_agent = tool_list[0]
            output = asyncio.run(ds_agent.async_run(input_data={
                'task': current_task_data.get('task', ''),
                'query': query
            }))
            return output['final_result']
        
        self.code_executor.set_variable("get_data", _get_data)
        self.code_executor.set_variable("get_analysis_result", _get_analysis_result)
        self.code_executor.set_variable("get_data_from_deep_search", _get_deepsearch_result)

        
    
    async def _prepare_init_prompt(self, input_data: dict) -> list[dict]:
        """
        为章节撰写准备初始 Prompt。
        
        该方法会汇总可用的数据源（数据集和分析报告）信息，并将其注入到 Prompt 中，
        以便 LLM 在撰写章节时能够引用这些数据。

        Args:
            input_data: 包含任务描述、章节大纲等信息的字典。

        Returns:
            包含初始用户消息的消息列表。
        """
        task = input_data.get('task')
        section_outline = input_data.get('section_outline')
        max_iterations = input_data.get('max_iterations', 10)
        if not task:
            raise ValueError("Input data must contain a 'task' key.")
        
        # 从 Prompt 加载器中获取数据 API 描述
        data_api_description = self.prompt_loader.get_prompt('data_api')
        
        # 为智能体准备数据信息
        collect_data_list = self.memory.get_collect_data(exclude_type=['search', 'click'])
        analysis_result_list = self.memory.get_analysis_result()
        data_info = "\n\n## Available Datas\n\n"
        for idx, item in enumerate(collect_data_list):
            data_info += f"**Data ID {idx}:**\n{item.brief_str()}\n\n"
        data_info += "\nYou can access these datasets using `get_data(data_id)` in your code.\n"
        data_info += "\n\n## Available Analysis Reports\n\n"
        for idx, item in enumerate(analysis_result_list):
            data_info += f"**Analysis Report ID {idx}:**\n{item.brief_str()}\n\n"
        data_info += "\nYou can access these analysis reports using `get_analysis_result(analysis_result_id)` in your code.\n"
        
        # 根据是否启用图表选择不同的撰写 Prompt
        if self.enable_chart:
            return [{
                "role": "user",
                "content": self.SECTION_WRITING_PROMPT.format(
                    task=task,
                    report_theme=input_data.get('task'),
                    section_description=section_outline,
                    data_api=data_api_description,
                    data_info=data_info,
                    max_iterations=max_iterations,
                    target_language=self.target_language_name
                )
            }]
        else:
            return [{
                "role": "user",
                "content": self.SECTION_WRITING_WO_CHART_PROMPT.format(
                    task=task,
                    report_theme=input_data.get('task'),
                    section_description=section_outline,
                    data_api=data_api_description,
                    data_info=data_info,
                    max_iterations=max_iterations,
                    target_language=self.target_language_name
                )
            }]

    async def _handle_search_action(self, action_content: str):
        """处理 'search' 动作，调用深度搜索工具。"""
        search_result = await self.tools[0].async_run(input_data={'query': action_content})
        return {
            'action': 'search',
            'action_content': action_content,
            'result': search_result['final_result'],
            'continue': True,
        }
    
    async def _handle_report_action(self, action_content: str):
        """处理 'report' 动作，返回章节生成结果。"""
        return {
            "action": "report",
            "action_content": action_content,
            "result": action_content,
            "continue": False,
        }
    async def _handle_outline_action(self, action_content: str):
        """处理 'outline' 动作，返回大纲生成结果。"""
        return {
            "action": "outline",
            "action_content": action_content,
            "result": action_content,
            "continue": False,
        }
    
    async def _handle_draft_action(self, action_content: str):
        """处理 'draft' 动作，通常在大纲生成的迭代中使用。"""
        return {
            "action": "draft",
            "action_content": action_content,
            "result": action_content,
            "continue": False,
        }

    
    async def _final_polish(self, section_input_data, draft_section: str):
        """
        对生成的章节内容进行最终润色。
        
        该过程会结合记忆中的分析结果（特别是图表信息），确保章节内容流畅，
        并能正确引用相关的图表占位符。

        Args:
            section_input_data: 章节输入数据。
            draft_section: 章节草稿内容。

        Returns:
            润色后的章节内容。
        """
        all_analysis_result = self.memory.get_analysis_result()
        all_image_list = []
        for analysis_result in all_analysis_result:
            all_image_list.extend(analysis_result.get_all_img())
        reference_image = '\n'.join(all_image_list)
        
        final_prompt = self.FINAL_POLISH_PROMPT.format(
            draft_report = draft_section,
            reference_image = reference_image,
            target_language = self.target_language_name
        )
        
        final_message = [{"role": "user", "content": final_prompt}]
        output = await self.llm.generate(messages = final_message)
        final_section = extract_markdown(output)
        return final_section
    
    async def _replace_image_path(self, report):
        """
        将报告中的占位符图片引用替换为实际的本地路径。
        
        该方法会：
        1. 收集所有分析结果中的图片路径。
        2. 使用 IndexBuilder 构建图片标题的向量索引。
        3. 在报告正文中查找 `@import` 占位符。
        4. 通过语义搜索匹配最相关的图片，并将其替换为 Markdown 图片语法。

        Args:
            report: 报告对象。

        Returns:
            替换图片路径后的报告对象。
        """
        # 如果禁用了图表，直接移除 @import 占位符
        if not self.enable_chart:
            for section in report.sections:
                section_new_content = []
                for p_paragraph in section._content:
                    # 将 @import.* 替换为空字符串
                    p_paragraph = re.sub(r'@import.*', '', p_paragraph, flags=re.DOTALL)
                    section_new_content.append(p_paragraph)
                section._content = section_new_content
            return report
        
        def remove_suffix(name: str):
            """移除文件扩展名"""
            return name.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace(".md", "")
        def is_image_file(name: str):
            """判断是否为图片文件"""
            return name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".md")
        
        all_analysis_result = self.memory.get_analysis_result()
        img_captions = []
        img_paths = []
        for analysis_result in all_analysis_result:
            short2long = {}
            img_dicts = {} # 映射：caption -> 绝对路径 
            chart_name_mapping = analysis_result.chart_name_mapping
            for long_name, short_name in chart_name_mapping.items():
                short2long[remove_suffix(short_name)] = remove_suffix(long_name)
            image_save_dir = analysis_result.image_save_dir
            for image_name in os.listdir(image_save_dir):
                if is_image_file(image_name):
                    img_path = os.path.join(image_save_dir, image_name)
                    img_name = remove_suffix(image_name)
                    long_image_name = short2long.get(img_name, "")
                    if long_image_name != "":
                        img_dicts[long_image_name] = img_path
            img_captions.extend(list(img_dicts.keys()))
            img_paths.extend(list(img_dicts.values()))
            
        if len(img_captions) == 0:
            self.logger.warning("No image captions found, skip image path replacement")
            return report
            
        self.logger.info(f"Building index for {len(img_captions)} images")
        index = IndexBuilder(config=self.config, embedding_model=self.use_embedding_name, working_dir=self.working_dir)
        await index._build_index(img_captions)

        used_img_list = []
        figure_idx = 1
        for section in report.sections:
            section_new_content = []
            for p_paragraph in section._content:
                match = re.findall(r'@import.*', p_paragraph,flags=re.DOTALL)
                try:
                    self.logger.debug(f"Section image placeholders: {len(match)}")
                except Exception:
                    pass
                if match and len(match) > 0:
                    for img_name in match:
                        # img_name 是简短的占位符字符串
                        most_similar_idx = (await index.search(img_name))[0]['id']
                        detect_img_name = img_captions[most_similar_idx]
                        detect_img_path = img_paths[most_similar_idx]
                        
                        if len(img_captions) == 1:
                            # 没有更多图片可以匹配
                            self.logger.warning("Available images are exhausted; stop replacing images.")
                            # 直接删除图片占位符
                            p_paragraph = p_paragraph.replace(img_name, "")
                            continue
                        
                        # 从候选列表中移除已使用的图片，并重建索引
                        del img_captions[most_similar_idx]
                        del img_paths[most_similar_idx]
                        await index._build_index(img_captions)

                        # 获取 Markdown 图片字符串并替换
                        new_string = get_md_img(detect_img_path, remove_suffix(os.path.basename(detect_img_path)), figure_idx)
                        figure_idx += 1
                        used_img_list.append(detect_img_name)
                        p_paragraph = p_paragraph.replace(img_name, new_string)

                section_new_content.append(p_paragraph)
            section._content = section_new_content
        return report


    
    async def _add_abstract(self, input_data, report):
        """
        为报告生成摘要并更新标题。
        
        Args:
            input_data: 输入数据。
            report: 报告对象。

        Returns:
            更新了摘要和标题的报告对象。
        """
        abstract_prompt = self.ABSTRACT_PROMPT
        title_prompt = self.TITLE_PROMPT

        # 生成摘要
        response_content = await self.llm.generate(
            messages = [
            {
                'role': 'user',
                'content': abstract_prompt.format(target_language=self.target_language_name, report_content=report.content)
            }
        ])
        response_content = extract_markdown(response_content)
        report.abstract = response_content
        
        # 生成更优的标题
        new_title = await self.llm.generate(
            messages = [
            {
                'role': 'user',
                'content': title_prompt.format(target_language=self.target_language_name, report_content=report.content)
            }
        ])
        new_title = new_title.replace("#","").strip()
        report._content = f"# {new_title}\n\n"

        return report

    async def _add_cover_page(self, input_data, report):
        """
        为公司调研类报告添加封面页和基本面数据。
        
        该方法会：
        1. 检查报告类型是否为 'company'。
        2. 从记忆中提取利润表、资产负债表、现金流量表和股东结构等数据。
        3. 调用 LLM 美化这些表格。
        4. 绘制股价走势图（如果可用）。
        5. 将这些内容作为报告的第一章节插入。

        Args:
            input_data: 输入数据，包含股票代码和报告类型。
            report: 报告对象。

        Returns:
            添加了封面信息的报告对象。
        """
        pipeline_type = input_data.get('target_type', 'company')
        if pipeline_type != 'company':
            return report
        stock_code = input_data.get('stock_code', '')
        if stock_code == "":
            return report

        output_str = "\n\n## Company Fundamentals\n\n"
        # 定义需要提取的财务报表和展示名称
        collect_data_list = self.memory.get_collect_data()
        table_configs = [
            ("Income statement", "Income Statement"),
            ("Balance sheet", "Balance Sheet"),
            ("Cash-flow statement", "Cash-Flow Statement"),
            ("Shareholding structure", "Shareholder Structure"),
        ]
        for keyword, display_name in table_configs:
            # 查找匹配股票代码和关键词的数据项
            target_item_list = [item for item in collect_data_list if keyword in item.name and stock_code in item.name]
            if len(target_item_list) == 0:
                print(f"No {display_name} data found")
                continue
            else:
                table_data = target_item_list[0].data
                if table_data is None:
                    print(f"{display_name} data is empty, skip formatting")
                    continue
                
                # 特殊处理财务报表的列名
                if keyword in ["Income statement", "Balance sheet", "Cash-flow statement"]:
                    if 'Category' in table_data.columns:
                        table_data.rename(columns={'Category': 'Line item (RMB mn)'}, inplace=True)
                
                # 调用 LLM 美化表格
                prompt = self.TABLE_BEAUTIFY_PROMPT.format(table_name=display_name, table_data=table_data.to_markdown(index=False))
                response = await self.llm.generate(
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                )
                table_string = "\n".join([line for line in response.split("\n") if line.strip() != ""])

                output_str += f'\n\n### {display_name}\n\n'
                output_str += table_string
                output_str += '\n\n'
        
        # 渲染股价走势图
        try:
            self.logger.info("Rendering stock-price chart for cover page")
            target_item_list = [item for item in collect_data_list if 'candlestick' in item.name.lower() and stock_code in item.name]
            if len(target_item_list) != 0:
                kline_data = target_item_list[0].data
                if kline_data is None:
                    self.logger.warning("Candlestick data is empty; skip price visualization")
                else:
                    if isinstance(kline_data, list) and len(kline_data) == 1:
                        kline_data = kline_data[0]
                    # 统一日期和收盘价的列名
                    if 'date' not in kline_data.columns:
                        if '\u65e5\u671f' in kline_data.columns:
                            kline_data.rename(columns={'\u65e5\u671f': 'date'}, inplace=True)
                        if '\u6536\u76d8' in kline_data.columns:
                            kline_data.rename(columns={'\u6536\u76d8': 'close'}, inplace=True)
                    fig_path = draw_kline_chart(kline_data, self.working_dir)
                    output_str += f'\n\n### Share Price Trend\n\n'
                    output_str += f'![Trailing price performance]({fig_path})\n\n'
        except Exception as e:
            self.logger.error(f"Failed to draw price trend: {e}", exc_info=True)
            pass

        # 将基本面信息作为第一章插入
        first_section = Section('Company Fundamentals', output_str)
        first_section.set_content(output_str)
        report.sections = [first_section] + report.sections

        return report
    

    async def _add_reference(self, report):
        """
        在报告末尾添加参考文献章节，并将正文中的引用占位符替换为数字索引。
        
        该方法会：
        1. 汇总所有采集到的原始数据源。
        2. 构建数据源的向量索引。
        3. 在报告各章节中查找 `[Source: ...]` 占位符。
        4. 通过语义搜索匹配最相关的数据源，并分配引用编号。
        5. 在报告最后添加 'Reference Data Sources' 章节。

        Args:
            report: 报告对象。

        Returns:
            添加了参考文献引用的报告对象。
        """
        collect_data_list = self.memory.get_collect_data() # 仅使用原始数据，不包含分析结果
        all_data = []
        for item in collect_data_list:
            # 提取名称和描述用于索引，提取来源用于展示
            name = item.name + '\n' + item.description
            content = item.source
            # 对于点击结果（网页），尝试获取网页标题
            if isinstance(item, ClickResult):
                url = item.link
                title = self.memory.get_url_title(url)
                if title == "":
                    title = item.name
                content = f"{title}\n{url}"

            if content not in [ii['content'] for ii in all_data]:
                all_data.append({
                    'name': name,
                    'content': content 
                })
        self.logger.info(f"Total data for reference: {len(all_data)}")
        
        # 构建参考文献的向量索引
        total_corpus = [item['name'] for item in all_data]
        index = IndexBuilder(config=self.config, embedding_model=self.use_embedding_name, working_dir=self.working_dir)
        await index._build_index(total_corpus)

        total_cited_dict = {} # 映射：原始数据索引 -> 引用编号
        for section in report.sections:
            try:
                self.logger.debug(f"Processing section, content length={len(section.content)}")
            except Exception:
                pass
            section_new_content = []
            for p_paragraph in section._content:
                content = p_paragraph
                # 查找引用占位符
                match_list = re.findall(r'\[[Ss]ource[：:]\s*(.*?)\]',content)
                self.logger.debug(f"Match list: {match_list}")
                for match_item in match_list:
                    # 通过向量搜索寻找最相关的数据源
                    search_result = await index.search(match_item, top_k=5)
                    score_list = [item['score'] for item in search_result]
                    id_list = [item['id'] for item in search_result]
                    
                    # 使用 softmax 归一化得分
                    sorted_idx = np.argsort(score_list)[::-1]
                    score_list = np.array(score_list)
                    score_list = np.exp(score_list) / np.sum(np.exp(score_list))

                    cite_list = []
                    for pos in sorted_idx:
                        pos = int(pos)
                        actual_idx = id_list[pos]
                        # 仅保留得分超过阈值的前 5 个引用
                        if score_list[pos] > 0.2 and len(cite_list) < 5:
                            cite_list.append(actual_idx)
                    if len(cite_list) == 0:
                        cite_list.append(id_list[sorted_idx[0]])
                    
                    # 分配或获取现有的引用编号
                    new_cite_list = []
                    for idx in cite_list:
                        if idx not in total_cited_dict:
                            total_cited_dict[idx] = len(total_cited_dict) + 1
                        new_cite_list.append(total_cited_dict[idx])
                    
                    # 替换占位符为 [1, 2, ...]
                    pattern_to_replace = r'\[[Ss]ource[：:]\s*' + re.escape(match_item) + r'\]'
                    content = re.sub(pattern_to_replace, f'[{",".join([str(item) for item in new_cite_list])}]', content)

                section_new_content.append(content)
            section._content = section_new_content

        # 构建参考文献章节的文本
        reference_str = "## Reference Data Sources\n\n"
        for old_index, new_index in total_cited_dict.items():
            content = all_data[old_index]['content']
            content = content.replace("\n", " ").replace("[PDF]", "")
            reference_str += f"{new_index}. {content}\n"
        
        # 添加为新的章节
        new_section = Section('Reference Data Sources', reference_str)
        new_section.set_content(reference_str)
        report.sections.append(new_section)
        return report


        

    async def post_process_report(self, input_data, report):
        """
        报告的后期处理流程，支持断点续传。
        
        处理阶段包括：
        0: 替换图片路径（将占位符改为本地绝对路径）。
        1: 添加摘要和优化标题。
        2: 添加封面和基本面数据页（针对公司调研报告）。
        3: 添加参考文献章节。
        4: 渲染报告为 Markdown, Docx 和 PDF 格式。

        Args:
            input_data: 输入数据。
            report: 报告对象。

        Returns:
            处理完成后的报告对象。
        """
        current_state = {
            'phase': 'post_process',
            'post_stage': self._post_stage,
            'report_obj': report,
        }
        # 阶段 0: 替换图片路径
        if self._post_stage <= 0:
            self.logger.info("[Phase2] Step 0: replace image paths")
            report = await self._replace_image_path(report)
            self._post_stage = 1
            current_state['report_obj_stage1'] = copy.deepcopy(report)
            current_state['report_obj'] = report
            current_state['post_stage'] = self._post_stage
            await self.save(state=current_state, checkpoint_name='report_latest.pkl')
            self.logger.info("[Phase2] Step 0 done, checkpoint saved")

        # 阶段 1: 添加摘要/标题（根据配置决定是否添加导言）
        if self._post_stage <= 1:
            if getattr(self, 'add_introduction', True):
                self.logger.info("[Phase2] Step 1: add abstract and title")
                report = await self._add_abstract(input_data, report)
            else:
                self.logger.info("[Phase2] Step 1: skipping abstract/introduction (add_introduction=False for general reports)")
                # 即使不添加摘要，也生成一个更好的标题
                new_title = await self.llm.generate(
                    messages = [
                    {
                        'role': 'user',
                        'content': self.TITLE_PROMPT.format(target_language=self.target_language_name, report_content=report.content)
                    }
                ])
                new_title = new_title.replace("#","").strip()
                report._content = f"# {new_title}\n\n"
            self._post_stage = 2
            current_state['report_obj_stage2'] = copy.deepcopy(report)
            current_state['report_obj'] = report
            current_state['post_stage'] = self._post_stage
            await self.save(state=current_state, checkpoint_name='report_latest.pkl')
            self.logger.info("[Phase2] Step 1 done, checkpoint saved")

        # 阶段 2: 添加封面/基本数据页
        if self._post_stage <= 2:
            self.logger.info("[Phase2] Step 2: add cover/basic data page")
            report = await self._add_cover_page(input_data, report)
            self._post_stage = 3
            current_state['report_obj_stage3'] = copy.deepcopy(report)
            current_state['report_obj'] = report
            current_state['post_stage'] = self._post_stage
            await self.save(state=current_state, checkpoint_name='report_latest.pkl')
            self.logger.info("[Phase2] Step 2 done, checkpoint saved")

        # 阶段 3: 添加参考文献（根据配置决定）
        if self._post_stage <= 3:
            if getattr(self, 'add_reference_section', True):
                self.logger.info("[Phase2] Step 3: add references")
                report = await self._add_reference(report)
            else:
                self.logger.info("[Phase2] Step 3: skipping reference section (add_reference_section=False)")
            self._post_stage = 4
            current_state['report_obj_stage4'] = copy.deepcopy(report)
            current_state['report_obj'] = report
            current_state['post_stage'] = self._post_stage
            await self.save(state=current_state, checkpoint_name='report_latest.pkl')
            self.logger.info("[Phase2] Step 3 done, checkpoint saved")

        # 阶段 4: 渲染为 docx
        if self._post_stage <= 4:
            self.logger.info("[Phase2] Step 4: render report to docx")
            working_dir = self.config.config['working_dir']
            md_path = os.path.join(working_dir, f'{report.title}.md')
            docx_path = os.path.join(working_dir, f'{report.title}.docx')
            content = report.content
            # 移除 Markdown 代码块标记（如果有）
            content = content.replace("```markdown", "").replace("```", "")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            media_dir = os.path.join(working_dir, "media")
            reference_doc = self.config.config['reference_doc_path']
            
            # 使用 pandoc 转换为 docx
            pandoc_cmd = [
                "pandoc",
                md_path,
                "-o",
                docx_path,
                "--standalone",
                "--toc",
                "--toc-depth=3",
                f"--resource-path={working_dir}",
                f"--reference-doc={reference_doc}"
            ]
            if os.path.exists(media_dir):
                pandoc_cmd.append(f"--extract-media={media_dir}")
            
            self.logger.info(f"Executing Pandoc: {' '.join(pandoc_cmd)}")
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            subprocess.run(pandoc_cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
            
            # 转换为 PDF
            pdf_path = docx_path.replace(".docx", ".pdf")
            try:
                docx2pdf.convert(docx_path, pdf_path)
            except Exception as e:
                self.logger.error(f"Failed to convert docx to pdf: {e}", exc_info=True)
            
            self._post_stage = 5
            current_state['rendered_md'] = md_path
            current_state['rendered_docx'] = docx_path
            current_state['finished'] = True
            await self.save(state=current_state, checkpoint_name='report_latest.pkl')
            self.logger.info(f"[Phase2] Step 4 done, rendered files: md={md_path}, docx={docx_path}, pdf={pdf_path}")
            
        return report

    def _get_persist_extra_state(self) -> Dict[str, Any]:
        """
        提供额外的状态信息，以便父类的持久化机制可以恢复流水线阶段。
        """
        return {
            'phase': getattr(self, '_phase', 'outline'),
            'section_index': getattr(self, '_section_index_done', 0),
            'post_stage': getattr(self, '_post_stage', 0),
        }

    def _load_persist_extra_state(self, state: Dict[str, Any]):
        """
        从检查点恢复阶段元数据。
        """
        # 从 _get_persist_extra_state 填充的 extra 字段中恢复
        extra = state.get('extra', {})
        
        phase = extra.get('phase') or state.get('phase')
        if isinstance(phase, str):
            self._phase = phase
        
        section_index = extra.get('section_index') or state.get('section_index')
        if section_index is not None:
            try:
                self._section_index_done = int(section_index)
            except Exception:
                pass
        
        post_stage = extra.get('post_stage') or state.get('post_stage')
        if post_stage is not None:
            try:
                self._post_stage = int(post_stage)
            except Exception:
                pass
        
        enable_chart = extra.get('enable_chart') or state.get('enable_chart')
        if enable_chart is not None:
            try:
                self.enable_chart = bool(enable_chart)
            except Exception:
                pass
        else:
            self.enable_chart = True

    
    async def _prepare_outline_prompt(self, input_data):
        """
        准备大纲生成的初始 Prompt。

        Args:
            input_data: 输入数据。

        Returns:
            消息列表。
        """
        max_iterations = input_data.get('max_iterations', 10)
        outline_template_path = self.config.config.get('outline_template_path', None)
        
        # 加载大纲模板（如果存在）
        if outline_template_path is None or not os.path.exists(outline_template_path):
            outline_template = ""
        else:
            with open(outline_template_path, 'r', encoding='utf-8') as f:
                outline_template = f.read()
                
        # 准备数据 API 描述和可用的分析信息
        data_api_description = self.prompt_loader.get_prompt('data_api_outline')
        analysis_result_list = self.memory.get_analysis_result()
        
        data_info = "You have access to the following analysis results:\n\n"
        for idx, result in enumerate(analysis_result_list):
            data_info += f"**Analysis Report ID {idx}:**\n{result.brief_str()}\n\n"
        data_info += "\nYou can retrieve detailed content using `get_analysis_result(analysis_id)` in your code.\n"
        
        # 构建大纲生成的初始 Prompt
        initial_prompt = self.DRAFT_GENERATOR_PROMPT.format(
            task=input_data['task'],
            report_requirements=outline_template,
            data_api=data_api_description,
            data_info=data_info,
            max_iterations=max_iterations,
            target_language=self.target_language_name
        )
        return [{"role": "user", "content": initial_prompt}]

    async def generate_outline(
        self, 
        input_data, 
        max_iterations: int = 10,
        stop_words: list[str] = [],
        echo=False,
        resume: bool = True,
        checkpoint_name: str = 'outline_latest.pkl'
    ):
        """
        通过 Agent 工作流生成报告大纲。

        Args:
            input_data: 包含任务元数据的字典。
            max_iterations: 最大交互轮数。
            stop_words: 停止词列表。
            echo: 是否回显输出。
            resume: 是否从检查点恢复。
            checkpoint_name: 检查点文件名。

        Returns:
            填充了大纲章节的 Report 对象。
        """
        # 为大纲生成准备执行环境
        await self._prepare_executor()

        self.logger.info(f"[Outline] Starting agentic outline generation (max {max_iterations} rounds)")
        
        outline_input_data = {
            'task': input_data['task'],
            'max_iterations': max_iterations
        }
        self.current_task_data = outline_input_data

        # 调用父类的 async_run 执行大纲生成逻辑
        outline_result = await super().async_run(
            input_data=outline_input_data,
            max_iterations=max_iterations,
            stop_words=stop_words,
            echo=echo,
            resume=resume,
            checkpoint_name=checkpoint_name,
            prompt_function=self._prepare_outline_prompt,
        )
    
        # 提取并创建报告对象
        outline_content = extract_markdown(outline_result['final_result'])
        return Report(outline_content) if outline_content else Report("# Error: Could not generate outline")

    async def async_run(
        self, 
        input_data: dict, 
        max_iterations: int = 10,
        stop_words: list[str] = [],
        echo=False,
        resume: bool = True,
        checkpoint_name: str = 'report_latest.pkl',
        enable_chart = True,
        add_introduction: bool = None,  # None 表示根据 target_type 自动检测
        add_reference_section: bool = True
    ) -> dict:
        """
        报告生成器的三阶段执行流：
        阶段 0: 大纲生成 (Outline Creation)
        阶段 1: 逐章节撰写 (Per-section Drafting)
        阶段 2: 后期处理 (Post Processing)

        Args:
            input_data: 输入数据，包含任务描述等。
            max_iterations: 智能体每阶段最大迭代次数。
            stop_words: 停止词列表。
            echo: 是否实时回显 LLM 输出。
            resume: 是否尝试从之前的检查点恢复。
            checkpoint_name: 持久化状态的文件名。
            enable_chart: 是否在报告中包含图表。
            add_introduction: 是否添加摘要/导言。
            add_reference_section: 是否添加参考文献章节。

        Returns:
            生成的报告对象（在完成所有阶段后）。
        """
        # 初始化或恢复阶段状态
        report = None
        start_index = 0
        self.enable_chart = enable_chart
        input_data['max_iterations'] = max_iterations
        
        # 根据 target_type 自动配置后期处理选项
        target_type = self.config.config.get('target_type', 'general')
        
        # 通用/深度搜索类报告默认不添加导言（用户通常有自己的结构）
        # 公司/财务类报告默认添加（标准格式）
        if add_introduction is None:
            self.add_introduction = target_type not in ['general']
        else:
            self.add_introduction = add_introduction
        
        self.add_reference_section = add_reference_section
        
        # 尝试从检查点恢复
        if resume:
            state = await self.load(checkpoint_name=checkpoint_name)
            if state is not None:
                self._load_persist_extra_state(state)
                self.logger.info(f"[Resume] phase={getattr(self, '_phase', None)}, section_index={getattr(self, '_section_index_done', None)}, post_stage={getattr(self, '_post_stage', None)}")
                
                # 如果已完成，直接返回
                if state.get('finished'):
                    restored_report = state.get('report_obj')
                    if restored_report:
                        self.logger.info("Report already completed, restoring from checkpoint")
                        return restored_report
                
                # 恢复进行中的报告对象
                restored_report = state.get('report_obj')
                if restored_report is not None:
                    report = restored_report
                    start_index = self._section_index_done
                    self.logger.info(f"[Resume] Restored report object, will resume from section_index={start_index}")
        
        # 阶段 0: 生成大纲
        if self._phase == 'outline' or report is None:
            self.logger.info("[Phase0] Generating Report Outline")
            report = await self.generate_outline(
                input_data, 
                max_iterations=max_iterations,
                stop_words=stop_words,
                echo=echo,
                resume=resume,
                checkpoint_name='outline_latest.pkl'
            )
            self._phase = 'sections'
            # 持久化大纲状态
            await self.save(
                state={
                    'phase': self._phase,
                    'report_obj': report,
                    'input_data': input_data,
                    'enable_chart': self.enable_chart,
                },
                checkpoint_name=checkpoint_name,
            )
            self.memory.save()
            self.logger.info(f"[Phase0] Completed: outline sections={len(report.sections)}")

        # 阶段 1: 逐章节生成
        if self._phase == 'sections':
            self.logger.info("[Phase1] Begin generating sections")
            # 遍历大纲中的章节进行撰写
            for idx, section in enumerate(report.sections):
                if idx < start_index:
                    continue
                section_input_data = input_data.copy()
                section_input_data['section_outline'] = section.outline
                self.logger.info(f"[Phase1] Section {idx+1}/{len(report.sections)} start")
                
                # 准备执行环境
                await self._prepare_executor()
                
                # 每个章节的运行都有自己的检查点
                section_result = await super().async_run(
                    input_data=section_input_data,
                    max_iterations=max_iterations,
                    stop_words=stop_words,
                    echo=echo,
                    resume=resume and idx == start_index,
                    checkpoint_name=f'section_{idx}.pkl'
                )
                draft_section = section_result['final_result']
                
                # 对章节内容进行最终润色
                final_section = await self._final_polish(section_input_data, draft_section)
                
                # 记录日志
                self.memory.add_log(
                    id=self.id,
                    type=self.type,
                    input_data=section_input_data,
                    output_data=section_result,
                    error=False,
                    note=f"Report generator executed successfully"
                )
                section.set_content(final_section) 
                
                # 每完成一章保存一次进度
                await self.save(
                    state={
                        'phase': 'sections',
                        'section_index': idx + 1,
                        'report_obj': report,
                        'input_data': input_data,
                    },
                    checkpoint_name=checkpoint_name,
                )
                self.memory.save()
                self._section_index_done = idx + 1
                self.logger.info(f"[Phase1] Section {idx+1} done, checkpoint saved (section_index={self._section_index_done})")
            
            # 所有章节完成后进入后期处理阶段
            self._phase = 'post_process'
            await self.save(
                state={
                    'phase': self._phase,
                    'section_index': self._section_index_done,
                    'post_stage': self._post_stage,
                    'report_obj': report,
                    'input_data': input_data,
                },
                checkpoint_name=checkpoint_name,
            )

        # 阶段 2: 后期处理
        if self._phase == 'post_process':
            self.logger.info("[Phase2] Begin post processing")
            report = await self.post_process_report(input_data, report)
            self.logger.info("[Phase2] Post processing completed")

        return report
    
