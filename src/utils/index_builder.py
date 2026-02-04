import os
import sys
import json
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

class IndexBuilder:
    """
    索引构建器：负责管理和检索文本向量（Embeddings）。
    
    该类实现了以下核心功能：
    1. 缓存管理：通过 JSON 文件缓存已生成的向量，避免重复调用 LLM API 产生费用。
    2. 索引持久化：使用 numpy (npz) 格式保存向量数据，支持快速加载。
    3. 向量检索：基于余弦相似度（点积）进行 Top-K 语义搜索。
    4. 批量处理：支持按批次（batch）请求向量，提高处理效率。
    """
    def __init__(
        self,
        config, 
        embedding_model: str = "qwen3-embedding",
        working_dir: str = "./agent_working/",
    ):
        """
        初始化索引构建器。

        Args:
            config: 配置对象，需包含 llm_dict 用于获取向量模型实例。
            embedding_model (str): 配置文件中指定的向量模型名称。
            working_dir (str): 数据存储的基准目录。
        """
        self.llm = config.llm_dict[embedding_model]
        self.embedding_model_name = embedding_model
        # 向量索引保存路径：存储所有数据的特征向量
        self.save_file_path = os.path.join(working_dir, "embeddings", "collect_data_list.npz")
        # 缓存文件路径：存储 原始文本 -> 向量 的映射，以及 搜索查询 -> 结果 的映射
        self.cache_file_path = os.path.join(working_dir, "embeddings", "cache.json")
        self.cache = {}
        self.embeddings = []
        
        # 加载持久化数据
        self._load_cache()
        self.load_index()

    def _load_cache(self):
        """
        加载搜索和向量缓存。
        包含向后兼容逻辑，能够自动迁移旧版本的扁平化缓存结构。
        """
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # 向后兼容处理：
                # 情况 1: 已经是新版结构 {"search": ..., "embeddings": ...}
                if isinstance(loaded, dict) and ("search" in loaded or "embeddings" in loaded):
                    self.cache = loaded
                # 情况 2: 旧版扁平结构，将其迁移到 "search" 字段下
                elif isinstance(loaded, dict):
                    self.cache = {"search": loaded, "embeddings": {}}
                else:
                    self.cache = {"search": {}, "embeddings": {}}
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load cache from {self.cache_file_path}. File might be corrupted: {e}")
                self.cache = {"search": {}, "embeddings": {}} # 损坏时重置
        else:
            print(f"No cache file found at {self.cache_file_path}. Starting with empty cache.")
            self.cache = {"search": {}, "embeddings": {}}

    def _save_cache(self):
        """将当前内存中的缓存持久化到磁盘文件。"""
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        try:
            with open(self.cache_file_path, "w", encoding="utf-8") as f:
                # 使用缩进提高 JSON 可读性
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Error: Could not save cache to {self.cache_file_path}: {e}")
        
    async def _get_embeddings_batch(self, batch: List[str], n_retries: int = 3):
        """
        批量获取文本的向量。优先从缓存读取，不存在的再调用 LLM API。

        Args:
            batch (List[str]): 待处理的文本列表。
            n_retries (int): API 调用失败时的重试次数。

        Returns:
            List: 与输入 batch 顺序一致的向量列表。
        """
        if not isinstance(self.cache, dict):
            self.cache = {"search": {}, "embeddings": {}}
        if "embeddings" not in self.cache or not isinstance(self.cache["embeddings"], dict):
            self.cache["embeddings"] = {}

        # 预分配结果列表，确保顺序与输入一致
        results: List = [None] * len(batch)
        to_compute_indices: List[int] = [] # 记录需要调用 API 的索引
        to_compute_texts: List[str] = []   # 记录需要调用 API 的文本

        # 检查缓存
        for idx, text in enumerate(batch):
            cached = self.cache["embeddings"].get(text)
            if cached is not None:
                results[idx] = cached
            else:
                to_compute_indices.append(idx)
                to_compute_texts.append(text)

        # 如果全部命中缓存，直接返回
        if not to_compute_texts:
            return results

        # 调用 LLM API 获取缺失的向量
        response = []
        for attempt in range(n_retries):
            try:
                response = await self.llm.generate_embeddings(to_compute_texts)
                break
            except Exception as e:
                print(f"Error getting embeddings (attempt {attempt + 1}/{n_retries}): {e}")
                if attempt == n_retries - 1:
                    print(f"Failed to get embeddings after {n_retries} attempts for a batch.")
                    response = []

        if not isinstance(response, list):
            response = []

        # 将 API 返回的结果填充回 results，并更新缓存
        for offset, idx in enumerate(to_compute_indices):
            if offset < len(response):
                emb = response[offset]
                # 确保 JSON 可序列化：将 numpy 数组转换为普通 list
                if isinstance(emb, np.ndarray):
                    emb = emb.tolist()
                results[idx] = emb
                self.cache["embeddings"][batch[idx]] = emb
            else:
                # 若获取失败，填充空列表避免下游崩溃
                results[idx] = []

        # 每次更新向量后即刻持久化缓存
        self._save_cache()

        return results

    async def build_index_from_analysis_result(self, analysis_result_list: List[dict], batch_size: int = 10, n_retries: int = 3):
        """
        根据分析结果列表构建向量索引。
        将标题和内容拼接后进行向量化。
        """
        texts = [f"{item['report_title']}\n{item['report_content']}" for item in analysis_result_list]
        await self._build_index(texts, batch_size, n_retries)

    async def build_index_from_collect_data_list(self, collect_data_list: List, batch_size: int = 10, n_retries: int = 3):
        """
        根据采集到的数据列表构建向量索引。
        调用数据的 brief_str() 方法获取其文本表示。
        """
        # 假设 CollectResult 对象具有 brief_str() 方法
        texts = [item.brief_str() for item in collect_data_list]
        await self._build_index(texts, batch_size, n_retries)

    async def _build_index(self, texts: List[str], batch_size: int=32, n_retries: int=2):
        """
        统一的索引构建内部逻辑。

        Args:
            texts (List[str]): 待向量化的文本列表。
            batch_size (int): 批处理大小。
            n_retries (int): 重试次数。
        """
        self.embeddings = [] # 重建索引前清空旧数据
        # 清空搜索缓存，因为索引 ID 已经改变，旧的搜索缓存将失效
        if isinstance(self.cache, dict) and "search" in self.cache:
            self.cache["search"] = {}
        
        # 使用 tqdm 显示进度条
        for i in tqdm(range(0, len(texts), batch_size), desc="Building index"):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._get_embeddings_batch(batch, n_retries)
            self.embeddings.extend(batch_embeddings)

        # 构建完成后保存到磁盘
        self._save_index()

    def _save_index(self):
        """将向量数据保存为 numpy 压缩文件 (.npz)"""
        if not self.embeddings:
            print("Warning: No embeddings to save. Index is empty.")
            return

        os.makedirs(os.path.dirname(self.save_file_path), exist_ok=True)
        try:
            np.savez(self.save_file_path, embeddings=np.array(self.embeddings))
        except IOError as e:
            print(f"Error: Could not save embeddings index to {self.save_file_path}: {e}")

    def load_index(self):
        """从磁盘加载现有的向量索引。"""
        if os.path.exists(self.save_file_path):
            try:
                embeddings_data = np.load(self.save_file_path)
                # 转换回列表，便于后续 extend 操作
                self.embeddings = embeddings_data['embeddings'].tolist()
                print(f"Successfully loaded index with {len(self.embeddings)} embeddings from {self.save_file_path}.")
            except Exception as e:
                print(f"Warning: Could not load index from {self.save_file_path}. File might be corrupted or in wrong format: {e}")
                self.embeddings = [] # 失败时重置
        else:
            print(f"No index file found at {self.save_file_path}. Starting with empty index.")

    async def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        在索引中搜索最相似的内容。

        Args:
            query (str): 查询文本。
            top_k (int): 返回最相关的结果数量。

        Returns:
            List[dict]: 包含 id (索引位置) 和 score (相似度分数) 的列表。
        """
        if self.embeddings is None:
            print("Warning: Embeddings index is empty. Cannot perform search.")
            return []

        # 确保缓存结构正确
        if not isinstance(self.cache, dict):
            self.cache = {"search": {}, "embeddings": {}}
        if "search" not in self.cache or not isinstance(self.cache["search"], dict):
            # 兼容性迁移：如果缓存是扁平的，则进行迁移
            if isinstance(self.cache, dict):
                flat = {k: v for k, v in self.cache.items() if k not in ("search", "embeddings")}
                self.cache = {"search": flat, "embeddings": self.cache.get("embeddings", {})}
            else:
                self.cache = {"search": {}, "embeddings": {}}

        # 优先返回搜索结果缓存
        if query in self.cache["search"]:
            return self.cache["search"][query]

        try:
            # 获取查询文本的向量（复用批处理方法以利用缓存）
            query_embedding_list = await self._get_embeddings_batch([query])
            if not query_embedding_list or not query_embedding_list[0]:
                return []
            query_embedding = np.array(query_embedding_list[0])
        except Exception as e:
            print(f"Error: Could not get embedding for query '{query}': {e}")
            return []

        # 确保索引向量已转换为 numpy 数组以进行高效的点积运算
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.array(self.embeddings)

        # 计算点积（假设向量已归一化，点积即余弦相似度）
        distances = np.dot(self.embeddings, query_embedding)
        
        if distances.size == 0:
            print("Warning: No distances computed. Embeddings array might be empty.")
            return []

        # 获取分数最高的 top_k 个索引（按分数降序排列）
        top_k_indices = np.argsort(distances)[::-1][:top_k]

        # 格式化结果并缓存
        results = [{'id': int(i), 'score': float(distances[i])} for i in top_k_indices]
        self.cache["search"][query] = results
        self._save_cache()

        return results
