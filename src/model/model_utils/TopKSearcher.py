import torch
import torch.nn as nn
import faiss
import numpy as np
from typing import List, Tuple
from ...Logger.logging import Logger

logger = Logger.get_logger('TopKSearcher')

class TopKSearcher:
    def __init__(self, k: int, use_gpu: bool = False):
        """
        初始化搜索器
        :param k: 每次召回最近邻的个数 (Top-K)
        :param use_gpu: 是否使用 GPU 资源进行 Faiss 搜索 (需要安装 faiss-gpu)
        """
        self.k = k
        self.index = None
        self.use_gpu = use_gpu
        self.dimension = None
        self.item_ids = None

    def update_embedding(self, embeddings: np.ndarray, ids: np.ndarray, normalize: bool = False):
        """
        更新存储的 Embedding 表到 Faiss 索引中
        :param embeddings: numpy array, shape (N, dim), 物料的 embedding
        :param ids: numpy array, shape (N, 1) 或 (N,), 物料对应的索引
        :param normalize: 是否对向量进行 L2 归一化 (如果为 True，内积等价于余弦相似度)
        """
        # 1. 转换数据类型
        embeddings = embeddings.astype('float32')
        self.item_ids = ids.flatten()
        
        self.dimension = embeddings.shape[1]
        num_embeddings = embeddings.shape[0]

        # 2. 如果需要余弦相似度，先进行归一化
        if normalize:
            faiss.normalize_L2(embeddings)

        # 3. 构建 Faiss 索引
        # IndexFlatIP (Inner Product) 适用于召回模型，计算点积
        index = faiss.IndexFlatIP(self.dimension)
        
        # 4. 如果启用 GPU 加速
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        # 5. 添加数据
        index.add(embeddings)
        self.index = index
        
        logger.info(f"[TopKSearcher] Index updated. Size: {num_embeddings}, Dim: {self.dimension}")

    def search(self, query_embeddings: torch.Tensor, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 Faiss 中查找最近的 K 个向量
        :param query_embeddings: Tensor, shape (B, dim), query 的 embedding
        :param normalize: 是否对 Query 进行归一化 (需与 update_embedding 保持一致)
        :return: (indices, scores)
                 indices: Tensor, shape (B, k), 查找到的 Item ID
                 scores: Tensor, shape (B, k), 对应的相似度分数 (内积值)
        """
        if self.index is None:
            raise ValueError("Index not initialized. Please call update_embedding first.")

        # 1. 数据预处理
        if query_embeddings.numel() == 0:
            return torch.empty((0, self.k), dtype=torch.long, device=query_embeddings.device), \
                   torch.empty((0, self.k), device=query_embeddings.device)

        device = query_embeddings.device
        # Tensor (Batch) -> Numpy
        query_batch = query_embeddings.detach().cpu().numpy().astype('float32')

        # 2. 如果 Item 侧归一化了，Query 侧通常也需要归一化
        if normalize:
            faiss.normalize_L2(query_batch)

        # 3. 执行搜索
        # D: Distances (Scores), I: Indices
        D, I = self.index.search(query_batch, self.k)

        # 4. 映射 ID 并转回 Tensor
        if self.item_ids is not None:
            # 将 Faiss 内部索引映射回原始 item_ids
            indices = self.item_ids[I]
        else:
            indices = I
            
        # 转回 Tensor
        indices_tensor = torch.from_numpy(indices).to(device)
        scores_tensor = torch.from_numpy(D).to(device)

        return indices_tensor, scores_tensor