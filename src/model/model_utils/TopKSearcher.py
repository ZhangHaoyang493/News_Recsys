import torch
import torch.nn as nn
import faiss
import numpy as np
from typing import List, Tuple

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

    def update_embedding(self, emb_layer: nn.Embedding, normalize: bool = False):
        """
        更新存储的 Embedding 表到 Faiss 索引中
        :param emb_layer: torch.nn.Embedding 层
        :param normalize: 是否对向量进行 L2 归一化 (如果为 True，内积等价于余弦相似度)
        """
        # 1. 获取权重并转换为 numpy (float32)
        # .detach() 从计算图分离，.cpu() 移至内存
        weights = emb_layer.weight.detach().cpu().numpy().astype('float32')
        
        self.dimension = weights.shape[1]
        num_embeddings = weights.shape[0]

        # 2. 如果需要余弦相似度，先进行归一化
        if normalize:
            faiss.normalize_L2(weights)

        # 3. 构建 Faiss 索引
        # IndexFlatIP (Inner Product) 适用于召回模型，计算点积
        index = faiss.IndexFlatIP(self.dimension)
        
        # 4. 如果启用 GPU 加速
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        # 5. 添加数据
        index.add(weights)
        self.index = index
        
        print(f"[TopKSearcher] Index updated. Size: {num_embeddings}, Dim: {self.dimension}")

    def search(self, query_embeddings: List[torch.Tensor], normalize: bool = False) -> Tuple[List[List[int]], List[List[float]]]:
        """
        在 Faiss 中查找最近的 K 个向量
        :param query_embeddings: List[Tensor]，每个 Tensor 代表一个 query 的 embedding
        :param normalize: 是否对 Query 进行归一化 (需与 update_embedding 保持一致)
        :return: (indices, scores)
                 indices: List[List[int]], 查找到的 Item ID
                 scores: List[List[float]], 对应的相似度分数 (内积值)
        """
        if self.index is None:
            raise ValueError("Index not initialized. Please call update_embedding first.")

        # 1. 数据预处理：List[Tensor] -> Tensor (Batch) -> Numpy
        # 假设输入的 list 中每个 tensor 是一维的 (dim,)
        if len(query_embeddings) == 0:
            return [], []

        # stack 将 [tensor(d), tensor(d)] 堆叠为 tensor(batch, d)
        query_batch = torch.stack(query_embeddings).detach().cpu().numpy().astype('float32')

        # 2. 如果 Item 侧归一化了，Query 侧通常也需要归一化
        if normalize:
            faiss.normalize_L2(query_batch)

        # 3. 执行搜索
        # D: Distances (Scores), I: Indices
        D, I = self.index.search(query_batch, self.k)

        # 4. 格式转换为 Python List
        # I 和 D 的形状都是 (batch_size, k)
        indices_list = I.tolist()
        scores_list = D.tolist()

        return indices_list, scores_list