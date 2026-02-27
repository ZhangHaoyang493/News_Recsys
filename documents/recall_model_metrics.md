# Recall 模型评估指标与逻辑详解

本文档详细介绍了 `BaseModelRecall` (`src/model/BaseModel/base_model_recall.py`) 中用于评估召回模型（Retrieval/Matching）性能的逻辑、评估流程及指标定义。

## 1. 核心职责

`BaseModelRecall` 继承自 `BaseModel`，专门针对召回阶段（Retrieval Stage）设计。
与排序模型（关注单个 (User, Item) 对的打分）不同，召回模型的核心任务是：**从海量物品库中快速检索出用户可能感兴趣的 Top-K 物品**。

因此，其评估流程包含两个关键步骤：
1.  **全量物品向量化**：计算所有候选物品的 Embedding 并建立索引。
2.  **Top-K 近邻搜索**：计算用户 Embedding，并在索引中搜索最相似的 K 个物品。

---

## 2. 评估流程详解

评估逻辑主要分布在 `validation_step` 和辅助函数中。PyTorch Lightning 的 `validation_step` 会被调用多次，这里通过 `dataloader_idx` 区分不同类型的数据流。

### 2.1 阶段一：建立物品索引 (`dataloader_idx == 0`)
*   **输入**：包含所有候选物品的 DataLoader。
*   **操作**：
    *   调用 `inference_item(batch)` 计算每个 Item 的向量。
    *   将向量 (`item_vector`) 和 ID (`item_id`) 暂存到内存列表 `self.item_embeddings` 和 `self.item_ids` 中。
*   **索引构建**：
    *   当检测到进入阶段二（即收到 `dataloader_idx=1` 的第一个 batch）或者在阶段一结束时，触发 `_build_item_index`。
    *   使用 `TopKSearcher`（通常基于 FAISS 或矩阵乘法）将所有物品向量加载到索引中，准备进行即时搜索。

### 2.2 阶段二：用户召回与打分 (`dataloader_idx == 1`)
*   **输入**：包含用户验证数据的 DataLoader (通常是 Impression 粒度)。
*   **去重优化**：
    *   由于验证集可能包含同一用户对不同物品的多次交互记录（展开后的 pointwise 数据），直接对所有样本计算用户向量极其低效。
    *   代码利用 `torch.unique_consecutive` 对 Batch 内的 `impression_id` 进行去重，只对唯一的 Impression 计算一次用户向量 (`inference_user`)。
*   **Top-K 搜索**：
    *   调用 `self.topk_searcher.search(user_emb_unique)`，检索出与当前用户最相似的 Top-K 物品列表。
*   **结果收集**：
    *   **过滤历史**：从召回结果中剔除用户历史点击过的物品（避免推荐重复内容）。
    *   **记录结果**：将 `(uid, recall_list, target_items)` 存入 `self.recall_res`。其中 `target_items` 是验证集中用户真实点击的物品（Ground Truth）。

---

## 3. 评估指标 (Metrics)

召回模型的评估通常基于 Top-K (默认 K=50, 100 等) 列表。

### 3.1 核心指标

*   **Hit Rate (HR@K)**:
    *   **定义**: 召回的 Top-K 列表中是否包含了用户真实点击的物品。
    *   **计算**: 只要有一个 Target Item 出现在 Recall List 中，HR = 1，否则为 0。
    *   **意义**: 衡量召回的“覆盖率”，即漏斗的口开得够不够大，是否漏掉了用户喜欢的物品。

*   **NDCG@K (Normalized Discounted Cumulative Gain)**:
    *   **定义**: 衡量召回列表的排序质量。即使是召回阶段，我们通常也希望最相关的物品排在前面。
    *   **计算**: 考虑命中物品在召回列表中的排名位置。
    *   **意义**: 召回截断（Truncation）的依据。如果 NDCG 高，说明头部物品质量好，后续排序阶段可以只处理更少的物品。

*   **MRR@K (Mean Reciprocal Rank)**:
    *   **定义**: 第一个正确命中的物品排名的倒数。
    *   **意义**: 衡量首个命中项出现的位置。

### 3.2 结果导出

*   `write_recall_res`: 代码并未将所有指标仅仅打印在日志中，而是支持将详细的召回结果写入文件 (`recall_results_epoch_X.txt`)。
    *   文件内容通常包含：`ImpressionID, LabedItems, RecallItems` 等信息。
    *   这对于后续做**Case Study**或将召回结果传给**精排模型**（作为候选集 Generating Candidates）非常重要。

---

## 4. 代码结构映射

| 方法名 | 作用 | 关键逻辑 |
| :--- | :--- | :--- |
| `_build_item_index` | 构建索引 | `self.topk_searcher.update_embedding(...)` |
| `validation_step` | 双流处理 | `if idx==0: inference_item` <br> `if idx==1: inference_user -> search -> filter history` |
| `_compute_recall_metrics` | 指标计算 | 计算单条记录的 HR, NDCG, MRR |

---

## 5. 二次开发提示

1.  **修改 Top-K 规模**:
    *   在 `__init__` 中，`TopKSearcher(k=self.config.k + 100)`。修改配置中的 `k` 即可改变召回数量。
    *   `+100` 是为了给过滤历史记录预留 buffer。

2.  **实现 Inference 接口**:
    *   子类（如 `DSSM`, `YubiYouTube`）必须实现 `inference_item` 和 `inference_user`，分别定义双塔的 Forward 逻辑。

3.  **多路召回融合**:
    *   目前的 `BaseModelRecall` 属于单路召回（Single Retrieval）。
    *   如果需要多路召回，通常是在模型外部将不同模型的 `recall_res` 文件进行归并，或者在 `validation_step` 中集成多个 Searcher。
