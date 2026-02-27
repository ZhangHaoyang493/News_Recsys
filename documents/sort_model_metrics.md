# Sort 模型评估指标与逻辑详解

本文档详细介绍了 `BaseModelSort` (`src/model/BaseModel/base_model_sort.py`) 中用于评估排序模型（Sorting/Ranking）性能的指标计算方法和整体评估流程。

## 1. 评估流程概览

`BaseModelSort` 的评估逻辑主要在 `on_validation_epoch_end` 钩子函数中执行，该函数会在每个验证 Epoch 结束时自动触发。

**整体流程如下：**

1.  **数据收集 (`validation_step`)**: 
    *   在每个 Validation Step 中，模型调用 `inference(batch)` 得到预测分数 (`scores`)。
    *   将 `(user_id, score, label)` 三元组收集到 `self.user_scores_dict` 字典中，按 `user_id` 进行分组汇总。这意味着我们会收集每个用户对应的所有候选物品的预测分和真实标签。

2.  **并行计算 (`_compute_wrapper`)**:
    *   利用 `ProcessPoolExecutor` 多进程并行处理每个用户的指标计算，以加速大规模验证集的评估。
    *   对每个用户，调用 `_compute_user_metrics` 函数计算该用户的 AUC, NDCG, HR, MRR 等指标。

3.  **冷热启动区分**:
    *   评估时会加载训练集用户列表 (`train_user_ids.json`)。
    *   **Warm Start**: 用户出现在训练集中。
    *   **Cold Start**: 用户未出现在训练集中。
    *   最终结果会分别报告 `Overall`, `Warm_Start`, `Cold_Start` 三组指标，帮助分析模型的泛化能力。

---

## 2. 评估指标释义

模型主要关注以下几类指标，其中 Top-K 指标默认 K=10。

### 2.1 分类/回归指标 (Classification/Regression Metrics)

这些指标衡量模型预测分数与真实标签的整体吻合程度，不考虑排序位置。

*   **AUC (Area Under ROC Curve)**:
    *   **定义**: ROC 曲线下的面积。衡量模型区分正例和负例的能力。
    *   **计算方式**: 使用 `sklearn.metrics.roc_auc_score`。
    *   **GAUC (Group AUC)**: 所有用户 AUC 的平均值。更能反映个性化排序的好坏（消除了用户活跃度差异带来的偏差）。
    *   **Overall AUC**: 将所有样本混合在一起计算的 AUC。

*   **LogLoss (Logarithmic Loss)**:
    *   **定义**: 对数损失，衡量由于错误分类导致的惩罚。
    *   **计算方式**: $-\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$。

### 2.2 排序指标 (Ranking Metrics) @ K=10

这些指标关注模型是否能将正例排在推荐列表的前面。对于每个用户，我们会根据预测分数对候选物品进行降序排列，取前 K 个。

*   **HR@K (Hit Rate / 命中率)**:
    *   **定义**: 前 K 个推荐结果中是否包含至少一个正例（点击/交互项目）。
    *   **计算**: 如果 Top-K 中有正例，则为 1，否则为 0。最终对所有用户求平均。
    *   **意义**: 衡量推荐系统是否“蒙对”了用户的兴趣。

*   **MRR@K (Mean Reciprocal Rank / 平均倒数排名)**:
    *   **定义**: 第一个正例出现位置的倒数。
    *   **计算**: 如果第一个正例排在第 $r$ 位，则分数为 $1/r$。如果没有正例，则为 0。
    *   **意义**: 衡量用户看到第一个感兴趣物品的快慢。排在第 1 位得分 1，排在第 2 位得分 0.5。

*   **NDCG@K (Normalized Discounted Cumulative Gain / 归一化折损累计增益)**:
    *   **定义**: 综合考虑了正例的数量和它们的位置。
    *   **DCG (Discounted Cumulative Gain)**: $\sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$，其中 $rel_i$ 是位置 $i$ 的相关性（通常点击为1，未点击为0）。位置越靠后，权重 $\frac{1}{\log_2(i+1)}$ 越小。
    *   **IDCG (Ideal DCG)**: 理想排序下的 DCG 值（正例全部排在最前面）。
    *   **NDCG**: $DCG / IDCG$。
    *   **意义**: 排序质量的黄金指标。不仅要求命中，还要求命中的物品排得越靠前越好。

---

## 3. 代码逻辑片段 (`_compute_user_metrics`)

```python
def _compute_user_metrics(uid, items, k, is_cold):
    # items: list of (score, label)
    
    # 1. 排序
    items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    top_k = items_sorted[:k]
    
    # 2. 计算 HR
    # 只要 Top K 中有一个 label为1 的，就是 Hit
    if any(x[1] == 1 for x in top_k):
        res['hr'] = 1.0
        
    # 3. 计算 MRR
    # 找到 Top K 中 label 为 1 的那些索引
    hits = np.where(top_k_labels == 1)[0]
    if len(hits) > 0:
        res['mrr'] = 1.0 / (hits[0] + 1) # 取第一个命中位置的倒数
        
    # 4. 计算 NDCG
    # ... (DCG / IDCG)
```

## 4. 日志输出示例

验证结束后，日志会对所有指标进行详细输出：

```text
==================== Epoch 0 Validation Results ====================
Overall:
  AUC:      0.6521
  LogLoss:  0.4123
  GAUC:     0.6105
  NDCG@10:  0.3214
  HR@10:    0.5421
  MRR@10:   0.2891
Warm Start Users (8542):
  AUC:      0.6612
  ...
Cold Start Users (1205):
  AUC:      0.5921
  ...
====================================================================
```
