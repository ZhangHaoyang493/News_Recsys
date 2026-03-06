# BaseModel 逻辑梳理与二次开发指南

本文档旨在梳理 `src/model/BaseModel` 中 `BaseModel` 及其子类 (`BaseModelSort`) 的核心逻辑，重点关注**配置读取**、**特征构造**以及**Embedding 层构建**流程，由于推荐系统的核心往往在于特征处理和 Embedding lookup，理解这部分逻辑对于后续进行模型结构的修改（例如修改塔结构、增加特征交互层）至关重要。

## 1. 核心类结构

*   `BaseModel` (`base_model.py`):
    *   作为所有推荐模型的基类（LightningModule）。
    *   负责加载配置文件 (`metrics`, `features`, `embeddings` 等)。
    *   负责根据配置自动构建 Embedding 层 (`nn.Embedding`)。
    *   负责计算输入到模型 User Tower 和 Item Tower 的维度 (`user_input_dim`, `item_input_dim`)。
*   `BaseModelSort` (`base_model_sort.py`):
    *   继承自 `BaseModel`。
    *   专注于排序（Ranking/CTR）任务。
    *   实现了通用的 `validation_step` 和 `on_validation_epoch_end`，用于计算 AUC, MRR, NDCG, HR 等指标。详细评估逻辑参见 [Sort 模型评估指标与逻辑详解](sort_model_metrics.md)。
    *   实现了冷启动用户与非冷启动用户的分别评估。

*   `BaseModelRecall` (`base_model_recall.py`):
    *   继承自 `BaseModel`。
    *   专注于召回（Retrieval/Matching）任务。
    *   实现了双流（Two-Tower）评估流程：先对所有物品向量进行编码并索引，再对用户向量进行 Top-K 近邻搜索。
    *   详细评估逻辑参见 [Recall 模型评估指标与逻辑详解](recall_model_metrics.md)。

---

## 2. 特征筛选逻辑 (Feature Selection)

模型最终使用的特征并非配置文件中列出的所有特征，而是由 `features.item_feature_names` 和 `features.user_feature_names` 显式指定的子集。

*   **`features.sparse_feature_names`**: 仅作为“定义”存在，列出了数据集中所有可用的离散特征。
*   **`features.dense_feature_names`**: 仅作为“定义”存在，列出了数据集中所有可用的连续特征。
*   **`features.array_feature_names`**: 仅作为“定义”存在，列出了数据集中所有可用的序列特征。
*   **`features.item_feature_names`**: **实际输入 Item Tower 的特征**。只有列在这里的特征，模型才会真正处理。
*   **`features.user_feature_names`**: **实际输入 User Tower 的特征**。只有列在这里的特征，模型才会真正处理。

**示例**：
即使 `sparse_feature_names` 中包含了 `city`, `device`，但如果 `user_feature_names` 里没有写它们，模型 Forward 阶段就不会读取这两个特征的 Embedding，它们实际上被忽略了。

---

## 3. 配置文件解析逻辑


`BaseModel` 在 `__init__` 中调用 `_load_config` 加载 YAML 配置，并将其解析为模型属性。

### 关键配置项与属性映射

为了方便对照开发，以下表格结合了真实配置文件 (`base_sort_conf.yaml`) 中的所有相关参数：

| Section | Parameter | Type | 示例值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`train_stage`** | | str | `sort` | 训练阶段标识，区分 `sort` (排序/精排) 或 `retrieval` (召回)。 |
| **`paths`** | `out_basedir` | str | `"src/tmp"` | 实验输出根目录，模型会自动在此目录下创建 `ckpts/`, `logs/`, `result_record/` 等。 |
| **`features`** | `sparse_feature_names` | list | `[user_id, item_id, category, ...]` | 定义所有可用的**离散特征**列表。 |
| | `dense_feature_names` | list | `[]` | 定义所有可用的**连续特征**列表（不需要 Embedding）。 |
| | `array_feature_names` | list | `[]` | 定义所有**序列特征**列表（如用户历史点击序列）。 |
| | `item_feature_names` | list | `[item_id, category, subcategory]` | **关键**：指定输入给 **Item Tower** 的特征子集。 |
| | `user_feature_names` | list | `[user_id, user_click_category]` | **关键**：指定输入给 **User Tower** 的特征子集。 |
| | `array_max_length` | dict | `{}` | 定义序列特征的最大长度，用于 Padding/Truncating。 |
| **`embeddings`** | `embedding_table_size` | dict | `{user_id: 94058, item_id: 65239, ...}` | **词表大小** (Vocabulary Size)。必须覆盖所有用到的 sparse features。 |
| | `embedding_tables` | dict | (如下详述) | 定义 Embedding Table 的分组与共享机制。 |
| **`dataset`** | `batch_size` | int | `512` | 训练批次大小。 |
| | `num_workers` | int | `4` | DataLoader 的工作线程数。 |
| | `pin_memory` | bool | `true` | 是否将 Tensor 锁页内存，加速 GPU 传输。 |
| | `only_test_warm_up_user` | bool | `true` | 测试时是否只包含预热过的用户（取决于具体 Dataset 实现）。 |
| **`train_hparams`** | `val_freq` | float | `0.5` | 验证频率 (0.5 epoch)。 |
| | `max_epoch` | int | `20` | 最大训练轮数。 |
| | `lr` | float | `1.0e-3` | 初始学习率。 |
| | `min_lr` | float | `5.0e-6` | 最小学习率 (配合 Scheduler)。 |
| | `lr_milestones` | list | `[40000, 200000]` | 学习率衰减的 Step 节点。 |
| | `device` | str | `"gpu"` | 使用 CPU 还是 GPU。 |
| | `gpus` | list | `[2]` | 指定使用的 GPU ID 列表。 |

---

## 3. Embedding 层构建与特征构造

这是 `BaseModel` 最核心的部分。它允许灵活地配置哪些特征共享 Embedding，哪些特征独立 Embedding。

### 3.1 真实配置解析 (`embeddings`)

根据 `src/model/cascade_recommendation/sort/base_sort_conf.yaml` 的内容，Embedding 配置结构如下：

```yaml
embeddings:
  embedding_table_size:
    user_id: 94058       # 用户ID总数
    item_id: 65239       # 物品ID总数
    category: 19
    subcategory: 270
    user_click_category: 18

  embedding_tables:
    base_embedding_table: # 表名，代码中对应 self.embedding_tables['base_embedding_table']
      embedding_dims:
        user_id: 32           # 产生的向量维度: [94058, 32]
        item_id: 32           # [65239, 32]
        category: 16          # [19, 16]
        subcategory: 16       # [270, 16]
        user_click_category: 16 # [18, 16]
      share_emb_table_features: {} # 如果有共享Embedding的需求，在此定义映射关系
      # 如果有预训练的Embedding，在此配置路径。支持 .pt, .pth, .ckpt (torch.save) 格式
      # 文件内容可以是 torch.Tensor 或 nn.Embedding
      pretrain_embedding:    
        # title_entity_embedding: "pretrain_embedding/title_entity_embedding.ckpt"
```

### 3.2 构建流程与维度计算

1.  **构建 Embedding Table (`_build_all_embedding_tables`)**：
    *   **关键机制**：模型构建的 Embedding 表完全取决于 `embeddings.embedding_tables.<table_name>.embedding_dims` 下定义的字段键（Keys）。只有出现在 `embedding_dims` 字典中的特征名，`BaseModel` 才会为其初始化 `nn.Embedding` 层。
    *   模型会读取 `base_embedding_table` 下的 `embedding_dims`。
    *   遍历每个特征，使用 `embedding_table_size` 中的大小和配置的维度创建 `nn.Embedding`。
    *   如果配置了 `pretrain_embedding`，则会尝试使用 `torch.load` 加载指定路径的权重文件。支持的文件内容格式为 `torch.Tensor` 或 `nn.Embedding` 对象。如果加载成功，将直接作为对应特征的 Embedding 层，并自动 freeze（`requires_grad=False`）。此时配置的维度仅作记录，实际维度以加载的权重为准。
    *   如果配置了 `share_emb_table_features` (例如 `target_iid: history_iid`)，则 `target_iid` 会复用 `history_iid` 的 Embedding Matrix，而不会创建新的 Embedding Layer。

2.  **自动计算输入维度 (`_calculate_input_dim`)**：
    
    模型会根据 `features.item_feature_names` 和 `features.user_feature_names` 自动计算全连接层的输入维度，无需手动硬编码。
    
    *   **Item Input Dim 计算**:
        *   `item_feature_names` = `[item_id, category, subcategory]`
        *   `item_id` (32) + `category` (16) + `subcategory` (16) = **64**
    
    *   **User Input Dim 计算**:
        *   `user_feature_names` = `[user_id, user_click_category]`
        *   `user_id` (32) + `user_click_category` (16) = **48**
    
    > **注意**：这意味着你的网络结构 (如 MLP) 的输入层大小将动态变为 64 和 48。

---

**二次开发提示**：
如果你在子类中定义了新的网络结构 (如 MLP)，其 `input_dim` 应当直接使用 `self.user_input_dim` 或 `self.item_input_dim`，而不需要手动硬编码数字。

---

## 4. 验证与指标计算 (`BaseModelSort`)

`BaseModelSort` 封装了标准的推荐系统评估流程。

### 4.1 推理 (`validation_step`)
*   调用子类实现的 `inference(batch)` 获取预测分数。
*   将 `user_id`, `score`, `label` 存储到 `self.user_scores_dict` 中，用于通过 User ID 进行 Group By 的指标计算。

### 4.2 指标计算 (`on_validation_epoch_end`)
1.  **并行计算**：使用 `ProcessPoolExecutor` 对每个用户并行计算指标。
2.  **冷启动区分**：
    *   模型加载 `train_user_ids.json` (如果存在)。
    *   验证集中的用户如果不在训练集中，被标记为 **Cold Start User**。
    *   最终会分别输出 `All`, `Warm` (Non-cold), `Cold` 三组指标 (AUC, MRR, NDCG@10, HR@10)。

---

## 5. 二次开发示例

假设你要添加一个新的特征 `user_age_level`：

1.  **配置文件修改**：
    *   在 `features.user_feature_names` 中添加 `user_age_level`。
    *   在 `embeddings.embedding_table_size` 中添加 `user_age_level: 10` (假设分了10个年龄段)。
    *   在 `embeddings.embedding_tables.base_embedding_table.embedding_dims` 中添加 `user_age_level: 32` (维度)。

2.  **模型代码**：
    *   不需要修改 `BaseModel` 代码。
    *   `self.user_input_dim` 会自动增加 32。
    *   在你的模型 `forward` 函数中，需要确保从 batch 中取出了 `user_age_level` 并传入 Embedding 层 lookup。
