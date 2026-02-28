# Scripts Usage Documentation

## `src/scripts/pretrain_embedding_table.py`

此脚本用于将预训练的向量文件（文本/TSV格式）转换为 PyTorch 的 `Embedding` 权重矩阵文件（`.pt` 或 `.npy`）。它需要结合特征提取阶段生成的 ID 映射字典使用，确保预训练向量的索引与模型输入的 ID 一致。

### 功能说明
1. 读取包含 Key-Vector 对的文本文件。
2. 读取特征提取生成的 `original_val_2_embedding_idx_dict.json` 映射文件。
3. 根据指定的 `feature_name` 查找对应的 ID 映射。
4. 创建一个形状为 `(vocab_size, emb_dim)` 的全零矩阵。
5. 遍历向量文件，将 Key 对齐到 ID，填充矩阵。
6. 保存为 PyTorch Tensor 或 Numpy Array。
7. 输出覆盖率（Hit Count）和稀疏度（Sparsity）。

### 命令行参数

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `--vec_file` | 是 | 原始预训练向量文件路径。格式：`Key <tab> Val1 <tab> Val2 ...` (无表头) |
| `--map_file` | 是 | ID 映射字典文件路径 (通常是 Feature Extractor 输出的 `original_val_2_embedding_idx_dict.json`) |
| `--output_path` | 是 | 输出文件保存路径 (建议以 `.pt` 或 `.ckpt` 结尾) |
| `--feature_name` | 是 | 映射字典中对应的特征名称 (例如 `item_title_entity_id`, `item_id`) |
| `--emb_dim` | 否 | Embedding 向量维度。如果不指定，脚本会自动读取文件第一行检测维度。 |
| `--vocab_size` | 否 | 词表大小。如果不指定，脚本会根据映射字典中的最大 ID 推断。 |

### 使用示例

假设：
- 预训练向量文件位于：`src/tmp/preprocess/entity_embedding_all.vec`
- 映射字典位于：`src/tmp/extractored_feature/original_val_2_embedding_idx_dict.json`
- 我们想处理名为 `item_title_entity_id` 的特征
- 输出到：`pretrain_embedding/title_entity_embedding.ckpt`

```bash
python src/scripts/pretrain_embedding_table.py \
    --vec_file src/tmp/preprocess/entity_embedding_all.vec \
    --map_file src/tmp/extractored_feature/original_val_2_embedding_idx_dict.json \
    --output_path pretrain_embedding/title_entity_embedding.ckpt \
    --feature_name item_title_entity_id
```

### 注意事项
1. **Sparsity (稀疏度)**: 脚本运行结束会打印 Sparsity。
   - `Sparsity = 1 - (Hit Count / Total Keys in Map)`
   * 这表示映射字典中有多少比例的 ID **没有** 在预训练向量文件中找到对应的向量。这些缺失的向量将保持为全 0 初始化。
2. **Key 匹配逻辑**: 脚本会尝试进行 string 到 int 的转换匹配，以兼容 JSON 键值必须为字符串的限制。
3. **输出格式**: 如果 `output_path` 以 `.npy` 结尾，保存为 Numpy 数组；否则保存为 PyTorch Tensor 对象。建议保存为 `.ckpt` 或 `.pt` 以便在 `base_model` 中直接加载。

---

## `src/scripts/log_analysis.py`

此脚本用于分析模型训练过程中生成的日志文件，提取验证集指标（如 AUC、NDCG、MRR），并输出最佳 Epoch 的详细性能表现。

### 功能说明
1. 解析日志文件，识别 Epoch 分隔符。
2. 提取 `Overall`、`Warm Start Users` 和 `Cold Start Users` 三个部分的指标数据。
3. 根据 **Warm Start Users 的 AUC** 指标自动挑选最佳 Epoch。
4. 以 Markdown 表格形式打印该 Epoch 下的所有指标详情。

### 命令行参数

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `log_file` | 是 | 训练日志文件的路径 (例如 `experiments/model_date/log.txt`) |

### 使用示例

```bash
python src/scripts/log_analysis.py experiments/dssm_20260228-112055/train.log
```

---

## `src/scripts/visiualize_user_history.py`

此脚本用于生成一个 HTML 报告，可视化用户的点击历史（History）和当前的曝光序列（Impressions）。通过该报告，可以直观地查看每个用户的历史兴趣偏好以及当前面对的候选新闻。

### 功能说明
1. 读取新闻数据 (`news.tsv`) 和用户行为数据 (`behaviors.tsv`)。
2. 将行为数据按时间戳排序。
3. 生成交互式 HTML 页面，包含：
   - 用户列表
   - 每次曝光的详情（时间、ID）
   - 用户历史点击新闻列表（显示标题、摘要、类别）
   - 当前曝光候选新闻列表（区分正样本和负样本）

### 命令行参数

| 参数 | 必选 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--news` | 是 | 新闻数据文件路径 (`news.tsv`) | - |
| `--behaviors` | 是 | 行为数据文件路径 (`behaviors.tsv`) | - |
| `--output` | 否 | 输出的 HTML 文件路径 | `src/tmp/user_history_viz.html` |

### 使用示例

```bash
python src/scripts/visiualize_user_history.py \
    --news Data/MIND/MINDsmall_dev/news.tsv \
    --behaviors Data/MIND/MINDsmall_dev/behaviors.tsv \
    --output user_viz.html
```

---

## `src/scripts/visualize_recall_html.py`

此脚本用于可视化召回模型的结果。它对比用户的点击历史、Ground Truth（真实点击项）以及模型召回的 Top-K 结果。

### 功能说明
1. 读取召回结果文件（格式：`imp_id \t user_id \t recall_items \t target_items \t history_items`）。
2. 读取新闻数据，提取标题、摘要、类别及实体信息。
3. 生成 HTML 报告，展示：
   - **Target (True Positive)**: 用户实际点击的新闻。
   - **Recall Results**: 模型召回的 Candidate 列表，并标注是否命中 (Hit)。
   - **User History**: 用户的近期点击历史。
   - **Entities**: 显示标题和摘要中提取的实体标签。

### 命令行参数

| 参数 | 必选 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--recall_file` / `-r` | 否 | 召回结果文件路径 (TSV) | `src/tmp/recall_result.tsv` |
| `--news_file` | 否 | 新闻信息文件路径 | `/data2/zhy/News_Recsys/src/tmp/preprocess/all_news_preprocess.csv` |
| `--output_file` | 否 | 输出 HTML 文件路径 | `recall_visualization.html` |
| `--sample_num` | 否 | 采样的用户数量（避免 HTML 过大） | 300 |

### 使用示例

```bash
python src/scripts/visualize_recall_html.py \
    -r src/tmp/recall_result.tsv \
    --news_file Data/MIND/MINDsmall_dev/news.tsv \
    --sample_num 50
```
