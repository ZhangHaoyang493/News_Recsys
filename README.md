# News_Recsys (基于 MIND 数据集的推荐系统)

本项目是一个基于深度学习的新闻推荐系统实现，目前主要针对 **MIND (Microsoft News Dataset)** 数据集进行实验。项目旨在构建一个完整的推荐链路，涵盖了从海量新闻中筛选候选集的**召回层 (Recall Layer)**，以及对候选集进行精细打分的**排序层 (Ranking Layer)**。此外，本项目后续计划探索并实现基于大模型的**端到端生成式推荐 (End-to-End Generative Recommendation)**。

## 1. 数据集 (Dataset)

本项目使用 **[MIND (Microsoft News Dataset)](https://learn.microsoft.com/zh-cn/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets)** 进行训练和评估。MIND 是一个用于新闻推荐研究的大规模数据集。

*   **来源**: 从 Microsoft News (MSN News) 的匿名用户行为日志中收集。
*   **内容**: 包含约 16 万篇英语新闻文章和超过 1500 万条印象日志（impression logs）。
*   **版本**: 本项目目前采用 **MIND-small** 版本。
    *   **MIND-large**: 完整数据集，包含 100 万用户的行为。
    *   **MIND-small**: 用于快速开发和测试的小型子集，包含 5 万用户。
*   **数据格式**:
    *   `news.tsv`: 新闻的详细信息（ID, 类别, 子类别, 标题, 摘要, 实体等）。
    *   `behaviors.tsv`: 用户的点击历史和当前印象日志（Impression ID, User ID, Time, History, Impressions）。

> **⚠️ 注意与说明**：
> 虽然本项目提供了基于双塔（DSSM）的召回模型实现，但 **MIND 数据集本质上是一个排序（Ranking/Re-ranking）任务数据集**。
> - 在 MIND 的 `behaviors.tsv` 中，每条样本已经给定了一个具体的**候选列表**（Impressions），其中包含了正样本（点击）和负样本（未点击）。
> - 任务目标是对这个给定的列表进行打分排序，而非从全库所有新闻中检索。
> - 因此，**本项目主要关注排序层（Deep, DCN, Wide&Deep 等）的研究与优化**。召回层模型仅作为完整推荐链路的补充实现或用于特定的全库检索实验，不作为核心评估对象。

### 数据准备
请从 [MIND 官方网站](https://msnews.github.io/) 下载 **MIND-small** 数据集，并解压到 `Data/MIND/` 目录下。

推荐的文件目录结构如下：

```text
Data/MIND
├── MINDsmall_dev
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   ├── news.tsv
│   └── relation_embedding.vec
├── MINDsmall_train
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   ├── news.tsv
│   └── relation_embedding.vec
```

```bash
mkdir -p Data/MIND/train Data/MIND/val
# 将下载的 train 和 dev 数据分别解压到对应文件夹
```

## 2. 环境要求 (Requirements)

*   Python 3.8.5+

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 3. 开发与逻辑文档

本项目包含完整的推荐系统开发文档，涵盖了从特征工程、模型定义到评估指标的详细说明，方便进行二次开发和逻辑查阅。

### 3.1 特征工程 (Feature Engineering)
*   **[Feature Extractor 开发逻辑详解](documents/feature_extractor_logic.md)**: 
    *   详细解析了 `src/dataset/FeaturesGenerator` 模块的架构。
    *   说明了如何通过 `feature_extractor_*.py` 添加新特征。
    *   解释了 ID Mapping、缓存机制以及 `config_fg.yaml` 配置文件的各项参数。

### 3.2 模型开发 (Model Development)
*   **[BaseModel 逻辑梳理与二次开发指南](documents/base_model_logic.md)**:
    *   核心文档。梳理了基类 `BaseModel` 的配置读取、Embedding 层自动构建以及输入维度计算逻辑。
    *   指导如何通过修改配置文件快速调整模型结构。

### 3.3 评估指标 (Evaluation Metrics)
*   **[Sort 模型评估指标与逻辑详解](documents/sort_model_metrics.md)**:
    *   针对排序模型 (`BaseModelSort`)，详细说明了 AUC, GAUC, LogLoss, NDCG@K, MRR@K 等指标的定义与计算实现。
*   **[Recall 模型评估指标与逻辑详解](documents/recall_model_metrics.md)**:
    *   针对召回模型 (`BaseModelRecall`)，解析了双塔模型评估流程（全量物品索引 + Top-K 搜索）以及 HR@K, NDCG@K 等召回指标。

### 3.4 脚本工具 (Scripts Usage)
*   **[脚本工具完整指南](documents/scripts_usage.md)**:
    *   涵盖 `src/scripts/` 目录下所有工具脚本的使用说明。
    *   包括 **Pretrain Embedding Table** (预训练权重转换)、**Log Analysis** (日志分析与最佳 Epoch 提取)、**Visualize User History** (用户行为可视化) 以及 **Visualize Recall Result** (召回结果可视化)。

---

## 4. 运行流程 (Workflow)

本项目使用 `Makefile` 统筹数据处理与模型训练流程。

### 4.1 数据预处理 (Preprocessing)
解析原始数据，生成中间格式：
```bash
make preprocess
```
配置文件：`src/dataset/FeaturesGenerator/config_fg.yaml`

### 4.2 特征提取 (Feature Extraction)
基于预处理数据提取模型特征：
```bash
make fe
```
配置文件：`src/dataset/FeaturesGenerator/config_fg.yaml`

### 4.3 txt 特征转 mmap (Feature TXT to MMAP)
将 `src/tmp/extractored_feature` 下的 `train_features.txt`、`dev_features.txt`、`item_features.txt` 转为 mmap 友好的目录格式：
```bash
make fe_npz
```
默认会生成：
- `src/tmp/extractored_feature/train_features_mmap/`
- `src/tmp/extractored_feature/dev_features_mmap/`
- `src/tmp/extractored_feature/item_features_mmap/`

每个目录内包含：
- `manifest.json`
- `part_000001/`
- `part_000001/*.npy`（按列存储的特征与 `label.npy`）

> 说明：`make fe_npz` 命令名保留历史命名，但当前实际生成的是 mmap 目录格式（不是单文件 `.npz`）。

### 4.4 模型训练 (Training)
指定模型名称进行训练。`model` 参数对应 `src/model/` (retrieval/sort) 下的模型文件夹名称。

示例（训练 deep 模型）：
```bash
make train stage=sort model=deep
```
配置文件：`src/model/cascade_recommendation/sort/deep/deep_conf.yaml`

stage 参数可选 `retrieval` 或 `sort`，分别对应召回层和排序层的训练。

#### 4.4.1 训练数据读取模式切换（txt / mmap）
通过配置项 `dataset.load_npz` 控制训练读取方式：
- `false`：使用 txt 读取（`DataReader`），读取 `train_features.txt/dev_features.txt/item_features.txt`
- `true`：使用 mmap 读取（`DataReaderMmap`），读取 `train_features_mmap/dev_features_mmap/item_features_mmap`

配置位置示例：
- `src/model/cascade_recommendation/sort/base_sort_conf.yaml`

推荐流程：
1. `make preprocess`
2. `make fe`
3. 若需要 mmap 训练，再执行 `make fe_npz`
4. 设置 `dataset.load_npz=true`
5. `make train stage=sort model=deep`

如果你希望回退到 txt 训练：
1. 设置 `dataset.load_npz=false`
2. 直接运行 `make train stage=sort model=deep`

> 当前代码库已同时支持 txt 训练与 mmap 训练。  
> 在样本规模较大、I/O 成为瓶颈时，优先尝试 mmap 训练链路。

#### 4.4.2 为什么需要 mmap 训练
在大规模样本下，txt 训练链路常见瓶颈是“数据读取与解析”而不是“模型前向/反向”本身。  
引入 mmap 训练的主要原因：

- 减少运行时解析开销：txt 训练需要频繁执行字符串切分、`str->int` 转换、数组字段解析与补齐；mmap 方案将这些前置到离线转换阶段。
- 降低 Python 层 CPU 压力：训练过程中直接读取二进制数组，减少 Python 文本处理逻辑。
- 改善随机访问效率：mmap 基于按需分页读取，通常比反复解析文本更适合训练阶段的高频访问。
- 提升训练稳定性：数据格式固定后，线上训练路径更简单，异常点更少，便于排查性能问题。

适用建议：
- 当你观察到 GPU 利用率低、DataLoader 耗时高、训练长时间卡在数据准备阶段时，优先切换到 mmap 训练。
- 如果当前实验规模较小，或只做功能验证，txt 流程依然可用且更直观。

### 4.5 日志分析 (Log Analysis)
自动寻找并分析指定模型最近一次实验的验证集日志，并打印出效果最好的那一次结果。
```bash
make log model=deep
```

### 4.6 辅助工具 (Utility)
*   **可视化用户历史**: `make visualize_history`
*   **召回结果可视化**: `make vis_recall` (需指定 path 参数)
*   **清理临时文件**: `make clean`


## 5. 结果 (Results)

### 排序模型 (Ranking Models)

#### 在Deep模型上的特征工程效果对比：

| Model | AUC | LogLoss | GAUC | MRR | NDCG@5 | HR@5 | NDCG@10 | HR@10 | Config Path | 备注 |
|-------|-----|---------|------|-----|--------|------|---------|-------|-------------|------|
| Deep  | 0.6344 | 0.1665 | 0.6023 | 0.3041 | 0.2533 | 0.4718 | 0.3135 | 0.6635 | [deep_conf.yaml](documents/cfg_record/deep_cfg_1.yaml) |基础的Deep模型|
| Deep-v1 | 0.6526 | 0.1692 | 0.6125 | 0.3223 | 0.2709 | 0.4973 | 0.3299 | 0.6817 | [deep_conf.yaml](documents/cfg_record/deep_cfg_2.yaml) | 引入序列特征（用户历史item id、用户历史item的category以及用户历史item的subcategory） |
| Deep-v2 | 0.6754 | 0.1632 | 0.6350 | 0.3417 | 0.2896 | 0.5306 | 0.3480 | 0.7076 | [deep_conf.yaml](documents/cfg_record/deep_cfg_3.yaml) | 在 Deep-v1 的基础上增加了数据集中预训练的title的embedding特征 |

#### 基于Deep-v1的特征配置在不同模型上的效果对比：
| Model | AUC | LogLoss | GAUC | MRR | NDCG@5 | HR@5 | NDCG@10 | HR@10 | Config Path | 备注 |
|-------|-----|---------|------|-----|--------|------|---------|-------|-------------|------|
| Deep | 0.6526 | 0.1692 | 0.6125 | 0.3223 | 0.2709 | 0.4973 | 0.3299 | 0.6817 | [deep_conf.yaml](documents/cfg_record/deep_cfg_2.yaml) | 引入序列特征（用户历史item id、用户历史item的category以及用户历史item的subcategory） |
| DCN | 0.6603 | 0.1661 | 0.6164 | 0.3315 | 0.2755 | 0.5041 | 0.3357 | 0.6922 | - |
| WideDeep | 0.6628 | 0.1803 | 0.6305 | 0.3298 | 0.2813 | 0.5232 | 0.3408 | 0.7063 | - |



## 数据集迁移说明 (Why MIND?)
本项目从 MovieLens 迁移至 MIND 数据集，主要基于以下原因：
*   **负样本选择 (Negative Sampling)**: MovieLens 数据集主要由用户评分组成，缺乏明确的曝光未点击数据，导致难以进行科学的负样本采样。
*   **模型适用性**: 传统的级联结构推荐模型更适合处理隐式反馈（点击/未点击）。MIND 数据集提供了详细的 Impression 日志，天然包含正负样本，更适合训练召回和排序模型。
