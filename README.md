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

---

## 4. 运行流程 (Workflow)

本项目使用 `Makefile` 统筹数据处理与模型训练流程。

### 4.1 数据预处理 (Preprocessing)
解析原始数据，生成中间格式：
```bash
make preprocess
```
配置文件：`src/dataset/FeaturesGenerator/config_fg.yaml`

### 3.2 特征提取 (Feature Extraction)
### 4.2 特征提取 (Feature Extraction)
基于预处理数据提取模型特征：
```bash
make fe
```
配置文件：`src/dataset/FeaturesGenerator/config_fg.yaml`


### 4.3 模型训练 (Training)
指定模型名称进行训练。`model` 参数对应 `src/model/` (retrieval/sort) 下的模型文件夹名称。

示例（训练 deep 模型）：
```bash
make train stage=sort model=deep
```
配置文件：`src/model/cascade_recommendation/sort/deep/deep_conf.yaml`

stage 参数可选 `retrieval` 或 `sort`，分别对应召回层和排序层的训练。

### 4.4 日志分析 (Log Analysis)
自动寻找并分析指定模型最近一次实验的验证集日志，并打印出效果最好的那一次结果。
```bash
make log model=deep
```

### 4.5 辅助工具 (Utility)
*   **可视化用户历史**: `make visualize_history`
*   **召回结果可视化**: `make vis_recall` (需指定 path 参数)
*   **清理临时文件**: `make clean`


## 5. 结果 (Results)

正在整理中...
| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-------|-----|-----|--------|---------|
| Base  | -   | -   | -      | -       |
| Ours  | -   | -   | -      | -       |


## 数据集迁移说明 (Why MIND?)
本项目从 MovieLens 迁移至 MIND 数据集，主要基于以下原因：
*   **负样本选择 (Negative Sampling)**: MovieLens 数据集主要由用户评分组成，缺乏明确的曝光未点击数据，导致难以进行科学的负样本采样。
*   **模型适用性**: 传统的级联结构推荐模型更适合处理隐式反馈（点击/未点击）。MIND 数据集提供了详细的 Impression 日志，天然包含正负样本，更适合训练召回和排序模型。
