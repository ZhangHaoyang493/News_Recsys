# News_Recsys

基于 MIND-small 数据集的新闻推荐实验框架，覆盖推荐系统中常见的两阶段链路：召回与排序。项目核心目标不是做一个完整线上服务，而是提供一套可复现、可扩展、便于做特征工程和模型对比的离线实验代码。

## 核心特性

- 支持 MIND-small 数据集的完整预处理、特征抽取、训练与评估流程
- 同时支持召回模型与排序模型的实验
- 支持 `txt` 特征训练链路与 `mmap` 高效读取链路
- 通过 YAML 配置驱动特征开关、Embedding 共享和训练参数
- 内置 warm/cold user 评估、结果落盘和实验日志记录

## 目录

- [技术栈](#技术栈)
- [项目目标](#项目目标)
- [仓库结构](#仓库结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [端到端流程](#端到端流程)
- [系统架构](#系统架构)
- [配置说明](#配置说明)
- [可用命令](#可用命令)
- [产物目录](#产物目录)
- [评估说明](#评估说明)
- [常见问题](#常见问题)
- [已知限制](#已知限制)
- [延伸阅读](#延伸阅读)

## 技术栈

- **Language**: Python 3.8.5+
- **Deep Learning**: PyTorch, Lightning
- **Data Processing**: pandas, NumPy
- **Configuration**: OmegaConf, YAML
- **Recall Search**: FAISS
- **Dataset**: Microsoft MIND-small
- **Experiment Management**: TensorBoard logger + local experiment directories
- **Deployment Style**: local offline training only; no production deployment config is included in this repository

## 项目目标

这个仓库更适合被理解为“推荐算法实验平台”，而不是“可直接上线的推荐服务”。

它主要解决三件事：

1. 把 MIND 原始数据统一加工成模型可消费的特征样本。
2. 基于统一的底座快速切换不同模型与特征组合。
3. 在排序阶段做离线评估与特征/模型对比，并保留召回底座能力以便后续扩展。

从数据集本身来看，MIND 更接近重排任务：每条样本已经自带 impression 候选列表，因此当前仓库的主训练链路是排序模型研究。

## 仓库结构

```text
.
├── Data/
│   ├── MIND/                         # MIND 原始数据目录
│   └── MovieLens_1M_data/            # 历史遗留数据目录
├── documents/                        # 设计与逻辑文档
├── experiments/                      # Lightning 训练产物
├── pretrain_embedding/               # 预训练 embedding 权重
├── src/
│   ├── dataset/
│   │   ├── DataReader/               # txt / mmap 数据读取
│   │   └── FeaturesGenerator/        # 预处理与特征工程
│   ├── Logger/                       # 日志封装
│   ├── model/
│   │   ├── BaseModel/                # 通用模型底座
│   │   ├── sort_models/
│   │   │   ├── classic_sort_models/  # 经典排序模型
│   │   │   └── generative_sort_models/
│   │   └── model_utils/              # 工具模块
│   └── scripts/                      # 辅助脚本
├── Makefile                          # 常用命令入口
├── requirements.txt                  # Python 依赖
└── README.md
```

## 数据集

项目当前围绕 **MIND-small** 数据集构建。你需要从 MIND 官方渠道下载数据，并放到如下目录：

```text
Data/MIND/
├── MINDsmall_train/
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   ├── news.tsv
│   └── relation_embedding.vec
└── MINDsmall_dev/
    ├── behaviors.tsv
    ├── entity_embedding.vec
    ├── news.tsv
    └── relation_embedding.vec
```

原始文件含义：

- `news.tsv`: 新闻内容与元信息，包括新闻 ID、类别、子类别、标题、摘要和实体信息
- `behaviors.tsv`: 用户行为日志，包括用户 ID、时间戳、点击历史和 impression 候选列表
- `entity_embedding.vec` / `relation_embedding.vec`: 实体与关系向量

## 环境要求

开始前请确保本机具备以下环境：

- Python 3.8.5 或更高版本
- 可用的 CUDA/PyTorch 运行环境，如果你希望使用 GPU 训练
- `pip`
- `make`
- 至少一个可用 GPU，当前训练脚本默认使用 GPU

推荐额外准备：

- `venv` 或 `conda` 用于隔离 Python 环境
- TensorBoard，用于查看训练日志

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd News_Recsys
```

### 2. 创建 Python 虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

如果你使用 Conda，也可以：

```bash
conda create -n news_recsys python=3.10 -y
conda activate news_recsys
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

当前仓库的主要依赖包括：

- `torch==2.2.2+cu118`
- `lightning==2.3.3`
- `pytorch_lightning==2.4.0`
- `faiss` / `faiss_cpu`
- `pandas`
- `omegaconf`

注意：

- `requirements.txt` 中同时包含 `faiss` 和 `faiss_cpu`，以及多个重复版本的 `PyYAML`
- 如果你在新环境安装失败，通常需要根据本机 CUDA、Python 版本重新整理这些依赖

### 4. 准备数据集

确保 `Data/MIND/` 目录下已经放好 MIND-small 数据，结构见上文 [Dataset](#dataset)。

### 5. 执行预处理

这一步会做几件事：

- 合并 train/dev 的新闻数据
- 构建全局 `news_id` / `user_id` 映射
- 将 `behaviors.tsv` 展开成单条样本
- 保存训练集用户集合，供 warm/cold 评估使用

运行命令：

```bash
make preprocess
```

产物默认写入 `src/tmp/preprocess/`。

### 6. 抽取特征

这一步会把预处理后的行为数据和新闻数据转成模型训练使用的特征格式：

```bash
make fe
```

默认产物：

```text
src/tmp/extractored_feature/
├── train_features.txt
├── dev_features.txt
├── item_features.txt
├── original_val_2_embedding_idx_dict.json
├── embedding_idx_2_original_val_dict.json
└── dataset_extract_info.yaml
```

### 7. 将 TXT 特征转换为 MMAP 格式

如果你希望使用更高效的 `mmap` 训练链路，再执行：

```bash
make fe_npz
```

虽然命令名仍然叫 `fe_npz`，但当前实现实际生成的是分片 `mmap` 目录，而不是单文件 `.npz`。

### 8. 训练模型

例如训练 `deep` 排序模型：

```bash
make train model=deep
```

训练日志与模型输出会写入 `experiments/` 下的时间戳目录。

## 端到端流程

推荐使用如下完整流程：

```bash
make preprocess
make fe
make fe_npz
make train model=deep
```

如果你只想走 `txt` 特征读取链路，可以在模型配置里将：

```yaml
dataset:
  load_npz: false
```

然后直接执行：

```bash
make train model=deep
```

## 系统架构

### 1. 数据预处理层

入口脚本：`src/dataset/FeaturesGenerator/preprocess.py`

这一层负责把原始 MIND 数据转成统一、稳定、可复用的中间结果。核心逻辑包括：

- 合并 train/dev 的 `news.tsv`
- 为 `news_id` 与 `user_id` 建立全局整数映射
- 将用户历史中的新闻 ID 同步映射成整数 ID
- 将 impression 候选列表按 `item-label` 展开成单条监督样本
- 保存训练用户列表 `train_user_ids.json`

预处理后的核心文件：

- `all_news_preprocess.csv`
- `train_behaviors_processed.csv`
- `dev_behaviors_processed.csv`
- `train_user_ids.json`

### 2. 特征工程层

入口脚本：`src/dataset/FeaturesGenerator/feature_extractor.py`

这层是整个项目最重要的模块之一。它的设计思路是：

- 在配置文件中声明要抽取哪些特征
- 在代码中提供同名的 `feature_extractor_<feature_name>` 函数
- 由基类自动调度这些函数，生成最终特征文本

当前支持的特征大致分为三类：

- 稀疏 ID 特征：`user_id`、`item_id`、`category`、`subcategory`
- 序列特征：`user_history`、`user_history_clicked_category`
- 实体相关特征：标题/摘要实体 ID、实体类型等

特征抽取器还做了两件关键的事情：

- 为原始特征值维护 `value -> embedding_idx` 映射
- 支持多个特征共享同一张 embedding 表

例如：

- `user_history` 复用 `item_id` 的 embedding
- `user_history_clicked_category` 复用 `category` 的 embedding

这使得“当前候选新闻”和“用户历史点击新闻”能落在统一语义空间中。

### 3. 数据读取层

入口模块：`src/dataset/DataReader/`

当前有两套读取链路：

- `DataReader`: 直接读取 `train_features.txt` 等文本文件
- `DataReaderMmap`: 读取 `mmap` 分片目录

`MINDDataModule` 会根据配置项 `dataset.load_npz` 自动选择使用哪套读取器。

数据样本在训练前会被解析成三种类型：

- 稀疏特征：整数 ID
- 稠密特征：浮点数
- 数组特征：序列 + mask

数组特征会自动补齐到配置指定的最大长度，并额外生成 `<feature>_mask`。

### 4. 模型层

统一底座位于 `src/model/BaseModel/`。

#### 基础模型层（BaseModel）

`BaseModel` 负责所有模型共用的基础能力：

- 加载 YAML 配置
- 构建 embedding table
- 计算 user/item 输入维度
- 支持共享 embedding 和预训练 embedding
- 将 batch 转换为拼接后的 embedding 向量
- 保存训练日志、loss 曲线和 checkpoint

#### 排序模型

排序模型位于：

```text
src/model/sort_models/classic_sort_models/
```

当前仓库中可见的排序模型包括：

- `lr`
- `fm`
- `deep`
- `deepfm`
- `dcn`
- `widedeep`
- `autoint`
- `din`

这类模型继承 `BaseModelSort`。训练阶段通常做点击率二分类；验证阶段会按用户聚合候选新闻，计算：

- AUC
- LogLoss
- GAUC
- MRR
- NDCG@5 / NDCG@10
- HR@5 / HR@10

`src/model/sort_models/generative_sort_models/` 已预留目录，用于后续扩展生成式排序模型。

### 5. 评估层

项目将 warm/cold user 区分开来做评估：

- warm user: 训练集中出现过的用户
- cold user: 验证时第一次出现的用户

训练开始前会读取 `src/tmp/preprocess/train_user_ids.json`，验证阶段据此拆分指标。

### 6. 实验日志

训练实验通常写入 `experiments/<model>_<timestamp>/`，目录中常见内容包括：

- `val_log.log`
- `train.log`
- `model_info.log`
- `ckpts/`
- `result_record/`
- `loss_figure/`

## 配置说明

### 特征抽取配置

文件：`src/dataset/FeaturesGenerator/config_fg.yaml`

主要控制：

- 原始数据路径
- 中间输出目录
- 要抽取的特征列表
- 哪些特征属于 item 特征
- 哪些特征是数组特征
- 数组最大长度
- 哪些特征共享 embedding

### 训练配置

排序阶段基础配置文件：

- `src/model/sort_models/classic_sort_models/base_sort_conf.yaml`

具体模型配置文件位于各模型目录下，例如：

- `src/model/sort_models/classic_sort_models/deep/deep_conf.yaml`

训练配置重点关注以下字段：

| 字段 | 作用 |
| --- | --- |
| `train_stage` | 当前仓库训练链路使用 `sort` |
| `features.sparse_feature_names` | 稀疏特征名 |
| `features.array_feature_names` | 序列特征名 |
| `features.user_feature_names` | 用户侧输入特征 |
| `features.item_feature_names` | 物品侧输入特征 |
| `embeddings.embedding_table_size` | 各特征 embedding 表大小 |
| `embeddings.embedding_tables` | embedding 维度、共享关系、预训练权重 |
| `dataset.load_npz` | 是否走 mmap 读取链路 |
| `dataset.batch_size` | batch size |
| `train_hparams` | 学习率、轮数、验证频率、设备等 |

## 可用命令

项目主要通过 `Makefile` 调度。

| Command | Description |
| --- | --- |
| `make preprocess` | 预处理原始 MIND 数据 |
| `make fe` | 抽取训练/验证/物品特征 |
| `make fe mode=debug` | 调试模式抽特征，仅处理有限样本 |
| `make fe_npz` | 将 txt 特征转换成 mmap 目录 |
| `make train model=deep` | 训练排序模型 |
| `make train model=deep model_group=classic_sort_models` | 显式指定模型组训练 |
| `make visualize_history` | 可视化用户历史 |
| `make vis_recall path=<recall_result_file>` | 生成召回结果可视化页面 |
| `make clean` | 清理 `src/tmp` |
| `make clean_exper` | 删除日志行数较少的实验目录 |
| `make server port=8000` | 启动静态文件服务器 |

说明：

- `make log model=<name>` 目标当前引用了 `src/scripts/log_analysis.py`
- 但该脚本在当前仓库中不存在，因此这一命令按现状不可用

## 产物目录

### 预处理产物

```text
src/tmp/preprocess/
├── all_news_preprocess.csv
├── train_behaviors_processed.csv
├── dev_behaviors_processed.csv
├── news_id_map.json
├── user_id_map.json
├── train_user_ids.json
├── entity_embedding_all.vec
└── relation_embedding_all.vec
```

### 特征产物

```text
src/tmp/extractored_feature/
├── train_features.txt
├── dev_features.txt
├── item_features.txt
├── train_features_mmap/
├── dev_features_mmap/
├── item_features_mmap/
├── original_val_2_embedding_idx_dict.json
├── embedding_idx_2_original_val_dict.json
└── dataset_extract_info.yaml
```

### 实验产物

```text
experiments/<model>_<timestamp>/
├── ckpts/
├── loss_figure/
├── result_record/
├── model_info.log
├── train.log
└── val_log.log
```

## 评估说明

### 排序阶段

排序模型验证流程如下：

1. 对验证集中的每条用户-候选新闻样本打分
2. 按用户聚合所有候选新闻
3. 对每个用户内部按分数降序排序
4. 计算用户级指标和全局指标
5. 输出 overall / warm / cold 三套结果

常见指标解释：

- **AUC**: 所有样本的全局 ROC-AUC
- **GAUC**: 用户粒度 AUC 的平均值
- **MRR**: 第一个正样本的倒数排名
- **HR@K**: Top-K 是否命中至少一个正样本
- **NDCG@K**: 考虑命中位置的排序质量

### 召回阶段

召回模型验证流程如下：

1. 先对全部 item 生成向量
2. 建立 Top-K 搜索索引
3. 对每个 impression 对应的用户向量做检索
4. 过滤掉用户历史点击过的新闻
5. 用 impression 中真实点击的新闻作为 target 做评估

## 常见问题

### `make preprocess` 报找不到数据文件

请检查：

- `Data/MIND/MINDsmall_train/` 和 `Data/MIND/MINDsmall_dev/` 是否存在
- 每个目录中是否包含 `news.tsv` 与 `behaviors.tsv`
- `src/dataset/FeaturesGenerator/config_fg.yaml` 中的 `paths.data_path` 是否正确

### `make fe` 失败，提示特征函数不存在

原因通常是：

- 你在 `config_fg.yaml` 中新增了某个特征名
- 但 `feature_extractor.py` 中没有实现对应的 `feature_extractor_<name>` 方法

### 训练时报特征缺失

请检查模型配置中的：

- `features.user_feature_names`
- `features.item_feature_names`
- `features.array_feature_names`

这些字段必须和特征抽取阶段实际产出的特征保持一致。

### 训练时报 embedding 配置错误

如果某个特征被加入模型输入，但没有在：

- `embeddings.embedding_table_size`
- `embeddings.embedding_tables.base_embedding_table.embedding_dims`

中声明，就会在模型初始化时报错。

### GPU 训练失败

当前训练脚本默认使用 GPU。请确认：

- 本机已安装可用的 CUDA 驱动
- `torch` 版本与 CUDA 环境匹配
- 模型配置中的设备参数与你的机器一致

如果只想临时调试代码，建议先降低 batch size，并将训练脚本中的设备配置改成 CPU 兼容形式。

### `make fe_npz` 后仍然没有走 mmap 训练

因为是否使用 mmap 读取，真正由训练配置中的以下字段决定：

```yaml
dataset:
  load_npz: true
```

仅执行 `make fe_npz` 不会自动切换训练链路。

## 已知限制

- 当前仓库没有标准化的部署配置，不适合作为生产服务直接发布
- 环境管理不够规范，`requirements.txt` 存在重复和潜在冲突依赖
- 训练脚本默认使用 GPU，对纯 CPU 环境不够友好
- `Makefile` 中的 `make log` 目标依赖的脚本缺失
- 目录中仍保留一些历史迁移痕迹，例如 MovieLens 相关路径或命名
- README 中描述的实验流程以本地离线实验为中心，不覆盖在线 serving、特征服务或在线召回系统

## 延伸阅读

仓库中已经有几份比较有价值的设计文档，建议结合本 README 阅读：

- `documents/feature_extractor_logic.md`
- `documents/base_model_logic.md`
- `documents/sort_model_metrics.md`
- `documents/recall_model_metrics.md`（召回指标说明，当前目录结构下无对应召回训练入口）
- `documents/scripts_usage.md`

如果你准备二次开发，推荐按以下顺序阅读源码：

1. `src/dataset/FeaturesGenerator/preprocess.py`
2. `src/dataset/FeaturesGenerator/feature_extractor_base.py`
3. `src/dataset/DataReader/pl_dataloader.py`
4. `src/model/BaseModel/base_model.py`
5. 具体模型目录，例如 `src/model/sort_models/classic_sort_models/deep/`
