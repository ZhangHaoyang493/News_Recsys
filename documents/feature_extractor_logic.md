# Feature Extractor 开发逻辑详解

本文档详细介绍了 `src/dataset/FeaturesGenerator` 模块中的特征提取逻辑，重点解析 `FeatureExtractorBase` 基类与 `FeatureExtractor` 实现类的设计模式、数据流以及如何进行二次开发。

## 1. 模块结构

*   **`feature_extractor_base.py` (抽象基类 FeatureExtractorBase)**
    *   **核心职责**: 负责通用的数据加载、路径管理、ID 映射 (Mapping) 维护、以及主流程控制（Run Pipeline）。
    *   **设计模式**: 模板方法模式 (Template Method)。基类定义了 `_process_behavior_file` 等骨架流程，而将具体的特征提取逻辑 (`feature_extractor_*` 方法) 留给子类实现。
    *   **数据管理**:
        *   `self.item_data_dict`: 将所有 News items 加载到内存。
        *   `self.feature_map_val2idx`: 维护 `Feature Name -> {Raw Value -> Embedding ID}` 的动态映射。

*   **`feature_extractor.py` (实现类 FeatureExtractor)**
    *   **核心职责**: 实现具体的特征提取函数。
    *   **约定**: 函数名必须为 `feature_extractor_{feature_name}`。

---

## 2. 配置文件参数详解 (`config_fg.yaml`)

特征工程不仅需要代码，还需要配置文件来定义哪些特征需要被提取，以及它们之间的关系。以下是核心配置项的详细说明：

### 2.1 Paths (路径配置)

*   **`out_basedir`**: 
    *   整个项目的输出根目录 (例如 `src/tmp`)。
    *   特征提取器会自动在该目录下寻找 `preprocess/` 文件夹读取原始数据，并在其下创建 `extractored_feature/` 文件夹存放提取结果。
*   **`user_history_path`** (可选): 用户历史行为文件的路径，如果某些特征依赖外部历史记录。

### 2.2 Features (特征定义)

*   **`feature_names`**: 
    *   **核心列表**。定义了本次运行需要生成的所有特征名。
    *   对于列表中的每一个名字 `xxx`，代码中**必须**存在一个名为 `feature_extractor_xxx` 的函数，否则运行时会报错 `NotImplementedError`。
    
*   **`item_feature_names`**: 
    *   一个子集列表。定义了哪些特征属于 **Item Side Feature** (纯物品特征)。
    *   主要用于 `_extract_item_features_only` 流程，生成 `item_features.txt`。这对于双塔召回模型的 Item Tower 或在线服务时的物品缓存非常重要。
    
*   **`share_emb_table_features`**: 
    *   **共享 Embedding 映射表** (Dict: `Source Feature -> Target Feature`)。
    *   **作用**: 让不同业务含义但语义空间相同的特征共享同一个 ID 映射表。
    *   **示例**: `history_category: category`。这意味着提取 `history_category` 特征时，系统会去查 `category` 的映射字典。如果遇到新词，会加入到 `category` 的词表中。

*   **`array_feature_names`**:
    *   明确标识哪些特征是 **序列/多值特征** (Array/Sequence)。
    *   虽然主要逻辑在提取函数中实现（如用逗号拼接），但在此声明有助于校验配置完整性。

*   **`array_max_length`**:
    *   (Dict) 定义序列特征的最大长度。
    *   虽然特征提取阶段通常只负责产出字符串（如 "1,2,3"），但此配置可用于校验或后续 Dataset 阶段截断。

---

## 3. 核心工作流 (Pipeline)

当调用 `extractor.run()` 时，系统按以下顺序执行：

1.  **加载元数据 (`_load_item_data`)**:
    *   读取 `all_news_preprocess.csv`。
    *   解析 Item ID, Title, Category, SubCategory, Entities (JSON) 等基础信息并存入内存字典。

2.  **处理训练/验证集 (`_process_behavior_file`)**:
    *   逐行读取 User Behaviors 文件 (train/dev)。
    *   **构造 Context**: 将每行数据解析为 `data_context` 字典，包含 `user_info` (ID, History), `item_info` (当前 Item), `timestamp` 等。
    *   **动态调用 (`_extract_single_row`)**:
        *   遍历配置 `feature_names` 中的每个特征名 (e.g., `user_id`, `category`)。
        *   反射调用对应的 `feature_extractor_{feature_name}(data_context, output_dict)` 方法。
    *   **生成输出**: 将提取后的特征 ID 序列化写入 `*_features.txt`。

3.  **处理物品特征 (`_extract_item_features_only`)**:
    *   遍历所有 Item，提取仅与 Item 相关的特征（用于 Serving 阶段的某些场景或双塔模型的 Item Tower 离线刷库）。
    *   输出到 `item_features.txt`。

4.  **保存映射表 (`_save_mappings`)**:
    *   将构建好的 `original_val_2_embedding_idx_dict.json` 写入磁盘。这是特征工程的核心产物，后续推理服务需要加载此字典将原始 Request 转换为 Model Input IDs。

---

## 3. ID 映射机制 (ID Mapping)

基类提供了 `get_feature_embedding_idx(feature_name, feature_value)` 方法：

*   **自动增长**: 如果遇到新的 Feature Value，自动分配 `current_max_id + 1`。
*   **共享字典**: 支持 `share_emb_table_features` 配置。例如配置 `history_category: category`，则提取 `history_category` 特征时，会去查 `category` 的字典，确保 ID 空间一致。
*   **缓存友好**: 维护了双向映射 (`val2idx` 和 `idx2val`) 并在结束后保存。

---

## 4. 二次开发指南

### 4.1 如何添加一个新特征？

假设你要添加一个特征：**用户历史点击物品的一级分类序列** (`user_history_category`)。

**步骤 1: 修改配置文件 `config_fg.yaml`**
在 `feature_names` 列表里添加 `user_history_category`，并在 `share_emb_table_features` 里指定它共享 `category` 的词表（如果需要）。

**步骤 2: 在 `FeatureExtractor` 类中实现提取函数**
在 `src/dataset/FeaturesGenerator/feature_extractor.py` 中添加方法：

```python
    def feature_extractor_user_history_category(self, data_line, extracted_features):
        """
        Input: data_line (包含 user_info, item_info 等)
        Output: extracted_features (写入结果字典)
        """
        # 1. 获取用户点击历史 ID 列表
        user_history_ids = data_line['user_info']['history']
        
        history_cat_indices = []
        for news_id in user_history_ids:
            # 2. 查 item_data_dict 获取每个 news 的 category
            news_info = self.item_data_dict.get(news_id, {})
            cat_str = news_info.get('category', 'unknown')
            
            # 3. 转换为 ID (这里共享 'category' 的词表)
            # 注意：第一个参数填映射目标的特征名 'category'，或者系统配置了 share 关系后填 'user_history_category' 
            # (系统会自动映射，建议直接填本特征名并配置 share)
            emb_id = self.get_feature_embedding_idx('user_history_category', cat_str)
            history_cat_indices.append(str(emb_id))
            
        # 4. 序列化 (Array 特征通常用逗号或空格拼接，根据下游 dataset 要求)
        extracted_features['user_history_category'] = ','.join(history_cat_indices)
```

**步骤 3: 运行提取脚本**
运行 Feature Generator，新特征会自动出现在输出文件中。

### 4.2 高级优化：缓存 (Caching)

如果你发现特征提取很慢（例如通过 Impression ID 重复计算用户历史统计特征），可以使用缓存机制。

*   `FeatureExtractor` 中维护了 `impression_id_now` 和各类 Cache 变量。
*   由于数据通常是按 Impression 排序的（同一个 Impression 的所有 Candidate 样本连在一起），可以在提取函数开头判断：

```python
if impression_id != self.impression_id_now['my_feature']:
    # ... 执行复杂计算 ...
    self.my_feature_cache = result
    self.impression_id_now['my_feature'] = impression_id
else:
    # ... 直接使用缓存 ...
    result = self.my_feature_cache
```

这样，对于同一个 Impression 下的 100 个候选物品，复杂的用户侧特征计算只需要执行 1 次，而不是 100 次。
