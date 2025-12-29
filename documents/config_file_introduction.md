# 关于配置文件的说明
本项目的配置文件在做特征工程、模型构建的时候都会使用到。配置文件是一个json文件，包含了下面内容：
- **特征工程配置**：定义了做特征工程所需要的参数：
  - 特征提取所需要的文件路径：例如元数据文件路径、评分数据文件路径、特征提取脚本路径等。
  - 需要提取的特征名称。
- **模型的Embedding Table配置**：定义了模型中使用的Embedding Table的相关参数，例如Embedding Size等。
- **模型的相关配置**：模型的相关配置主要有两类：
  - 召回模型在跑测试集的时候，对于召回的物料可能需要进行历史消重，需要传入一些预先准备好的用户历史文件等。
  - 有些模型可能需要额外指定一些相关参数，例如DeepFM模型的FM侧的Embedding的维度。

下面我们将详细介绍配置文件的各个部分。

## 特征工程配置

### 关于特征提取所需要的文件路径的配置字段
首先给出一个特征工程文件路径配置的示例：
```json
{
  "ratings_path": [
      "/data2/zhy/Movie_Recsys/MovieLens_1M_data/train_ratings.dat",
      "/data2/zhy/Movie_Recsys/MovieLens_1M_data/test_ratings.dat"
    ],
  "movies_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/movies.dat",
  "users_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/users.dat",
  "feature_extractor_path": "/data2/zhy/Movie_Recsys/FeaturesGenerator/feature_extractor.py",
  "out_basedir": "/data2/zhy/Movie_Recsys/FeatureFiles"
}
```
我们对于以上字段进行解释：
|字段名 | 值类型 | 说明 |
|-------|-------|-------|
|`ratings_path` | 列表 | 评分元数据的路径，每一行的数据是以下格式`user_id::movie_id::rating::timestamp`，和MovieLens数据集提供的ratings.dat文件的格式保持一致，它的值是列表类型，可以包含多个ratings文件的路径，因此可以同时对多个ratings文件进行特征处理。一般来说，`ratings_path`的值包含训练集和测试集的评分数据文件路径，**注意，训练集的路径一定要在测试集路径的前面。**|
|`movies_path` | 字符串 | 电影的元信息文件路径，是一个字符串，指向电影的元信息文件。数据行格式为`movie_id::name::movie_genre`，和MovieLens数据集提供的movies.dat文件的格式保持一致。|
|`users_path` | 字符串 | 用户的元信息文件路径，是一个字符串，指向用户的元信息文件。数据行格式为`user_id::gender::age::occupation::zip_code`，和MovieLens数据集提供的users.dat文件的格式保持一致。|
|`feature_extractor_path` | 字符串 | 特征工程脚本的路径，是一个字符串，指向用户自定义的特征提取脚本的文件路径。有关特征提取脚本的详细说明，请参考[特征工程框架介绍](./feature_engineering_introduction.md)|
|`out_basedir` | 字符串 | 特征文件的输出目录，通常是一个字符串，指向特征文件的输出目录。|

### 需要提取的特征名称配置字段
关于要提取的每个特征，需要在配置文件中进行指定，仅涉及到一个字段`feature_names`，下面给出一个示例：
```json
{
  "feature_names": [
    "user_id",
    "movie_id",
    "user_gender",
    "user_age",
    "movie_year",
    "user_recent_10_movies",
    "user_occupation",
    "movie_genres",
    "user_recent_positive_score_10_movie",
    "user_history_favourite_genre"
  ]
}
```
在上面的示例中，`feature_names`字段是一个列表，包含了需要提取的特征名称。这里涉及到的每个特征名称，都需要在`FeatureExtractor`类中提供对应的特征提取方法，具体请参考[特征工程框架介绍](./feature_engineering_introduction.md)。

## 模型的Embedding Table配置
Embedding Table涉及到多个字段，主要关系到每个特征属于Sparse特征、Dense特征还是Array特征，每个特征的Embedding Size、Embedding Table的Size，以及多个特征是否共享同一个Embedding Table等。下面给出一个完整的Embedding Table配置示例：
```json
{
  "share_emb_table_features": { // 共享embedding表的特征名称
    "user_recent_10_movies": "movie_id",
    "user_recent_positive_score_10_movie": "movie_id",
    "user_history_favourite_genre": "movie_genres"
  },
  "sparse_feature_names": [
    "user_id",
    "movie_id",
    "user_gender",
    "user_age",
    "movie_year",
    "user_occupation",
    "user_history_favourite_genre"
  ],
  "dense_feature_names": [],
  "array_feature_names": [
    "user_recent_10_movies",
    "movie_genres",
    "user_recent_positive_score_10_movie"
  ],
  "embedding_size": {
    "user_id": 4, //16,
    "movie_id": 4, //16,
    "user_gender": 2, //8,
    "user_age": 2, //8,
    "movie_year": 2, //8,
    "user_occupation": 2, //8,
    "movie_genres": 2 //8
  },
  "embedding_table_size": {
    "user_id": 6040,
    "movie_id": 3883,
    "user_gender": 2,
    "user_age": 7,
    "movie_year": 18,
    "user_occupation": 21,
    "movie_genres": 18
  },
  "array_max_length": {
    "user_recent_10_movies": 10,
    "movie_genres": 5,
    "user_recent_positive_score_10_movie": 10
  },
  "item_feature_names": [
    "movie_id",
    "movie_year",
    "movie_genres"
  ],
  "user_feature_names": [
    "user_id",
    "user_gender",
    "user_age",
    "user_recent_10_movies",
    "user_occupation",
    "user_recent_positive_score_10_movie",
    "user_history_favourite_genre"
  ]
}
```
我们对于以上字段进行解释：
|字段名 | 值类型 | 说明 |
|-------|-------|-------|
|`share_emb_table_features` | 字典 | 在特征工程的时候，尤其是当数据量不足的话，如果让每个特征都使用一个自己的Embedding Table，可能会导致Embedding Table过大且稀疏，导致Embedding学习不够充分。通过该字段，可以指定多个相关特征共享同一个Embedding Table，从而减少Embedding Table的数量和大小。字典的键是共享Embedding Table的特征名称，值是它们共享的目标Embedding Table对应的特征名称。例如，对于`user_recent_10_movies`这个特征，代表用户最近打分的10个电影，它对应的Embedding表就可以和`movie_id`共享。|
|`sparse_feature_names` | 列表 | 稀疏特征名称列表，包含了所有的稀疏特征名称。什么是稀疏特征（Sparse Feature）？参考[特征工程框架介绍](./feature_engineering_introduction.md)|
|`dense_feature_names` | 列表 | 密集特征名称列表，包含了所有的密集特征名称。什么是密集特征（Dense Feature）？参考[特征工程框架介绍](./feature_engineering_introduction.md)|
|`array_feature_names` | 列表 | 数组特征名称列表，包含了所有的数组特征名称。什么是数组特征（Array Feature）？参考[特征工程框架介绍](./feature_engineering_introduction.md)|
|`embedding_size` | 字典 | 每个特征对应的Embedding Size，字典的键是特征名称，值是对应的Embedding Size。只有稀疏特征和数组特征才有Embedding Size，密集特征没有Embedding Size。当然，如果数组特征指定了共享的特征，那么就不需要给出Embedding size。 |
|`embedding_table_size` | 字典 | 每个特征对应的Embedding Table的大小，这里的大小指的是Embedding Table中Embedding的个数，和每个Sparse特征的可能取值的个数有关。例如`user_gender`特征只涉及到男女两种取值，因此它的Embedding Table Size是2。而`user_id`对应的Embedding Table的大小则是`user_id`的可能取值个数，共有6040个用户，因此`user_id`对应的table size就是6040. |
|`array_max_length` | 字典 | 数组特征的最大长度，字典的键是数组特征名称，值是对应的最大长度。例如`user_recent_10_movies`这个特征，代表用户最近打分的10个电影，因此它的最大长度是10。|
|`item_feature_names` | 列表 | 物料特征名称列表，包含了所有的物料特征名称。物料特征是指与推荐物料相关的特征，例如电影的ID、类型等。 |
|`user_feature_names` | 列表 | 用户特征名称列表，包含了所有的用户特征名称。用户特征是指与用户相关的特征，例如用户的ID、性别、年龄等。 |

## 模型的相关配置
模型的相关配置主要有两类，第一类是关于召回模型在跑测试集的时候，可能需要进行历史消重，需要传入一些预先准备好的用户历史文件等；第二类是有些模型可能需要额外指定一些相关参数，例如DeepFM模型的FM侧的Embedding的维度。
### 召回模型的相关配置字段
召回模型在做召回的时候需要对历史物料进行消重，因此需要提供一些预先准备好的用户历史文件等。下面给出一个示例：
```json
{
  "user_history_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/user_history.json",
  "embedding_idx_2_original_val_dict_path": "/data2/zhy/Movie_Recsys/FeatureFiles/embedding_idx_2_original_val_dict.json",
  "original_val_2_embedding_idx_dict_path": "/data2/zhy/Movie_Recsys/FeatureFiles/original_val_2_embedding_idx_dict.json",
}
```
我们对于以上字段进行解释：
|字段名 | 值类型 | 说明 |
|-------|-------|-------|
|`user_history_path` | 字符串 | 用户历史文件路径，是一个字符串，指向用户历史文件的路径。用户历史文件包含了每个用户过去交互过的物料信息，用于在召回阶段进行历史消重。历史文件的格式参考`MovieLens_1M_data/user_history.json`|
|`embedding_idx_2_original_val_dict_path` | 字符串 | Embedding索引到原始值的映射字典文件路径，是一个字符串，指向该映射字典文件的路径。该字典用于将Embedding索引映射回原始的特征值。为什么会产生映射？参考[特征工程框架介绍](./feature_engineering_introduction.md)|
|`original_val_2_embedding_idx_dict_path` | 字符串 | 原始值到Embedding索引的映射字典文件路径，是一个字符串，指向该映射字典文件的路径。该字典用于将原始的特征值映射到Embedding索引。为什么会产生映射？参考[特征工程框架介绍](./feature_engineering_introduction.md)|


### 其他模型的相关配置字段
有些模型可能需要额外指定一些相关参数，例如DeepFM模型的FM侧的Embedding的维度。下面给出一个示例：
```json
{
  "deepfm_cfg": {
    "fm_feature_names": [
      "movie_id",
      "movie_year",
      "movie_genres",
      "user_id",
      "user_gender",
      "user_age",
      "user_recent_10_movies",
      "user_occupation",
      "user_recent_positive_score_10_movie",
      "user_history_favourite_genre"
    ],
    "fm_dim": 2 // fm的隐向量的维度
  }
}
```
这只是针对DeepFM模型的一个示例配置，其他模型可能会有不同的配置需求，这些有关模型的配置字段并不是必须的，只有在模型需要除了之前提到的配置字段之外的其他配置时，才需要在配置文件中添加相应的字段。有关这些自定义的配置字段的读取，可以参考各个模型的实现代码。

对于示例配置，由于DeepFM模型涉及到Deep侧和FM侧，不同侧的可能需要使用到不同的特征，因此需要指定FM侧使用的特征名称`fm_feature_names`，以及FM侧的隐向量维度`fm_dim`。