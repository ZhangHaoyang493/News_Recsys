# 特征工程说明

## 什么是特征工程
特征工程是机器学习和数据科学中的一个关键步骤，涉及从原始数据中提取、转换和选择特征，以提高模型的性能。特征工程的目标是创建能够更好地表示数据的特征，从而帮助机器学习算法更有效地学习模式和关系。

而在推荐算法中，特征工程更是非常重要的，因为推荐系统通常需要处理大量的用户行为数据、物品属性数据以及上下文信息。通过有效的特征工程，可以提升推荐系统的准确性和用户满意度。一般的特征工程包括对用户特征、物品特征以及用户-物品交互特征的构建。

## 在本项目中怎样做特征工程？
整体来说，特征工程框架的使用可以分为以下几个步骤：
- 定义特征并指定特征生成逻辑
- 生成特征
- 使用生成的特征进行模型训练

下面我们将逐步介绍每个步骤的具体操作。
## 定义特征并指定特征生成逻辑
### 特征定义
我们的特征提取框架涉及到了如下三种主要的特征类型：
- **密集特征（Dense Features）**：这些特征通常是数值型的，表示用户或物品的某些属性。例如，用户的年龄、物品的价格、用户最近评分的均值等。
- **稀疏特征（Sparse Features）**：也就是在推荐系统中常用的类别特征。例如，用户的职业、用户id、电影的id等。值的注意的是，数值特征也可以通过分桶的方式转化为类别特征，从而作为稀疏特征使用。
- **数组特征（Array Features）**：这些特征表示为一个数组或列表，通常用于表示用户的历史行为序列或物品的多值属性。例如，用户最近浏览的商品列表、电影的类型列表等。

不管是哪种类型的特征，第一步都需要在配置文件中的`feature_names`字段中指定需要提取的特征名称:
```json
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
```
在上述的实例中，我们指定了10个需要提取的特征名称：
|特征名称|特征类型|说明|
|-------|-------|-------|
|user_id|稀疏特征|用户ID|
|movie_id|稀疏特征|电影ID|
|user_gender|稀疏特征|用户性别|
|user_age|稀疏特征|用户年龄段|
|movie_year|稀疏特征|电影上映年份|
|user_recent_10_movies|数组特征|用户最近评分的10部电影ID|
|user_occupation|稀疏特征|用户职业|
|movie_genres|数组特征|电影类型|
|user_recent_positive_score_10_movie|数组特征|用户最近评分大于等于4.0的10部电影ID|
|user_history_favourite_genre|数组特征|用户历史评分最高的电影类型|

特征名称的命名由你自己指定。

第二步则是在配置文件中指定上述在`feature_names`中定义的每个特征的类型，这里涉及到三个字段，分别是`sparse_feature_names`、`dense_feature_names`和`array_feature_names`，每个字段对应一种特征类型，分别指定该类型下的特征名称列表。例如：
```json
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
```
上述配置中，我们将`sparse_feature_names`字段指定了7个稀疏特征名称，`dense_feature_names`字段为空（表示没有密集特征），`array_feature_names`字段指定了3个数组特征名称。

第三步则是对Array类型的特征，指定其最大长度，这里通过`array_feature_max_length`字段来实现。例如：
```json
"array_max_length": {
    "user_recent_10_movies": 10,
    "movie_genres": 5,
    "user_recent_positive_score_10_movie": 10
  }
```
上述配置中，我们指定了`user_recent_10_movies`和`user_recent_positive_score_10_movie`这两个数组特征的最大长度为10，`movie_genres`的最大长度为5。

第四步是对定义的特征进行分类，也就是指定`item_feature_names`和`user_feature_names`字段，分别指定物品侧特征和用户侧特征的名称列表。例如：
```json
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
```

### 指定特征生成逻辑
为了让特征提取框架知道如何生成每个特征，我们需要创建一个`.py`的特征生成脚本文件，并在配置文件中通过`feature_extractor_path`字段指定该脚本的路径。例如：
```json
"feature_extractor_path": "/data2/zhy/Movie_Recsys/FeaturesGenerator/feature_extractor.py"
```

#### 最终生成的特征是什么样子的？
特征是从用户-电影的打分数据、以及用户和电影的属性文件中提取出来的，会生成一个`.txt`文件，其中每一行包含了一条用户-电影的交互数据所对应的所有特征以及label，在模型训练阶段，模型读取这个特征文件的每一行，然后进行解析并且训练模型。在以下文档中，我们将特征文件中的每一行称之为**特征行**。

特征行由多个特征字段和一个label字段两部分组成：
```
feature_1 feature_2 feature_3 ... feature_n<\t>label
```
每个特征字段的格式如下：
```
feature_name:feature_value
```
其中，`feature_name`是特征的名称，`feature_value`是该特征对应的值。对于不同类型的特征，`feature_value`的格式有所不同：
- **稀疏特征**：`feature_value`是一个单一的值，表示该特征在最终Embedding Table中的索引。因为稀疏特征最终会映射到一个Embedding上，所以在特征提取的过程中，我们直接对稀疏特征的每一个值进行映射，映射到其最终在Embedding Table中的索引值。例如，对于电影类别这一特征，如果电影类别是“动作片”，我们可能会将“动作片”映射到索引5，那么最终的特征表示为`movie_genres:5`。
- **密集特征**：`feature_value`是一个浮点数值，表示该特征的实际数值。例如，对于用户年龄这一特征，如果用户年龄是25岁，那么最终的特征表示为`user_age:25`。
- **数组特征**：`feature_value`是一个逗号分隔的索引列表，表示该特征在最终Embedding Table中的索引序列。例如，对于用户最近评分的10部电影ID这一特征，如果用户最近评分的10部电影ID分别映射到索引3,7,15,23,42,56,78,89,90,101，那么最终的特征表示为`user_recent_10_movies:3,7,15,23,42,56,78,89,90,101`。
  
多个特征间使用空格进行分隔。

label字段是一个用逗号分隔的浮点数值列表，这是考虑到我们可能会进行多任务学习。例如，如果我们只需要一个label来表示最终的score分数，那么label字段可能是`4`；如果我们还需要另外一个label来表示用户对电影的喜好程度（0-1 label），那么我们的label字段可能是`4,1`。

我们给出一个具体的特征行示例：
```
user_id:6 movie_id:543 user_gender:0 user_age:0 movie_year:1 user_recent_10_movies:17,31,586,387,398,592,189,164,593,171 user_occupation:4 movie_genres:1,2,10 user_recent_positive_score_10_movie:388,17,31,586,387,398,592,189,164,171 user_history_favourite_genre:10	5.0 1
```


#### 怎样定义特征生成逻辑
特征生成脚本需要实现继承自`FeatureExtractorBase`类的`FeatureExtractor`类，在`FeatureExtractor`类中需要对每个特征定义一个对应的提取函数。函数的命名需要和特征名称保持一致。例如，对于`user_id`特征，我们需要定义一个名为`feature_extractor_user_id`的函数。对于label字段，需要指定一个名为`label_extractor`的函数来提取label。除此之外，你还可以选择性的在`initialization`函数中定义一些你提取特征时可能用到的全局变量。一个模版如下所示：
```python
from feature_extractor_base import FeatureExtractorBase

class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, config: dict):
        super().__init__(config)

    # 对于一些定制化的feature的特征提取函数，可以在这里对需要的特殊的类变量进行初始化
    def initialization(self):
        pass

    def feature_extractor_xxx(self, data_line):
        pass

    # 提取标签，返回一个列表形式
    def label_extractor(self, data_line):
        pass
```

注意，对于每个特征的提取函数`feature_extractor_xxx`，以及`label_extractor`函数，都需要传入一个参数`data_line`，表示当前处理的用户-电影交互数据行。data_line是一个字典类型，其格式如下：
```python
{
    'rating': [user_id, movie_id, float(rating), int(timestamp)],
    'movie_info': {
        'movie_id': movie_id,
        'title': title,
        'year': year,
        'genres': ['genre1', 'genre2', ...]
    },
    'user_info': {
        'user_id': user_id,
        'gender': gender,
        'age': age,
        'occupation': occupation,
        'zip_code': zip_code
    }
}
```
在data_line涉及到的所有value中，除了`rating`字段中的第三个值是浮点数，第四个值是整数之外，其他的值全部是字符串类型。

我们给出几个具体的特征提取函数示例：
##### user_id
```python
def feature_extractor_user_id(self, data_line):  # 提取用户的id
    user_id = data_line['rating'][0]
    # 获取对应的feature_name 'user_id'的embedding索引映射字典
    embedding_idx = self.get_feature_embedding_idx('user_id', user_id)
    # 将hash后的字符串存储到extracted_feature中
    self.extracted_feature['user_id'] = embedding_idx
```
我们逐行进行解释：
- 首先从data_line中获取用户ID。
`user_id = data_line['rating'][0]`
- 然后通过`get_feature_embedding_idx`函数获取该用户ID在最终Embedding Table中的索引映射。该函数的作用是根据特征名称和特征值，返回该特征值在Embedding Table中的索引。对于稀疏特征和数组特征，这个函数的调用是必须的，因为这些特征最终会映射到Embedding Table上。
`embedding_idx = self.get_feature_embedding_idx('user_id', user_id)`
- 最后将该索引值存储到`self.extracted_feature`字典中，键为特征名称，值为该特征的索引值。**每一个特征提取函数都需要将提取到的特征值存储到`self.extracted_feature`字典中，框架会在最后将该字典中的所有特征值组合成最终的特征行。**
`self.extracted_feature['user_id'] = embedding_idx`

##### movie_year
```python
def feature_extractor_movie_year(self, data_line):  # 提取电影年份
    movie_year = data_line['movie_info'].get('year', '')
    # 将hash后的字符串存储到extracted_feature中
    embedding_idx = self.get_feature_embedding_idx('movie_year', int(movie_year) // 5)  # 将年份除以5，5年为一个区间
    self.extracted_feature['movie_year'] = embedding_idx
```

movie_year本身是一个数值特征，但是我们将其作为稀疏特征来处理，因此我们需要先对年份进行分桶处理（这里是每5年为一个区间），然后再通过`get_feature_embedding_idx`函数获取其在Embedding Table中的索引映射。

##### user_recent_10_movies
```python
def initialization(self):
    self.user_recent_10_movies = {}  # user_id: [movie_id1, movie_id2, ..., movie_id10]

def feature_extractor_user_recent_10_movies(self, data_line):  # 提取用户最近观看的10部电影
    user_id = data_line['rating'][0]
    movie_id = data_line['rating'][1]
    if user_id not in self.user_recent_10_movies:
        self.user_recent_10_movies[user_id] = []
    
    recent_movies = self.user_recent_10_movies[user_id]

    if len(recent_movies) > self.array_max_length.get('user_recent_10_movies', 10):
        recent_movies.pop(0)  # 保持只保存最近的10部电影
    
    recent_movies_str = ','.join(map(str, recent_movies))  # 转换为字符串存储
    self.extracted_feature['user_recent_10_movies'] = recent_movies_str

    recent_movies.append(
        self.get_feature_embedding_idx(
            self.share_emb_table_features.get('user_recent_10_movies', 'user_recent_10_movies'), # 如果share_emb_table_features中没有配置user_recent_10_movies的共享feature，则使用默认的feature name
            movie_id
        )
    )  # 使用电影的embedding索引来表示电影
```
在这个函数中，我们首先在`initialization`函数中定义了一个字典`self.user_recent_10_movies`，用于存储每个用户最近评分的10部电影ID列表。在特征提取函数中，我们从data_line中获取用户ID和电影ID，然后更新该用户的最近电影列表，确保只保留最近的10部电影。最后，我们将这些电影ID转换为字符串形式存储到`self.extracted_feature`字典中。在获取电影ID的Embedding索引时，我们通过`share_emb_table_features`字段来检查是否需要和其他特征共享Embedding Table。

## 生成特征
在完成特征定义和特征生成逻辑的编写之后，我们就可以使用特征提取框架来生成特征文件了。

在生成特征之前，我们需要在配置文件中指定一些必要的参数：
- `ratings_path`：一个列表，包含训练集和验证集的用户-电影打分数据文件路径。训练集的数据路径要放在第一个位置，验证集的数据路径要放在第二个位置。
- `movies_path`：电影属性数据文件路径。
- `users_path`：用户属性数据文件路径。
- `out_basedir`：特征文件的输出目录，生成的特征文件将会保存在该目录下。

示例配置如下所示：
```json
"ratings_path": [
    "/data2/zhy/Movie_Recsys/MovieLens_1M_data/sort_train_val_data/train_ratings_for_sort.dat",
    "/data2/zhy/Movie_Recsys/MovieLens_1M_data/sort_train_val_data/val_ratings_for_sort.dat"
],
"movies_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/movies.dat",
"users_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/users.dat",
"out_basedir": "/data2/zhy/Movie_Recsys/FeatureFilesForSort"
```

在配置完成之后，我们运行如下命令来生成特征文件：
```bash
python FeaturesGenerator/data_generator.py -c path_to_your_config.json
```

输出的特征文件将会保存在`out_basedir`目录下，根据示例配置生成的特征文件路径如下所示：
```
./FeatureFilesForSort
├── embedding_idx_2_original_val_dict.json
├── movie_features.txt
├── original_val_2_embedding_idx_dict.json
├── train_ratings_for_sort_features.txt
└── val_ratings_for_sort_features.txt
```

我们先解释一下我们是怎样将特征的值映射到Embedding Table中的索引的。在特征提取过程中，框架会为每个稀疏特征和数组特征创建一个映射字典，将每个特征值映射到一个唯一的整数索引。当一个新的特征值出现时，框架会为其分配下一个可用的索引。例如，我们举一个例子，假设我们在提取`user_id`特征的过程中，读取了5个数据行，遇到的用户ID分别是`1, 3, 2, 1, 7`，那么每个ID的映射关系如下所示：
|user_id|embedding_idx|
|-------|-------------|
|1      |0            |
|3      |1            |
|2      |2            |
|7      |3            |
这是因为用户ID `1` 是第一个出现的ID，因此被映射到索引 `0`；用户ID `3` 是第二个出现的ID，因此被映射到索引 `1`；用户ID `2` 是第三个出现的ID，因此被映射到索引 `2`；第四个用户ID是`1`，之前已经出现过了，所以仍然被映射到索引 `0`；用户ID `7` 是第四个出现的ID，因此被映射到索引 `3`。

解释完映射关系之后，我们来看一下输出的特征文件：
- `train_ratings_for_sort_features.txt`：训练集的特征文件，每一行对应一条用户-电影交互数据的特征行。
- `val_ratings_for_sort_features.txt`：验证集的特征文件，每一行对应一条用户-电影交互数据的特征行。
- `movie_features.txt`：电影的特征文件，每一行对应一部电影的电影侧特征的特征行，这个文件的生成是为了在召回的时候生成物料侧的Embedding提前存入到向量数据库中。
- `original_val_2_embedding_idx_dict.json`：特征值到Embedding索引的映射字典文件。
- `embedding_idx_2_original_val_dict.json`：Embedding索引到特征值的映射字典文件。


## 使用生成的特征进行模型训练
在生成特征文件之后，我们就可以使用这些特征文件来进行模型的训练了。具体的模型训练步骤请参考[模型训练说明](./model_training.md)文档。



