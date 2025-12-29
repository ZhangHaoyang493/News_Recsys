# 模型训练

## 定义每个特征Embedding Table的维度和大小
通过在配置文件中定义每个特征的Embedding维度和大小，模型训练代码会根据这些配置来创建相应的Embedding Table。例如，假设我们有如下的配置片段：
```json
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
}
```
这里，`embedding_size`字段定义了每个特征的Embedding维度，例如`user_id`的Embedding维度是4，`movie_id`的Embedding维度也是4。`embedding_table_size`字段定义了每个特征的Embedding Table的大小，例如`user_id`的Embedding Table大小是6040，表示有6040个不同的用户ID。

## 模型定义
模型需要继承`BaseModel/base_model.py`中的`BaseModel`类，并实现相应的方法。有关模型代码和训练代码可以参考`sort`文件夹中所涉及的代码实现。