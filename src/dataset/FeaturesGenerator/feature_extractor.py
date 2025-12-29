from tqdm import tqdm
import hashlib
import os
from omegaconf import OmegaConf
from .feature_extractor_base import FeatureExtractorBase

class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, config: dict):
        super().__init__(config)

    # 对于一些定制化的feature name的特征提取函数，可以在这里对需要的特殊的类变量进行初始化
    def initialization(self):
        pass

    def feature_extractor_user_id(self, data_line, extracted_features):  # 提取用户的id
        user_id = data_line['user_info']['user_id']
        embedding_idx = int(user_id)
        extracted_features['user_id'] = embedding_idx

    def feature_extractor_item_id(self, data_line, extracted_features):  # 提取电影的id
        item_id = data_line['item_info']['news_id']
        embedding_idx = int(item_id)
        extracted_features['item_id'] = embedding_idx

    def feature_extractor_category(self, data_line, extracted_features):  # 提取电影的一级分类
        first_category = data_line['item_info']['category']
        embedding_idx = self.get_feature_embedding_idx('category', first_category)
        extracted_features['category'] = embedding_idx
    
    def feature_extractor_subcategory(self, data_line, extracted_features):  # 提取电影的二级分类
        second_category = data_line['item_info']['subcategory']
        embedding_idx = self.get_feature_embedding_idx('subcategory', second_category)
        extracted_features['subcategory'] = embedding_idx

    def feature_extractor_user_click_category(self, data_line, extracted_features):  # 提取用户点击次数最多的电影一级分类
        """
        提取用户点击历史中出现次数最多的一级分类作为特征。
        逻辑：
        1. 获取用户点击历史列表。
        2. 遍历历史物品，统计各分类（转换为embedding index后）的出现频次。
        3. 选出频次最高的分类索引作为特征值。
        4. 若无历史记录，使用'unknown'对应的索引。
        """
        user_history = data_line['user_info']['history']
        category_count = {}
        for news_id in user_history:
            news_info = self.item_data_dict.get(news_id, {})
            category = news_info.get('category', 'unknown')
            embedding_idx = self.get_feature_embedding_idx('user_click_category', category)
            category_count[embedding_idx] = category_count.get(embedding_idx, 0) + 1
        if category_count:
            most_clicked_category = max(category_count, key=category_count.get)
            extracted_features['user_click_category'] = most_clicked_category
        else:
            extracted_features['user_click_category'] = self.get_feature_embedding_idx('user_click_category', 'unknown')

    

    # 提取标签，返回一个列表形式
    def label_extractor(self, data_line):
        return [data_line['label']]
    

if __name__ == "__main__":
    import argparse
    from .feature_extractor_base import FeatureExtractorBase

    parser = argparse.ArgumentParser(description="Feature Extractor")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    # 加载配置文件
    config = OmegaConf.load(args.config)

    # 初始化FeatureExtractor
    feature_extractor = FeatureExtractor(config)
    # 执行特征提取
    feature_extractor.run()