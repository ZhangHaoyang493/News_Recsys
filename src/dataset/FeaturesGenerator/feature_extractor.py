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

    def initialize_caches(self):
        self.user_click_category_cache = None
        self.user_history_cache = None
        self.user_history_clicked_category_cache = None
        self.user_history_clicked_subcategory_cache = None
        self.user_click_subcategory_cache = None
        
        self.item_abstract_entity_type_cache = {}
        self.item_title_entity_type_cache = {}
        self.item_abstract_entity_vector_cache = {}
        self.item_title_entity_vector_cache = {}

        self.impression_id_now = {
            'user_click_category': -1,
            'user_history': -1,
            'user_history_clicked_category': -1,
            'user_history_clicked_subcategory': -1
        }

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
        impression_id = data_line['impression_id']
        # if impression_id not in self.user_click_category_cache:
        if impression_id != self.impression_id_now['user_click_category']:
            self.impression_id_now['user_click_category'] = impression_id
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

            self.user_click_category_cache = extracted_features['user_click_category']
            
        else:
            extracted_features['user_click_category'] = self.user_click_category_cache

    def feature_extractor_user_click_subcategory(self, data_line, extracted_features):  # 提取用户点击次数最多的电影二级分类
        """
        提取用户点击历史中出现次数最多的二级分类作为特征。
        逻辑：
        1. 获取用户点击历史列表。
        2. 遍历历史物品，统计各子分类（转换为embedding index后）的出现频次。
        3. 选出频次最高的子分类索引作为特征值。
        4. 若无历史记录，使用'unknown'对应的索引。
        """
        impression_id = data_line['impression_id']
        # if impression_id not in self.user_click_category_cache:
        if impression_id != self.impression_id_now['user_click_subcategory']:
            self.impression_id_now['user_click_subcategory'] = impression_id

            user_history = data_line['user_info']['history']
            subcategory_count = {}
            for news_id in user_history:
                news_info = self.item_data_dict.get(news_id, {})
                subcategory = news_info.get('subcategory', 'unknown')
                embedding_idx = self.get_feature_embedding_idx('user_click_subcategory', subcategory)
                subcategory_count[embedding_idx] = subcategory_count.get(embedding_idx, 0) + 1
            if subcategory_count:
                most_clicked_subcategory = max(subcategory_count, key=subcategory_count.get)
                extracted_features['user_click_subcategory'] = most_clicked_subcategory
            else:
                extracted_features['user_click_subcategory'] = self.get_feature_embedding_idx('user_click_subcategory', 'unknown')

            self.user_click_subcategory_cache = extracted_features['user_click_subcategory']
            
        else:
            extracted_features['user_click_subcategory'] = self.user_click_subcategory_cache
    
    def feature_extractor_user_history(self, data_line, extracted_features):  # 提取用户点击历史
        impression_id = data_line['impression_id']
        # if impression_id not in self.user_history_cache:
        if impression_id != self.impression_id_now['user_history']:
            self.impression_id_now['user_history'] = impression_id
            user_history = data_line['user_info']['history']
            history_indices = []
            for news_id in user_history:
                embedding_idx = str(news_id)
                history_indices.append(embedding_idx)
            
            extracted_features['user_history'] = ','.join(history_indices)

            self.user_history_cache = extracted_features['user_history']
        else:
            extracted_features['user_history'] = self.user_history_cache

    def feature_extractor_impression_id(self, data_line, extracted_features):  # 提取每条样本的impression id
        impression_id = data_line['impression_id']
        extracted_features['impression_id'] = impression_id


    def feature_extractor_user_history_clicked_category(self, data_line, extracted_features):  # 提取用户点击历史中的电影一级分类
        impression_id = data_line['impression_id']
        # if impression_id not in self.user_history_clicked_category_cache:
        if impression_id != self.impression_id_now['user_history_clicked_category']:
            self.impression_id_now['user_history_clicked_category'] = impression_id

            user_history = data_line['user_info']['history']
            history_categories = []
            for news_id in user_history:
                news_info = self.item_data_dict.get(news_id, {})
                category = news_info.get('category', 'unknown')
                embedding_idx = self.get_feature_embedding_idx('user_history_clicked_category', category)
                history_categories.append(str(embedding_idx))
            
            extracted_features['user_history_clicked_category'] = ','.join(history_categories)

            self.user_history_clicked_category_cache = extracted_features['user_history_clicked_category']
        else:
            extracted_features['user_history_clicked_category'] = self.user_history_clicked_category_cache

    def feature_extractor_user_history_clicked_subcategory(self, data_line, extracted_features):  # 提取用户点击历史中的电影一级分类
        impression_id = data_line['impression_id']
        # if impression_id not in self.user_history_clicked_subcategory_cache:
        if impression_id != self.impression_id_now['user_history_clicked_subcategory']:
            self.impression_id_now['user_history_clicked_subcategory'] = impression_id
            user_history = data_line['user_info']['history']
            history_subcategories = []
            for news_id in user_history:
                news_info = self.item_data_dict.get(news_id, {})
                subcategory = news_info.get('subcategory', 'unknown')
                embedding_idx = self.get_feature_embedding_idx('user_history_clicked_subcategory', subcategory)
                history_subcategories.append(str(embedding_idx))
            
            extracted_features['user_history_clicked_subcategory'] = ','.join(history_subcategories)

            self.user_history_clicked_subcategory_cache = extracted_features['user_history_clicked_subcategory']
        else:
            extracted_features['user_history_clicked_subcategory'] = self.user_history_clicked_subcategory_cache
    
    def feature_extractor_item_abstract_entity_type(self, data_line, extracted_features):  # 提取电影摘要中的实体类型
        news_id = data_line['item_info']['news_id']
        if news_id in self.item_abstract_entity_type_cache:
            extracted_features['item_abstract_entity_type'] = self.item_abstract_entity_type_cache[news_id]
            return
        news_info = self.item_data_dict.get(news_id, {})
        abstract_entities = news_info.get('abstract_entities', [])
        entity_types = set()
        for entity in abstract_entities:
            entity_type = entity.get('Type', 'unknown')
            embedding_idx = self.get_feature_embedding_idx('item_abstract_entity_type', entity_type)
            entity_types.add(str(embedding_idx))
        
        extracted_features['item_abstract_entity_type'] = ','.join(entity_types)
        self.item_abstract_entity_type_cache[news_id] = extracted_features['item_abstract_entity_type']


    def feature_extractor_item_title_entity_type(self, data_line, extracted_features):  # 提取电影标题中的实体类型
        news_id = data_line['item_info']['news_id']
        if news_id in self.item_title_entity_type_cache:
            extracted_features['item_title_entity_type'] = self.item_title_entity_type_cache[news_id]
            return
        news_info = self.item_data_dict.get(news_id, {})
        title_entities = news_info.get('title_entities', [])
        entity_types = set()
        for entity in title_entities:
            entity_type = entity.get('Type', 'unknown')
            embedding_idx = self.get_feature_embedding_idx('item_title_entity_type', entity_type)
            entity_types.add(str(embedding_idx))
        
        extracted_features['item_title_entity_type'] = ','.join(entity_types)
        self.item_title_entity_type_cache[news_id] = extracted_features['item_title_entity_type']

    def feature_extractor_item_abstract_entity_vector(self, data_line, extracted_features):  # 提取电影摘要中的实体类型的embedding
        news_id = data_line['item_info']['news_id']
        if news_id in self.item_abstract_entity_vector_cache:
            extracted_features['item_abstract_entity_vector'] = self.item_abstract_entity_vector_cache[news_id]
            return
        news_info = self.item_data_dict.get(news_id, {})
        abstract_entities = news_info.get('abstract_entities', [])
        entity_vectors = []
        dim = 0
        for entity in abstract_entities:
            entity_vector_id = entity.get('WikidataId', [])
            if entity_vector_id:
                if entity_vector_id in self.entity_embedding_dict:
                    entity_vector = self.entity_embedding_dict[entity_vector_id]
                else:
                    continue
                entity_vector_str = ','.join(map(str, entity_vector))
                if dim == 0:
                    dim = len(entity_vector)
                else:
                    assert dim == len(entity_vector), f"Entity vector dimension mismatch for news_id {news_id}"
                entity_vectors.append(entity_vector_str)
        num = len(entity_vectors)
        entity_vectors = [str(dim), str(num)] + entity_vectors
        extracted_features['item_abstract_entity_vector'] = ','.join(entity_vectors)
        self.item_abstract_entity_vector_cache[news_id] = extracted_features['item_abstract_entity_vector']

    def feature_extractor_item_title_entity_vector(self, data_line, extracted_features):  # 提取电影标题中的实体类型的embedding
        news_id = data_line['item_info']['news_id']
        if news_id in self.item_title_entity_vector_cache:
            extracted_features['item_title_entity_vector'] = self.item_title_entity_vector_cache[news_id]
            return
        news_info = self.item_data_dict.get(news_id, {})
        title_entities = news_info.get('title_entities', [])
        entity_vectors = []
        dim = 0
        for entity in title_entities:
            entity_vector_id = entity.get('WikidataId', [])
            if entity_vector_id:
                if entity_vector_id in self.entity_embedding_dict:
                    entity_vector = self.entity_embedding_dict[entity_vector_id]
                else:
                    continue
                entity_vector_str = ','.join(map(str, entity_vector))
                if dim == 0:
                    dim = len(entity_vector)
                else:
                    assert dim == len(entity_vector), f"Entity vector dimension mismatch for news_id {news_id}"
                entity_vectors.append(entity_vector_str)
        num = len(entity_vectors)
        entity_vectors = [str(dim), str(num)] + entity_vectors
        extracted_features['item_title_entity_vector'] = ','.join(entity_vectors)
        self.item_title_entity_vector_cache[news_id] = extracted_features['item_title_entity_vector']

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