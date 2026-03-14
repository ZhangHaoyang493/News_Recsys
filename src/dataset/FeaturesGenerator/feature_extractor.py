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

    def initialize_caches_user(self):
        self.user_click_category_cache = {}
        self.user_history_cache = {}
        self.user_history_clicked_category_cache = {}
        self.user_history_clicked_subcategory_cache = {}
        self.user_click_subcategory_cache = {}
        self.user_history_title_entity_id_cache = {}
        self.user_history_title_entity_type_cache = {}
        self.user_history_abstract_entity_type_cache = {}
    
    def initialize_caches_item(self):
        self.item_abstract_entity_type_cache = {}
        self.item_title_entity_type_cache = {}
        self.item_abstract_entity_id_cache = {}
        self.item_title_entity_id_cache = {}

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
        """
        impression_id = data_line['impression_id']
        if impression_id in self.user_click_category_cache:
            extracted_features['user_click_category'] = self.user_click_category_cache[impression_id]
            return

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

        self.user_click_category_cache[impression_id] = extracted_features['user_click_category']

    def feature_extractor_user_click_subcategory(self, data_line, extracted_features):  # 提取用户点击次数最多的电影二级分类
        """
        提取用户点击历史中出现次数最多的二级分类作为特征。
        """
        impression_id = data_line['impression_id']
        if impression_id in self.user_click_subcategory_cache:
            extracted_features['user_click_subcategory'] = self.user_click_subcategory_cache[impression_id]
            return

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

        self.user_click_subcategory_cache[impression_id] = extracted_features['user_click_subcategory']
    
    def feature_extractor_user_history(self, data_line, extracted_features):  # 提取用户点击历史
        impression_id = data_line['impression_id']
        if impression_id in self.user_history_cache:
            extracted_features['user_history'] = self.user_history_cache[impression_id]
            return

        user_history = data_line['user_info']['history']
        history_indices = []
        for news_id in user_history:
            embedding_idx = str(news_id)
            history_indices.append(embedding_idx)
        
        extracted_features['user_history'] = ','.join(history_indices)
        self.user_history_cache[impression_id] = extracted_features['user_history']

    def feature_extractor_impression_id(self, data_line, extracted_features):  # 提取每条样本的impression id
        impression_id = data_line['impression_id']
        extracted_features['impression_id'] = impression_id


    def feature_extractor_user_history_clicked_category(self, data_line, extracted_features):  # 提取用户点击历史中的电影一级分类
        impression_id = data_line['impression_id']
        if impression_id in self.user_history_clicked_category_cache:
            extracted_features['user_history_clicked_category'] = self.user_history_clicked_category_cache[impression_id]
            return

        user_history = data_line['user_info']['history']
        history_categories = []
        for news_id in user_history:
            news_info = self.item_data_dict.get(news_id, {})
            category = news_info.get('category', 'unknown')
            embedding_idx = self.get_feature_embedding_idx('user_history_clicked_category', category)
            history_categories.append(str(embedding_idx))
        
        extracted_features['user_history_clicked_category'] = ','.join(history_categories)
        self.user_history_clicked_category_cache[impression_id] = extracted_features['user_history_clicked_category']

    def feature_extractor_user_history_clicked_subcategory(self, data_line, extracted_features):  # 提取用户点击历史中的电影一级分类
        impression_id = data_line['impression_id']
        if impression_id in self.user_history_clicked_subcategory_cache:
            extracted_features['user_history_clicked_subcategory'] = self.user_history_clicked_subcategory_cache[impression_id]
            return

        user_history = data_line['user_info']['history']
        history_subcategories = []
        for news_id in user_history:
            news_info = self.item_data_dict.get(news_id, {})
            subcategory = news_info.get('subcategory', 'unknown')
            embedding_idx = self.get_feature_embedding_idx('user_history_clicked_subcategory', subcategory)
            history_subcategories.append(str(embedding_idx))
        
        extracted_features['user_history_clicked_subcategory'] = ','.join(history_subcategories)
        self.user_history_clicked_subcategory_cache[impression_id] = extracted_features['user_history_clicked_subcategory']
    
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

    def feature_extractor_item_abstract_entity_id(self, data_line, extracted_features):  # 提取电影摘要中的实体类型的id
        news_id = data_line['item_info']['news_id']
        if news_id in self.item_abstract_entity_id_cache:
            extracted_features['item_abstract_entity_id'] = self.item_abstract_entity_id_cache[news_id]
            return
        news_info = self.item_data_dict.get(news_id, {})
        abstract_entities = news_info.get('abstract_entities', [])
        entity_ids = []
        for entity in abstract_entities:
            entity_vector_id = entity.get('WikidataId', 'UNKNOWN')
            embedding_idx = self.get_feature_embedding_idx('item_abstract_entity_id', entity_vector_id)
            entity_ids.append(str(embedding_idx))
        extracted_features['item_abstract_entity_id'] = ','.join(entity_ids)
        self.item_abstract_entity_id_cache[news_id] = extracted_features['item_abstract_entity_id']

    def feature_extractor_item_title_entity_id(self, data_line, extracted_features):  # 提取电影标题中的实体类型的id
        news_id = data_line['item_info']['news_id']
        if news_id in self.item_title_entity_id_cache:
            extracted_features['item_title_entity_id'] = self.item_title_entity_id_cache[news_id]
            return
        news_info = self.item_data_dict.get(news_id, {})
        title_entities = news_info.get('title_entities', [])
        entity_ids = []
        for entity in title_entities:
            entity_vector_id = entity.get('WikidataId', 'UNKNOWN')
            embedding_idx = self.get_feature_embedding_idx('item_title_entity_id', entity_vector_id)
            entity_ids.append(str(embedding_idx))
            
        extracted_features['item_title_entity_id'] = ','.join(entity_ids)
        self.item_title_entity_id_cache[news_id] = extracted_features['item_title_entity_id']

    def feature_extractor_user_history_title_entity_id(self, data_line, extracted_features):  # 提取用户点击历史中的电影标题实体id
        impression_id = data_line['impression_id']
        if impression_id in self.user_history_title_entity_id_cache:
            extracted_features['user_history_title_entity_id'] = self.user_history_title_entity_id_cache[impression_id]
            return

        user_history = data_line['user_info']['history']
        history_entity_ids = []
        for news_id in user_history:
            if news_id in self.item_title_entity_id_cache:
                cached_entity_ids = self.item_title_entity_id_cache[news_id]
                if cached_entity_ids == '':  # 如果缓存的实体ID列表为空，说明之前处理过但没有实体，直接跳过
                    continue
                history_entity_ids.append(cached_entity_ids)
                continue
            news_info = self.item_data_dict.get(news_id, {})
            title_entities = news_info.get('title_entities', [])
            for entity in title_entities:
                entity_vector_id = entity.get('WikidataId', 'UNKNOWN')
                embedding_idx = self.get_feature_embedding_idx('user_history_title_entity_id', entity_vector_id)
                history_entity_ids.append(str(embedding_idx))

        
        extracted_features['user_history_title_entity_id'] = ','.join(history_entity_ids)
        self.user_history_title_entity_id_cache[impression_id] = extracted_features['user_history_title_entity_id']

    def feature_extractor_user_history_title_entity_type(self, data_line, extracted_features):  # 提取用户点击历史中的电影标题实体类型
        impression_id = data_line['impression_id']
        if impression_id in self.user_history_title_entity_type_cache:
            extracted_features['user_history_title_entity_type'] = self.user_history_title_entity_type_cache[impression_id]
            return

        user_history = data_line['user_info']['history']
        history_entity_types = []
        for news_id in user_history:
            if news_id in self.item_title_entity_type_cache:
                cached_entity_types = self.item_title_entity_type_cache[news_id]
                if cached_entity_types == '':  # 如果缓存的实体类型列表为空，说明之前处理过但没有实体类型，直接跳过
                    continue
                history_entity_types.append(cached_entity_types)
                continue
            news_info = self.item_data_dict.get(news_id, {})
            title_entities = news_info.get('title_entities', [])
            for entity in title_entities:
                entity_type = entity.get('Type', 'unknown')
                embedding_idx = self.get_feature_embedding_idx('user_history_title_entity_type', entity_type)
                history_entity_types.append(str(embedding_idx))
        
        extracted_features['user_history_title_entity_type'] = ','.join(history_entity_types)
        self.user_history_title_entity_type_cache[impression_id] = extracted_features['user_history_title_entity_type']

    def feature_extractor_user_history_abstract_entity_type(self, data_line, extracted_features):  # 提取用户点击历史中的电影摘要实体id
        impression_id = data_line['impression_id']
        if impression_id in self.user_history_abstract_entity_type_cache:
            extracted_features['user_history_abstract_entity_type'] = self.user_history_abstract_entity_type_cache[impression_id]
            return

        user_history = data_line['user_info']['history']
        history_entity_types = []
        for news_id in user_history:
            if news_id in self.item_abstract_entity_type_cache:
                cached_entity_types = self.item_abstract_entity_type_cache[news_id]
                if cached_entity_types == '':  # 如果缓存的实体类型列表为空，说明之前处理过但没有实体类型，直接跳过
                    continue
                history_entity_types.append(cached_entity_types)
                continue
            news_info = self.item_data_dict.get(news_id, {})
            abstract_entities = news_info.get('abstract_entities', [])
            for entity in abstract_entities:
                entity_type = entity.get('Type', 'unknown')
                embedding_idx = self.get_feature_embedding_idx('user_history_abstract_entity_type', entity_type)
                history_entity_types.append(str(embedding_idx))
        
        extracted_features['user_history_abstract_entity_type'] = ','.join(history_entity_types)
        self.user_history_abstract_entity_type_cache[impression_id] = extracted_features['user_history_abstract_entity_type']

    # 提取标签，返回一个列表形式
    def label_extractor(self, data_line):
        return [data_line['label']]
    

if __name__ == "__main__":
    import argparse
    from .feature_extractor_base import FeatureExtractorBase

    parser = argparse.ArgumentParser(description="Feature Extractor")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'debug'], help='Run mode: normal or debug')
    args = parser.parse_args()

    # 加载配置文件
    config = OmegaConf.load(args.config)
    config.run_mode = args.mode

    # 初始化FeatureExtractor
    feature_extractor = FeatureExtractor(config)
    # 执行特征提取
    feature_extractor.run()
