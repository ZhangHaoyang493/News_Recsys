import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import os.path as osp

from ....BaseModel.base_model_recall import BaseModelRecall
from ....model_utils.lr_schedule import CosinDecayLR
from .....Logger.logging import Logger

logger = Logger.get_logger("InvertIndex")

# 倒排索引召回
class InvertIndex(BaseModelRecall):
    def __init__(self, config):
        super().__init__(config)

        self.train_file_path = osp.join(self.out_basedir, 'extractored_feature', 'train_features.txt')
        self.item_file_path = osp.join(self.out_basedir, 'extractored_feature', 'item_features.txt')
        self.test_file_path = osp.join(self.out_basedir, 'extractored_feature', 'dev_features.txt')
        self.k = self.config.k
        self.only_test_warm_up_user = self.config.dataset.get('only_test_warm_up_user', False)

        self.get_item_click_num_info()
        self.build_invert_index()
        self.sort_invert_index()
        

    def build_invert_index(self):
        self.category_invert_index = {}
        self.subcategory_invert_index = {}
        self.item_category_subcategory = {}

        logger.info("Building invert index...")

        with open(self.item_file_path, 'r') as f:
            for line in f:
                features_, labels_ = line.strip().split('\t')
                features = features_.split(' ')
                features_dict = {}
                for fea in features:
                    key, value = fea.split(':')
                    features_dict[key] = float(value)
                category = int(features_dict.get('category', -1))
                subcategory = int(features_dict.get('subcategory', -1))
                item_id = int(features_dict.get('item_id', -1))

                if category not in self.category_invert_index:
                    self.category_invert_index[category] = []
                self.category_invert_index[category].append(item_id)
                if subcategory not in self.subcategory_invert_index:
                    self.subcategory_invert_index[subcategory] = []
                self.subcategory_invert_index[subcategory].append(item_id)

                self.item_category_subcategory[item_id] = (category, subcategory)


    def sort_invert_index(self):
        logger.info("Sorting invert index...")
        for category, item_list in self.category_invert_index.items():
            item_list.sort(key=lambda x: self.item_click_num.get(x, 0), reverse=True)
        for subcategory, item_list in self.subcategory_invert_index.items():
            item_list.sort(key=lambda x: self.item_click_num.get(x, 0), reverse=True)

    def get_item_click_num_info(self):
        logger.info("Getting item click num info...")

        self.item_click_num = {}
        if self.only_test_warm_up_user:
            self.train_user = set()
        with open(self.train_file_path, 'r') as f:
            for line in f:
                features_, labels_ = line.strip().split('\t')
                features = features_.split(' ')
                features_dict = {}
                for fea in features:
                    key, value = fea.split(':')
                    features_dict[key] = value
                user_id = int(features_dict.get('user_id', -1))
                item_id = int(features_dict.get('item_id', -1))
                if item_id not in self.item_click_num:
                    self.item_click_num[item_id] = 0
                self.item_click_num[item_id] += 1
                if self.only_test_warm_up_user:
                    self.train_user.add(user_id)

    def get_user_click_category_and_subcategory(self, user_history):
        categories, subcategories = {}, {}
        for item_id in user_history:
            if item_id in self.item_category_subcategory:
                category, subcategory = self.item_category_subcategory[item_id]
                categories[category] = categories.get(category, 0) + 1
                subcategories[subcategory] = subcategories.get(subcategory, 0) + 1
        return categories, subcategories
    

    def get_recall_res(self):
        self.recall_res = {}
        logger.info("Getting recall results...")
        with open(self.test_file_path, 'r') as f:
            for line in f:
                features_, labels_ = line.strip().split('\t')
                features = features_.split(' ')
                features_dict = {}
                for fea in features:
                    key, value = fea.split(':')
                    features_dict[key] = value
                user_id = int(features_dict.get('user_id', -1))
                if self.only_test_warm_up_user and user_id not in self.train_user:
                    continue
                item_id = int(features_dict.get('item_id', -1))
                impression_id = int(features_dict.get('impression_id', -1))
                user_history_str = features_dict.get('user_history', '')

                if impression_id not in self.recall_res:


                    if user_history_str:
                        user_history = [int(x) for x in user_history_str.split(',')]
                    else:
                        user_history = []

                    self.recall_res[impression_id] = {'uid': user_id, 'recall': set(), 'target': []}

                    categories, subcategories = self.get_user_click_category_and_subcategory(user_history)

                    recall_res_now = {}

                    # 根据类别倒排索引进行召回
                    for category in categories:
                        if category in self.category_invert_index:
                            for rec_item in self.category_invert_index[category][:self.k*2]:  # 提前多取一些，避免用户历史重复过多导致召回不足
                                if rec_item not in user_history:
                                    recall_res_now[rec_item] = self.item_click_num.get(rec_item, 0) * categories[category]

                    recall_res_now_list = sorted(recall_res_now.items(), key=lambda x: x[1], reverse=True)[:self.k]
                    for rec_item, _ in recall_res_now_list:
                        self.recall_res[impression_id]['recall'].add(rec_item)

                if int(labels_) == 1:
                    self.recall_res[impression_id]['target'].append(item_id)

        
    def get_recall_metrics(self):
        self.get_recall_res()
        self.on_validation_epoch_end()
