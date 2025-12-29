import os
import shutil
import logging
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import pyjson5 as json
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from ...Logger.logging import Logger

# 配置日志
logger = Logger.get_logger("FeatureExtractor")

class FeatureExtractorBase(ABC):
    """
    特征提取基类 (Base Feature Extractor)
    
    职责：
    1. 管理配置与路径
    2. 加载 Item (News) 基础数据
    3. 维护 Feature Name 到 Embedding Index 的映射字典
    4. 调度数据读取与特征提取流程
    
    注意：这是一个抽象基类，具体的特征提取逻辑 (feature_extractor_*) 需要在子类中实现。
    """

    def __init__(self, config: DictConfig):
        """
        初始化特征提取器
        
        Args:
            config (DictConfig): 由 OmegaConf 加载的配置对象
        """
        self.config = config
        
        # --- 1. 路径配置 ---
        self.out_basedir = Path(config.paths.out_basedir)
        self.user_history_path = config.paths.get('user_history_path', None)
        
        # --- 2. 特征配置 ---
        self.features_cfg = config.features
        self.feature_names: List[str] = self.features_cfg.feature_names
        self.item_feature_names: List[str] = self.features_cfg.get('item_feature_names', [])
        self.share_emb_table_features: Dict[str, str] = self.features_cfg.get('share_emb_table_features', {})
        self.array_max_length: Dict[str, int] = self.features_cfg.get('array_max_length', {})
        self.array_feature_names: List[str] = self.features_cfg.get('array_feature_names', [])
        
        # --- 3. 内部状态初始化 ---
        self.item_data_dict: Dict[int, Dict[str, Any]] = {}
        
        # 映射字典: Feature Name -> ({原始值: ID}, 当前最大ID)
        # 结构: {'user_id': ({'u1': 1, 'u2': 2}, 2), ...}
        self.feature_map_val2idx: Dict[str, list] = {fea: [{}, 0] for fea in self.feature_names}
        
        # 反向映射: Feature Name -> {ID: 原始值}
        # 结构: {'user_id': {1: 'u1', 2: 'u2'}, ...}
        self.feature_map_idx2val: Dict[str, Dict[int, Any]] = {fea: {} for fea in self.feature_names}

        # --- 4. 执行初始化流程 ---
        self._validate_config()
        self._resolve_paths()
        self._prepare_output_dir()
        
        # 加载数据
        self._load_item_data()
        
        # 子类钩子：用于初始化某些特定的逻辑（如加载预训练词向量等）
        self.initialization()

    def initialization(self):
        """子类可重写此方法以进行自定义初始化"""
        pass

    @abstractmethod
    def label_extractor(self, data_line: Dict[str, Any]) -> List[int]:
        """
        [抽象方法] 从数据行中提取标签
        必须在子类中实现
        """
        pass

    def _validate_config(self):
        """验证配置文件的完整性"""
        for fea in self.array_feature_names:
            if fea not in self.array_max_length:
                raise ValueError(f"Array feature '{fea}' declared but max_length not defined in config.")
        
        if not self.out_basedir:
            raise ValueError("out_basedir is missing in config.")

    def _resolve_paths(self):
        """解析并组装相关文件路径"""
        preprocess_dir = self.out_basedir / 'preprocess'
        self.item_path = preprocess_dir / 'all_news_preprocess.csv'
        self.train_behavior_path = preprocess_dir / 'train_behaviors_processed.csv'
        self.val_behavior_path = preprocess_dir / 'dev_behaviors_processed.csv'
        self.output_feature_dir = self.out_basedir / 'extractored_feature'

    def _prepare_output_dir(self):
        """准备输出目录，如果存在则清理重建"""
        if self.output_feature_dir.exists():
            logger.warning(f"Cleaning existing output directory: {self.output_feature_dir}")
            shutil.rmtree(self.output_feature_dir)
        self.output_feature_dir.mkdir(parents=True, exist_ok=True)

    def _load_item_data(self):
        """加载新闻/物品基础数据到内存"""
        if not self.item_path.exists():
            raise FileNotFoundError(f"Item data not found at {self.item_path}")

        logger.info(f"Loading item data from {self.item_path}...")
        with open(self.item_path, 'r', encoding='utf-8') as f:
            # 这里的ncols=100是为了防止进度条在某些终端换行
            for line in tqdm(f, desc="Reading Items", ncols=100):
                try:
                    parts = line.strip().split('\t')
                    # 确保数据列数符合预期，防止IndexError
                    if len(parts) < 8: 
                        continue
                        
                    news_id = int(parts[0])
                    self.item_data_dict[news_id] = {
                        'news_id': news_id,
                        'category': parts[1],
                        'subcategory': parts[2],
                        'title': parts[3],
                        'abstract': parts[4],
                        'url': parts[5],
                        'title_entities': parts[6],
                        'abstract_entities': parts[7]
                    }
                except ValueError as e:
                    logger.warning(f"Skipping malformed line in item data: {e}")

    def get_feature_embedding_idx(self, feature_name: str, feature_value: Any) -> int:
        """
        获取特征值的 Embedding 索引。
        如果是新值，会自动分配新的 ID 并更新映射表。
        
        Args:
            feature_name: 特征名称
            feature_value: 特征原始值
        
        Returns:
            int: 映射后的 ID
        """
        # 处理共享 Embedding 表的情况
        target_map_name = self.share_emb_table_features.get(feature_name, feature_name)
        
        if target_map_name not in self.feature_map_val2idx:
             # 如果配置了 feature_names 但没在代码里处理，可能会报错，这里做个防御
             raise KeyError(f"Feature name '{target_map_name}' not found in initialization map.")

        val2idx_entry = self.feature_map_val2idx[target_map_name]
        val_dict, current_max_idx = val2idx_entry[0], val2idx_entry[1]

        if feature_value not in val_dict:
            # 分配新 ID (从 1 开始，0通常留给 padding/unknown)
            new_idx = current_max_idx + 1
            val_dict[feature_value] = new_idx
            
            # 更新状态
            val2idx_entry[1] = new_idx
            self.feature_map_idx2val[target_map_name][new_idx] = feature_value
            return new_idx
        
        return val_dict[feature_value]

    def _extract_single_row(self, data_context: Dict[str, Any]) -> Tuple[str, str]:
        """
        处理单行数据：提取特征并生成 Label
        
        Args:
            data_context: 包含 user_info, item_info, timestamp 等的上下文字典
            
        Returns:
            Tuple[str, str]: (特征字符串, Label字符串)
        """
        extracted_features = {}
        
        # 1. 动态调用特征提取函数
        for fea in self.feature_names:
            func_name = f"feature_extractor_{fea}"
            if not hasattr(self, func_name):
                raise NotImplementedError(f"Method '{func_name}' required for feature '{fea}' is not implemented.")
            
            extractor_func = getattr(self, func_name)
            # 这里调用子类实现的具体逻辑，通常会调用 self.get_feature_embedding_idx
            extractor_func(data_context, extracted_features)

        # 2. 提取 Label
        labels = self.label_extractor(data_context)

        # 3. 格式化输出 (LibSVM-like: feature_name:feature_id)
        # 注意：这里仅作示例，具体格式需根据下游模型调整
        feature_str = ' '.join([f"{k}:{v}" for k, v in extracted_features.items()])
        label_str = ' '.join(map(str, labels))
        
        return feature_str, label_str

    def _process_behavior_file(self, input_path: Path):
        """处理用户行为文件 (Train/Val)"""
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return

        output_filename = input_path.stem.split('_')[0] + '_features.txt' # e.g., train_features.txt
        output_path = self.output_feature_dir / output_filename
        
        logger.info(f"Processing behaviors: {input_path} -> {output_path}")

        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in tqdm(fin, desc=f"Extracting {input_path.stem}", ncols=100):
                try:
                    # 解析原始行
                    parts = line.strip().split('\t')
                    # 假设格式: req_id, uid, time, hist, item_id, label
                    user_id = int(parts[1])
                    timestamp = int(parts[2])
                    history_str = parts[3]
                    history = [int(x) for x in history_str.split(' ')] if history_str else []
                    item_id = int(parts[4])
                    label = int(parts[5])

                    item_info = self.item_data_dict.get(item_id, {})
                    
                    # 构建上下文对象
                    data_context = {
                        'item_info': item_info,
                        'user_info': {
                            'user_id': user_id,
                            'history': history
                        },
                        'timestamp': timestamp,
                        'label': label
                    }
                    
                    # 提取特征
                    feat_str, label_str = self._extract_single_row(data_context)
                    fout.write(f"{feat_str}\t{label_str}\n")
                    
                except Exception as e:
                    logger.error(f"Error processing line: {line[:50]}... Error: {e}")
                    continue

    def _extract_item_features_only(self):
        """单独提取物品特征 (用于 Item Tower 或 推理)"""
        output_path = self.output_feature_dir / 'item_features.txt'
        logger.info(f"Extracting static item features to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as fout:
            for _, item_info in tqdm(self.item_data_dict.items(), desc="Extracting Item Feats", ncols=100):
                extracted = {}
                # 仅提取 item_feature_names 中定义的特征
                for fea in self.item_feature_names:
                    func_name = f"feature_extractor_{fea}"
                    if hasattr(self, func_name):
                        getattr(self, func_name)({'item_info': item_info}, extracted)
                
                # 格式化
                feat_str = ' '.join([f"{k}:{v}" for k, v in extracted.items()])
                # Item Feature 通常没有 Label，这里用 -1 占位
                fout.write(f"{feat_str}\t-1\n")

    def _save_mappings(self):
        """保存 ID 映射表，供推理使用"""
        logger.info("Saving feature mappings...")
        
        # 保存 val -> idx
        with open(self.output_feature_dir / 'original_val_2_embedding_idx_dict.json', 'w', encoding='utf-8') as f:
            # json 不支持 tuple key，直接 dump 整个结构
            json.dump(self.feature_map_val2idx, f, indent=2)
            
        # 保存 idx -> val
        with open(self.output_feature_dir / 'embedding_idx_2_original_val_dict.json', 'w', encoding='utf-8') as f:
            json.dump(self.feature_map_idx2val, f, indent=2)
            
        # 保存本次运行的配置
        with open(self.output_feature_dir / 'dataset_extract_info.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(OmegaConf.to_container(self.config), f)

    def run(self):
        """
        [主入口] 执行完整的特征提取流程
        """
        logger.info(">>> Starting Feature Extraction Pipeline <<<")
        
        # 1. 处理训练集
        self._process_behavior_file(self.train_behavior_path)
        
        # 2. 处理验证集
        self._process_behavior_file(self.val_behavior_path)
        
        # 3. 处理物品特征
        self._extract_item_features_only()
        
        # 4. 保存映射表
        self._save_mappings()
        
        logger.info(">>> Pipeline Completed Successfully <<<")

# ==========================================
# 使用示例 (子类实现)
# ==========================================
# class MyMINDFeatureExtractor(FeatureExtractorBase):
#     def label_extractor(self, data_line):
#         return [data_line['label']]
#
#     def feature_extractor_user_id(self, context, output_dict):
#         uid = context['user_info']['user_id']
#         output_dict['user_id'] = self.get_feature_embedding_idx('user_id', uid)
#
#     def feature_extractor_movie_id(self, context, output_dict):
#         # ... 实现逻辑
#         pass

