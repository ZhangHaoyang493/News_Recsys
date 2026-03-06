import sys
import os
import logging
import math
import json
from typing import Dict, Any, List, Optional, Set, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities.model_summary import ModelSummary

from ..model_utils.FeatureIdMapper import FeatureIdMapper
from ...Logger.logging import Logger

# 配置 logging
logger = Logger.get_logger("BaseModel")

class BaseModel(L.LightningModule):
    """
    推荐系统基础模型类 (Base Recommendation Model) - YAML 配置版
    作为 BaseModelSort 和 BaseModelRecall 的父类，封装共有逻辑。
    """

    def __init__(self, config_or_path: Union[str, DictConfig, Dict]):
        """
        初始化 BaseModel

        Args:
            config_or_path: 模型配置文件路径 (str) 或 配置对象 (Dict/DictConfig)。
        """
        super().__init__()
        self._load_config(config_or_path)
        self._validate_config()

        # 构建 Embedding 层
        self.share_emb_table_features_dict = {}
        self._build_all_embedding_tables()

        # 计算输入维度
        self.item_input_dim = self._calculate_input_dim(self.share_emb_table_features_dict['base_embedding_table'], self.item_feature_names)
        self.user_input_dim = self._calculate_input_dim(self.share_emb_table_features_dict['base_embedding_table'], self.user_feature_names)
        logger.info(f"Input Dimensions - Item: {self.item_input_dim}, User: {self.user_input_dim}")
        
        # 初始化训练状态变量
        self._init_metrics_state()
        
        # 保存超参数
        self.save_hyperparameters(OmegaConf.to_container(self.config, resolve=True))
        
        # 特征 ID 映射器 (延迟加载)
        self.feature_id_mapper = None 

    def _load_config(self, config_or_path):
        """加载 YAML 配置文件并解析参数"""
        if isinstance(config_or_path, str):
            if not os.path.exists(config_or_path):
                raise FileNotFoundError(f"Config file not found: {config_or_path}")
            self.config = OmegaConf.load(config_or_path)
        else:
            self.config = config_or_path

        # --- 1. Paths ---
        paths_cfg = self.config.get('paths', {})
        self.out_basedir: str = paths_cfg.get('out_basedir', '')
        self.user_history_path: str = paths_cfg.get('user_history_path', '')

        # --- 2. Features ---
        features_cfg = self.config.get('features', {})
        self.sparse_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('sparse_feature_names', []), resolve=True) or [])
        self.dense_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('dense_feature_names', []), resolve=True) or [])
        self.array_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('array_feature_names', []), resolve=True) or [])
        
        self.item_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('item_feature_names', []), resolve=True) or [])
        self.user_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('user_feature_names', []), resolve=True) or [])
        
        self.array_max_length: Dict[str, int] = OmegaConf.to_container(features_cfg.get('array_max_length', {}), resolve=True) or {}
        self.dense_feature_dim: int = features_cfg.get('dense_feature_dim', 1)

        # --- 3. Embeddings ---
        emb_cfg = self.config.get('embeddings', {})
        self.embedding_table_size: Dict[str, int] = OmegaConf.to_container(emb_cfg.get('embedding_table_size', {}), resolve=True) or {}
        self.embedding_tables_cfg = OmegaConf.to_container(emb_cfg.get('embedding_tables', {}), resolve=True) or {}


        # --- 4. Dataset ---
        self.dataset_cfg = self.config.get('dataset', {})

        # --- 5. Train Hyperparams ---
        self.train_hparams = self.config.get('train_hparams', {})

    def _validate_config(self):
        """校验关键配置是否存在"""
        if not self.out_basedir:
            logger.warning("out_basedir is not set in config.")

        if 'base_embedding_table' not in self.embedding_tables_cfg:
            logger.error("base_embedding_table is not set in config.")


    def _get_emb_feature_name(self, share_emb_table_features: Dict[str, str], feature_name: str) -> str:
        return share_emb_table_features.get(feature_name, feature_name)

    def _calculate_input_dim(self, share_emb_table_features: Dict[str, str], feature_names: Set[str]) -> int:
        total_dim = 0
        for fname in feature_names:
            if fname in self.dense_feature_names:
                total_dim += self.dense_feature_dim
            else:
                emb_fname = self._get_emb_feature_name(share_emb_table_features, fname)
                dim = self.embedding_tables_cfg['base_embedding_table']['embedding_dims'].get(emb_fname)
                if dim is None:
                    logger.error(f"Feature '{fname}' mapped to '{emb_fname}' has no embedding size config.")
                    raise ValueError(f"Feature '{fname}' mapped to '{emb_fname}' has no embedding size config.")
                total_dim += dim
        return total_dim

    def _build_embedding_table(
        self, 
        feature_names: Set[str], 
        embedding_table_size: Dict[str, int], 
        embedding_size: Dict[str, int],
        pretrain_embedding: Dict[str, str] = {},
        share_emb_table_features: Dict[str, str] = {}
        ) -> nn.ModuleDict:
        tables = nn.ModuleDict()
        all_emb_features = feature_names
        
        for fname in all_emb_features:
            emb_fname = self._get_emb_feature_name(share_emb_table_features, fname)
            if emb_fname in self.dense_feature_names: continue
            if emb_fname in tables: continue
            if emb_fname in pretrain_embedding:
                emb_path = pretrain_embedding[emb_fname]
                if not os.path.exists(emb_path):
                    logger.error(f"Pretrained embedding file not found for {emb_fname}: {emb_path}")
                    raise FileNotFoundError(f"Pretrained embedding file not found for {emb_fname}: {emb_path}")
                
                loaded_data = torch.load(emb_path, map_location='cpu')
                
                if isinstance(loaded_data, torch.Tensor):
                    # 如果加载的是 Tensor，则使用 from_pretrained
                    tables[emb_fname] = nn.Embedding.from_pretrained(loaded_data, freeze=True, padding_idx=0)
                elif isinstance(loaded_data, nn.Embedding):
                    # 如果加载的是 nn.Embedding 直接使用
                    tables[emb_fname] = loaded_data
                    tables[emb_fname].weight.requires_grad = False
                else:
                    raise TypeError(f"Unsupported pretrained embedding type: {type(loaded_data)}")
                
                logger.info(f"Loaded pretrained embedding for {emb_fname} from {emb_path}")
                continue
                
            size = embedding_table_size.get(emb_fname)
            dim = embedding_size.get(emb_fname)

            if size is None or dim is None:
                logger.error(f"Missing embedding config (size/dim) for feature: {emb_fname}")
                raise ValueError(f"Missing embedding config (size/dim) for feature: {emb_fname}")
            
            tables[emb_fname] = nn.Embedding(size, dim, padding_idx=0)
        return tables

    def _build_all_embedding_tables(self):
        self.embedding_tables = nn.ModuleDict()
        for table_name, table_cfg in self.embedding_tables_cfg.items():
            embedding_dims = table_cfg.get('embedding_dims', {})
            feature_names = set(embedding_dims.keys())
            share_emb_table_features = table_cfg.get('share_emb_table_features', {})
            pretrain_embedding = table_cfg.get('pretrain_embedding', {})
            
            tables = self._build_embedding_table(feature_names, self.embedding_table_size, embedding_dims, pretrain_embedding, share_emb_table_features)
            self.embedding_tables[table_name] = tables
            self.share_emb_table_features_dict[table_name] = share_emb_table_features

    def _init_metrics_state(self):
        # 基类只初始化容器，具体指标由子类定义
        self.best_metrics = {}
        # 记录训练损失历史，用于绘图
        self.train_loss_history = []
        self.train_step_history = []

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每一步结束时记录 Loss"""
        if outputs is None: return
        
        # 获取 Loss 值
        loss = None
        if isinstance(outputs, torch.Tensor):
            loss = outputs.item()
        elif isinstance(outputs, dict):
            if 'loss' in outputs:
                loss = outputs['loss'].item()
        
        if loss is not None:
            self.train_loss_history.append(loss)
            self.train_step_history.append(self.global_step)

    def setup(self, stage: str):
        if self.logger and self.logger.log_dir:
            self.log_dir = self.logger.log_dir
        else:
            self.log_dir = self.out_basedir if self.out_basedir else "./logs"
            
        self.ckpt_dir = os.path.join(self.log_dir, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        val_log_path = os.path.join(self.log_dir, "val_log.log")
        if not os.path.exists(val_log_path):
            with open(val_log_path, 'w') as f: pass

        self.res_show_path = os.path.join(self.log_dir, "result_record")
        if not os.path.exists(self.res_show_path):
            os.makedirs(self.res_show_path, exist_ok=True)
        

        if stage == 'fit':
            emb_idx_2_val = os.path.join(self.out_basedir, 'extractored_feature', 'embedding_idx_2_original_val_dict.json')
            val_2_emb_idx = os.path.join(self.out_basedir, 'extractored_feature', 'original_val_2_embedding_idx_dict.json')
            
            if os.path.exists(emb_idx_2_val) and os.path.exists(val_2_emb_idx):
                self.feature_id_mapper = FeatureIdMapper(emb_idx_2_val, val_2_emb_idx)
            else:
                logger.warning(f"Feature ID Mapper files not found at {self.out_basedir}/extractored_feature/")
            
        self.user_in_train_path = os.path.join(self.out_basedir, 'preprocess', 'train_user_ids.json')
        if os.path.exists(self.user_in_train_path):
            with open(self.user_in_train_path, 'r') as f:
                self.user_in_train_set = set(json.load(f))
        else:
            self.user_in_train_set = set()

        if self.trainer.is_global_zero:
            summary = ModelSummary(self, max_depth=3)
            with open(os.path.join(self.log_dir, 'model_info.log'), "w") as f:
                f.write('\n' + str(summary) + '\n') 

    def on_train_start(self):
        if self.logger:
            log_dir = self.logger.log_dir or self.out_basedir
            self.model_save_path = os.path.join(log_dir, 'checkpoints')
            self.log_file_path = os.path.join(log_dir, 'training_log.log')
            os.makedirs(self.model_save_path, exist_ok=True)

    def on_train_epoch_end(self):
        if not self.trainer or not hasattr(self, 'log_dir'): return
        metrics = self.trainer.callback_metrics
        train_metrics = {k: v.item() for k, v in metrics.items() if not k.startswith('val_')}
        if not train_metrics: return

        log_msg = f"Epoch {self.current_epoch} Training Metrics:\n"
        for k, v in train_metrics.items():
            log_msg += f"  {k}: {v:.4f}\n"
        log_msg += "-" * 20 + "\n"

        try:
            with open(os.path.join(self.log_dir, "train.log"), "a") as f:
                f.write(log_msg)
        except Exception as e:
            print(f"Failed to write train log: {e}")

        # --- 绘制 Loss 曲线 ---
        if self.train_loss_history:
            try:
                import matplotlib.pyplot as plt
                
                loss_fig_dir = os.path.join(self.log_dir, "loss_figure")
                os.makedirs(loss_fig_dir, exist_ok=True)
                
                plt.figure(figsize=(10, 6))
                plt.plot(self.train_step_history, self.train_loss_history, label='Training Loss')
                plt.xlabel('Global Step')
                plt.ylabel('Loss')
                plt.title(f'Training Loss Curve (Epoch {self.current_epoch})')
                plt.legend()
                plt.grid(True)
                
                save_path = os.path.join(loss_fig_dir, f"loss_curve_epoch_{self.current_epoch}.png")
                plt.savefig(save_path)
                plt.close() # 释放内存
                logger.info(f"Loss curve saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to plot loss curve: {e}")

    def get_feature_embedding(self, embedding_tables_name: str, feature_name: str, feature_value: torch.Tensor) -> torch.Tensor:
        if feature_name in self.dense_feature_names:
            return feature_value.float().unsqueeze(1)
            
        emb_fname = self._get_emb_feature_name(self.share_emb_table_features_dict[embedding_tables_name], feature_name)
        if emb_fname not in self.embedding_tables[embedding_tables_name]:
             raise ValueError(f"Embedding table not found for {feature_name} (mapped to {emb_fname})")
        
        return self.embedding_tables[embedding_tables_name][emb_fname](feature_value.long())

    def array_feature_pooling(self, embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return embedding.mean(dim=1)
        mask = mask.unsqueeze(-1)
        masked_emb = embedding * mask
        sum_emb = masked_emb.sum(dim=1) 
        sum_mask = mask.sum(dim=1) + 1e-8 
        return sum_emb / sum_mask

    def get_embeddings_from_batch(self, embedding_tables_name: str, batch: Dict[str, torch.Tensor], feature_names: Set[str]) -> Tuple[torch.Tensor, List[int], List[str]]:
        sorted_features = sorted(list(feature_names))
        emb_list = []
        dims = []
        
        for fname in sorted_features:
            if fname not in batch:
                logger.error(f"Feature '{fname}' not found in batch.")
                raise ValueError(f"Feature '{fname}' not found in batch.")
            
            if fname in self.array_feature_names:
                mask = batch.get(f"{fname}_mask", None)

            val = batch[fname] if not (fname in self.array_feature_names) else batch[fname] * mask
            emb = self.get_feature_embedding(embedding_tables_name, fname, val)
            
            if fname in self.array_feature_names:
                emb = self.array_feature_pooling(emb, mask)
            
            emb_list.append(emb)
            dims.append(emb.shape[1])
            
        if not emb_list:
            return torch.tensor([]).to(self.device), [], []

        return torch.cat(emb_list, dim=1), dims, sorted_features

    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info(f"Loading weights from {path}...")
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")


    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def on_validation_start(self):
        # 保存模型
        if not hasattr(self, 'ckpt_dir'):
            logger.warning("ckpt_dir not set. Skipping model save on validation start.")
            raise RuntimeWarning("ckpt_dir not set. Skipping model save on validation start.")
        ckpt_path = os.path.join(self.ckpt_dir, f'epoch_{self.current_epoch}_step_{self.global_step}.ckpt')
        self.save_model(ckpt_path)