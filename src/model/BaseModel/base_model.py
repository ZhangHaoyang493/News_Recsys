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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities.model_summary import ModelSummary

from ..model_utils.FeatureIdMapper import FeatureIdMapper
from ...Logger.logging import Logger

# 配置 logging
logger = Logger.get_logger("BaseModel")

class BaseModel(L.LightningModule):
    """
    推荐系统基础模型类 (Base Recommendation Model) - YAML 配置版

    功能：
    1. 负责 Embedding 层的统一构建与管理 (支持 Sparse, Dense, Array 特征)。
    2. 提供特征 Embedding 的查询接口。
    3. 封装通用的训练、验证、测试和指标计算流程。
    4. 集成 PyTorch Lightning。
    """

    def __init__(self, config_path: str):
        """
        初始化 BaseModel

        Args:
            config_path (str): 模型配置文件 (YAML) 的路径。
        """
        super(BaseModel, self).__init__()
        self._load_config(config_path)
        self._validate_config()

        # 计算输入维度 (用于构建后续的 MLP 或 Attention 层)
        self.item_input_dim = self._calculate_input_dim(self.item_feature_names)
        self.user_input_dim = self._calculate_input_dim(self.user_feature_names)
        logger.info(f"Input Dimensions - Item: {self.item_input_dim}, User: {self.user_input_dim}")

        # 构建 Embedding 层
        self.embedding_tables = self._build_embedding_tables()

        # 加载用户历史 (可选)
        if self.user_history_path and os.path.exists(self.user_history_path):
            import json # 历史文件通常还是json
            with open(self.user_history_path, 'r') as f:
                self.user_history = json.load(f)
        
        # 初始化训练状态变量
        self._init_metrics_state()
        
        # 保存超参数供 PL 自动记录 (OmegaConf对象需要转为dict或直接保存)
        self.save_hyperparameters(OmegaConf.to_container(self.config, resolve=True))
        
        # 特征 ID 映射器 (延迟加载)
        self.feature_id_mapper = None 

    def _load_config(self, config_path: str):
        """加载 YAML 配置文件并解析参数"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # 使用 OmegaConf 加载 YAML
        self.config = OmegaConf.load(config_path)

        # --- 1. Paths ---
        paths_cfg = self.config.get('paths', {})
        self.out_basedir: str = paths_cfg.get('out_basedir', '')
        self.user_history_path: str = paths_cfg.get('user_history_path', '')

        # --- 2. Features ---
        features_cfg = self.config.get('features', {})
        # 使用 OmegaConf 的 to_container 转为标准 list，避免类型问题
        self.sparse_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('sparse_feature_names', []), resolve=True) or [])
        self.dense_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('dense_feature_names', []), resolve=True) or [])
        self.array_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('array_feature_names', []), resolve=True) or [])
        
        self.item_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('item_feature_names', []), resolve=True) or [])
        self.user_feature_names: Set[str] = set(OmegaConf.to_container(features_cfg.get('user_feature_names', []), resolve=True) or [])
        
        self.array_max_length: Dict[str, int] = OmegaConf.to_container(features_cfg.get('array_max_length', {}), resolve=True) or {}
        # 你的 yaml 里没有 dense_feature_dim，我先设个默认值，如果 yaml 后续加了这个字段，可以从 features_cfg 获取
        # self.dense_feature_dim: int = features_cfg.get('dense_feature_dim', 1) 

        # --- 3. Embeddings ---
        emb_cfg = self.config.get('embeddings', {})
        self.embedding_size: Dict[str, int] = OmegaConf.to_container(emb_cfg.get('embedding_size', {}), resolve=True) or {}
        self.embedding_table_size: Dict[str, int] = OmegaConf.to_container(emb_cfg.get('embedding_table_size', {}), resolve=True) or {}
        self.share_emb_table_features: Dict[str, str] = OmegaConf.to_container(emb_cfg.get('share_emb_table_features', {}), resolve=True) or {}

        # --- 4. Dataset (如有需要可在 DataModule 中用，模型里可能用不到，暂存) ---
        self.dataset_cfg = self.config.get('dataset', {})

        # --- 5. Train Hyperparams ---
        self.train_hparams = self.config.get('train_hparams', {})

    def _validate_config(self):
        """校验关键配置是否存在"""
        if not self.out_basedir:
            logger.warning("out_basedir is not set in config.")
        
        # 简单的校验：确保稀疏特征都有对应的 embedding size 配置
        for fname in self.sparse_feature_names:
            emb_name = self._get_emb_feature_name(fname)
            if emb_name not in self.embedding_size:
                logger.warning(f"Embedding size for feature '{emb_name}' (from '{fname}') is missing!")

    def _get_emb_feature_name(self, feature_name: str) -> str:
        """获取特征对应的 Embedding 表名 (处理共享 Embedding 的情况)"""
        # 注意：YAML 解析后的 Dict key 查找
        return self.share_emb_table_features.get(feature_name, feature_name)

    def _calculate_input_dim(self, feature_names: Set[str]) -> int:
        """计算指定特征集合的总输入维度"""
        total_dim = 0
        for fname in feature_names:
            if fname in self.dense_feature_names:
                total_dim += self.dense_feature_dim
            else:
                # Sparse 或 Array 特征
                emb_fname = self._get_emb_feature_name(fname)
                dim = self.embedding_size.get(emb_fname)
                if dim is None:
                    # 如果配置里找不到，给个默认值或者报错，这里为了稳健给个默认值并打印警告
                    logger.warning(f"Feature '{fname}' mapped to '{emb_fname}' has no embedding size config. Using default 8.")
                    dim = 8
                total_dim += dim
        return total_dim

    def _build_embedding_tables(self) -> nn.ModuleDict:
        """构建所有需要的 Embedding 表"""
        tables = nn.ModuleDict()
        
        # 合并需要 Embedding 的特征 (Sparse + Array)
        all_emb_features = self.sparse_feature_names.union(self.array_feature_names)
        
        for fname in all_emb_features:
            emb_fname = self._get_emb_feature_name(fname)
            
            # 避免重复创建共享表
            if emb_fname in tables:
                continue
                
            size = self.embedding_table_size.get(emb_fname)
            dim = self.embedding_size.get(emb_fname)

            if size is None or dim is None:
                # 可以在这里抛出错误或者记录错误
                logger.error(f"Missing embedding config (size/dim) for feature: {emb_fname}")
                continue
            
            # padding_idx=0 是常见做法
            tables[emb_fname] = nn.Embedding(size, dim, padding_idx=0)
            
        return tables

    def _init_metrics_state(self):
        """初始化评估指标状态"""
        self.best_metrics = {
            'AUC': 0.0,
            'LogLoss': float('inf'),
            'GAUC': 0.0,
            'HR@10': 0.0,
            'NDCG@10': 0.0,
            'MRR@10': 0.0,
            'Step': -1
        }
        self.user_scores_dict = {}

    def setup(self, stage: str):
        """PL 生命周期钩子"""
        # 确定 log_dir 并创建相关文件/文件夹
        if self.logger and self.logger.log_dir:
            self.log_dir = self.logger.log_dir
        else:
            self.log_dir = self.out_basedir if self.out_basedir else "./logs"
            
        # 创建 ckpts 文件夹
        self.ckpt_dir = os.path.join(self.log_dir, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # 创建 val_log.log 文件
        val_log_path = os.path.join(self.log_dir, "val_log.log")
        if not os.path.exists(val_log_path):
            with open(val_log_path, 'w') as f:
                pass

        if stage == 'fit':
            # 构造字典文件路径
            emb_idx_2_val = os.path.join(self.out_basedir, 'extractored_feature', 'embedding_idx_2_original_val_dict.json')
            val_2_emb_idx = os.path.join(self.out_basedir, 'extractored_feature', 'original_val_2_embedding_idx_dict.json')
            
            if os.path.exists(emb_idx_2_val) and os.path.exists(val_2_emb_idx):
                self.feature_id_mapper = FeatureIdMapper(emb_idx_2_val, val_2_emb_idx)
            else:
                logger.warning(f"Feature ID Mapper files not found at {self.out_basedir}/extractored_feature/")
            
        self.user_in_train_path = os.path.join(self.out_basedir, 'preprocess', 'train_user_ids.json')
        with open(self.user_in_train_path, 'r') as f:
            self.user_in_train_set = set(json.load(f))


        # 将模型结构表写入日志
        if self.trainer.is_global_zero:
            summary = ModelSummary(self, max_depth=2)
            with open(os.path.join(self.log_dir, 'model_info.log'), "w") as f:
                f.write('\n' + str(summary) + '\n') 


    def on_train_start(self):
        """训练开始时设置日志路径"""
        if self.logger:
            log_dir = self.logger.log_dir or self.out_basedir
            self.model_save_path = os.path.join(log_dir, 'checkpoints')
            self.log_file_path = os.path.join(log_dir, 'training_log.log')
            os.makedirs(self.model_save_path, exist_ok=True)

    def on_train_epoch_end(self):
        """在每个训练轮次结束后被调用，记录训练指标到 train.log"""
        if not self.trainer or not hasattr(self, 'log_dir'):
            return

        # 获取当前 epoch 的指标
        # 注意：self.trainer.callback_metrics 包含了当前 epoch 累积的指标
        metrics = self.trainer.callback_metrics
        
        # 筛选出训练相关的指标 (通常不带 val_ 前缀，或者根据具体命名习惯)
        # 这里假设所有非 val_ 开头的都是训练指标，或者直接记录所有指标
        train_metrics = {k: v.item() for k, v in metrics.items() if not k.startswith('val_')}
        
        if not train_metrics:
            return

        log_msg = f"Epoch {self.current_epoch} Training Metrics:\n"
        for k, v in train_metrics.items():
            log_msg += f"  {k}: {v:.4f}\n"
        log_msg += "-" * 20 + "\n"

        # 写入 train.log
        try:
            train_log_path = os.path.join(self.log_dir, "train.log")
            with open(train_log_path, "a") as f:
                f.write(log_msg)
        except Exception as e:
            print(f"Failed to write train log: {e}")

    # ==========================
    # Embedding 获取与处理逻辑 (保持不变)
    # ==========================

    def get_feature_embedding(self, feature_name: str, feature_value: torch.Tensor) -> torch.Tensor:
        """获取单个特征的 Embedding 或处理后的 Dense 值"""
        if feature_name in self.dense_feature_names:
            return feature_value.float().unsqueeze(1)
            
        emb_fname = self._get_emb_feature_name(feature_name)
        if emb_fname not in self.embedding_tables:
             raise ValueError(f"Embedding table not found for {feature_name} (mapped to {emb_fname})")
        
        return self.embedding_tables[emb_fname](feature_value.long())

    def array_feature_pooling(self, embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对 Array 特征进行 Pooling (Mean Pooling)"""
        if mask is None:
            return embedding.mean(dim=1)
        
        mask = mask.unsqueeze(-1) # (B, L, 1)
        masked_emb = embedding * mask
        sum_emb = masked_emb.sum(dim=1) 
        sum_mask = mask.sum(dim=1) + 1e-8 
        return sum_emb / sum_mask

    def get_embeddings_from_batch(self, batch: Dict[str, torch.Tensor], feature_names: Set[str]) -> Tuple[torch.Tensor, List[int], List[str]]:
        """从 Batch 中提取并拼接指定特征集合的 Embedding"""
        sorted_features = sorted(list(feature_names))
        emb_list = []
        dims = []
        
        for fname in sorted_features:
            if fname not in batch:
                # 容错处理：如果 batch 里没有这个特征，跳过或报错
                continue
                
            val = batch[fname]
            emb = self.get_feature_embedding(fname, val)
            
            if fname in self.array_feature_names:
                mask = batch.get(f"{fname}_mask", None)
                emb = self.array_feature_pooling(emb, mask)
            
            emb_list.append(emb)
            dims.append(emb.shape[1])
            
        if not emb_list:
            return torch.tensor([]).to(self.device), [], []

        return torch.cat(emb_list, dim=1), dims, sorted_features

    # ==========================
    # 抽象接口
    # ==========================
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def inference(self, batch):
        raise NotImplementedError("Subclasses must implement inference()")


    def validation_step(self, batch, batch_idx):
        scores = self.inference(batch)

        user_ids = batch['user_id'].view(-1).cpu().numpy()
        labels = batch['label'].view(-1).cpu().numpy()
        scores = scores.view(-1).cpu().numpy()

        for uid, score, label in zip(user_ids, scores, labels):
            if uid not in self.user_scores_dict:
                self.user_scores_dict[uid] = []
            self.user_scores_dict[uid].append((score, label))


    def on_validation_epoch_end(self):
        # 1. 准备数据容器
        all_preds = []
        all_labels = []
        
        # 新增：分别收集冷热启动用户的预测值和标签
        warm_preds = []
        warm_labels = []
        cold_preds = []
        cold_labels = []
        
        # 用户级指标列表
        metrics_all = {'auc': [], 'ndcg': [], 'hr': [], 'mrr': []}
        metrics_warm = {'auc': [], 'ndcg': [], 'hr': [], 'mrr': []}
        metrics_cold = {'auc': [], 'ndcg': [], 'hr': [], 'mrr': []}
        
        k = 10

        # 2. 遍历用户计算指标
        for uid, items in tqdm(self.user_scores_dict.items(), desc="Calculating validation metrics", ncols=80):
            if not items:
                continue
                
            preds = [x[0] for x in items]
            labels = [x[1] for x in items]
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # 区分冷热启动
            is_cold = False
            if hasattr(self, 'user_in_train_set') and self.user_in_train_set:
                # 尝试多种类型匹配以防万一 (int vs str)
                if uid not in self.user_in_train_set and str(uid) not in self.user_in_train_set:
                    is_cold = True
            
            # 新增：分类收集
            if is_cold:
                cold_preds.extend(preds)
                cold_labels.extend(labels)
            else:
                warm_preds.extend(preds)
                warm_labels.extend(labels)
            
            target_metrics = metrics_cold if is_cold else metrics_warm
            
            # --- AUC ---
            if len(set(labels)) > 1:
                try:
                    auc = roc_auc_score(labels, preds)
                    metrics_all['auc'].append(auc)
                    target_metrics['auc'].append(auc)
                except ValueError:
                    pass

            # --- TopK Metrics (NDCG, HR, MRR) ---
            # 按分数降序排序
            sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
            top_k = sorted_items[:k]
            
            # 统计该用户总共有多少个正样本
            num_positives = sum(1 for x in items if x[1] == 1)
            
            if num_positives == 0:
                # 如果没有正样本，跳过或记为0
                metrics_all['hr'].append(0.0)
                target_metrics['hr'].append(0.0)
                metrics_all['ndcg'].append(0.0)
                target_metrics['ndcg'].append(0.0)
                metrics_all['mrr'].append(0.0)
                target_metrics['mrr'].append(0.0)
                continue

            # HR
            has_hit = any(x[1] == 1 for x in top_k)
            hr = 1.0 if has_hit else 0.0
            metrics_all['hr'].append(hr)
            target_metrics['hr'].append(hr)
            
            # NDCG
            dcg = 0.0
            for rank, (_, label) in enumerate(top_k, start=1):
                if label == 1:
                    dcg += 1.0 / np.log2(rank + 1)
            
            # IDCG: 理想情况下，前 min(num_positives, k) 个位置都是正样本
            idcg = 0.0
            for rank in range(1, min(num_positives, k) + 1):
                idcg += 1.0 / np.log2(rank + 1)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            metrics_all['ndcg'].append(ndcg)
            target_metrics['ndcg'].append(ndcg)
            
            # MRR (通常只看第一个命中的)
            mrr = 0.0
            for rank, (_, label) in enumerate(top_k, start=1):
                if label == 1:
                    mrr = 1.0 / rank
                    break 
            
            metrics_all['mrr'].append(mrr)
            target_metrics['mrr'].append(mrr)

        # 3. 汇总计算
        def get_mean(l):
            return np.mean(l) if l else 0.0

        def calc_auc_logloss(preds, labels):
            auc = 0.0
            logloss = 0.0
            if len(preds) > 0:
                try:
                    if len(set(labels)) > 1:
                        auc = roc_auc_score(labels, preds)
                except:
                    pass
                try:
                    eps = 1e-15
                    preds_arr = np.clip(preds, eps, 1 - eps)
                    labels_arr = np.array(labels)
                    logloss = -np.mean(labels_arr * np.log(preds_arr) + (1 - labels_arr) * np.log(1 - preds_arr))
                except:
                    pass
            return auc, logloss

        # 计算各组的整体 AUC & LogLoss
        overall_auc, overall_logloss = calc_auc_logloss(all_preds, all_labels)
        warm_auc, warm_logloss = calc_auc_logloss(warm_preds, warm_labels)
        cold_auc, cold_logloss = calc_auc_logloss(cold_preds, cold_labels)

        results = {
            "Overall": {
                "AUC": overall_auc,
                "LogLoss": overall_logloss,
                "GAUC": get_mean(metrics_all['auc']),
                f"NDCG@{k}": get_mean(metrics_all['ndcg']),
                f"HR@{k}": get_mean(metrics_all['hr']),
                f"MRR@{k}": get_mean(metrics_all['mrr'])
            },
            "Warm_Start": {
                "AUC": warm_auc,
                "LogLoss": warm_logloss,
                "GAUC": get_mean(metrics_warm['auc']),
                f"NDCG@{k}": get_mean(metrics_warm['ndcg']),
                f"HR@{k}": get_mean(metrics_warm['hr']),
                f"MRR@{k}": get_mean(metrics_warm['mrr']),
                "User_Count": len(metrics_warm['hr'])
            },
            "Cold_Start": {
                "AUC": cold_auc,
                "LogLoss": cold_logloss,
                "GAUC": get_mean(metrics_cold['auc']),
                f"NDCG@{k}": get_mean(metrics_cold['ndcg']),
                f"HR@{k}": get_mean(metrics_cold['hr']),
                f"MRR@{k}": get_mean(metrics_cold['mrr']),
                "User_Count": len(metrics_cold['hr'])
            }
        }
        
        # 4. 记录到日志文件
        log_msg = (
            f"\n{'='*20} Epoch {self.current_epoch} Validation Results {'='*20}\n"
            f"Overall:\n"
            f"  AUC:      {results['Overall']['AUC']:.4f}\n"
            f"  LogLoss:  {results['Overall']['LogLoss']:.4f}\n"
            f"  GAUC:     {results['Overall']['GAUC']:.4f}\n"
            f"  NDCG@{k}:  {results['Overall'][f'NDCG@{k}']:.4f}\n"
            f"  HR@{k}:    {results['Overall'][f'HR@{k}']:.4f}\n"
            f"  MRR@{k}:   {results['Overall'][f'MRR@{k}']:.4f}\n"
            f"Warm Start Users ({results['Warm_Start']['User_Count']}):\n"
            f"  AUC:      {results['Warm_Start']['AUC']:.4f}\n"
            f"  LogLoss:  {results['Warm_Start']['LogLoss']:.4f}\n"
            f"  GAUC:     {results['Warm_Start']['GAUC']:.4f}\n"
            f"  NDCG@{k}:  {results['Warm_Start'][f'NDCG@{k}']:.4f}\n"
            f"  HR@{k}:    {results['Warm_Start'][f'HR@{k}']:.4f}\n"
            f"  MRR@{k}:   {results['Warm_Start'][f'MRR@{k}']:.4f}\n"
            f"Cold Start Users ({results['Cold_Start']['User_Count']}):\n"
            f"  AUC:      {results['Cold_Start']['AUC']:.4f}\n"
            f"  LogLoss:  {results['Cold_Start']['LogLoss']:.4f}\n"
            f"  GAUC:     {results['Cold_Start']['GAUC']:.4f}\n"
            f"  NDCG@{k}:  {results['Cold_Start'][f'NDCG@{k}']:.4f}\n"
            f"  HR@{k}:    {results['Cold_Start'][f'HR@{k}']:.4f}\n"
            f"  MRR@{k}:   {results['Cold_Start'][f'MRR@{k}']:.4f}\n"
            f"{'='*60}\n"
        )
        
        # 打印到控制台
        print(log_msg)
        
        # 写入文件
        if hasattr(self, 'log_dir'):
             val_log_path = os.path.join(self.log_dir, "val_log.log")
             with open(val_log_path, "a") as f:
                 f.write(log_msg)
        

    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info(f"Loading weights from {path}...")
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict, strict=True)