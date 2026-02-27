import sys
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .base_model import BaseModel
from ...Logger.logging import Logger

# 配置 logging
logger = Logger.get_logger("BaseModelSort")

def _compute_user_metrics(uid, items, k, is_cold):
    if not items:
        return None

    preds = [float(x[0]) for x in items]
    labels = [int(x[1]) for x in items]
    
    res = {
        'uid': uid,
        'preds': preds,
        'labels': labels,
        'is_cold': is_cold,
        'auc': None,
        'hr': 0.0,
        'ndcg': 0.0,
        'mrr': 0.0
    }

    # --- AUC ---
    if len(set(labels)) > 1:
        try:
            res['auc'] = roc_auc_score(labels, preds)
        except ValueError: 
            pass

    # --- TopK Metrics ---
    # items is list of (score, label)
    items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    top_k = items_sorted[:k]
    
    num_positives = sum(1 for x in items if x[1] == 1)
    
    if num_positives == 0:
        return res

    # HR
    if any(x[1] == 1 for x in top_k):
        res['hr'] = 1.0
    
    # MRR & NDCG
    top_k_labels = np.array([x[1] for x in top_k])
    
    # MRR
    hits = np.where(top_k_labels == 1)[0]
    if len(hits) > 0:
        res['mrr'] = 1.0 / (hits[0] + 1)

    # NDCG
    ranks = np.arange(1, len(top_k_labels) + 1)
    discounts = 1.0 / np.log2(ranks + 1)
    dcg = np.sum(top_k_labels * discounts)
    
    n_ideal = min(num_positives, k)
    ideal_ranks = np.arange(1, n_ideal + 1)
    idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
    
    if idcg > 0:
        res['ndcg'] = dcg / idcg
        
    return res

def _compute_wrapper(args):
    return _compute_user_metrics(*args)

class BaseModelSort(BaseModel):
    """
    推荐系统排序模型基类 (Base Sorting Model)
    继承自 BaseModel，实现排序任务特定的验证和指标计算逻辑。
    """

    def __init__(self, config):
        super().__init__(config)
        self.user_scores_dict = {}

    def _init_metrics_state(self):
        super()._init_metrics_state()
        self.user_scores_dict = {}

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
        
        warm_preds = []
        warm_labels = []
        cold_preds = []
        cold_labels = []
        
        metrics_all = {'auc': [], 'ndcg': [], 'hr': [], 'mrr': []}
        metrics_warm = {'auc': [], 'ndcg': [], 'hr': [], 'mrr': []}
        metrics_cold = {'auc': [], 'ndcg': [], 'hr': [], 'mrr': []}
        
        k = 10

        # 2. 准备并行任务
        tasks = []
        has_train_set = hasattr(self, 'user_in_train_set') and self.user_in_train_set is not None
        
        for uid, items in self.user_scores_dict.items():
            if not items: continue
            
            is_cold = False
            if has_train_set:
                if uid not in self.user_in_train_set and str(uid) not in self.user_in_train_set:
                    is_cold = True
            
            tasks.append((uid, items, k, is_cold))

        # 3. 并行计算
        results = []
        if len(tasks) < 1000:
            for task in tqdm(tasks, desc="Calculating validation metrics (Serial)", ncols=80):
                results.append(_compute_wrapper(task))
        else:
            num_workers = max(1, min(multiprocessing.cpu_count(), 28))
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                chunksize = max(1, len(tasks) // (num_workers * 4))
                results = list(tqdm(executor.map(_compute_wrapper, tasks, chunksize=chunksize), 
                                  total=len(tasks), 
                                  desc=f"Calculating validation metrics (Parallel {num_workers})", 
                                  ncols=120))

        # 4. 汇总结果
        for res in results:
            if res is None: continue
            
            preds = res['preds']
            labels = res['labels']
            is_cold = res['is_cold']
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            if is_cold:
                cold_preds.extend(preds)
                cold_labels.extend(labels)
            else:
                warm_preds.extend(preds)
                warm_labels.extend(labels)
            
            target_metrics = metrics_cold if is_cold else metrics_warm
            
            if res['auc'] is not None:
                metrics_all['auc'].append(res['auc'])
                target_metrics['auc'].append(res['auc'])
            
            metrics_all['hr'].append(res['hr'])
            metrics_all['ndcg'].append(res['ndcg'])
            metrics_all['mrr'].append(res['mrr'])
            
            target_metrics['hr'].append(res['hr'])
            target_metrics['ndcg'].append(res['ndcg'])
            target_metrics['mrr'].append(res['mrr'])

        # 5. 汇总计算
        def get_mean(l): return np.mean(l) if l else 0.0

        def calc_auc_logloss(preds, labels):
            auc, logloss = 0.0, 0.0
            if len(preds) > 0:
                try:
                    if len(set(labels)) > 1: auc = roc_auc_score(labels, preds)
                except: pass
                try:
                    eps = 1e-15
                    preds_arr = np.clip(preds, eps, 1 - eps)
                    labels_arr = np.array(labels)
                    logloss = -np.mean(labels_arr * np.log(preds_arr) + (1 - labels_arr) * np.log(1 - preds_arr))
                except: pass
            return auc, logloss

        overall_auc, overall_logloss = calc_auc_logloss(all_preds, all_labels)
        warm_auc, warm_logloss = calc_auc_logloss(warm_preds, warm_labels)
        cold_auc, cold_logloss = calc_auc_logloss(cold_preds, cold_labels)

        results = {
            "Overall": {
                "AUC": overall_auc, "LogLoss": overall_logloss, "GAUC": get_mean(metrics_all['auc']),
                f"NDCG@{k}": get_mean(metrics_all['ndcg']), f"HR@{k}": get_mean(metrics_all['hr']), f"MRR@{k}": get_mean(metrics_all['mrr'])
            },
            "Warm_Start": {
                "AUC": warm_auc, "LogLoss": warm_logloss, "GAUC": get_mean(metrics_warm['auc']),
                f"NDCG@{k}": get_mean(metrics_warm['ndcg']), f"HR@{k}": get_mean(metrics_warm['hr']), f"MRR@{k}": get_mean(metrics_warm['mrr']),
                "User_Count": len(metrics_warm['hr'])
            },
            "Cold_Start": {
                "AUC": cold_auc, "LogLoss": cold_logloss, "GAUC": get_mean(metrics_cold['auc']),
                f"NDCG@{k}": get_mean(metrics_cold['ndcg']), f"HR@{k}": get_mean(metrics_cold['hr']), f"MRR@{k}": get_mean(metrics_cold['mrr']),
                "User_Count": len(metrics_cold['hr'])
            }
        }
        
        # 4. 记录日志
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
        
        logger.info(log_msg)
        if hasattr(self, 'log_dir'):
             with open(os.path.join(self.log_dir, "val_log.log"), "a") as f:
                 f.write(log_msg)

        self._init_metrics_state()