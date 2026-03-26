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

def _compute_user_metrics(uid, items, k_list, is_cold):
    if not items:
        return None

    preds = [float(x[0]) for x in items]
    labels = [int(x[1]) for x in items]
    
    # 默认值
    res = {
        'uid': uid,
        'preds': preds,
        'labels': labels,
        'is_cold': is_cold,
        'auc': None,
        'mrr': 0.0,
        'hr': {k: 0.0 for k in k_list},
        'ndcg': {k: 0.0 for k in k_list},
    }

    # --- AUC ---
    if len(set(labels)) > 1:
        try:
            res['auc'] = roc_auc_score(labels, preds)
        except ValueError: 
            pass

    # --- Sorting ---
    # items is list of (score, label)
    items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    all_sorted_labels = np.array([x[1] for x in items_sorted])
    
    num_positives = sum(labels)
    if num_positives == 0:
        return res

    # --- Global MRR (Mean Reciprocal Rank over full list) ---
    hits = np.where(all_sorted_labels == 1)[0]
    if len(hits) > 0:
        # hits[0] 是第一个命中位置的索引(0-based)，排名为 hits[0] + 1
        res['mrr'] = 1.0 / (hits[0] + 1)
        
    # --- TopK Metrics (HR@K, NDCG@K) ---
    for k in k_list:
        top_k_labels = all_sorted_labels[:k]
        
        # HR@K
        if np.sum(top_k_labels) > 0:
            res['hr'][k] = 1.0
        
        # NDCG@K
        ranks = np.arange(1, len(top_k_labels) + 1)
        discounts = 1.0 / np.log2(ranks + 1)
        dcg = np.sum(top_k_labels * discounts)
        
        n_ideal = min(num_positives, k)
        ideal_ranks = np.arange(1, n_ideal + 1)
        idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
        
        if idcg > 0:
            res['ndcg'][k] = dcg / idcg
            
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

    # def _init_metrics_state(self):
    #     super()._init_metrics_state()
        

    def inference(self, batch):
        raise NotImplementedError("Subclasses must implement inference()")

    def validation_step(self, batch, batch_idx):
        scores = self.inference(batch)

        user_ids = batch['user_id'].view(-1).cpu().numpy()
        labels = batch['label'].view(-1).cpu().numpy()
        scores = scores.view(-1).cpu().numpy()
        
        # 尝试获取 item_id
        item_ids = None
        for key in ['news_id', 'item_id', 'movie_id', 'doc_id', 'mid']:
            if key in batch:
                item_ids = batch[key].view(-1).cpu().numpy()
                break
        
        if item_ids is None and hasattr(self, 'item_feature_names'):
            for name in self.item_feature_names:
                if name.endswith('_id') and name in batch:
                    item_ids = batch[name].view(-1).cpu().numpy()
                    break
        
        if item_ids is None:
            # Fallback if no ID found, using index as placeholder, or just 0
            # This ensures the code doesn't break, though IDs will be meaningless
            item_ids = np.zeros_like(user_ids)

        for uid, item_id, score, label in zip(user_ids, item_ids, scores, labels):
            if uid not in self.user_scores_dict:
                self.user_scores_dict[uid] = []
            self.user_scores_dict[uid].append((item_id, score, label))

    def on_validation_epoch_end(self):
        # 1. 准备数据容器
        f_res = None
        
        # 定义需要计算的 K 值列表
        k_list = [5, 10]
        
        # 指标容器初始化
        # hr, ndcg 现在是 dict: {k1: [], k2: []}
        metrics_all = {'auc': [], 'mrr': [], 'hr': {k: [] for k in k_list}, 'ndcg': {k: [] for k in k_list}}
        metrics_warm = {'auc': [], 'mrr': [], 'hr': {k: [] for k in k_list}, 'ndcg': {k: [] for k in k_list}}
        
        # Cold start metric initialization
        metrics_cold = {'auc': [], 'mrr': [], 'hr': {k: [] for k in k_list}, 'ndcg': {k: [] for k in k_list}}

        # 2. 准备并行任务
        tasks = []
        has_train_set = hasattr(self, 'user_in_train_set') and self.user_in_train_set is not None
        
        # 3. 结果文件写入初始化
        result_file = None
        if hasattr(self, 'log_dir'):
            try:
                result_dir = os.path.join(self.log_dir, "result_record")
                os.makedirs(result_dir, exist_ok=True)
                result_file = os.path.join(result_dir, f"sort_results_epoch_{self.current_epoch}_step_{self.global_step}.txt")
                f_res = open(result_file, 'w', encoding='utf-8')
                logger.info(f"Saving sort results to {result_file}")
            except Exception as e:
                logger.error(f"Failed to open result file: {e}")
                f_res = None

        try:
            for uid, items in self.user_scores_dict.items():
                if not items: continue
                
                # items 元素: (item_id, score, label)
                # 排序: score 降序
                items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
                
                # 写入结果文件 (如果文件已打开)
                if f_res:
                    item_ids_str = " ".join([str(x[0]) for x in items_sorted])
                    scores_str = " ".join([f"{x[1]:.6f}" for x in items_sorted])
                    labels_str = " ".join([str(int(x[2])) for x in items_sorted])
                    f_res.write(f"{uid}\t{item_ids_str}\t{scores_str}\t{labels_str}\n")
                
                is_cold = False
                if has_train_set:
                    if uid not in self.user_in_train_set and str(uid) not in self.user_in_train_set:
                        is_cold = True
                
                # 提取 (score, label) 用于指标计算
                metric_items = [(x[1], x[2]) for x in items]
                tasks.append((uid, metric_items, k_list, is_cold))
        finally:
            if f_res:
                f_res.close()

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
        all_preds = []
        all_labels = []
        warm_preds = []
        warm_labels = []
        cold_preds = []
        cold_labels = []

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
            
            # AUC
            if res['auc'] is not None:
                metrics_all['auc'].append(res['auc'])
                target_metrics['auc'].append(res['auc'])
            
            # MRR (Global)
            metrics_all['mrr'].append(res['mrr'])
            target_metrics['mrr'].append(res['mrr'])
            
            # TopK Metrics
            for k in k_list:
                # HR
                hr_val = res['hr'].get(k, 0.0)
                metrics_all['hr'][k].append(hr_val)
                target_metrics['hr'][k].append(hr_val)
                
                # NDCG
                ndcg_val = res['ndcg'].get(k, 0.0)
                metrics_all['ndcg'][k].append(ndcg_val)
                target_metrics['ndcg'][k].append(ndcg_val)

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

        # Calculate LogLoss and Overall AUC
        overall_auc, overall_logloss = calc_auc_logloss(all_preds, all_labels)
        warm_auc, warm_logloss = calc_auc_logloss(warm_preds, warm_labels)
        cold_auc, cold_logloss = calc_auc_logloss(cold_preds, cold_labels)

        # Helper to format metrics dict
        def build_result_dict(name, auc, logloss, metrics_container, user_count=None):
            res_dict = {
                "AUC": auc,
                "LogLoss": logloss,
                "GAUC": get_mean(metrics_container['auc']),
                "MRR": get_mean(metrics_container['mrr'])
            }
            if user_count is not None:
                res_dict["User_Count"] = user_count
                
            for k in k_list:
                res_dict[f"NDCG@{k}"] = get_mean(metrics_container['ndcg'][k])
                res_dict[f"HR@{k}"] = get_mean(metrics_container['hr'][k])
            return res_dict

        results_formatted = {
            "Overall": build_result_dict("Overall", overall_auc, overall_logloss, metrics_all),
            "Warm_Start": build_result_dict("Warm_Start", warm_auc, warm_logloss, metrics_warm, len(metrics_warm['mrr'])),
            "Cold_Start": build_result_dict("Cold_Start", cold_auc, cold_logloss, metrics_cold, len(metrics_cold['mrr']))
        }
        
        # 4. 记录日志
        def format_log_section(title, data):
            msg = f"{title}:\n"
            msg += f"  AUC:      {data['AUC']:.4f}\n"
            msg += f"  LogLoss:  {data['LogLoss']:.4f}\n"
            msg += f"  GAUC:     {data['GAUC']:.4f}\n"
            msg += f"  MRR:      {data['MRR']:.4f}\n"
            for k in k_list:
                msg += f"  NDCG@{k}:  {data[f'NDCG@{k}']:.4f}\n"
                msg += f"  HR@{k}:    {data[f'HR@{k}']:.4f}\n"
            return msg

        log_msg = f"\n{'='*20} Epoch {self.current_epoch} Validation Results {'='*20}\n"
        log_msg += format_log_section("Overall", results_formatted['Overall'])
        log_msg += format_log_section(f"Warm Start Users ({results_formatted['Warm_Start']['User_Count']})", results_formatted['Warm_Start'])
        log_msg += format_log_section(f"Cold Start Users ({results_formatted['Cold_Start']['User_Count']})", results_formatted['Cold_Start'])
        log_msg += f"{'='*60}\n"
        
        logger.info(log_msg)
        if hasattr(self, 'log_dir'):
             with open(os.path.join(self.log_dir, "val_log.log"), "a") as f:
                 f.write(log_msg)

        # --- Update Validation LogLoss History and Plot ---
        # 记录验证集 LogLoss 并重新绘图 (Sort模型特有)
        if 'Overall' in results_formatted and 'LogLoss' in results_formatted['Overall']:
            val_logloss = results_formatted['Overall']['LogLoss']
            # 使用列表存储 (step, loss)
            if not hasattr(self, 'val_logloss_history'):
                self.val_logloss_history = []
            self.val_logloss_history.append((self.global_step, val_logloss))
            self._plot_loss_curve()

        self.user_scores_dict = {}