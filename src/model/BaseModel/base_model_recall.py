import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .base_model import BaseModel
from ..model_utils.TopKSearcher import TopKSearcher
from ...Logger.logging import Logger

# 配置 logging
logger = Logger.get_logger("BaseModelRecall")

def _compute_recall_metrics(info_dict, user_in_train_set, k):
    uid, recall, target_items = info_dict['uid'], info_dict['recall'], info_dict['target']
    
    if len(target_items) == 0:
        return None

    is_cold = False
    if user_in_train_set is not None:
        if uid not in user_in_train_set and str(uid) not in user_in_train_set:
            is_cold = True

    num_positives = len(target_items)
    
    # HR
    has_hit = any(item in recall for item in target_items)
    hr = 1.0 if has_hit else 0.0
    
    # NDCG
    dcg = 0.0
    for rank, (item_id) in enumerate(recall, start=1):
        if item_id in target_items: dcg += 1.0 / np.log2(rank + 1)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(num_positives, k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # MRR
    mrr = 0.0
    for rank, (item_id) in enumerate(recall, start=1):
        if item_id in target_items:
            mrr = 1.0 / rank
            break
            
    return {
        'is_cold': is_cold,
        'hr': hr,
        'ndcg': ndcg,
        'mrr': mrr
    }

def _compute_recall_wrapper(args):
    return _compute_recall_metrics(*args)

class BaseModelRecall(BaseModel):
    """
    推荐系统召回模型基类 (Base Recall Model)
    继承自 BaseModel，实现召回任务特定的 TopK 搜索和验证逻辑。
    """

    def __init__(self, config):
        super().__init__(config)
        
        # TopK 搜索器 (用于召回阶段)
        self.topk_searcher = TopKSearcher(k=self.config.k + 100)
        self.impression_user_history = {}

    def inference_item(self, batch):
        raise NotImplementedError("Subclasses must implement inference_item()")
    
    def inference_user(self, batch):
        raise NotImplementedError("Subclasses must implement inference_user()")

    def on_validation_epoch_start(self):
        self.item_embeddings = []
        self.item_ids = []

    def _build_item_index(self):
        """辅助函数：构建 TopK 索引"""
        if len(self.item_embeddings) > 0:
            logger.info("Building TopK index from validation set 0...")
            all_embeddings = torch.cat(self.item_embeddings, dim=0).numpy()
            all_ids = torch.cat(self.item_ids, dim=0).numpy()
            
            if hasattr(self, 'topk_searcher'):
                self.topk_searcher.update_embedding(all_embeddings, all_ids, normalize=True)
            
            # 清空缓存，防止重复构建
            self.item_embeddings = []
            self.item_ids = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # idx=0的是item数据集，用于生成候选物品的embedding
        if dataloader_idx == 0:
            item_vector = self.inference_item(batch)
            self.item_embeddings.append(item_vector.detach().cpu())
            self.item_ids.append(batch['item_id'].detach().cpu())
            
        elif dataloader_idx == 1:
            # 技巧：在 idx=1 的第一个 batch 处，检测并处理 idx=0 收集到的数据
            if len(self.item_embeddings) > 0:
                self._build_item_index()
                self.recall_res = {}

            user_emb = self.inference_user(batch)
            # # 由于相同的用户总是相邻的，并且具有相同的user_emb
            # user_emb_unique, user_counts = torch.unique_consecutive(user_emb, return_counts=True, dim=0)
            # # 对每个唯一用户进行TopK搜索
            # idxs, scores = self.topk_searcher.search(user_emb_unique, normalize=True)  # idxs: (B, K), scores: (B, K)
            # # 每个用户开始的索引号，计算user_counts的累积和
            # user_start_idx = torch.cumsum(torch.cat([torch.tensor([0]).to(user_counts.device), user_counts[:-1]]), dim=0)
            # 相同的impression总是相邻的，并且具有相同的user_emb
            impression_id_tensor = batch['impression_id']
            impression_ids, impression_counts = torch.unique_consecutive(impression_id_tensor, return_counts=True, dim=0)
            impression_start_idx = torch.cumsum(torch.cat([torch.tensor([0]).to(impression_counts.device), impression_counts[:-1]]), dim=0)
            user_emb_unique = user_emb[impression_start_idx]
            # 对每个唯一impression进行TopK搜索
            idxs, scores = self.topk_searcher.search(user_emb_unique, normalize=True)  # idxs: (B, K), scores: (B, K)

            # user_id = batch['user_id'].view(-1).cpu().numpy().tolist()
            # impression_id = batch['impression_id'].view(-1).cpu().numpy().tolist()
            user_history = batch['user_history']
            user_history_mask = batch['user_history_mask']

            for i in range(len(impression_start_idx)):
                impre_id = int(batch['impression_id'][impression_start_idx[i]])
                uid = int(batch['user_id'][impression_start_idx[i]])
                if impre_id not in self.recall_res:
                    # recall_list 用于存储最终的召回结果列表
                    # recall 使用 set 结构，方便去重和快速查找
                    self.recall_res[impre_id] = {'uid': uid, 'recall': set(), 'target': [], 'history': [], 'scores': [], 'recall_list': []}
                    
                    impression_recall_res = idxs[i].cpu().numpy().tolist()
                    # 构建用户历史集合
                    if impre_id not in self.impression_user_history:
                        history_length = int(user_history_mask[impression_start_idx[i]].sum().item())
                        impression_history = user_history[impression_start_idx[i]].cpu().numpy().tolist()[:history_length]
                        self.impression_user_history[impre_id] = impression_history
                    else:
                        impression_history = self.impression_user_history[impre_id]

                    self.recall_res[impre_id]['history'] = list(impression_history)

                    impression_history_set = set(impression_history)
                    # 过滤掉历史交互过的物品
                    for idx_, rec_item in enumerate(impression_recall_res):
                        if rec_item not in impression_history_set:
                            self.recall_res[impre_id]['recall'].add(rec_item)
                            self.recall_res[impre_id]['recall_list'].append(rec_item)
                            self.recall_res[impre_id]['scores'].append(float(scores[i][idx_]))
                            if len(self.recall_res[impre_id]['recall']) >= self.config.k:
                                break
            
            positive_label = batch['label'][:, 0] == 1
            positive_impression_ids = batch['impression_id'][positive_label]#.view(-1).cpu().numpy().tolist()
            positive_item_ids = batch['item_id'][positive_label]#.view(-1).cpu().numpy().tolist()

            for i in range(len(positive_impression_ids)):
                impre_id = int(positive_impression_ids[i])
                item = int(positive_item_ids[i])
                if impre_id not in self.recall_res:
                    raise ValueError(f"Impression ID {impre_id} not found in recall results.")
                self.recall_res[impre_id]['target'].append(item)
            # for impre_id, item in zip(positive_impression_ids, positive_item_ids):
            #     if impre_id not in self.recall_res:
            #         raise ValueError(f"Impression ID {impre_id} not found in recall results.")
            #     self.recall_res[impre_id]['target'].append(item)

            # for i, (impre_id, label, item) in enumerate(zip(impression_id, labels, item_id)):
            #     if impre_id not in self.recall_res:
            #         self.recall_res[impre_id] = {'uid': -1, 'recall': set(), 'target': [], 'history': []}
                
            #         impression_recall_res = idxs[i].cpu().numpy().tolist()
            #         # 构建用户历史集合
            #         if impre_id not in self.impression_user_history:
            #             history_length = int(user_history_mask[i].sum().item())
            #             impression_history = user_history[i].cpu().numpy().tolist()[:history_length]
            #             self.impression_user_history[impre_id] = impression_history
            #         else:
            #             impression_history = self.impression_user_history[impre_id]

            #         self.recall_res[impre_id]['history'] = list(impression_history)

            #         impression_history_set = set(impression_history)
            #         # 过滤掉历史交互过的物品
            #         for rec_item in impression_recall_res:
            #             if rec_item not in impression_history_set:
            #                 self.recall_res[impre_id]['recall'].add(rec_item)
            #                 if len(self.recall_res[impre_id]['recall']) >= self.config.k:
            #                     break
            #         self.recall_res[impre_id]['uid'] = user_id[i]
            #     if label == 1:
            #         self.recall_res[impre_id]['target'].append(item)

    def write_recall_res(self):
        # 如果当前的训练轮数模5=0，那么就把self.recall_res写到文件里
        if self.current_epoch % 5 == 0:
            res_file_path = os.path.join(self.res_show_path, f"recall_results_epoch_{self.current_epoch}.txt")
            with open(res_file_path, "w") as f:
                for impre_id, info_dict in self.recall_res.items():
                    recall_items_str = ",".join([str(item) for item in info_dict['recall_list']])
                    target_items_str = ",".join([str(item) for item in info_dict['target']])
                    history_items_str = ",".join([str(item) for item in info_dict['history']])
                    scores_str = ",".join([str(score) for score in info_dict['scores']])
                    f.write(f"{impre_id}\t{info_dict['uid']}\t{recall_items_str}\t{target_items_str}\t{history_items_str}\t{scores_str}\n")
            logger.info(f"Recall results for epoch {self.current_epoch} written to {res_file_path}")

    def on_validation_epoch_end(self):
        self.write_recall_res()
        # 1. 准备数据容器
        metrics_all = {'ndcg': [], 'hr': [], 'mrr': []}
        metrics_warm = {'ndcg': [], 'hr': [], 'mrr': []}
        metrics_cold = {'ndcg': [], 'hr': [], 'mrr': []}
        
        k = self.config.k

        # 2. 准备并行任务
        tasks = []
        user_in_train_set = self.user_in_train_set if hasattr(self, 'user_in_train_set') else None
        
        for impre_id, info_dict in self.recall_res.items():
            tasks.append((info_dict, user_in_train_set, k))

        # 3. 并行计算
        results = []
        if len(tasks) < 1000:
             for task in tqdm(tasks, desc="Calculating validation metrics (Serial)", ncols=80):
                results.append(_compute_recall_wrapper(task))
        else:
             num_workers = max(1, min(multiprocessing.cpu_count(), 28))
             with ProcessPoolExecutor(max_workers=num_workers) as executor:
                chunksize = max(1, len(tasks) // (num_workers * 4))
                results = list(tqdm(executor.map(_compute_recall_wrapper, tasks, chunksize=chunksize), 
                                    total=len(tasks), 
                                    desc=f"Calculating validation metrics (Parallel {num_workers})", 
                                    ncols=120))

        # 4. 汇总结果
        for res in results:
            if res is None: continue
            
            is_cold = res['is_cold']
            target_metrics = metrics_cold if is_cold else metrics_warm
            
            metrics_all['hr'].append(res['hr']); target_metrics['hr'].append(res['hr'])
            metrics_all['ndcg'].append(res['ndcg']); target_metrics['ndcg'].append(res['ndcg'])
            metrics_all['mrr'].append(res['mrr']); target_metrics['mrr'].append(res['mrr'])

        # 5. 汇总计算
        def get_mean(l): return np.mean(l) if l else 0.0

        results = {
            "Overall": {
                f"NDCG@{k}": get_mean(metrics_all['ndcg']), f"HR@{k}": get_mean(metrics_all['hr']), f"MRR@{k}": get_mean(metrics_all['mrr'])
            },
            "Warm_Start": {
                f"NDCG@{k}": get_mean(metrics_warm['ndcg']), f"HR@{k}": get_mean(metrics_warm['hr']), f"MRR@{k}": get_mean(metrics_warm['mrr']),
                "User_Count": len(metrics_warm['hr'])
            },
            "Cold_Start": {
                f"NDCG@{k}": get_mean(metrics_cold['ndcg']), f"HR@{k}": get_mean(metrics_cold['hr']), f"MRR@{k}": get_mean(metrics_cold['mrr']),
                "User_Count": len(metrics_cold['hr'])
            }
        }
        
        # 4. 记录日志 
        log_msg = (
            f"\n{'='*20} Epoch {self.current_epoch} Validation Results {'='*20}\n"
            f"Overall:\n"
            f"  NDCG@{k}:  {results['Overall'][f'NDCG@{k}']:.4f}\n"
            f"  HR@{k}:    {results['Overall'][f'HR@{k}']:.4f}\n"
            f"  MRR@{k}:   {results['Overall'][f'MRR@{k}']:.4f}\n"
            f"Warm Start Users ({results['Warm_Start']['User_Count']}):\n"
            f"  NDCG@{k}:  {results['Warm_Start'][f'NDCG@{k}']:.4f}\n"
            f"  HR@{k}:    {results['Warm_Start'][f'HR@{k}']:.4f}\n"
            f"  MRR@{k}:   {results['Warm_Start'][f'MRR@{k}']:.4f}\n"
            f"Cold Start Users ({results['Cold_Start']['User_Count']}):\n"
            f"  NDCG@{k}:  {results['Cold_Start'][f'NDCG@{k}']:.4f}\n"
            f"  HR@{k}:    {results['Cold_Start'][f'HR@{k}']:.4f}\n"
            f"  MRR@{k}:   {results['Cold_Start'][f'MRR@{k}']:.4f}\n"
            f"{'='*60}\n"
        )

        logger.info(log_msg)
        if hasattr(self, 'log_dir'):
             with open(os.path.join(self.log_dir, "val_log.log"), "a") as f:
                 f.write(log_msg)