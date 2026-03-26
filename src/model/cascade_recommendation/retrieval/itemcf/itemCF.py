# 基于物品的协同过滤  
from tqdm import tqdm
from .....Logger.logging import Logger
import numpy as np

logger = Logger.get_logger("ItemCF")

# 构造用户和物料的交互历史
def build_user_item_history(behavours_file_path: str):
    user_item_history = {}
    logger.info("Building user-item interaction history from behaviours file.")
    with open(behavours_file_path, 'r') as f:
        for line in tqdm(f, desc="Building user-item history"):
            impression_id, user_id, time, history, interactions = line.strip().split('\t')
            history = history.split() if history != '' else []
            interactions = interactions.split() if interactions != '' else []
            interactions = [interaction.split('-') for interaction in interactions]

            if user_id not in user_item_history:
                user_item_history[user_id] = {}
            for item_id in history:
                user_item_history[user_id][item_id] = 1
            for item_id, click in interactions:
                if click == '1':
                    user_item_history[user_id][item_id] = 1

    return user_item_history


# 计算物品相似度矩阵
def compute_item_similarity(user_item_history):
    logger.info("Computing item similarity matrix.")
    item_co_occurrence = {}
    item_count = {}

    for user, items in tqdm(user_item_history.items(), desc="Computing co-occurrence"):
        # items是一个字典，key是item_id，value是(rating, timestamp)
        for item_i in items:
            item_count[item_i] = item_count.get(item_i, 0) + 1
            for item_j in items:
                if item_i == item_j:
                    continue
                item_co_occurrence.setdefault(item_i, {})
                item_co_occurrence[item_i][item_j] = item_co_occurrence[item_i].get(item_j, 0) + 1

    # 计算相似度矩阵
    item_similarity = {}
    for item_i, related_items in tqdm(item_co_occurrence.items(), desc="Computing similarity"):
        item_similarity.setdefault(item_i, {})
        for item_j, co_count in related_items.items():
            sim_score = co_count / ((item_count[item_i] * item_count[item_j]) ** 0.5)
            item_similarity[item_i][item_j] = sim_score

    return item_similarity

# 根据物品相似度推荐k个物品，和用户历史消重
def recall_top_k_items(user_id, user_item_history, item_similarity, k=10):
    if user_id not in user_item_history:
        return []

    interacted_items = user_item_history[user_id]
    scores = {}
    for item_i in interacted_items:
        for item_j, sim_score in item_similarity.get(item_i, {}).items():
            if item_j in interacted_items:
                continue  # 跳过用户已经交互过的物品
            scores[item_j] = scores.get(item_j, 0) + sim_score

    # 按照得分排序，取前k个物品
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_items = [item for item, score in ranked_items[:k]]
    return recommended_items

# 输入测试集计算命中率
def eval(test_file_path, user_item_history, item_similarity, k=20):
    hits = 0
    total = 0
    user_retrieval_res = {}
    with open(test_file_path, 'r') as f:
        for line in tqdm(f, desc="Calculating hit rate"):
            impression_id, user_id, time, history, interactions = line.strip().split('\t')
            interactions = interactions.split() if interactions != '' else []
            interactions = [interaction.split('-') for interaction in interactions]
            ground_truth_items = [item_id for item_id, click in interactions if click == '1']

            if user_id not in user_retrieval_res:
                recommended_items = recall_top_k_items(user_id, user_item_history, item_similarity, k)
                user_retrieval_res[user_id] = [recommended_items, ground_truth_items]
            else:
                user_retrieval_res[user_id][1].extend(ground_truth_items)
            
            
            
            


    warmup_metrics = {'HR@20': [], 'NDCG@20': [], 'MRR@20': []}
    cold_metrics = {'HR@20': [], 'NDCG@20': [], 'MRR@20': []}
    for user_id, (recommended_items, ground_truth_items) in user_retrieval_res.items():
        if user_id not in user_item_history:
            metrics = cold_metrics
        else:
            metrics = warmup_metrics
        # HR
        hit = any(item in recommended_items for item in ground_truth_items)
        metrics['HR@20'].append(1.0 if hit else 0.0)
        # NDCG
        dcg = 0.0
        for rank, item_id in enumerate(recommended_items, start=1):
            if item_id in ground_truth_items:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(ground_truth_items), k) + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics['NDCG@20'].append(ndcg)
        # MRR
        mrr = 0.0
        for rank, item_id in enumerate(recommended_items, start=1):
            if item_id in ground_truth_items:
                mrr = 1.0 / rank
                break
        metrics['MRR@20'].append(mrr)

    metrics_str = ""
    for metric_name, metric_values in warmup_metrics.items():
        mean_value = np.mean(metric_values) if metric_values else 0.0
        metrics_str += f"Warmup {metric_name}: {mean_value:.4f}  "
    for metric_name, metric_values in cold_metrics.items():
        mean_value = np.mean(metric_values) if metric_values else 0.0
        metrics_str += f"Cold {metric_name}: {mean_value:.4f}  "
    logger.info("Evaluation Results: " + metrics_str)


if __name__ == "__main__":
    # 构建用户-物品交互历史
    train_file_path = '/data2/zhy/News_Recsys/Data/MIND/MINDsmall_train/behaviors.tsv'
    test_file_path = '/data2/zhy/News_Recsys/Data/MIND/MINDsmall_dev/behaviors.tsv'
    user_item_history = build_user_item_history(train_file_path)

    # 计算物品相似度矩阵
    item_similarity = compute_item_similarity(user_item_history)

    # 计算测试集的命中率
    eval(test_file_path, user_item_history, item_similarity, k=20)