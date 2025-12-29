# 基于物品的协同过滤  
# Hit Rate@50: 0.149418
from tqdm import tqdm

# 构造用户和物料的交互历史
def build_user_item_history(rating_file_path):
    user_item_history = {}
    with open(rating_file_path, 'r', encoding='ISO-8859-1') as f:
        for line in tqdm(f, desc="Building user-item history"):
            user_id, item_id, rating, timestamp = line.strip().split('::')
            if user_id not in user_item_history:
                user_item_history[user_id] = {}
            user_item_history[user_id][item_id] = (float(rating), timestamp)
    return user_item_history


# 计算物品相似度矩阵
def compute_item_similarity(user_item_history):
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
def hit_rate(test_file_path, user_item_history, item_similarity, k=10):
    hits = 0
    total = 0
    # 记录每个用户的推荐物品
    recall_history = {}
    with open(test_file_path, 'r', encoding='ISO-8859-1') as f:
        for line in tqdm(f, desc="Calculating hit rate"):
            user_id, item_id, rating, timestamp = line.strip().split('::')
            if user_id not in recall_history:
                recall_history[user_id] = set(recall_top_k_items(user_id, user_item_history, item_similarity, k))
            if item_id in recall_history[user_id]:
                hits += 1
            total += 1
    return hits / total if total > 0 else 0.0


if __name__ == "__main__":
    # 构建用户-物品交互历史
    train_file_path = '/data2/zhy/Movie_Recsys/MovieLens_1M_data/train_ratings.dat'
    test_file_path = '/data2/zhy/Movie_Recsys/MovieLens_1M_data/test_ratings.dat'
    user_item_history = build_user_item_history(train_file_path)

    # 计算物品相似度矩阵
    item_similarity = compute_item_similarity(user_item_history)

    # 计算测试集的命中率
    hr = hit_rate(test_file_path, user_item_history, item_similarity, k=10)
    print(f"Hit Rate@10: {hr:.6f}")