import argparse
import pandas as pd
import json  # 用于保存结果文件 (map.json)，不再用于读取配置文件
import os
from pathlib import Path
from omegaconf import OmegaConf # 引入 OmegaConf
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Process MIND data with OmegaConf (YAML).")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file (yaml)')
    return parser.parse_args()

def build_and_save_maps(data_root, out_base):
    """
    Step 1: 构建全局映射 (NewsID -> Int, UserID -> Int)
    同时保存仅出现在训练集中的 User ID (mapped)
    """
    print("Step 1: 构建全局 ID 映射字典...")
    
    sub_datasets = ['MINDsmall_train', 'MINDsmall_dev']
    
    # --- 1. 构建 News ID 映射 ---
    news_id_list = []
    print("  - 正在读取所有 News ID...")
    for sub in sub_datasets:
        path = os.path.join(data_root, sub, 'news.tsv')
        if os.path.exists(path):
            df_tmp = pd.read_csv(path, sep='\t', header=None, usecols=[0], names=['news_id'])
            news_id_list.append(df_tmp['news_id'])
    
    if not news_id_list:
        raise FileNotFoundError(f"未在 {data_root} 下找到任何 news.tsv 文件，请检查 data_path 配置。")

    all_news_ids = pd.concat(news_id_list).unique()
    print(f"  - 全局 News 数量: {len(all_news_ids)}")
    news_map = {nid: int(i + 1) for i, nid in enumerate(all_news_ids)}
    
    # --- 2. 构建 User ID 映射 & 提取训练集 User ---
    user_id_list = []
    train_raw_users = set()  # 用于暂存训练集的原始User ID

    print("  - 正在读取所有 User ID...")
    for sub in sub_datasets:
        path = os.path.join(data_root, sub, 'behaviors.tsv')
        if os.path.exists(path):
            df_tmp = pd.read_csv(path, sep='\t', header=None, usecols=[1], names=['user_id'])
            user_id_list.append(df_tmp['user_id'])
            
            # 检测是否为训练集 (包含 'train' 关键字)
            if 'train' in sub:
                print(f"    -> 检测到训练集: {sub}，正在记录训练集用户...")
                train_raw_users.update(df_tmp['user_id'].unique())
            
    all_user_ids = pd.concat(user_id_list).unique()
    print(f"  - 全局 User 数量: {len(all_user_ids)}")
    user_map = {uid: int(i + 1) for i, uid in enumerate(all_user_ids)}

    # 将训练集原始ID转换为Map后的ID
    print(f"  - 正在转换并保存训练集用户列表 (原始数量: {len(train_raw_users)})...")
    train_user_ids_mapped = [user_map[uid] for uid in train_raw_users if uid in user_map]

    # --- 3. 保存字典 ---
    preprocess_dir = os.path.join(out_base, 'preprocess')
    Path(preprocess_dir).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(preprocess_dir, 'news_id_map.json'), 'w', encoding='utf-8') as f:
        json.dump(news_map, f)
    with open(os.path.join(preprocess_dir, 'user_id_map.json'), 'w', encoding='utf-8') as f:
        json.dump(user_map, f)
    
    # 保存训练集用户ID文件
    train_users_path = os.path.join(preprocess_dir, 'train_user_ids.json')
    with open(train_users_path, 'w', encoding='utf-8') as f:
        json.dump(train_user_ids_mapped, f)
    print(f"  - [完成] 训练集用户ID已保存至: {train_users_path}")
        
    return news_map, user_map

def strict_map_series(series, mapping_dict, col_name):
    """
    辅助函数：对 Series 进行映射，如果发现未知 Key，立即报错
    """
    mapped = series.map(mapping_dict)
    if mapped.isna().any():
        unknown_ids = series[mapped.isna()].unique()
        raise KeyError(f"在列 '{col_name}' 中发现未知 ID，不在全局字典中！\n示例未知 ID: {unknown_ids[:5]}...")
    return mapped.astype(int)

def process_and_save_all_news(data_root, sub_datasets, output_path, news_map):
    print(f"\nStep 2: 处理并合并所有 News 文件...")
    col_names = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    
    df_list = []
    for sub in sub_datasets:
        path = os.path.join(data_root, sub, 'news.tsv')
        if os.path.exists(path):
            print(f"  - 读取: {path}")
            df = pd.read_csv(path, sep='\t', names=col_names, quoting=3)
            df_list.append(df)
            
    if not df_list:
        print("  ! 未找到任何 News 文件，跳过处理。")
        return

    # 1. 合并
    print("  - 正在合并 News 数据...")
    full_news_df = pd.concat(df_list, ignore_index=True)
    
    # 2. 去重
    print(f"  - 去重前数量: {len(full_news_df)}")
    full_news_df.drop_duplicates(subset=['news_id'], inplace=True)
    print(f"  - 去重后数量: {len(full_news_df)}")

    # 3. 映射 ID
    print("  - 映射 News ID...")
    full_news_df['news_id'] = strict_map_series(full_news_df['news_id'], news_map, 'news_id')

    # 4. 保存
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    full_news_df.to_csv(output_path, index=False, sep='\t', header=False)
    print(f"  - [完成] 合并后的 News 已保存至: {output_path}")

def process_behaviors(input_path, output_path, user_map, news_map):
    if not os.path.exists(input_path): return

    print(f"正在处理 Behaviors 文件: {input_path}")
    col_names = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    df = pd.read_csv(input_path, sep='\t', names=col_names, quoting=3)
    
    # 1. 时间处理 & 排序
    df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df = df.sort_values(by='time', ascending=True)
    df['time'] = df['time'].astype('int64') // 10**9

    # 2. 映射 User ID
    df['user_id'] = strict_map_series(df['user_id'], user_map, 'user_id')

    # 3. 映射 History ID
    df['history'] = df['history'].fillna('')

    def map_history_strict(hist_str):
        if not hist_str:
            return ""
        raw_ids = hist_str.split(' ')
        try:
            mapped_ids = [str(news_map[nid]) for nid in raw_ids]
        except KeyError as e:
            raise KeyError(f"数据错误: History 序列 '{hist_str}' 中发现未知 ID {e}")
        return " ".join(mapped_ids)

    df['history'] = df['history'].apply(map_history_strict)

    # 4. 裂变 (Explode) Impressions
    df['impressions'] = df['impressions'].str.split(' ')
    df_exploded = df.explode('impressions').reset_index(drop=True)

    # 5. 拆分 Impression Item ID
    split_cols = df_exploded['impressions'].str.rsplit('-', n=1, expand=True)
    item_id_str = split_cols[0]
    
    # 6. 映射 Item ID
    df_exploded['item_id'] = strict_map_series(item_id_str, news_map, 'impression_item_id')

    # 处理 Label
    df_exploded['label'] = pd.to_numeric(split_cols[1])
    
    # 删除原始列
    df_exploded.drop(columns=['impressions'], inplace=True)

    # 7. 保存
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    df_exploded.to_csv(output_path, index=False, sep='\t', header=False)
    print(f"  - [完成] Behaviors 已保存至: {output_path}")

def main():
    args = parse_args()
    
    # ==========================================
    # 修改点：使用 OmegaConf 读取 YAML 配置
    # ==========================================
    print(f"Loading config from: {args.config}")
    config = OmegaConf.load(args.config)
    
    # 从 config.paths 中获取路径
    data_root = config.paths.data_path
    out_base = config.paths.out_basedir
    
    print(f"Data Root: {data_root}")
    print(f"Output Base: {out_base}")

    if not os.path.exists(os.path.join(out_base, 'preprocess')):
        os.makedirs(os.path.join(out_base, 'preprocess'))
    else:
        shutil.rmtree(os.path.join(out_base, 'preprocess'))
        os.makedirs(os.path.join(out_base, 'preprocess'))
    
    # 这里假设 MINDsmall 的目录结构是固定的
    sub_datasets = ['MINDsmall_train', 'MINDsmall_dev']

    # Step 1: 构建字典 (同时保存 train_user_ids.json)
    news_map, user_map = build_and_save_maps(data_root, out_base)
    
    # Step 2: 集中处理所有 News 并保存为一个文件
    all_news_output = os.path.join(out_base, 'preprocess', 'all_news_preprocess.csv')
    process_and_save_all_news(data_root, sub_datasets, all_news_output, news_map)

    # Step 3: 分别处理 Behaviors
    print("\nStep 3: 开始处理 Behaviors 文件...")
    for sub_ds in sub_datasets:
        file_suffix = sub_ds.split('_')[-1] # train 或 dev
        
        input_behaviors = os.path.join(data_root, sub_ds, 'behaviors.tsv')
        output_behaviors = os.path.join(out_base, 'preprocess', f'{file_suffix}_behaviors_processed.csv')
        
        process_behaviors(input_behaviors, output_behaviors, user_map, news_map)
        
    print("\n所有任务全部完成。")

if __name__ == "__main__":
    main()