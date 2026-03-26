import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm

def process_pretrained_embedding(vec_file, map_file, output_path, feature_name, emb_dim=None, vocab_size=None):
    """
    将预训练的 vec 文件转换为 PyTorch Embedding 权重并保存。
    
    Args:
        vec_file (str): 原始 embedding 向量文件路径 (TSV格式: key\tval1\tval2...)
        map_file (str): 特征各值到 ID 的映射字典文件路径 (JSON: {feature_name: {raw_val: idx, ...}})
        output_path (str): 处理后的权重保存路径 (.pt 或 .npy)
        feature_name (str): 在映射字典中对应的特征名称 (如 'item_title_entity_id')
        emb_dim (int, optional): Embedding 维度。如果不填则自动从文件中读取。
        vocab_size (int, optional): 词表大小。如果不填则根据映射字典的最大 ID 确定。
    """
    
    print(f"Loading mapping dictionary from {map_file}...")
    with open(map_file, 'r', encoding='utf-8') as f:
        full_map = json.load(f)
    
    if feature_name not in full_map:
        # 尝试查找是否在 tuple 形式的 key 中 (feature_map_val2idx 的结构通常是 {feat: [dict, max_idx]})
        # 这里假设输入的 json 是 feature_extractor 输出的 original_val_2_embedding_idx_dict.json
        # 它的结构通常是 {feature_name: {val: idx}} 或者 {feature_name: [{val: idx}, max_idx]}
        # 我们可以做一些兼容性处理
        raise KeyError(f"Feature name '{feature_name}' not found in mapping file keys: {list(full_map.keys())}")

    # 获取该特征的映射字典
    val2idx_entry = full_map[feature_name]
    
    # 兼容 feature_extractor_base.py 输出的结构
    # 结构示例: "category": [{"entertainment": 1, ...}, 19]
    if isinstance(val2idx_entry, list):
        if len(val2idx_entry) >= 1:
            val2idx = val2idx_entry[0]
            # 尝试获取 max_idx
            if len(val2idx_entry) >= 2:
                max_idx_in_map = val2idx_entry[1]
            else:
                max_idx_in_map = max(val2idx.values()) if val2idx else 0
        else:
             raise ValueError(f"List for feature '{feature_name}' is empty.")
    elif isinstance(val2idx_entry, dict):
        val2idx = val2idx_entry
        max_idx_in_map = max(val2idx.values()) if val2idx else 0
    else:
        raise ValueError(f"Unexpected format for feature '{feature_name}' in mapping file.")

    actual_vocab_size = max_idx_in_map + 1
    if vocab_size is None:
        vocab_size = actual_vocab_size
        print(f"Auto-detected vocab size: {vocab_size}")
    elif vocab_size < actual_vocab_size:
        print(f"Warning: Provided vocab_size ({vocab_size}) is smaller than max ID ({max_idx_in_map}) in mapping. Indices out of bound will be ignored.")

    # 第一次读取以确定维度 (如果未指定)
    if emb_dim is None:
        with open(vec_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    emb_dim = len(parts) - 1 # 第一列是 Key
                    print(f"Auto-detected embedding dimension: {emb_dim}")
                    break
        if emb_dim is None:
            raise ValueError("Could not detect embedding dimension from file.")

    # 初始化全 0 的 Embedding 矩阵
    # 使用 padding_idx=0 的习惯，通常 index 0 也是全 0
    embedding_matrix = torch.zeros((vocab_size, emb_dim), dtype=torch.float32)
    
    print(f"Processing embedding file: {vec_file}")
    print(f"Target Feature: {feature_name}")
    print(f"Matrix Shape: {embedding_matrix.shape}")

    hit_count = 0
    miss_count = 0

    with open(vec_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Vectors"):
            parts = line.strip().split('\t')
            if len(parts) < emb_dim + 1:
                continue
            
            key = parts[0]
            
            # 尝试在映射表中查找 ID
            # 原始映射表中的 Key可能是字符串或数字，这里做统一处理
            if key in val2idx:
                idx = val2idx[key]
            else:
                # 尝试类型转换查找
                try:
                    # 如果 key 是数字字符串 '123' 但字典里存的是 int 123
                    if isinstance(list(val2idx.keys())[0], int): 
                         if int(key) in val2idx:
                             idx = val2idx[int(key)]
                         else:
                             miss_count += 1
                             continue
                    else:
                        miss_count += 1
                        continue
                except:
                    miss_count += 1
                    continue

            if idx >= vocab_size:
                continue

            # 解析向量
            try:
                vec_values = [float(x) for x in parts[1:]]
                if len(vec_values) != emb_dim:
                    # 某些行可能截断，尝试截取或补全，这里选择跳过
                    print(f"Warning: Dimension mismatch for key {key}. Expected {emb_dim}, got {len(vec_values)}")
                    continue
                
                embedding_matrix[idx] = torch.tensor(vec_values)
                hit_count += 1
            except ValueError:
                print(f"Warning: Failed to parse vector for key {key}")
                continue

    print(f"Processing complete.")
    print(f"Total keys in mapping: {len(val2idx)}")
    print(f"Keys found in pretrained file: {hit_count}")
    print(f"Keys missing/not found: {miss_count}")
    print(f"Sparsity: {1 - (hit_count / len(val2idx)):.2%}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith('.npy'):
        np.save(output_path, embedding_matrix.numpy())
    else:
        torch.save(embedding_matrix, output_path)
    
    print(f"Saved embedding matrix to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pretrained vector file to PyTorch embedding weights.")
    
    parser.add_argument('--vec_file', type=str, required=True, 
                        help='Path to the pretrained vector file (TSV: key val1 val2...).')
    parser.add_argument('--map_file', type=str, required=True, 
                        help='Path to the ID mapping JSON file.')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to save the output (.pt or .npy).')
    parser.add_argument('--feature_name', type=str, required=True, 
                        help='The feature name in the mapping file to use (e.g., item_id, category).')
    parser.add_argument('--emb_dim', type=int, default=None, 
                        help='Embedding dimension. Auto-detected if not provided.')
    parser.add_argument('--vocab_size', type=int, default=None, 
                        help='Vocabulary size. Auto-detected from mapping if not provided.')

    args = parser.parse_args()

    process_pretrained_embedding(
        args.vec_file, 
        args.map_file, 
        args.output_path, 
        args.feature_name, 
        args.emb_dim, 
        args.vocab_size
    )
