import json
import os
from typing import Any, Union, Optional

class FeatureIdMapper:
    def __init__(self, idx2val_path: str, val2idx_path: str):
        """
        初始化映射器，加载两个字典文件
        :param idx2val_path: embedding_idx_2_original_val_dict.json 的路径
        :param val2idx_path: original_val_2_embedding_idx_dict.json 的路径
        """
        self.idx2val_path = idx2val_path
        self.val2idx_path = val2idx_path
        
        # 加载数据
        self.idx2val_dict = self._load_json(idx2val_path)
        self.val2idx_dict = self._load_json(val2idx_path)
        
        # 打印加载信息
        print(f"[FeatureIdMapper] Loaded mappings for features: {list(self.idx2val_dict.keys())}")

    def _load_json(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary file not found: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON from {path}. Error: {e}")

    def get_emb_idx(self, feature_name: str, real_value: Any) -> Optional[int]:
        """
        输入真实值，获取 embedding index
        :param feature_name: 特征名称，例如 "user_id"
        :param real_value: 真实值，例如 "u1001" 或 25
        :return: embedding index (int) 或 None
        """
        # 1. 检查特征是否存在
        feature_map = self.val2idx_dict.get(feature_name)
        if feature_map is None:
            print(f"Warning: Feature '{feature_name}' not found in mapping.")
            return None

        # 2. 尝试查找
        # JSON 的 Key 总是字符串，所以我们将输入转为 str 尝试查找
        # 很多时候真实值可能是 int (如 age: 25)，但在 JSON key 中是 "25"
        key = str(real_value)
        
        if key in feature_map:
            return feature_map[key]
        
        return None

    def get_real_val(self, feature_name: str, emb_idx: int) -> Optional[Any]:
        """
        输入 embedding index，获取真实值
        :param feature_name: 特征名称，例如 "user_id"
        :param emb_idx: 整数索引，例如 0
        :return: 真实值 (str/int/float) 或 None
        """
        # 1. 检查特征是否存在
        feature_map = self.idx2val_dict.get(feature_name)
        if feature_map is None:
            print(f"Warning: Feature '{feature_name}' not found in mapping.")
            return None
            
        # 2. 尝试查找
        # 同样，emb_idx 在 JSON 中作为 Key 保存时变成了字符串
        # 例如 index 0 在 JSON 里是 "0"
        key = str(emb_idx)
        
        if key in feature_map:
            return feature_map[key]
            
        return None