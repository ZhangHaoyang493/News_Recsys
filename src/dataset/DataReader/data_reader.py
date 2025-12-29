import torch
import os
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from typing import Dict, Union, List

class DataReader(Dataset):
    """
    通用特征数据加载器 (General Feature Data Loader)
    
    解析格式为 "feature_name:value" 的文本文件，并根据配置文件将其转换为
    Sparse(Embedding ID), Dense(Float), 或 Array(Sequence) 类型的 Tensor。
    """

    def __init__(self, config_path: str, feature_file_path: str = None):
        """
        初始化数据集

        Args:
            config_path (str): YAML 配置文件路径
            feature_file_path (str, optional): 数据文件路径. 默认为 None.
        
        Raises:
            ValueError: 如果未提供数据路径或数组特征缺少长度配置
            FileNotFoundError: 如果数据文件不存在
        """
        # 加载配置
        config = OmegaConf.load(config_path)

        # 使用 set 替代 list 以优化 __getitem__ 中的查找速度 (O(1) vs O(N))
        self.sparse_features = set(config.features.sparse_feature_names)
        self.dense_features = set(config.features.dense_feature_names)
        self.array_features = set(config.features.array_feature_names)
        
        # 获取数组特征的最大长度配置
        self.array_max_length = config.features.array_max_length

        # 校验数据路径
        self.data_path = feature_file_path
        if self.data_path is None:
            raise ValueError("Data file path must be provided.")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # 读取数据到内存
        # 注意：如果数据集过大无法放入内存，建议改写为基于 seek/tell 的索引式读取或使用 IterableDataset
        with open(self.data_path, 'r', encoding='utf-8') as f:
            # 过滤空行并去除首尾空白
            self.data_lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.data_lines)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        # 获取原始文本行
        raw_line = self.data_lines[idx]
        
        # 数据格式预设: "feat1:val1 feat2:val2 ... \t label1 label2"
        try:
            feature_part, label_part = raw_line.split('\t')
        except ValueError:
            raise ValueError(f"Line {idx} format error: missing tab separator between features and labels.")

        feature_items = feature_part.split(' ')
        ret_datas = {}
        
        for item in feature_items:
            # 解析 key:value 对
            if ':' not in item:
                raise ValueError(f"Feature item format error: '{item}' does not contain ':' separator.")
            feature_name, val_str = item.split(':', 1)

            # 1. 处理稀疏特征 (Sparse / ID)
            if feature_name in self.sparse_features:
                ret_datas[feature_name] = int(val_str)

            # 2. 处理稠密特征 (Dense / Float)
            elif feature_name in self.dense_features:
                ret_datas[feature_name] = float(val_str)

            # 3. 处理变长数组特征 (Array / Sequence)
            elif feature_name in self.array_features:
                max_len = self.array_max_length.get(feature_name)
                if max_len is None:
                    raise ValueError(f"Max length for array feature '{feature_name}' missing in config.")

                # 解析数组字符串 (假设逗号分隔: "1,2,3")
                if val_str:
                    indices = [int(x) for x in val_str.split(',')]
                else:
                    indices = []
                
                seq_len = len(indices)

                # 构造 Padding 和 Mask
                if seq_len < max_len:
                    # 长度不足：补0
                    pad_len = max_len - seq_len
                    indices.extend([0] * pad_len)
                    # Mask: 真实数据为1，Padding为0
                    mask = [1.0] * seq_len + [0.0] * pad_len
                else:
                    # 长度超出：截断
                    indices = indices[:max_len]
                    mask = [1.0] * max_len

                # 转为 Tensor
                ret_datas[feature_name] = torch.tensor(indices, dtype=torch.long)
                ret_datas[f"{feature_name}_mask"] = torch.tensor(mask, dtype=torch.float32)

        # 处理标签 (支持多目标)
        labels = [float(l) for l in label_part.strip().split(' ')]
        ret_datas['label'] = torch.tensor(labels, dtype=torch.float32)

        return ret_datas