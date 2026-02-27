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

    def __init__(self, config, feature_file_path: str = None, train_user_id_set: set = None, filter_negative: bool = False):
        """
        初始化数据集

        Args:
            config (str): YAML 配置
            feature_file_path (str, optional): 数据文件路径. 默认为 None.
        
        Raises:
            ValueError: 如果未提供数据路径或数组特征缺少长度配置
            FileNotFoundError: 如果数据文件不存在
        """
        # 使用 set 替代 list 以优化 __getitem__ 中的查找速度 (O(1) vs O(N))
        self.sparse_features = set(config.features.sparse_feature_names)
        self.dense_features = set(config.features.dense_feature_names)
        self.array_features = set(config.features.array_feature_names)
        self.vector_features = set(config.features.vector_feature_names)
        
        # 获取数组特征的最大长度配置
        self.array_max_length = config.features.array_max_length
        self.vector_max_length = config.features.vector_max_length

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

        if train_user_id_set is not None:
            self.filter_data_by_user_id(train_user_id_set)

        if filter_negative:
            self.filter_negative_data()

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
                    # 长度不足：补-1
                    pad_len = max_len - seq_len
                    indices.extend([-1] * pad_len)
                    # Mask: 真实数据为1，Padding为0
                    mask = [1.0] * seq_len + [0.0] * pad_len
                else:
                    # 长度超出：截断
                    indices = indices[:max_len]
                    mask = [1.0] * max_len

                # 转为 Tensor
                ret_datas[feature_name] = torch.tensor(indices, dtype=torch.long)
                ret_datas[f"{feature_name}_mask"] = torch.tensor(mask, dtype=torch.float32)

            # 4. 处理向量列表特征 (Vector List)
            elif feature_name in self.vector_features:
                dim = self.vector_feature_dim.get(feature_name)
                if dim is None:
                    raise ValueError(f"Dim for vector feature '{feature_name}' missing in config.")

                if val_str:
                     # val_str 格式: "dim,num,ele1,ele2..."
                    parts = [float(x) for x in val_str.split(',')]
                    # 至少包含 dim, num
                    if len(parts) >= 2:
                        input_dim = int(parts[0])
                        num = int(parts[1])
                        max_num = self.vector_max_length.get(feature_name)
                        if max_num is None:
                            raise ValueError(f"Max num for vector feature '{feature_name}' missing in config.")
                        
                        elements = parts[2:]
                        
                        if input_dim != dim:
                            raise ValueError(f"Dimension mismatch for vector feature '{feature_name}': expected {dim}, got {input_dim}")
                        
                        if len(elements) != dim * num:
                            raise ValueError(f"Element count mismatch for vector feature '{feature_name}': expected {dim*num}, got {len(elements)}")
                        
                        # 重塑 Tensor: (num, dim)
                        tensor = torch.tensor(elements, dtype=torch.float32).view(num, dim)

                        # Truncation & Padding
                        if num >= max_num:
                            # 截断或刚好
                            tensor = tensor[:max_num, :]
                            mask = [1.0] * max_num
                        else:
                            # Padding: 补 0 向量
                            pad_len = max_num - num
                            pad_tensor = torch.zeros((pad_len, dim), dtype=torch.float32)
                            tensor = torch.cat([tensor, pad_tensor], dim=0)
                            mask = [1.0] * num + [0.0] * pad_len
                            
                        ret_datas[feature_name] = tensor
                        # 添加 Mask: (max_num,)
                        ret_datas[f"{feature_name}_mask"] = torch.tensor(mask, dtype=torch.float32)
                    else:
                        raise ValueError(f"Invalid format for vector feature '{feature_name}': expected at least 2 headers (dim,num)")
                else:
                    raise ValueError(f"Empty value for vector feature '{feature_name}'")

        # 处理标签 (支持多目标)
        labels = [float(l) for l in label_part.strip().split(' ')]
        ret_datas['label'] = torch.tensor(labels, dtype=torch.float32)

        return ret_datas

    def get_user_id_set(self, feature_name: str = 'user_id') -> set:
        """
        获取数据集中包含的所有 User ID 的集合。
        
        Args:
            feature_name (str): User ID 在数据文件中的特征名称，默认为 'user_id'。
            
        Returns:
            set: 包含所有 User ID (int 或 str) 的集合。
        """
        unique_ids = set()
        prefix = f"{feature_name}:"
        
        for line in self.data_lines:
            try:
                # 获取特征部分 (TAB之前)
                if '\t' in line:
                    feature_part = line.split('\t', 1)[0]
                else:
                    feature_part = line
                
                # 查找包含 feature_name 的项
                for item in feature_part.strip().split(' '):
                    if item.startswith(prefix):
                        val_str = item.split(':', 1)[1]
                        # 尝试转换为 int
                        try:
                            val = int(val_str)
                        except ValueError:
                            val = val_str
                        unique_ids.add(val)
                        break
            except Exception:
                continue
                
        return unique_ids

    def filter_data_by_user_id(self, user_id_set: set, feature_name: str = 'user_id'):
        """
        根据提供的 User ID 集合过滤数据，只保留 User ID 在集合中的数据行。
        
        Args:
            user_id_set (set): 合法的 User ID 集合。
            feature_name (str): User ID 的特征名称，默认为 'user_id'。
        """
        filtered_lines = []
        prefix = f"{feature_name}:"

        for line in self.data_lines:
            try:
                # 快速检查
                if prefix not in line:
                    continue

                feature_part = line.split('\t', 1)[0]
                for item in feature_part.strip().split(' '):
                    if item.startswith(prefix):
                        val_str = item.split(':', 1)[1]
                        try:
                            val = int(val_str)
                        except ValueError:
                            val = val_str
                        
                        if val in user_id_set:
                            filtered_lines.append(line)
                        break
            except Exception:
                continue
        self.data_lines = filtered_lines


    def filter_negative_data(self):
        """
        如果当前是召回阶段，过滤掉负样本，因为召回阶段的负样本来自于随机采样/batch 内负样本
        """
        filtered_lines = []
        for line in self.data_lines:
            feature_part, label_part = line.split('\t')
            labels = float(label_part)
            # 只保留正样本
            if labels > 0:
                filtered_lines.append(line)
        self.data_lines = filtered_lines
        
        
