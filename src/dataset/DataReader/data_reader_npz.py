import bisect
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class DataReaderNPZ(Dataset):
    """
    NPZ 特征数据加载器。

    支持两种输入：
    1. 单个 `.npz` 文件
    2. 分片目录（包含多个 `.npz` 文件，按文件名排序后拼接）
    """

    def __init__(
        self,
        config,
        feature_file_path: str = None,
        train_user_id_set: set = None,
        filter_negative: bool = False,
    ):
        """
        初始化 NPZ 数据集读取器。

        参数说明：
        - config: 全局配置对象，至少需要包含 `features` 下的三类特征定义
          （sparse_feature_names / dense_feature_names / array_feature_names）
          以及 `array_max_length`。
        - feature_file_path: 可以是单个 npz 文件路径，也可以是包含多个分片 npz 的目录。
        - train_user_id_set: 若提供，则仅保留 user_id 在该集合中的样本（常用于 warm user 评估）。
        - filter_negative: 若为 True，则仅保留正样本（label > 0）。
        """
        self.sparse_features = set(config.features.sparse_feature_names)
        self.dense_features = set(config.features.dense_feature_names)
        self.array_features = set(config.features.array_feature_names)
        self.array_max_length = config.features.array_max_length

        if feature_file_path is None:
            raise ValueError("Data file path must be provided.")
        self.data_path = Path(feature_file_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file path does not exist: {self.data_path}")

        self.npz_files = self._resolve_npz_files(self.data_path)
        if not self.npz_files:
            raise FileNotFoundError(f"No npz files found from path: {self.data_path}")

        self.file_row_counts, self.cum_row_counts = self._build_file_row_meta(self.npz_files)
        self.total_rows = int(self.cum_row_counts[-1]) if self.cum_row_counts else 0

        self._active_file_idx: Optional[int] = None
        self._active_npz = None

        # None 表示全量样本；不为 None 时存放“全局行号”子集
        self.selected_indices: Optional[np.ndarray] = None

        if train_user_id_set is not None:
            self.filter_data_by_user_id(train_user_id_set)
        if filter_negative:
            self.filter_negative_data()

    def _resolve_npz_files(self, path: Path) -> List[Path]:
        """
        解析输入路径，返回需要读取的 npz 文件列表。

        规则：
        - 若输入是文件：必须是 `.npz`，返回单元素列表。
        - 若输入是目录：收集目录下所有 `.npz` 文件，并按文件名排序。
          排序可确保 `part_000001, part_000002, ...` 的自然拼接顺序稳定。
        """
        if path.is_file():
            if path.suffix.lower() != ".npz":
                raise ValueError(f"Expected a .npz file, got: {path}")
            return [path]

        # 目录模式：按文件名排序，保证分片顺序稳定
        files = sorted([p for p in path.glob("*.npz") if p.is_file()])
        return files

    def _build_file_row_meta(self, files: List[Path]):
        """
        构建分片文件的行数元信息。

        返回：
        - row_counts: 每个文件各自的样本数
        - cum_counts: 前缀和数组，用于全局下标 -> 分片下标的二分定位
          例如 row_counts=[3,5,2]，则 cum_counts=[3,8,10]
        """
        row_counts = []
        cum_counts = []
        running = 0
        for f in files:
            with np.load(f, allow_pickle=False) as data:
                if "label" not in data.files:
                    raise KeyError(f"'label' not found in npz file: {f}")
                n = int(data["label"].shape[0])
            row_counts.append(n)
            running += n
            cum_counts.append(running)
        return row_counts, cum_counts

    def _locate_global_idx(self, global_idx: int):
        """
        将“全局样本下标”映射到“分片文件下标 + 分片内局部下标”。

        通过 `bisect_right(cum_row_counts, global_idx)` 二分定位所属分片，
        时间复杂度 O(log N_files)。
        """
        file_idx = bisect.bisect_right(self.cum_row_counts, global_idx)
        file_start = 0 if file_idx == 0 else self.cum_row_counts[file_idx - 1]
        local_idx = global_idx - file_start
        return file_idx, int(local_idx)

    def _ensure_active_file(self, file_idx: int):
        """
        确保目标分片文件已被打开并缓存为当前活动文件。

        该方法实现“懒加载 + 单文件缓存”：
        - 连续访问同一分片时避免重复 open；
        - 切换分片时关闭旧句柄，减少文件句柄占用。
        """
        if self._active_file_idx == file_idx and self._active_npz is not None:
            return
        if self._active_npz is not None:
            self._active_npz.close()
        self._active_npz = np.load(self.npz_files[file_idx], allow_pickle=False)
        self._active_file_idx = file_idx

    def __len__(self) -> int:
        """
        返回当前数据集长度。

        - 未做过滤时：返回全量样本数
        - 做过过滤时：返回筛选子集长度
        """
        if self.selected_indices is None:
            return self.total_rows
        return int(self.selected_indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        """
        根据样本下标读取并构造模型输入字典。

        返回字段包含：
        - 稀疏特征: int
        - 稠密特征: float
        - 数组特征: torch.long 向量
        - 数组 mask: torch.float32 向量
        - label: torch.float32 向量
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        # 若启用过滤，idx 先映射到全局行号；否则 idx 本身即全局行号
        global_idx = int(self.selected_indices[idx]) if self.selected_indices is not None else idx
        # 再从全局行号定位到具体分片和局部行号
        file_idx, local_idx = self._locate_global_idx(global_idx)
        self._ensure_active_file(file_idx)
        data = self._active_npz

        ret_datas: Dict[str, Union[torch.Tensor, int, float]] = {}

        # 稀疏特征
        for fea in self.sparse_features:
            if fea in data.files:
                ret_datas[fea] = int(data[fea][local_idx])
            else:
                ret_datas[fea] = 0

        # 稠密特征
        for fea in self.dense_features:
            if fea in data.files:
                ret_datas[fea] = float(data[fea][local_idx])
            else:
                ret_datas[fea] = 0.0

        # 数组特征与 mask
        for fea in self.array_features:
            max_len = self.array_max_length.get(fea)
            if max_len is None:
                raise ValueError(f"Max length for array feature '{fea}' missing in config.")

            if fea in data.files:
                arr = np.asarray(data[fea][local_idx], dtype=np.int64)
            else:
                arr = np.full((int(max_len),), -1, dtype=np.int64)

            if arr.shape[0] != int(max_len):
                raise ValueError(
                    f"Array feature '{fea}' length mismatch: got {arr.shape[0]}, expected {int(max_len)}"
                )

            mask_name = f"{fea}_mask"
            if mask_name in data.files:
                mask = np.asarray(data[mask_name][local_idx], dtype=np.float32)
            else:
                # 兼容历史文件：若未保存 mask，则按 -1 padding 规则在线构造
                mask = (arr != -1).astype(np.float32)

            ret_datas[fea] = torch.tensor(arr, dtype=torch.long)
            ret_datas[mask_name] = torch.tensor(mask, dtype=torch.float32)

        # 标签
        if "label" not in data.files:
            raise KeyError(f"'label' not found in npz file: {self.npz_files[file_idx]}")
        label_row = np.asarray(data["label"][local_idx], dtype=np.float32)
        ret_datas["label"] = torch.tensor(label_row, dtype=torch.float32)
        return ret_datas

    def _merge_selected(self, new_indices: np.ndarray):
        """
        合并筛选结果。

        语义为“与已有筛选条件取交集”：
        - 第一次筛选：selected = new
        - 后续筛选：selected = selected ∩ new
        """
        new_indices = np.asarray(new_indices, dtype=np.int64)
        if self.selected_indices is None:
            self.selected_indices = np.sort(new_indices)
            return
        # 与已有筛选结果取交集，保持升序
        mask = np.isin(self.selected_indices, new_indices)
        self.selected_indices = self.selected_indices[mask]

    def get_user_id_set(self, feature_name: str = "user_id") -> set:
        """
        扫描所有分片，收集指定特征（默认 user_id）的唯一值集合。
        """
        unique_ids = set()
        for f in self.npz_files:
            with np.load(f, allow_pickle=False) as data:
                if feature_name not in data.files:
                    continue
                vals = np.asarray(data[feature_name])
                for v in np.unique(vals):
                    unique_ids.add(int(v))
        return unique_ids

    def filter_data_by_user_id(self, user_id_set: set, feature_name: str = "user_id"):
        """
        根据给定 user_id 集合过滤样本。

        实现方式：
        - 逐分片读取 user_id 列
        - 找到匹配行的局部下标并映射为全局下标
        - 与当前 selected_indices 做交集合并
        """
        if not user_id_set:
            self.selected_indices = np.empty((0,), dtype=np.int64)
            return

        candidate_globals = []
        start = 0
        target_ids = np.asarray(list(user_id_set))
        for f, rows in zip(self.npz_files, self.file_row_counts):
            with np.load(f, allow_pickle=False) as data:
                if feature_name not in data.files:
                    start += rows
                    continue
                ids = np.asarray(data[feature_name])
                keep_mask = np.isin(ids, target_ids)
                local_keep = np.nonzero(keep_mask)[0]
                if local_keep.size > 0:
                    candidate_globals.append(local_keep + start)
            start += rows

        selected = (
            np.concatenate(candidate_globals).astype(np.int64)
            if candidate_globals
            else np.empty((0,), dtype=np.int64)
        )
        self._merge_selected(selected)

    def filter_negative_data(self):
        """
        过滤负样本，仅保留正样本。

        判定规则：
        - 若 label 为一维：label > 0
        - 若 label 为二维：取第一列 > 0
        """
        candidate_globals = []
        start = 0
        for f, rows in zip(self.npz_files, self.file_row_counts):
            with np.load(f, allow_pickle=False) as data:
                if "label" not in data.files:
                    start += rows
                    continue
                labels = np.asarray(data["label"], dtype=np.float32)
                if labels.ndim == 1:
                    keep_mask = labels > 0
                else:
                    keep_mask = labels[:, 0] > 0
                local_keep = np.nonzero(keep_mask)[0]
                if local_keep.size > 0:
                    candidate_globals.append(local_keep + start)
            start += rows

        selected = (
            np.concatenate(candidate_globals).astype(np.int64)
            if candidate_globals
            else np.empty((0,), dtype=np.int64)
        )
        self._merge_selected(selected)

    def __del__(self):
        """
        析构时释放当前活动 npz 文件句柄。
        """
        if getattr(self, "_active_npz", None) is not None:
            try:
                self._active_npz.close()
            except Exception:
                pass
