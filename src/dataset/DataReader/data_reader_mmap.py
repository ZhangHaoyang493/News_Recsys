import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class DataReaderMmap(Dataset):
    """
    基于 mmap 目录的数据读取器（单 part 版本）。

    期望目录结构：
    - split_dir/
      - manifest.json
      - part_000001/
        - <feature>.npy
        - <feature>_mask.npy (array feature)
        - label.npy
    """

    def __init__(self, config, feature_file_path: str = None, train_user_id_set: set = None, filter_negative: bool = False):
        self.sparse_features = set(config.features.sparse_feature_names)
        self.dense_features = set(config.features.dense_feature_names)
        self.array_features = set(config.features.array_feature_names)
        self.array_max_length = config.features.array_max_length

        if feature_file_path is None:
            raise ValueError("Data file path must be provided.")
        self.data_path = Path(feature_file_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file path does not exist: {self.data_path}")

        self.manifest = self._load_manifest(self.data_path)
        self.part_info = self._resolve_single_part(self.manifest, self.data_path)
        self.part_dir = self.data_path / self.part_info["path"]
        self.total_rows = int(self.part_info["rows"])
        self.arrays = self._load_single_part_arrays(self.part_dir)

        self.selected_indices: Optional[np.ndarray] = None
        if train_user_id_set is not None:
            self.filter_data_by_user_id(train_user_id_set)
        if filter_negative:
            self.filter_negative_data()

    def _load_manifest(self, split_dir: Path) -> Dict:
        manifest_path = split_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in mmap directory: {split_dir}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "npy_mmap":
            raise ValueError(f"Unsupported manifest format: {manifest.get('format')}, expected 'npy_mmap'")
        if "parts" not in manifest or not manifest["parts"]:
            raise ValueError(f"Invalid manifest parts in: {manifest_path}")
        return manifest

    def _resolve_single_part(self, manifest: Dict, split_dir: Path) -> Dict:
        """
        解析并校验单 part 信息。
        若 manifest 里有多个 part，直接报错，避免读到与当前约定不一致的数据格式。
        """
        parts = manifest["parts"]
        if len(parts) != 1:
            raise ValueError(
                f"DataReaderMmap(single-part) expects exactly 1 part, "
                f"but got {len(parts)} in {split_dir / 'manifest.json'}"
            )

        part_info = parts[0]
        part_path = split_dir / part_info["path"]
        if not part_path.exists():
            raise FileNotFoundError(f"Part directory does not exist: {part_path}")

        rows = int(part_info.get("rows", 0))
        if rows <= 0:
            raise ValueError(f"Invalid part rows={rows} in manifest: {split_dir / 'manifest.json'}")
        return part_info

    def _load_single_part_arrays(self, part_dir: Path) -> Dict[str, np.ndarray]:
        """
        一次性 mmap 加载单 part 的全部需要字段。
        """
        arrays = {}

        # 只加载 __getitem__ 会用到的字段，统一 mmap_mode='r' 避免额外内存拷贝。
        for fea in self.sparse_features:
            fp = part_dir / f"{fea}.npy"
            arrays[fea] = np.load(fp, mmap_mode="r") if fp.exists() else None
        for fea in self.dense_features:
            fp = part_dir / f"{fea}.npy"
            arrays[fea] = np.load(fp, mmap_mode="r") if fp.exists() else None
        for fea in self.array_features:
            f_fp = part_dir / f"{fea}.npy"
            m_fp = part_dir / f"{fea}_mask.npy"
            arrays[fea] = np.load(f_fp, mmap_mode="r") if f_fp.exists() else None
            arrays[f"{fea}_mask"] = np.load(m_fp, mmap_mode="r") if m_fp.exists() else None

        label_fp = part_dir / "label.npy"
        if not label_fp.exists():
            raise FileNotFoundError(f"label.npy missing in part directory: {part_dir}")
        arrays["label"] = np.load(label_fp, mmap_mode="r")
        return arrays

    def __len__(self) -> int:
        if self.selected_indices is None:
            return self.total_rows
        return int(self.selected_indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        local_idx = int(self.selected_indices[idx]) if self.selected_indices is not None else idx
        arrs = self.arrays

        ret_datas: Dict[str, Union[torch.Tensor, int, float]] = {}

        for fea in self.sparse_features:
            data = arrs.get(fea)
            ret_datas[fea] = int(data[local_idx]) if data is not None else 0

        for fea in self.dense_features:
            data = arrs.get(fea)
            ret_datas[fea] = float(data[local_idx]) if data is not None else 0.0

        for fea in self.array_features:
            max_len = self.array_max_length.get(fea)
            if max_len is None:
                raise ValueError(f"Max length for array feature '{fea}' missing in config.")

            data = arrs.get(fea)
            if data is None:
                seq = np.full((int(max_len),), -1, dtype=np.int64)
            else:
                seq = np.asarray(data[local_idx], dtype=np.int64)
            if seq.shape[0] != int(max_len):
                raise ValueError(
                    f"Array feature '{fea}' length mismatch: got {seq.shape[0]}, expected {int(max_len)}"
                )

            mask_data = arrs.get(f"{fea}_mask")
            if mask_data is None:
                mask = (seq != -1).astype(np.float32)
            else:
                mask = np.asarray(mask_data[local_idx], dtype=np.float32)

            ret_datas[fea] = torch.tensor(seq, dtype=torch.long)
            ret_datas[f"{fea}_mask"] = torch.tensor(mask, dtype=torch.float32)

        label_row = np.asarray(arrs["label"][local_idx], dtype=np.float32)
        ret_datas["label"] = torch.tensor(label_row, dtype=torch.float32)
        return ret_datas

    def _merge_selected(self, new_indices: np.ndarray):
        new_indices = np.asarray(new_indices, dtype=np.int64)
        if self.selected_indices is None:
            self.selected_indices = np.sort(new_indices)
            return
        mask = np.isin(self.selected_indices, new_indices)
        self.selected_indices = self.selected_indices[mask]

    def get_user_id_set(self, feature_name: str = "user_id") -> set:
        unique_ids = set()
        fp = self.part_dir / f"{feature_name}.npy"
        if not fp.exists():
            return unique_ids
        vals = np.load(fp, mmap_mode="r")
        for v in np.unique(vals):
            unique_ids.add(int(v))
        return unique_ids

    def filter_data_by_user_id(self, user_id_set: set, feature_name: str = "user_id"):
        if not user_id_set:
            self.selected_indices = np.empty((0,), dtype=np.int64)
            return

        candidate_indices = []
        target_ids = np.asarray(list(user_id_set))
        fp = self.part_dir / f"{feature_name}.npy"
        if fp.exists():
            ids = np.load(fp, mmap_mode="r")
            keep_mask = np.isin(ids, target_ids)
            local_keep = np.nonzero(keep_mask)[0]
            if local_keep.size > 0:
                candidate_indices.append(local_keep)

        selected = (
            np.concatenate(candidate_indices).astype(np.int64)
            if candidate_indices
            else np.empty((0,), dtype=np.int64)
        )
        self._merge_selected(selected)

    def filter_negative_data(self):
        candidate_indices = []
        fp = self.part_dir / "label.npy"
        if fp.exists():
            labels = np.asarray(np.load(fp, mmap_mode="r"), dtype=np.float32)
            if labels.ndim == 1:
                keep_mask = labels > 0
            else:
                keep_mask = labels[:, 0] > 0
            local_keep = np.nonzero(keep_mask)[0]
            if local_keep.size > 0:
                candidate_indices.append(local_keep)

        selected = (
            np.concatenate(candidate_indices).astype(np.int64)
            if candidate_indices
            else np.empty((0,), dtype=np.int64)
        )
        self._merge_selected(selected)
