import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from ...Logger.logging import Logger


logger = Logger.get_logger("Txt2NpzConverter")


def parse_line(raw_line: str) -> Tuple[Dict[str, str], List[float]]:
    """
    解析单行 txt 特征。

    输入格式：
    `feat1:val1 feat2:val2 ... \\t label1 [label2 ...]`
    """
    line = raw_line.strip()
    if not line:
        raise ValueError("Empty line")
    if "\t" not in line:
        raise ValueError("Missing tab separator between features and labels")

    feature_part, label_part = line.split("\t", 1)
    feature_dict: Dict[str, str] = {}
    for token in feature_part.strip().split(" "):
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid feature token without ':': {token}")
        key, val = token.split(":", 1)
        feature_dict[key] = val

    labels = [float(x) for x in label_part.strip().split(" ") if x != ""]
    if not labels:
        raise ValueError("Empty label field")
    return feature_dict, labels


def parse_scalar_int(value: str) -> int:
    """标量特征转 int，缺失或非法值回填为 0。"""
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_array_ints(value: str) -> List[int]:
    """数组特征字符串转 int 列表，空串返回空列表。"""
    if value is None or value == "":
        return []
    return [int(x) for x in value.split(",") if x != ""]


def allocate_buffers(
    rows: int,
    label_dim: int,
    feature_names: List[str],
    array_feature_names: List[str],
    array_max_length: Dict[str, int],
):
    """
    预分配分片缓冲区。
    - 标量：0
    - 数组：-1
    - mask：0.0
    - label：0.0
    """
    features = {}
    masks = {}
    for fea in feature_names:
        if fea in array_feature_names:
            max_len = int(array_max_length[fea])
            features[fea] = np.full((rows, max_len), -1, dtype=np.int64)
            masks[f"{fea}_mask"] = np.zeros((rows, max_len), dtype=np.float32)
        else:
            features[fea] = np.zeros((rows,), dtype=np.int64)
    labels = np.zeros((rows, label_dim), dtype=np.float32)
    return features, masks, labels


def save_part_to_npy(
    part_dir: Path,
    feature_names: List[str],
    array_feature_names: List[str],
    features: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    labels_arr: np.ndarray,
):
    """
    将单个分片写为列式 .npy 文件。
    """
    part_dir.mkdir(parents=True, exist_ok=True)
    for fea in feature_names:
        np.save(part_dir / f"{fea}.npy", features[fea], allow_pickle=False)
        if fea in array_feature_names:
            np.save(part_dir / f"{fea}_mask.npy", masks[f"{fea}_mask"], allow_pickle=False)
    np.save(part_dir / "label.npy", labels_arr, allow_pickle=False)


def convert_chunk_to_mmap_part(
    chunk_lines: List[str],
    part_idx: int,
    split_out_dir: str,
    feature_names: List[str],
    array_feature_names: List[str],
    array_max_length: Dict[str, int],
    label_dim: int,
):
    """
    子进程：把一个文本分块转换为一个 part 目录（列式 .npy）。
    """
    rows = len(chunk_lines)
    features, masks, labels_arr = allocate_buffers(
        rows=rows,
        label_dim=label_dim,
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
    )

    bad_lines = 0
    for row_idx, line in enumerate(chunk_lines):
        try:
            feature_dict, row_labels = parse_line(line)
            for fea in feature_names:
                raw_value = feature_dict.get(fea, "")
                if fea in array_feature_names:
                    max_len = int(array_max_length[fea])
                    arr_values = parse_array_ints(raw_value)
                    if len(arr_values) < max_len:
                        seq_len = len(arr_values)
                        if seq_len > 0:
                            features[fea][row_idx, :seq_len] = np.asarray(arr_values, dtype=np.int64)
                            masks[f"{fea}_mask"][row_idx, :seq_len] = 1.0
                    else:
                        arr_values = arr_values[-max_len:]
                        features[fea][row_idx, :] = np.asarray(arr_values, dtype=np.int64)
                        masks[f"{fea}_mask"][row_idx, :] = 1.0
                else:
                    features[fea][row_idx] = parse_scalar_int(raw_value)

            max_copy = min(len(row_labels), label_dim)
            labels_arr[row_idx, :max_copy] = np.asarray(row_labels[:max_copy], dtype=np.float32)
        except Exception:
            bad_lines += 1

    part_dir = Path(split_out_dir) / f"part_{part_idx:06d}"
    save_part_to_npy(part_dir, feature_names, array_feature_names, features, masks, labels_arr)

    return {
        "part_idx": part_idx,
        "rows": rows,
        "bad_lines": bad_lines,
        "part_dir": str(part_dir),
    }


def write_manifest(
    split_name: str,
    out_dir: Path,
    total_rows: int,
    max_rows_per_part: int,
    label_dim: int,
    feature_names: List[str],
    array_feature_names: List[str],
    array_max_length: Dict[str, int],
    parts_meta: List[Dict],
):
    """
    写 split 级 manifest，供后续读取端构建全局索引与随机采样。
    """
    parts_meta = sorted(parts_meta, key=lambda x: x["part_idx"])
    manifest = {
        "split": split_name,
        "format": "npy_mmap",
        "total_rows": int(total_rows),
        "max_rows_per_part": int(max_rows_per_part),
        "label_dim": int(label_dim),
        "feature_names": feature_names,
        "array_feature_names": array_feature_names,
        "array_max_length": array_max_length,
        "parts": [
            {
                "part_idx": int(m["part_idx"]),
                "rows": int(m["rows"]),
                "bad_lines": int(m["bad_lines"]),
                "path": str(Path(m["part_dir"]).name),
            }
            for m in parts_meta
        ],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def merge_parts_into_single_part(
    out_dir: Path,
    parts_meta: List[Dict],
    feature_names: List[str],
    array_feature_names: List[str],
    total_rows: int,
    label_dim: int,
) -> List[Dict]:
    """
    将多个 part_* 合并为单个 part_000001。

    说明：
    - 并行转换阶段仍按分块执行；
    - 最终产物只保留一个 part，便于后续读取端减少分片切换卡顿。
    """
    parts_sorted = sorted(parts_meta, key=lambda x: x["part_idx"])
    if len(parts_sorted) <= 1:
        return parts_sorted

    logger.info(f"开始合并分片：{len(parts_sorted)} -> 1")
    merged_dir_tmp = out_dir / "part_000001_tmp"
    if merged_dir_tmp.exists():
        shutil.rmtree(merged_dir_tmp)
    merged_dir_tmp.mkdir(parents=True, exist_ok=True)

    # 1) 先创建目标 memmap（单个大 part）
    merged_arrays: Dict[str, np.memmap] = {}
    for fea in feature_names:
        first_part_dir = Path(parts_sorted[0]["part_dir"])
        first_arr = np.load(first_part_dir / f"{fea}.npy", mmap_mode="r")
        if first_arr.ndim == 1:
            shape = (total_rows,)
        else:
            shape = (total_rows, first_arr.shape[1])
        merged_arrays[fea] = np.lib.format.open_memmap(
            merged_dir_tmp / f"{fea}.npy",
            mode="w+",
            dtype=first_arr.dtype,
            shape=shape,
        )
        if fea in array_feature_names:
            first_mask = np.load(first_part_dir / f"{fea}_mask.npy", mmap_mode="r")
            merged_arrays[f"{fea}_mask"] = np.lib.format.open_memmap(
                merged_dir_tmp / f"{fea}_mask.npy",
                mode="w+",
                dtype=first_mask.dtype,
                shape=(total_rows, first_mask.shape[1]),
            )

    first_label = np.load(Path(parts_sorted[0]["part_dir"]) / "label.npy", mmap_mode="r")
    merged_arrays["label"] = np.lib.format.open_memmap(
        merged_dir_tmp / "label.npy",
        mode="w+",
        dtype=first_label.dtype,
        shape=(total_rows, first_label.shape[1] if first_label.ndim > 1 else label_dim),
    )

    # 2) 按 part 序号顺序拷贝到目标大数组
    cursor = 0
    total_bad = 0
    for p in parts_sorted:
        part_dir = Path(p["part_dir"])
        rows = int(p["rows"])
        total_bad += int(p.get("bad_lines", 0))
        end = cursor + rows

        for fea in feature_names:
            arr = np.load(part_dir / f"{fea}.npy", mmap_mode="r")
            merged_arrays[fea][cursor:end] = arr
            if fea in array_feature_names:
                mask = np.load(part_dir / f"{fea}_mask.npy", mmap_mode="r")
                merged_arrays[f"{fea}_mask"][cursor:end, :] = mask

        label_arr = np.load(part_dir / "label.npy", mmap_mode="r")
        if label_arr.ndim == 1:
            merged_arrays["label"][cursor:end, 0] = label_arr
        else:
            merged_arrays["label"][cursor:end, :] = label_arr

        logger.info(f"合并 part={p['part_idx']} -> 全量区间[{cursor}, {end})")
        cursor = end

    # 3) 清理旧 part 目录，仅保留 part_000001
    for p in parts_sorted:
        part_dir = Path(p["part_dir"])
        if part_dir.exists():
            shutil.rmtree(part_dir)
    merged_final = out_dir / "part_000001"
    if merged_final.exists():
        shutil.rmtree(merged_final)
    merged_dir_tmp.rename(merged_final)

    logger.info(f"分片合并完成：总行数={cursor}, 总坏行={total_bad}，保留 {merged_final}")
    return [{
        "part_idx": 1,
        "rows": int(cursor),
        "bad_lines": int(total_bad),
        "part_dir": str(merged_final),
    }]


def scan_file_rows_and_label_dim(input_path: Path) -> Tuple[int, int]:
    """
    预扫描文件：统计有效非空行数和标签维度。
    """
    total_lines = 0
    label_dim = 1
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"Scanning {input_path.name}", ncols=100):
            if not line.strip():
                continue
            total_lines += 1
            if total_lines == 1 or label_dim == 1:
                try:
                    _, labels = parse_line(line)
                    label_dim = max(label_dim, len(labels))
                except Exception:
                    continue
    return total_lines, label_dim


def convert_one_file_to_mmap(
    split_name: str,
    input_path: Path,
    out_dir: Path,
    feature_names: List[str],
    array_feature_names: List[str],
    array_max_length: Dict[str, int],
    max_rows_per_part: int = 1000000,
):
    """
    将一个 txt 文件转换为 mmap 友好的分片列式数据目录。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input feature file not found: {input_path}")

    logger.info("------------------------------------------------------------")
    logger.info(f"开始转换 split={split_name}，输入文件：{input_path}")
    logger.info(f"输出目录：{out_dir}")
    logger.info("步骤 1/5：预扫描行数与标签维度")
    total_rows, label_dim = scan_file_rows_and_label_dim(input_path)
    if total_rows == 0:
        raise ValueError(f"No valid content lines found in file: {input_path}")
    logger.info(f"预扫描完成：rows={total_rows}, label_dim={label_dim}")

    logger.info("步骤 2/5：准备输出目录")
    if out_dir.exists():
        logger.info(f"清理旧目录：{out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只启用最多4路并行；CPU不足时自动降级。
    total_cpu = os.cpu_count() or 1
    worker_count = 1 if total_cpu < 2 else min(4, int(total_cpu))
    logger.info(f"步骤 3/5：开始分片转换，worker_count={worker_count}, max_rows_per_part={max_rows_per_part}")
    logger.info("分配规则：按 part 序号取模，CPU1->1,5,9... CPU2->2,6,10... 依此类推")

    parts_meta: List[Dict] = []
    total_bad_lines = 0
    total_done_rows = 0

    if worker_count == 1:
        # 单进程路径：顺序转，逻辑简单稳定
        part_idx = 1
        chunk_lines: List[str] = []
        with open(input_path, "r", encoding="utf-8") as fin, tqdm(total=total_rows, desc=f"Parsing {input_path.name}", ncols=100) as pbar:
            for line in fin:
                if not line.strip():
                    continue
                chunk_lines.append(line)
                if len(chunk_lines) >= max_rows_per_part:
                    result = convert_chunk_to_mmap_part(
                        chunk_lines, part_idx, str(out_dir),
                        feature_names, array_feature_names, array_max_length, label_dim
                    )
                    parts_meta.append(result)
                    total_bad_lines += int(result["bad_lines"])
                    total_done_rows += int(result["rows"])
                    logger.info(f"分片完成：part={part_idx}, rows={result['rows']}, bad={result['bad_lines']}")
                    pbar.update(len(chunk_lines))
                    chunk_lines = []
                    part_idx += 1
            if chunk_lines:
                result = convert_chunk_to_mmap_part(
                    chunk_lines, part_idx, str(out_dir),
                    feature_names, array_feature_names, array_max_length, label_dim
                )
                parts_meta.append(result)
                total_bad_lines += int(result["bad_lines"])
                total_done_rows += int(result["rows"])
                logger.info(f"尾部分片完成：part={part_idx}, rows={result['rows']}, bad={result['bad_lines']}")
                pbar.update(len(chunk_lines))
    else:
        # 多进程路径：最多4路，并按 part 序号取模路由到固定 worker
        pending_by_worker = {wid: [] for wid in range(worker_count)}
        executors = [ProcessPoolExecutor(max_workers=1) for _ in range(worker_count)]
        try:
            part_idx = 1
            chunk_lines = []

            def collect_one(worker_id: int):
                nonlocal total_bad_lines, total_done_rows
                future, f_part_idx = pending_by_worker[worker_id].pop(0)
                result = future.result()
                parts_meta.append(result)
                total_bad_lines += int(result["bad_lines"])
                total_done_rows += int(result["rows"])
                logger.info(
                    f"CPU{worker_id+1} 完成分片：part={f_part_idx}, rows={result['rows']}, "
                    f"bad={result['bad_lines']}"
                )

            with open(input_path, "r", encoding="utf-8") as fin, tqdm(total=total_rows, desc=f"Parsing {input_path.name}", ncols=100) as pbar:
                for line in fin:
                    if not line.strip():
                        continue
                    chunk_lines.append(line)
                    if len(chunk_lines) >= max_rows_per_part:
                        wid = (part_idx - 1) % worker_count
                        future = executors[wid].submit(
                            convert_chunk_to_mmap_part,
                            chunk_lines,
                            part_idx,
                            str(out_dir),
                            feature_names,
                            array_feature_names,
                            array_max_length,
                            label_dim,
                        )
                        pending_by_worker[wid].append((future, part_idx))
                        logger.info(f"提交分片：part={part_idx} -> CPU{wid+1}, rows={len(chunk_lines)}")
                        if len(pending_by_worker[wid]) >= 2:
                            collect_one(wid)
                        pbar.update(len(chunk_lines))
                        chunk_lines = []
                        part_idx += 1

                if chunk_lines:
                    wid = (part_idx - 1) % worker_count
                    future = executors[wid].submit(
                        convert_chunk_to_mmap_part,
                        chunk_lines,
                        part_idx,
                        str(out_dir),
                        feature_names,
                        array_feature_names,
                        array_max_length,
                        label_dim,
                    )
                    pending_by_worker[wid].append((future, part_idx))
                    logger.info(f"提交尾部分片：part={part_idx} -> CPU{wid+1}, rows={len(chunk_lines)}")
                    pbar.update(len(chunk_lines))

            logger.info("主进程读取结束，等待所有 worker 完成")
            for wid in range(worker_count):
                while pending_by_worker[wid]:
                    collect_one(wid)
        finally:
            for ex in executors:
                ex.shutdown(wait=True)

    logger.info("步骤 4/5：写 manifest.json")
    # 按你的要求，最终只保留一个 part_000001。
    parts_meta = merge_parts_into_single_part(
        out_dir=out_dir,
        parts_meta=parts_meta,
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        total_rows=total_done_rows,
        label_dim=label_dim,
    )
    write_manifest(
        split_name=split_name,
        out_dir=out_dir,
        total_rows=total_done_rows,
        max_rows_per_part=max_rows_per_part,
        label_dim=label_dim,
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
        parts_meta=parts_meta,
    )

    logger.info(
        f"步骤 5/5：split={split_name} 转换完成，rows={total_done_rows}, "
        f"bad_lines={total_bad_lines}, parts={len(parts_meta)}"
    )


def main():
    """
    命令行入口：把 extractored_feature 下 txt 特征转为 mmap 友好分片目录。
    """
    parser = argparse.ArgumentParser(description="Convert extracted txt feature files to mmap-friendly sharded npy.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to feature config yaml.")
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="src/tmp/extractored_feature",
        help="Directory containing train_features.txt/dev_features.txt/item_features.txt",
    )
    parser.add_argument(
        "--max_rows_per_part",
        type=int,
        default=1000000,
        help="Maximum rows per shard part.",
    )
    args = parser.parse_args()

    logger.info("Txt2NpzConverter（mmap模式）启动")
    logger.info(
        f"输入参数：config={args.config}, feature_dir={args.feature_dir}, "
        f"max_rows_per_part={args.max_rows_per_part}"
    )

    logger.info("步骤 A：加载配置")
    conf = OmegaConf.load(args.config)
    feature_names: List[str] = list(conf.features.feature_names)
    array_feature_names: List[str] = list(conf.features.get("array_feature_names", []))
    array_max_length: Dict[str, int] = dict(conf.features.get("array_max_length", {}))
    for fea in array_feature_names:
        if fea not in array_max_length:
            raise ValueError(f"Array feature '{fea}' is missing max length in config.")
    logger.info(
        f"配置加载完成：feature_names={len(feature_names)}, array_feature_names={len(array_feature_names)}"
    )

    feature_dir = Path(args.feature_dir)
    if not feature_dir.exists():
        raise FileNotFoundError(f"特征目录不存在：{feature_dir}")
    logger.info(f"步骤 B：输入目录检查通过：{feature_dir}")

    logger.info("步骤 C：开始转换 train/dev/item 为 mmap 分片目录")
    convert_one_file_to_mmap(
        split_name="train",
        input_path=feature_dir / "train_features.txt",
        out_dir=feature_dir / "train_features_mmap",
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
        max_rows_per_part=args.max_rows_per_part,
    )
    convert_one_file_to_mmap(
        split_name="dev",
        input_path=feature_dir / "dev_features.txt",
        out_dir=feature_dir / "dev_features_mmap",
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
        max_rows_per_part=args.max_rows_per_part,
    )
    # item 默认也走同一流程（通常只会生成一个 part）
    convert_one_file_to_mmap(
        split_name="item",
        input_path=feature_dir / "item_features.txt",
        out_dir=feature_dir / "item_features_mmap",
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
        max_rows_per_part=args.max_rows_per_part,
    )
    logger.info("全部转换完成：已生成 train/dev/item 的 mmap 分片目录")


if __name__ == "__main__":
    main()
