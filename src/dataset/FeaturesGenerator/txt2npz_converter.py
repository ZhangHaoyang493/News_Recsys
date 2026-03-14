import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from ...Logger.logging import Logger


logger = Logger.get_logger("Txt2NpzConverter")


def parse_line(raw_line: str) -> Tuple[Dict[str, str], List[float]]:
    """
    解析一行 txt 特征数据。

    输入行格式约定：
    - `feat1:val1 feat2:val2 ... \\t label1 [label2 ...]`
    - 特征与标签之间用 `\\t` 分隔
    - 特征内部用空格分隔，单个特征为 `name:value`

    Returns:
        Tuple[Dict[str, str], List[float]]:
            - feature_dict: 原始字符串特征字典（value 保持字符串，后续按类型解析）
            - labels: 标签列表（float）
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
    """
    将标量特征值转为 int。

    约定：
    - 缺失值/空串/非法值回填为 0
    - 与当前 embedding 稀疏特征默认处理语义保持一致
    """
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_array_ints(value: str) -> List[int]:
    """
    将数组特征字符串解析为 int 列表。

    输入格式示例：
    - `"1,2,3"` -> [1, 2, 3]
    - `""` -> []
    """
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
    按样本行数预分配 numpy 缓冲区，避免逐行 append 带来的反复扩容。

    缓冲区默认值：
    - 标量特征：0
    - 数组特征：-1（padding）
    - 数组 mask：0.0
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


def convert_chunk_to_npz_file(
    chunk_lines: List[str],
    part_idx: int,
    output_chunk_dir: str,
    output_prefix: str,
    feature_names: List[str],
    array_feature_names: List[str],
    array_max_length: Dict[str, int],
    label_dim: int,
):
    """
    子进程执行：将一个文本分块直接转换并落盘为一个 npz 文件。

    返回：
    - rows: 当前分块行数
    - bad_lines: 当前分块坏行数
    - file_name: 输出文件名
    - part_idx: 分块序号
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

            if row_labels:
                max_copy = min(len(row_labels), label_dim)
                labels_arr[row_idx, :max_copy] = np.asarray(row_labels[:max_copy], dtype=np.float32)
        except Exception:
            bad_lines += 1

    to_save = {}
    for fea in feature_names:
        if fea in array_feature_names:
            to_save[fea] = features[fea]
            to_save[f"{fea}_mask"] = masks[f"{fea}_mask"]
        else:
            to_save[fea] = features[fea]
    to_save["label"] = labels_arr

    part_file = Path(output_chunk_dir) / f"{output_prefix}_part_{part_idx:06d}.npz"
    np.savez(part_file, **to_save)
    return {
        "rows": rows,
        "bad_lines": bad_lines,
        "file_name": part_file.name,
        "part_idx": part_idx,
    }


def convert_one_file(
    input_path: Path,
    output_path: Path,
    feature_names: List[str],
    array_feature_names: List[str],
    array_max_length: Dict[str, int],
    max_rows_per_file: int = 500000,
    output_chunk_dir: Path = None,
    output_prefix: str = "",
):
    """
    将单个 txt 特征文件转换为 npz 文件。

    处理流程：
    1. 第一遍扫描：统计有效行数并探测 label 维度（用于预分配）
    2. 第二遍解析：逐行写入预分配数组
    3. 保存 npz：包含所有特征、数组 mask 和 label

    数组特征规则：
    - 长度不足：右侧补 -1，mask 前 seq_len 为 1.0
    - 长度超出：保留最近 max_len 个，mask 全 1.0
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input feature file not found: {input_path}")

    logger.info("------------------------------------------------------------")
    logger.info(f"开始转换文件：{input_path}")
    logger.info(f"目标输出文件：{output_path}")
    logger.info("步骤 1/5：预扫描文件，统计行数与标签维度")

    total_lines = 0
    label_dim = 1
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"Scanning {input_path.name}", ncols=100):
            if not line.strip():
                continue
            total_lines += 1
            # 从首个可解析标签行推断 label 维度
            if total_lines == 1 or label_dim == 1:
                try:
                    _, labels = parse_line(line)
                    label_dim = max(label_dim, len(labels))
                except Exception:
                    continue

    if total_lines == 0:
        raise ValueError(f"No valid content lines found in file: {input_path}")
    logger.info(f"预扫描完成：有效非空行数={total_lines}, 标签维度={label_dim}")
    logger.info("步骤 2/5：初始化写出模式与分块缓冲区参数")
    bad_lines = 0
    row_idx = 0

    if output_chunk_dir is None:
        logger.info("当前为单文件输出模式（会按总行数一次性分配缓冲区）")
        chunk_rows = total_lines
    else:
        logger.info(
            f"当前为分块输出模式：目录={output_chunk_dir}，每个 npz 最多 {max_rows_per_file} 行"
        )
        if output_chunk_dir.exists():
            logger.info(f"清理旧分块目录：{output_chunk_dir}")
            shutil.rmtree(output_chunk_dir)
        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_rows = max_rows_per_file

    # 分块模式下：如果 CPU 数>=2，启用多进程并发转换（最多4路）
    total_cpu = os.cpu_count() or 1
    if output_chunk_dir is not None and total_cpu >= 2:
        worker_count = max(1, min(4, int(total_cpu)))
        num_parts = (total_lines + max_rows_per_file - 1) // max_rows_per_file
        logger.info(f"步骤 3/5：启用最多4路并行分块转换（当前 worker 数={worker_count}）")
        logger.info("分配规则：按 part 序号取模分配到不同 CPU")
        for w in range(worker_count):
            # part 从1开始；worker 索引从0开始
            if w < worker_count - 1:
                logger.info(f"- CPU{w + 1} 负责 part 序号满足 (part-1)%{worker_count} == {w}")
            else:
                logger.info(f"- CPU{w + 1} 负责 part 序号满足 (part-1)%{worker_count} == {w}")
        logger.info("当前模式：多 CPU 并行（每个 CPU 固定处理各自取模分片）")

        bad_lines_total = 0
        rows_total = 0
        submitted_parts = 0
        completed_parts = 0
        pending_by_worker = {wid: [] for wid in range(worker_count)}

        executors = [ProcessPoolExecutor(max_workers=1) for _ in range(worker_count)]
        try:
            part_idx = 1
            current_chunk_lines: List[str] = []

            def collect_one_pending(worker_id: int):
                nonlocal bad_lines_total, rows_total, completed_parts
                pending_list = pending_by_worker[worker_id]
                future, f_part_idx = pending_list.pop(0)
                result = future.result()
                completed_parts += 1
                bad_lines_total += int(result["bad_lines"])
                rows_total += int(result["rows"])
                logger.info(
                    f"CPU{worker_id + 1} 完成分块：part={f_part_idx}, file={result['file_name']}, "
                    f"rows={result['rows']}, chunk_bad_lines={result['bad_lines']}, "
                    f"累计完成={completed_parts}/{num_parts}"
                )

            with open(input_path, "r", encoding="utf-8") as fin, tqdm(
                total=total_lines, desc=f"Parsing {input_path.name}", ncols=100
            ) as pbar:
                for line in fin:
                    if not line.strip():
                        continue
                    current_chunk_lines.append(line)
                    if len(current_chunk_lines) >= max_rows_per_file:
                        task_lines = current_chunk_lines
                        current_chunk_lines = []
                        target_worker = (part_idx - 1) % worker_count
                        future = executors[target_worker].submit(
                            convert_chunk_to_npz_file,
                            task_lines,
                            part_idx,
                            str(output_chunk_dir),
                            output_prefix,
                            feature_names,
                            array_feature_names,
                            array_max_length,
                            label_dim,
                        )
                        pending_by_worker[target_worker].append((future, part_idx))
                        logger.info(
                            f"已提交分块 part={part_idx} 给 CPU{target_worker + 1}，rows={len(task_lines)}"
                        )
                        if len(pending_by_worker[target_worker]) >= 2:
                            collect_one_pending(target_worker)

                        submitted_parts += 1
                        pbar.update(len(task_lines))
                        part_idx += 1

                if current_chunk_lines:
                    target_worker = (part_idx - 1) % worker_count
                    future = executors[target_worker].submit(
                        convert_chunk_to_npz_file,
                        current_chunk_lines,
                        part_idx,
                        str(output_chunk_dir),
                        output_prefix,
                        feature_names,
                        array_feature_names,
                        array_max_length,
                        label_dim,
                    )
                    pending_by_worker[target_worker].append((future, part_idx))
                    logger.info(
                        f"已提交尾部分块 part={part_idx} 给 CPU{target_worker + 1}，rows={len(current_chunk_lines)}"
                    )
                    submitted_parts += 1
                    pbar.update(len(current_chunk_lines))

            logger.info("主进程读取完成，等待所有 CPU 完成剩余分块")
            for wid in range(worker_count):
                while pending_by_worker[wid]:
                    collect_one_pending(wid)
        finally:
            for ex in executors:
                ex.shutdown(wait=True)

        logger.info(
            f"多 CPU 并行转换完成：提交分块数={submitted_parts}, 完成分块数={completed_parts}, "
            f"总行数={rows_total}, 坏行数={bad_lines_total}"
        )
        logger.info(
            f"文件转换完成：{input_path.name} -> {output_chunk_dir}（总行数={rows_total}, "
            f"坏行数={bad_lines_total}, 分块数={completed_parts}）"
        )
        logger.info("步骤 4/5：写出流程结束，开始汇总")
        logger.info("步骤 5/5：当前文件转换任务完成")
        return

    logger.info(
        f"步骤 3/5：单 CPU 顺序解析并写入分块缓冲区（每块缓冲区行数={chunk_rows}）"
    )
    logger.info("当前模式：单 CPU（无多线程/多进程）")

    features, masks, labels_arr = allocate_buffers(
        rows=chunk_rows,
        label_dim=label_dim,
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
    )
    logger.info("首块缓冲区分配完成")

    part_idx = 1
    chunk_row_idx = 0
    num_parts_written = 0

    def flush_chunk(valid_rows: int, current_part_idx: int):
        if valid_rows <= 0:
            return
        to_save = {}
        for fea in feature_names:
            if fea in array_feature_names:
                to_save[fea] = features[fea][:valid_rows, :]
                to_save[f"{fea}_mask"] = masks[f"{fea}_mask"][:valid_rows, :]
            else:
                to_save[fea] = features[fea][:valid_rows]
        to_save["label"] = labels_arr[:valid_rows, :]

        if output_chunk_dir is None:
            np.savez(output_path, **to_save)
            logger.info(
                f"文件写入完成：{output_path}（总行数={valid_rows}, 坏行数={bad_lines}, "
                f"标签维度={to_save['label'].shape[1]}）"
            )
        else:
            part_file = output_chunk_dir / f"{output_prefix}_part_{current_part_idx:06d}.npz"
            np.savez(part_file, **to_save)
            logger.info(f"分块写入完成：{part_file.name}（rows={valid_rows}）")

    with open(input_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(tqdm(fin, desc=f"Parsing {input_path.name}", ncols=100), start=1):
            if not line.strip():
                continue

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
                                features[fea][chunk_row_idx, :seq_len] = np.asarray(arr_values, dtype=np.int64)
                                masks[f"{fea}_mask"][chunk_row_idx, :seq_len] = 1.0
                        else:
                            arr_values = arr_values[-max_len:]
                            features[fea][chunk_row_idx, :] = np.asarray(arr_values, dtype=np.int64)
                            masks[f"{fea}_mask"][chunk_row_idx, :] = 1.0
                    else:
                        features[fea][chunk_row_idx] = parse_scalar_int(raw_value)

                if row_labels:
                    max_copy = min(len(row_labels), label_dim)
                    labels_arr[chunk_row_idx, :max_copy] = np.asarray(row_labels[:max_copy], dtype=np.float32)
            except Exception as e:
                bad_lines += 1
                if bad_lines <= 20:
                    logger.warning(f"第 {line_no} 行解析失败，已使用默认值占位。错误：{e}")
            finally:
                chunk_row_idx += 1
                row_idx += 1

            if chunk_row_idx >= chunk_rows:
                flush_chunk(chunk_row_idx, part_idx)
                num_parts_written += 1
                part_idx += 1
                chunk_row_idx = 0
                # 重新分配下一块缓冲区（避免保留过多内存）
                features, masks, labels_arr = allocate_buffers(
                    rows=chunk_rows,
                    label_dim=label_dim,
                    feature_names=feature_names,
                    array_feature_names=array_feature_names,
                    array_max_length=array_max_length,
                )

    if chunk_row_idx > 0:
        flush_chunk(chunk_row_idx, part_idx)
        num_parts_written += 1

    logger.info(f"顺序解析完成：总行数={row_idx}, 坏行总数={bad_lines}")
    logger.info("步骤 4/5：写出流程结束，开始汇总")
    if output_chunk_dir is None:
        logger.info(f"文件转换完成：{input_path.name} -> {output_path.name}")
    else:
        logger.info(
            f"文件转换完成：{input_path.name} -> {output_chunk_dir}（总行数={row_idx}, "
            f"坏行数={bad_lines}, 分块数={num_parts_written}）"
        )
    logger.info("步骤 5/5：当前文件转换任务完成")


def main():
    """
    命令行入口：
    - 从配置读取特征定义
    - 将 feature_dir 下 train/dev/item 三个 txt 文件转换为 npz
    """
    parser = argparse.ArgumentParser(description="Convert extracted txt feature files to npz format.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to feature config yaml.")
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="src/tmp/extractored_feature",
        help="Directory containing train_features.txt/dev_features.txt/item_features.txt",
    )
    args = parser.parse_args()
    logger.info("Txt2NpzConverter 启动")
    logger.info(f"输入参数：config={args.config}, feature_dir={args.feature_dir}")

    logger.info("步骤 A：加载配置文件")
    conf = OmegaConf.load(args.config)
    feature_names: List[str] = list(conf.features.feature_names)
    array_feature_names: List[str] = list(conf.features.get("array_feature_names", []))
    array_max_length: Dict[str, int] = dict(conf.features.get("array_max_length", {}))
    logger.info(
        f"配置加载完成：feature_names={len(feature_names)}, "
        f"array_feature_names={len(array_feature_names)}"
    )

    logger.info("步骤 B：校验序列特征的 max_length 配置")
    for fea in array_feature_names:
        if fea not in array_max_length:
            raise ValueError(f"Array feature '{fea}' is missing max length in config.")
    logger.info("序列特征配置校验通过")

    feature_dir = Path(args.feature_dir)
    if not feature_dir.exists():
        raise FileNotFoundError(f"特征目录不存在：{feature_dir}")
    logger.info(f"步骤 C：检查输入目录通过：{feature_dir}")

    logger.info("步骤 D：本次转换使用单 CPU 顺序执行（无并行）")

    logger.info("步骤 E：开始转换 train/dev/item")
    logger.info("其中 train/dev 使用分块输出（每块最多 500000 行）")

    # train: 分块输出到 train_features_npz/
    convert_one_file(
        input_path=feature_dir / "train_features.txt",
        output_path=feature_dir / "train_features.npz",
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
        max_rows_per_file=500000,
        output_chunk_dir=feature_dir / "train_features_npz",
        output_prefix="train_features",
    )

    # dev: 分块输出到 dev_features_npz/
    convert_one_file(
        input_path=feature_dir / "dev_features.txt",
        output_path=feature_dir / "dev_features.npz",
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
        max_rows_per_file=500000,
        output_chunk_dir=feature_dir / "dev_features_npz",
        output_prefix="dev_features",
    )

    # item: 仍保留单文件输出
    convert_one_file(
        input_path=feature_dir / "item_features.txt",
        output_path=feature_dir / "item_features.npz",
        feature_names=feature_names,
        array_feature_names=array_feature_names,
        array_max_length=array_max_length,
    )

    logger.info("全部文件转换完成：train/dev 已分块，item 为单文件 npz")


if __name__ == "__main__":
    main()
