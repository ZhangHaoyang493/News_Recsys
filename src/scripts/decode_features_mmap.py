import os
import json
import argparse
from pathlib import Path

import numpy as np
import yaml


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_share_map(config_path):
    share_map = {}
    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        cfg = load_yaml(config_path)
        if "features" in cfg and "share_emb_table_features" in cfg["features"]:
            share_map = cfg["features"]["share_emb_table_features"]
        elif "share_emb_table_features" in cfg:
            share_map = cfg["share_emb_table_features"]
    else:
        print("Warning: Config file not found. Shared features might not be decoded correctly.")
    return share_map


def decode_one_id(map_name, val_idx, idx2val, passthrough_id_maps):
    decoded_val = None
    if map_name in idx2val:
        decoded_val = idx2val[map_name].get(int(val_idx))
    if decoded_val is not None:
        return f"{decoded_val} (ID:{int(val_idx)})"
    if map_name in passthrough_id_maps:
        return f"{int(val_idx)} (ID:{int(val_idx)})"
    return f"UNKNOWN_ID:{int(val_idx)}"


def format_label(label_row):
    label_arr = np.asarray(label_row)
    if label_arr.ndim == 0:
        return str(float(label_arr))
    if label_arr.ndim == 1 and label_arr.shape[0] == 1:
        return str(float(label_arr[0]))
    return "[" + ", ".join(str(float(x)) for x in label_arr.tolist()) + "]"


def main():
    parser = argparse.ArgumentParser(description="Decode mmap feature indices to original values for verification.")
    parser.add_argument("-f", "--feature_file", type=str, required=True, help="Path to mmap feature directory (e.g., train_features_mmap)")
    parser.add_argument("-m", "--mapping_file", type=str, required=True, help="Path to embedding_idx_2_original_val_dict.json")
    parser.add_argument("-c", "--config_file", type=str, default=None, help="Path to dataset_extract_info.yaml. If not provided, looks in the mapping_file directory.")
    parser.add_argument("-s", "--start_line", type=int, default=1, help="1-based line number to start decoding from.")
    parser.add_argument("-n", "--num_lines", type=int, default=20, help="Number of lines to decode.")
    args = parser.parse_args()

    if args.start_line < 1:
        raise ValueError("--start_line must be >= 1")
    if args.num_lines < 1:
        raise ValueError("--num_lines must be >= 1")

    feature_dir = Path(args.feature_file)
    mapping_path = Path(args.mapping_file)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature mmap directory not found: {feature_dir}")

    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    print(f"Loading mappings from {mapping_path}...")
    idx2val_raw = load_json(mapping_path)
    idx2val = {feat_name: {int(k): v for k, v in mapping.items()} for feat_name, mapping in idx2val_raw.items()}
    passthrough_id_maps = {"impression_id", "user_id", "item_id"}

    config_path = args.config_file
    if config_path is None:
        potential_path = mapping_path.parent / "dataset_extract_info.yaml"
        if potential_path.exists():
            config_path = potential_path
    share_map = load_share_map(config_path)

    manifest = load_json(manifest_path)
    parts = manifest.get("parts", [])
    if len(parts) != 1:
        raise ValueError(f"decode_features_mmap expects exactly 1 part, got {len(parts)} in {manifest_path}")

    part_dir = feature_dir / parts[0]["path"]
    if not part_dir.exists():
        raise FileNotFoundError(f"Part directory not found: {part_dir}")

    feature_names = list(manifest.get("feature_names", []))
    if not feature_names:
        feature_names = sorted(
            [
                p.stem
                for p in part_dir.glob("*.npy")
                if p.name != "label.npy" and not p.name.endswith("_mask.npy")
            ]
        )
    array_feature_names = set(manifest.get("array_feature_names", []))

    arrays = {}
    for fea in feature_names:
        fp = part_dir / f"{fea}.npy"
        arrays[fea] = np.load(fp, mmap_mode="r") if fp.exists() else None
        if fea in array_feature_names:
            mfp = part_dir / f"{fea}_mask.npy"
            arrays[f"{fea}_mask"] = np.load(mfp, mmap_mode="r") if mfp.exists() else None

    label_fp = part_dir / "label.npy"
    if not label_fp.exists():
        raise FileNotFoundError(f"label.npy not found: {label_fp}")
    labels = np.load(label_fp, mmap_mode="r")

    total_rows = int(parts[0]["rows"])
    start_idx = args.start_line - 1
    if start_idx >= total_rows:
        print(f"Start line {args.start_line} exceeds total rows {total_rows}. Nothing to decode.")
        return
    end_idx = min(start_idx + args.num_lines, total_rows)

    print(f"Decoding lines [{args.start_line}, {end_idx}] from {feature_dir}...")
    decoded_count = 0

    for local_idx in range(start_idx, end_idx):
        line_no = local_idx + 1
        decoded_features = {}

        for key in feature_names:
            data = arrays.get(key)
            if data is None:
                continue

            map_name = share_map.get(key, key)
            row_val = np.asarray(data[local_idx])

            # 标量特征
            if row_val.ndim == 0:
                decoded_features[key] = decode_one_id(map_name, int(row_val), idx2val, passthrough_id_maps)
                continue

            # 数组特征：优先按 mask 过滤有效位；若无 mask 则过滤 padding(-1)
            values = row_val.astype(np.int64).tolist()
            mask_arr = arrays.get(f"{key}_mask")
            if mask_arr is not None:
                mask_vals = np.asarray(mask_arr[local_idx]).astype(np.float32).tolist()
                valid_ids = [v for v, m in zip(values, mask_vals) if m > 0]
            else:
                valid_ids = [v for v in values if v != -1]

            decoded_tokens = [decode_one_id(map_name, int(v), idx2val, passthrough_id_maps) for v in valid_ids]
            decoded_features[key] = "[" + ", ".join(decoded_tokens) + "]"

        print(f"Line {line_no}:")
        print(f"  Label: {format_label(labels[local_idx])}")
        print("  Features:")
        for k, v in decoded_features.items():
            print(f"    - {k}: {v}")
        print("-" * 40)
        decoded_count += 1

    print(f"Done. Printed {decoded_count} decoded lines.")


if __name__ == "__main__":
    main()
