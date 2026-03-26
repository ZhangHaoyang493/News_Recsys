import sys
import os
import json
import argparse
import yaml
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Decode feature indices to original values for verification.")
    parser.add_argument('-f', '--feature_file', type=str, required=True, help="Path to the feature file (e.g., train_features.txt)")
    parser.add_argument('-m', '--mapping_file', type=str, required=True, help="Path to embedding_idx_2_original_val_dict.json")
    parser.add_argument('-c', '--config_file', type=str, default=None, help="Path to dataset_extract_info.yaml. If not provided, looks in the mapping_file directory.")
    parser.add_argument('-s', '--start_line', type=int, default=1, help="1-based line number to start decoding from.")
    parser.add_argument('-n', '--num_lines', type=int, default=20, help="Number of lines to decode.")
    
    args = parser.parse_args()
    
    feature_path = Path(args.feature_file)
    mapping_path = Path(args.mapping_file)

    if args.start_line < 1:
        raise ValueError("--start_line must be >= 1")
    if args.num_lines < 1:
        raise ValueError("--num_lines must be >= 1")
    
    # 1. Load Mappings
    print(f"Loading mappings from {mapping_path}...")
    idx2val_raw = load_json(mapping_path)
    # Convert string keys to int for lookup
    idx2val = {}
    for feat_name, mapping in idx2val_raw.items():
        idx2val[feat_name] = {int(k): v for k, v in mapping.items()}
    passthrough_id_maps = {"impression_id", "user_id", "item_id"}

    # 2. Load Config for Shared Embeddings
    config_path = args.config_file
    if config_path is None:
        # Try to find in the same directory as mapping file
        potential_path = mapping_path.parent / 'dataset_extract_info.yaml'
        if potential_path.exists():
            config_path = potential_path
    
    share_map = {}
    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        cfg = load_yaml(config_path)
        # Assuming structure details from feature_extractor_base.py
        # variable 'share_emb_table_features' might be in features section
        if 'features' in cfg and 'share_emb_table_features' in cfg['features']:
            share_map = cfg['features']['share_emb_table_features']
        elif 'share_emb_table_features' in cfg:
             share_map = cfg['share_emb_table_features']
    else:
        print("Warning: Config file not found. Shared features might not be decoded correctly.")

    # 3. Process Check
    end_line = args.start_line + args.num_lines - 1
    print(f"Decoding lines [{args.start_line}, {end_line}] from {feature_path}...")

    decoded_count = 0
    with open(feature_path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            line_no = i + 1
            if line_no < args.start_line:
                continue
            if decoded_count >= args.num_lines:
                break
                
            parts = line.strip().split('\t')
            feature_part = parts[0]
            label_part = parts[1] if len(parts) > 1 else "[No Label]"
            
            decoded_features = {}
            
            # Feature string format: "feat1:idx1 feat2:idx2 ..."
            # Note: idx might be single int or something else? Usually int.
            
            for item in feature_part.split(' '):
                if ':' not in item:
                    continue
                
                # Split only on the last colon to handle keys like "user:name" if ever needed, 
                # but standard is key:value.
                # Assuming key:value
                key, val_str = item.rsplit(':', 1)

                # Determine referencing map name
                map_name = share_map.get(key, key)

                # 数组特征：形如 "1,2,3"。逐个反查成明文。
                if "," in val_str:
                    raw_tokens = [tok for tok in val_str.split(",") if tok != ""]
                    decoded_tokens = []
                    for tok in raw_tokens:
                        try:
                            tok_idx = int(tok)
                        except ValueError:
                            decoded_tokens.append(f"{tok} (RAW)")
                            continue

                        decoded_tok = None
                        if map_name in idx2val:
                            decoded_tok = idx2val[map_name].get(tok_idx)
                        if decoded_tok is not None:
                            decoded_tokens.append(f"{decoded_tok} (ID:{tok_idx})")
                        elif map_name in passthrough_id_maps:
                            decoded_tokens.append(f"{tok_idx} (ID:{tok_idx})")
                        else:
                            decoded_tokens.append(f"UNKNOWN_ID:{tok_idx}")

                    decoded_features[key] = "[" + ", ".join(decoded_tokens) + "]"
                else:
                    # 标量特征：保持原逻辑
                    try:
                        val_idx = int(val_str)
                    except ValueError:
                        decoded_features[key] = f"{val_str} (RAW)"
                        continue

                    decoded_val = None
                    if map_name in idx2val:
                        decoded_val = idx2val[map_name].get(val_idx)

                    if decoded_val is not None:
                        decoded_features[key] = f"{decoded_val} (ID:{val_idx})"
                    elif map_name in passthrough_id_maps:
                        decoded_features[key] = f"{val_idx} (ID:{val_idx})"
                    else:
                        decoded_features[key] = f"UNKNOWN_ID:{val_idx}"
            
            print(f"Line {line_no}:")
            print(f"  Label: {label_part}")
            print("  Features:")
            for k, v in decoded_features.items():
                print(f"    - {k}: {v}")
            print("-" * 40)
            decoded_count += 1

    print(f"Done. Printed {decoded_count} decoded lines.")

if __name__ == "__main__":
    main()
