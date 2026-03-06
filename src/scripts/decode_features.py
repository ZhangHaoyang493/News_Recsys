import sys
import os
import json
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Decode feature indices to original values for verification.")
    parser.add_argument('--feature_file', type=str, required=True, help="Path to the feature file (e.g., train_features.txt)")
    parser.add_argument('--mapping_file', type=str, required=True, help="Path to embedding_idx_2_original_val_dict.json")
    parser.add_argument('--config_file', type=str, default=None, help="Path to dataset_extract_info.yaml. If not provided, looks in the mapping_file directory.")
    parser.add_argument('--output_file', type=str, default="decoded_features_sample.txt", help="Path to save the decoded output.")
    parser.add_argument('--num_lines', type=int, default=20, help="Number of lines to decode.")
    
    args = parser.parse_args()
    
    feature_path = Path(args.feature_file)
    mapping_path = Path(args.mapping_file)
    output_path = Path(args.output_file)
    
    # 1. Load Mappings
    print(f"Loading mappings from {mapping_path}...")
    idx2val_raw = load_json(mapping_path)
    # Convert string keys to int for lookup
    idx2val = {}
    for feat_name, mapping in idx2val_raw.items():
        idx2val[feat_name] = {int(k): v for k, v in mapping.items()}

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
    print(f"Decoding first {args.num_lines} lines from {feature_path}...")
    
    with open(feature_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(fin):
            if i >= args.num_lines:
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
                
                try:
                    val_idx = int(val_str)
                except ValueError:
                    decoded_features[key] = f"{val_str} (RAW)"
                    continue
                
                # Determine referencing map name
                map_name = share_map.get(key, key)
                
                decoded_val = None
                if map_name in idx2val:
                    decoded_val = idx2val[map_name].get(val_idx)
                
                if decoded_val is not None:
                    decoded_features[key] = f"{decoded_val} (ID:{val_idx})"
                else:
                    decoded_features[key] = f"UNKNOWN_ID:{val_idx}"
            
            # Write to output
            fout.write(f"Line {i+1}:\n")
            fout.write(f"  Label: {label_part}\n")
            fout.write(f"  Features:\n")
            for k, v in decoded_features.items():
                fout.write(f"    - {k}: {v}\n")
            fout.write("-" * 40 + "\n")

    print(f"Done. Decoded results saved to {output_path}")

if __name__ == "__main__":
    main()
