import re
import sys
import os
import argparse

def parse_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    epochs = []
    current_epoch = None
    current_section = None
    
    # Regex patterns
    # Matches: ==================== Epoch 0 Validation Results ====================
    epoch_start_pattern = re.compile(r"=+ Epoch (\d+) Validation Results =+")
    # Matches: Overall:, Warm Start Users (5943):, Cold Start Users (44057):
    section_pattern = re.compile(r"^\s*(Overall|Warm Start Users|Cold Start Users).*:$")
    # Matches: AUC: 0.5261, LogLoss: nan, NDCG@10: 0.2558
    # Updated regex to match stripped lines (removed leading \s+)
    metric_pattern = re.compile(r"^([a-zA-Z0-9@]+):\s+([0-9\.\-eE]+|nan|inf|-inf)")
    # Matches: ============================================================
    block_end_pattern = re.compile(r"={10,}")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for epoch start
        epoch_match = epoch_start_pattern.match(line)
        if epoch_match:
            # If we were already parsing an epoch, save it before starting new one
            if current_epoch is not None:
                epochs.append(current_epoch)
            
            current_epoch = {
                'epoch': int(epoch_match.group(1)),
                'data': {}
            }
            current_section = None
            continue

        if current_epoch is None:
            continue

        # Check for block end (separator line)
        if block_end_pattern.match(line) and "Epoch" not in line:
            epochs.append(current_epoch)
            current_epoch = None
            current_section = None
            continue

        # Check for section header
        section_match = section_pattern.match(line)
        if section_match:
            raw_section = section_match.group(1)
            if "Warm Start Users" in raw_section:
                current_section = "Warm Start Users"
            elif "Cold Start Users" in raw_section:
                current_section = "Cold Start Users"
            else:
                current_section = "Overall"
            
            current_epoch['data'][current_section] = {}
            continue

        # Check for metrics
        metric_match = metric_pattern.match(line)
        if metric_match and current_section:
            key = metric_match.group(1)
            value_str = metric_match.group(2)
            try:
                val_float = float(value_str)
            except ValueError:
                val_float = float('nan')
            
            current_epoch['data'][current_section][key] = val_float

    # Append the last epoch if file ends without separator
    if current_epoch is not None:
        epochs.append(current_epoch)

    return epochs

def print_best_epoch(epochs, model_name="Unknown"):
    best_epoch = None
    max_auc = -1.0

    for epoch in epochs:
        try:
            auc = epoch['data']['Warm Start Users']['AUC']
            # Handle nan
            if auc != auc: 
                continue
            if auc > max_auc:
                max_auc = auc
                best_epoch = epoch
        except KeyError:
            continue

    if best_epoch:
        print(f"Best Epoch: {best_epoch['epoch']} (Warm Start AUC: {max_auc:.4f})")
        print()
        
        sections = ['Overall', 'Warm Start Users', 'Cold Start Users']
        # Collect all unique metrics found in the best epoch across sections
        metrics = []
        if 'Overall' in best_epoch['data']:
            metrics = list(best_epoch['data']['Overall'].keys())
        elif 'Warm Start Users' in best_epoch['data']:
            metrics = list(best_epoch['data']['Warm Start Users'].keys())
        
        # Markdown Table
        header = "| Model | Metric | " + " | ".join(sections) + " |"
        separator = "| :--- | :--- | " + " | ".join([":---"] * len(sections)) + " |"
        
        print(header)
        print(separator)
        
        for i, metric in enumerate(metrics):
            # Simulate rowspan by only printing model name in the first row
            current_model_name = model_name if i == 0 else ""
            row = [current_model_name, metric]
            for section in sections:
                val = best_epoch['data'].get(section, {}).get(metric, "N/A")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            print("| " + " | ".join(row) + " |")
    else:
        print("No valid epoch data found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze log file for best Warm Start AUC.")
    parser.add_argument("log_file", type=str, help="Path to the log file")
    args = parser.parse_args()
    
    log_file = args.log_file
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
    else:
        # Extract model name from directory path
        model_name = os.path.basename(os.path.dirname(os.path.abspath(log_file))).split('_')[0]
        epochs = parse_log(log_file)
        print_best_epoch(epochs, model_name)
