import pandas as pd
import os
import html
import json
from tqdm import tqdm
from collections import Counter

# ================= 配置路径 =================
# 请根据实际情况修改这些路径
RECALL_RESULT_FILE = "src/tmp/recall_result.tsv"  # 假设的召回结果文件路径
NEWS_INFO_FILE = "Data/MIND/MINDsmall_dev/news.tsv" # 假设的物料信息文件路径
OUTPUT_HTML_FILE = "recall_visualization.html"

# 采样展示的用户数量，防止HTML过大卡顿
SAMPLE_NUM = 100 

def clean_json_str(s):
    """
    Clean the entity string which might be double-quoted or have escaped quotes.
    Example: '"[{""Label"": ...}]"' -> '[{"Label": ...}]'
    """
    if not s:
        return []
    
    s = s.strip()
    # Remove outer quotes if they exist (common in some CSV formats)
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    
    # Replace double double-quotes with single double-quotes
    s = s.replace('""', '"')
    
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return []
    except Exception as e:
        # print(f"Error parsing json: {s[:20]}... {e}")
        return []

def load_news_info(news_file):
    """
    读取物料信息文件
    格式: item_id \t category \t subcategory \t title \t abstract \t url \t title_entities \t abstract_entities
    """
    if not os.path.exists(news_file):
        print(f"Warning: News file not found at {news_file}. Titles will be missing.")
        return {}
    
    news_dict = {}
    print(f"Loading news info from {news_file}...")
    try:
        with open(news_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                item_id, category, subcategory, title = parts[0], parts[1], parts[2], parts[3]
                
                abstract = parts[4] if len(parts) > 4 else ""
                # abstract_ents usually at index 7, title_ents at index 6 in MIND
                title_ents_str = parts[6] if len(parts) > 6 else "[]"
                abs_ents_str = parts[7] if len(parts) > 7 else "[]"
                
                title_ents = clean_json_str(title_ents_str)
                abs_ents = clean_json_str(abs_ents_str)

                # Extract labels for simple display
                t_labels = [{'Label': e.get('Label'), 'Type': e.get('Type')} for e in title_ents if e.get('Label')]
                a_labels = [{'Label': e.get('Label'), 'Type': e.get('Type')} for e in abs_ents if e.get('Label')]

                news_dict[item_id] = {
                    'cat': category,
                    'subcat': subcategory,
                    'title': title,
                    'abstract': abstract,
                    't_labels': t_labels,
                    'a_labels': a_labels
                }
    except Exception as e:
        print(f"Error reading news file: {e}")
    return news_dict

def load_recall_results(result_file, sample_num=100):
    """
    读取召回结果文件
    格式: impression_id \t user_id \t recall_items(opt) \t target_items(opt) \t history_items(opt)
    注意：您描述的格式是第3列召回，第4列目标，第5列历史。
    但有时候会有列缺失，需要做健壮性处理。
    """
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Recall result file not found: {result_file}")

    data = []
    print(f"Loading recall results from {result_file}...")
    
    hit_data = []
    normal_data = []

    with open(result_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            # 补全缺失列
            while len(parts) < 6:
                parts.append("")
                
            imp_id = parts[0]
            user_id = parts[1]
            recall_str = parts[2]
            target_str = parts[3]
            history_str = parts[4]
            scores_str = parts[5]

            recall_list = recall_str.split(',') if recall_str else []
            scores_list = scores_str.split(',') if scores_str else []
            target_list = target_str.split(',') if target_str else []
            
            # Check for hit (HR=1 logic: at least one target item is in recall items)
            # Assuming set intersection. ID matching.
            # Convert to sets for faster lookup
            recall_set = set(recall_list)
            is_hit = any(t in recall_set for t in target_list)

            # 简单的长度对齐
            if len(scores_list) < len(recall_list):
                 scores_list.extend([""] * (len(recall_list) - len(scores_list)))

            entry = {
                'imp_id': imp_id,
                'user_id': user_id,
                'recall_list': recall_list,
                'score_list': scores_list,
                'target_list': target_list,
                'history_list': history_str.split(',') if history_str else [],
                'is_hit': is_hit
            }
            
            if is_hit:
                hit_data.append(entry)
            else:
                normal_data.append(entry)
    
            if sample_num > 0:
                data.append(entry)
            
            # Use sample_num broadly or just for normal?
            # Re-read requirement: "First visualize all hit rate=1 users, then show sample_num data"
            # It's likely sample_num refers to the "extra" normal data.
            # But the loop logic above was trying to be memory efficient. Now we read all.
            # If the file is huge (Millions), reading all is bad.
            # But "visualize ALL HIT users" implies we must scan the whole file to find hits.
            pass

    # 1. Extend ALL hits
    final_data = hit_data
    print(f"Loaded {len(hit_data)} HIT samples.")

    # 2. Add some normal samples
    added_count = 0
    if sample_num > 0:
        take_num = min(len(normal_data), sample_num)
        final_data.extend(normal_data[:take_num])
        added_count = take_num
        print(f"Added {added_count} normal samples.")
            
    return final_data

def generate_html(data, news_map, output_file):
    """生成可视化 HTML"""
    
    # CSS 样式
    style = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 25px; padding: 20px; }
        .header { display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }
        .meta-info { color: #666; font-size: 0.9em; }
        .section-title { font-weight: bold; margin-top: 10px; margin-bottom: 8px; color: #333; border-left: 4px solid #007bff; padding-left: 8px; }
        
        .item-row { display: flex; flex-wrap: wrap; gap: 10px; }
        .item-box { 
            border: 1px solid #ddd; border-radius: 4px; padding: 8px; width: 220px; font-size: 0.85em; 
            background: #fff; display: flex; flex-direction: column; justify-content: flex-start;
            position: relative;
        }
        .item-box:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); transform: translateY(-2px); transition: all 0.2s; z-index: 10; }
        
        .cat-tag { font-size: 0.75em; color: #fff; background-color: #6c757d; padding: 2px 6px; border-radius: 4px; align-self: flex-start; margin-bottom: 4px;}
        .title-text { color: #333; font-weight: 600; margin-bottom: 4px; line-height: 1.2; }
        .abstract-text { 
            color: #666; font-size: 0.8em; margin-bottom: 6px; 
            display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
            line-height: 1.3;
        }
        .id-text { color: #aaa; font-size: 0.75em; margin-top: auto; padding-top: 4px; border-top: 1px solid #eee;}

        .entities-container { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 4px; }
        .ent-tag { font-size: 0.7em; padding: 1px 4px; border-radius: 3px; border: 1px solid #ddd; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .ent-title { background-color: #e3f2fd; color: #0d47a1; border-color: #90caf9; }
        .ent-abs { background-color: #f3f3f3; color: #555; }
        
        /* Specific Colors */
        .box-history { border-color: #d1ecf1; background-color: #f0faff; }
        .box-recall { border-color: #e2e3e5; }
        .box-target { border-color: #c3e6cb; background-color: #f0fff4; border-width: 2px; }
        
        .hit { border-color: #28a745; background-color: #d4edda; }
        .hit::after { content: "HIT"; position: absolute; top: -8px; right: -8px; background: #28a745; color: white; font-size: 0.7em; padding: 2px 6px; border-radius: 10px; font-weight: bold;}
        
        .empty-msg { color: #999; font-style: italic; padding: 10px; }
    </style>
    """

    html_content = [f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Recall Visualization</title>
        {style}
    </head>
    <body>
        <div class="container">
            <h1>Recall Visualization Report</h1>
            <p>Displaying top {len(data)} records.</p>
    """]

    for row in tqdm(data, desc="Generating HTML"):
        imp_id = row['imp_id']
        user_id = row['user_id']
        history_list = row['history_list']
        recall_list = row['recall_list']
        score_list = row.get('score_list', [])
        target_list = set(row['target_list']) # Use set for fast lookup

        # 统计用户历史偏好 Category
        hist_cats = []
        for iid in history_list:
            if iid in news_map:
                hist_cats.append(news_map[iid]['cat'])
        top_cats = Counter(hist_cats).most_common(10)
        top_cats_str = "".join([f'<span style="background:#dee2e6;padding:2px 5px;border-radius:3px;margin-right:5px;font-size:0.8em;display:inline-block;margin-bottom:2px;">{c} <b>{n}</b></span>' for c,n in top_cats])

        # 计算指标
        hit_num = sum(1 for item in recall_list if item in target_list)
        recall_len = len(recall_list)
        precision = (hit_num / recall_len * 100) if recall_len > 0 else 0
        recall_rate = (hit_num / len(target_list) * 100) if len(target_list) > 0 else 0

        html_content.append(f"""
        <div class="card">
            <div class="header">
                <div>
                    <strong>User ID:</strong> {user_id} <br>
                    <span class="meta-info">Impression ID: {imp_id}</span>
                    <div style="margin-top:8px; color:#555; font-size:0.9em;">
                        <span style="display:inline-block; margin-bottom:4px;">History Pref:</span><br> 
                        {top_cats_str}
                    </div>
                </div>
                <div style="text-align: right;">
                    <strong>Hits: {hit_num} / {len(target_list)}</strong><br>
                    <span class="meta-info">Recall@K: {recall_len} | Precision: {precision:.1f}%</span>
                </div>
            </div>
        """)

        # --- Helper to render items ---
        def render_items(items, box_class, highlight_set=None, scores=None):
            if not items:
                return '<div class="empty-msg">No Data</div>'
            
            html_parts = ['<div class="item-row">']
            for i, item_id in enumerate(items):
                # Get info
                info = news_map.get(item_id, {})
                title = html.escape(info.get('title', 'Unknown'))
                cat = html.escape(f"{info.get('cat', '-')} > {info.get('subcat', '-')}")
                abstract = html.escape(info.get('abstract', ''))
                t_labels = info.get('t_labels', [])
                a_labels = info.get('a_labels', [])
                
                # Check hit
                is_hit = False
                if highlight_set and item_id in highlight_set:
                    is_hit = True
                
                extra_class = "hit " if is_hit else ""
                if is_hit:
                    box_class = box_class.replace("hit", "") # avoid dup
                
                # Score display
                score_html = ""
                if scores and i < len(scores) and scores[i]:
                     try:
                        score_val = float(scores[i])
                        score_html = f'<div style="font-size:0.8em; color:#0056b3; font-weight:bold; margin-top:2px;">Sc: {score_val:.4f}</div>'
                     except:
                        score_html = f'<div style="font-size:0.8em; color:#0056b3; font-weight:bold; margin-top:2px;">Sc: {scores[i]}</div>'

                # Entities HTML
                ents_html = '<div class="entities-container">'
                # Show first few entities
                for ent in t_labels[:3]:
                     lbl = str(ent.get('Label', ''))
                     typ = str(ent.get('Type', ''))
                     # Display Label (Type)
                     display_text = f"{html.escape(lbl)} <span style='font-size:0.8em; opacity:0.7;'>({html.escape(typ)})</span>"
                     ents_html += f'<span class="ent-tag ent-title" title="Title Entity: {html.escape(lbl)} (Type: {html.escape(typ)})">{display_text}</span>'
                for ent in a_labels[:2]:
                     lbl = str(ent.get('Label', ''))
                     typ = str(ent.get('Type', ''))
                     display_text = f"{html.escape(lbl)} <span style='font-size:0.8em; opacity:0.7;'>({html.escape(typ)})</span>"
                     ents_html += f'<span class="ent-tag ent-abs" title="Abstract Entity: {html.escape(lbl)} (Type: {html.escape(typ)})">{display_text}</span>'
                ents_html += '</div>'

                html_parts.append(f"""
                <div class="item-box {box_class} {extra_class}" title="{title}&#10;{abstract}">
                    <span class="cat-tag">{cat}</span>
                    <div class="title-text">{title}</div>
                    <div class="abstract-text">{abstract}</div>
                    {ents_html}
                    {score_html}
                    <div class="id-text">{item_id}</div>
                </div>
                """)
            html_parts.append('</div>')
            return "".join(html_parts)

        # 1. Target Items (Ground Truth)
        html_content.append(f'<div class="section-title">Target (True Positive)</div>')
        html_content.append(render_items(list(target_list), "box-target"))

        # 2. Recall Results (Predictions)
        html_content.append(f'<div class="section-title">Recall Results (Top {recall_len})</div>')
        html_content.append(render_items(recall_list, "box-recall", target_list, scores=score_list))


        # 3. User History
        html_content.append(f'<div class="section-title">User History (Last {len(history_list)})</div>')
        html_content.append(render_items(history_list, "box-history"))

        html_content.append("</div>") # End Card

    html_content.append("""
        </div>
    </body>
    </html>
    """)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Recall Visualization HTML")
    parser.add_argument("--recall_file", "-r", type=str, default="src/tmp/recall_result.tsv", help="Path to recall result TSV file")
    parser.add_argument("--news_file", type=str, default="/data2/zhy/News_Recsys/src/tmp/preprocess/all_news_preprocess.csv", help="Path to news TSV file")
    parser.add_argument("--output_file", type=str, default="recall_visualization.html", help="Path to output HTML file")
    parser.add_argument("--sample_num", type=int, default=300, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    RECALL_RESULT_FILE = args.recall_file
    NEWS_INFO_FILE = args.news_file
    OUTPUT_HTML_FILE = args.output_file
    SAMPLE_NUM = args.sample_num

    # Load Data
    news_map = load_news_info(NEWS_INFO_FILE)
    recall_data = load_recall_results(RECALL_RESULT_FILE, sample_num=SAMPLE_NUM)
    
    if recall_data:
        generate_html(recall_data, news_map, OUTPUT_HTML_FILE)
    else:
        print("No recall data found or file is empty.")
