import pandas as pd
import os
import html
import argparse
import json
from ..Logger.logging import Logger

# 配置 logging
logger = Logger.get_logger("VisiualizeUserHistory")

def load_news_data(news_path):
    """
    读取新闻数据，返回一个字典映射: NewsID -> {Title, Category, SubCategory, Abstract}
    """
    logger.info(f"Loading news data from {news_path}...")
    # 假设没有header，列顺序符合MIND数据集格式
    # News ID, Category, Subcategory, Title, Abstract, URL, Title Entities, Abstract Entities
    df = pd.read_csv(news_path, sep='\t', header=None, names=[
        'news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'
    ], usecols=['news_id', 'category', 'subcategory', 'title', 'abstract'])
    
    news_map = {}
    for _, row in df.iterrows():
        news_map[row['news_id']] = {
            'category': row['category'],
            'subcategory': row['subcategory'],
            'title': row['title'],
            'abstract': row['abstract']
        }
    return news_map

def load_behaviors_data(behaviors_path):
    """
    读取行为数据。
    """
    logger.info(f"Loading behaviors data from {behaviors_path}...")
    # Impression ID, User ID, Time, History, Impressions
    df = pd.read_csv(behaviors_path, sep='\t', header=None, names=[
        'impression_id', 'user_id', 'time', 'history', 'impressions'
    ])
    
    # 填充空值
    df['history'] = df['history'].fillna('')
    
    # 转换时间列为 datetime 对象以便正确排序
    try:
        df['time'] = pd.to_datetime(df['time'])
    except Exception as e:
        logger.warning(f"Failed to parse time column: {e}. Sorting might be incorrect.")

    # 返回所有数据，不再采样
    return df

def generate_html_report(behaviors_df, news_map, output_path):
    logger.info("Generating HTML report...")
    
    # 1. 准备数据供前端使用 (JSON)
    # 为了减小文件体积，我们只提取必要字段并使用简短的键名
    
    # 转换 news_map
    # 前端只需要: title, category, subcategory, abstract
    js_news_map = {}
    for nid, info in news_map.items():
        js_news_map[nid] = {
            't': info.get('title', ''),
            'c': info.get('category', ''),
            's': info.get('subcategory', ''),
            'a': info.get('abstract', '')
        }

    # 转换 behaviors
    # 结构: [ { uid: 'u1', imps: [ { iid: '...', t: '...', h: ['n1', 'n2'], c: [['n3', 1], ['n4', 0]] } ] } ]
    users_data = []
    grouped = behaviors_df.groupby('user_id')
    
    logger.info(f"Processing {len(grouped)} users for visualization...")
    
    for user_id, group in grouped:
        user_obj = {'uid': user_id, 'imps': []}
        
        # Sort impressions by time (ascending)
        group = group.sort_values('time')
        
        for _, row in group.iterrows():
            history_ids = row['history'].split() if isinstance(row['history'], str) else []
            impressions_raw = row['impressions'].split() if isinstance(row['impressions'], str) else []
            
            candidates = []
            for imp_str in impressions_raw:
                try:
                    nid, label = imp_str.rsplit('-', 1)
                    candidates.append([nid, int(label)])
                except ValueError:
                    continue
            
            user_obj['imps'].append({
                'iid': row['impression_id'],
                't': str(row['time']), # Convert timestamp back to string for JSON
                'h': history_ids,
                'c': candidates
            })
        users_data.append(user_obj)

    # 序列化数据
    logger.info("Serializing data to JSON (this might take a moment)...")
    json_news = json.dumps(js_news_map)
    json_users = json.dumps(users_data)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>User News History Visualization</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 0; height: 100vh; overflow: hidden; }}
            
            /* Layout */
            .app-container {{ display: flex; height: 100vh; }}
            
            /* Sidebar: Users */
            .col-users {{ width: 250px; background: #fff; border-right: 1px solid #ddd; display: flex; flex-direction: column; }}
            .col-header {{ padding: 15px; background: #2c3e50; color: #fff; font-weight: bold; }}
            .list-container {{ flex: 1; overflow-y: auto; }}
            
            /* Middle: Impressions */
            .col-impressions {{ width: 300px; background: #f9f9f9; border-right: 1px solid #ddd; display: flex; flex-direction: column; }}
            
            /* Right: Details */
            .col-details {{ flex: 1; background: #fff; overflow-y: auto; padding: 20px; }}
            
            /* List Items */
            .list-item {{ padding: 12px 15px; border-bottom: 1px solid #eee; cursor: pointer; transition: background 0.2s; }}
            .list-item:hover {{ background-color: #f0f0f0; }}
            .list-item.active {{ background-color: #3498db; color: white; border-color: #3498db; }}
            .list-item-meta {{ font-size: 0.8em; color: #888; margin-top: 4px; }}
            .list-item.active .list-item-meta {{ color: #e0e0e0; }}
            
            /* Detail View Styles */
            .detail-header {{ margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #eee; }}
            .section-title {{ font-weight: bold; margin: 25px 0 15px; color: #555; text-transform: uppercase; font-size: 0.9em; letter-spacing: 1px; border-left: 4px solid #3498db; padding-left: 10px; }}
            
            /* Grid for History and Candidates */
            .history-container, .candidates-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            
            /* News Card */
            .news-card {{ 
                background: #fff; 
                border-radius: 8px; 
                padding: 15px; 
                font-size: 0.9em; 
                display: flex; 
                flex-direction: column; 
                min-width: 240px; 
                max-width: 240px; 
                position: relative; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border: 1px solid #eee;
                transition: all 0.2s ease;
                height: 180px; /* Fixed height for uniformity */
            }}
            .news-card:hover {{ 
                transform: translateY(-4px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.12); 
                z-index: 10;
            }}
            
            .news-meta {{ 
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                margin-bottom: 10px; 
                font-size: 0.75em; 
            }}
            .news-cat {{ 
                color: #555; 
                background: #f2f4f6;
                padding: 3px 8px;
                border-radius: 100px;
                font-weight: 600;
                font-size: 0.85em;
                max-width: 160px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .news-title {{ 
                font-weight: 700; 
                line-height: 1.4; 
                color: #1a1a1a;
                margin-bottom: auto; 
                /* Line clamping */
                display: -webkit-box;
                -webkit-line-clamp: 4;
                -webkit-box-orient: vertical;
                overflow: hidden;
                font-size: 1em;
            }}
            .news-footer {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 12px;
                padding-top: 8px;
                border-top: 1px solid #f7f7f7;
            }}
            .news-id {{ 
                font-size: 0.75em; 
                color: #ccc; 
                font-family: 'Consolas', monospace;
            }}
            
            /* Status Styles */
            .clicked {{ border-top: 4px solid #2ecc71; }}
            .not-clicked {{ border-top: 4px solid #e0e0e0; opacity: 0.9; }}
            .history-item {{ border-top: 4px solid #3498db; background: linear-gradient(to bottom, #fbfdff, #fff); }}
            
            .badge {{ 
                padding: 3px 8px; 
                border-radius: 4px; 
                color: #fff; 
                font-weight: bold; 
                font-size: 0.7em; 
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .clicked-badge {{ background-color: #2ecc71; color: white; }}
            .ignored-badge {{ background-color: #ecf0f1; color: #95a5a6; }}
            .history-badge {{ background-color: #3498db; color: white; }}
            
            .empty-state {{ text-align: center; color: #999; margin-top: 50px; font-size: 1.2em; }}
            
            /* Search Box */
            .search-box {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            .search-box input {{ width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="app-container">
            <!-- Column 1: Users -->
            <div class="col-users">
                <div class="col-header">Users (<span id="user-count">0</span>)</div>
                <div class="search-box">
                    <input type="text" id="user-search" placeholder="Search User ID...">
                </div>
                <div id="user-list" class="list-container"></div>
            </div>
            
            <!-- Column 2: Impressions -->
            <div class="col-impressions">
                <div class="col-header">Impressions</div>
                <div id="impression-list" class="list-container">
                    <div class="empty-state">Select a user</div>
                </div>
            </div>
            
            <!-- Column 3: Details -->
            <div class="col-details">
                <div id="detail-view">
                    <div class="empty-state">Select an impression to view details</div>
                </div>
            </div>
        </div>

        <script>
            const newsMap = {json_news};
            const usersData = {json_users};
            
            // DOM Elements
            const userListEl = document.getElementById('user-list');
            const impListEl = document.getElementById('impression-list');
            const detailViewEl = document.getElementById('detail-view');
            const userCountEl = document.getElementById('user-count');
            const userSearchInput = document.getElementById('user-search');
            
            let selectedUserIndex = -1;
            let selectedImpIndex = -1;

            // Initialize
            userCountEl.textContent = usersData.length;
            renderUserList(usersData);

            // Search functionality
            userSearchInput.addEventListener('input', (e) => {{
                const term = e.target.value.toLowerCase();
                const filtered = usersData.filter(u => u.uid.toLowerCase().includes(term));
                renderUserList(filtered, term ? true : false);
            }});

            function escapeHtml(text) {{
                if (!text) return "";
                return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
            }}

            function renderUserList(data, isFiltered=false) {{
                userListEl.innerHTML = '';
                // If filtered, we need to map back to original index or handle selection differently.
                // For simplicity, we store original index in dataset.
                
                data.forEach((user, idx) => {{
                    // Find original index if filtered
                    const originalIndex = isFiltered ? usersData.indexOf(user) : idx;
                    
                    const div = document.createElement('div');
                    div.className = 'list-item';
                    if (originalIndex === selectedUserIndex) div.classList.add('active');
                    div.innerHTML = `
                        <div>${{user.uid}}</div>
                        <div class="list-item-meta">${{user.imps.length}} impressions</div>
                    `;
                    div.onclick = () => selectUser(originalIndex, div);
                    userListEl.appendChild(div);
                }});
            }}

            function selectUser(index, element) {{
                selectedUserIndex = index;
                selectedImpIndex = -1;
                
                // Update UI active state
                document.querySelectorAll('#user-list .list-item').forEach(el => el.classList.remove('active'));
                if(element) element.classList.add('active');
                
                renderImpressionList(index);
                
                // Auto-select first impression
                const user = usersData[index];
                if (user && user.imps.length > 0) {{
                    const firstImpEl = impListEl.firstElementChild;
                    if (firstImpEl) {{
                        selectImpression(index, 0, firstImpEl);
                    }}
                }} else {{
                    detailViewEl.innerHTML = '<div class="empty-state">Select an impression to view details</div>';
                }}
            }}

            function renderImpressionList(userIndex) {{
                impListEl.innerHTML = '';
                const user = usersData[userIndex];
                
                if (!user || user.imps.length === 0) {{
                    impListEl.innerHTML = '<div class="empty-state">No impressions</div>';
                    return;
                }}

                user.imps.forEach((imp, idx) => {{
                    const div = document.createElement('div');
                    div.className = 'list-item';
                    div.innerHTML = `
                        <div>${{imp.iid}}</div>
                        <div class="list-item-meta">${{imp.t}}</div>
                    `;
                    div.onclick = () => selectImpression(userIndex, idx, div);
                    impListEl.appendChild(div);
                }});
            }}

            function selectImpression(userIndex, impIndex, element) {{
                selectedImpIndex = impIndex;
                
                document.querySelectorAll('#impression-list .list-item').forEach(el => el.classList.remove('active'));
                element.classList.add('active');
                
                renderDetail(userIndex, impIndex);
            }}

            function createCard(nid, type, clicked=false) {{
                const info = newsMap[nid] || {{ t: 'Unknown', c: '-', s: '-', a: '' }};
                let cardClass = "news-card";
                let badge = "";

                if (type === 'history') {{
                    cardClass += " history-item";
                    badge = '<span class="badge history-badge">History</span>';
                }} else {{
                    if (clicked) {{
                        cardClass += " clicked";
                        badge = '<span class="badge clicked-badge">Clicked</span>';
                    }} else {{
                        cardClass += " not-clicked";
                        badge = '<span class="badge ignored-badge">Ignored</span>';
                    }}
                }}

                return `
                <div class="${{cardClass}}" title="${{escapeHtml(info.a)}}">
                    <div class="news-meta">
                        <span class="news-cat" title="${{escapeHtml(info.c)}} / ${{escapeHtml(info.s)}}">${{escapeHtml(info.c)}} / ${{escapeHtml(info.s)}}</span>
                        ${{badge}}
                    </div>
                    <div class="news-title">${{escapeHtml(info.t)}}</div>
                    <div class="news-footer">
                        <span class="news-id">${{nid}}</span>
                    </div>
                </div>
                `;
            }}

            function renderDetail(userIndex, impIndex) {{
                const imp = usersData[userIndex].imps[impIndex];
                
                let html = `
                    <div class="detail-header">
                        <h2>Impression Details</h2>
                        <div><strong>ID:</strong> ${{imp.iid}}</div>
                        <div><strong>Time:</strong> ${{imp.t}}</div>
                    </div>
                    
                    <div class="section-title">User History (Prior Clicks)</div>
                    <div class="history-container">
                `;
                
                if (imp.h.length === 0) {{
                    html += '<div style="padding:10px; color:#999;">No history available</div>';
                }} else {{
                    imp.h.forEach(nid => {{
                        html += createCard(nid, 'history');
                    }});
                }}
                
                html += `
                    </div>
                    <div class="section-title">Impression Candidates</div>
                    <div class="candidates-container">
                `;
                
                // Sort candidates: clicked (1) first
                const sortedCandidates = [...imp.c].sort((a, b) => b[1] - a[1]);
                
                sortedCandidates.forEach(item => {{
                    const [nid, label] = item;
                    html += createCard(nid, 'candidate', label === 1);
                }});
                
                html += `</div>`;
                
                detailViewEl.innerHTML = html;
            }}
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize User News History")
    parser.add_argument('--news', type=str, required=True, help='Path to news.tsv')
    parser.add_argument('--behaviors', type=str, required=True, help='Path to behaviors.tsv')
    parser.add_argument('--output', type=str, default='src/tmp/user_history_viz.html', help='Output HTML file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.news) or not os.path.exists(args.behaviors):
        logger.error("Error: Input files not found.")
    else:
        news_map = load_news_data(args.news)
        behaviors_df = load_behaviors_data(args.behaviors)
        generate_html_report(behaviors_df, news_map, args.output)
