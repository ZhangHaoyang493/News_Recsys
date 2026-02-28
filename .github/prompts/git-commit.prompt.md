---
title: 中文 Git 提交并推送
description: 自动生成符合规范的中文 Commit Message 并提供推送指令
---

请根据我当前暂存区（Staged Changes）的代码改动，完成以下任务：

1. **生成提交信息**：
   - 使用 **中文** 编写。
   - 严格遵循 Conventional Commits 规范（例如：`feat:`, `fix:`, `docs:`, `refactor:` 等）。
   - 第一行是精炼的摘要，如果改动较多，请在换行后列出具体的改动点。

2. **提供操作指令**：
   - 请直接给出可以直接复制执行的命令，格式如下：
     ```bash
     git commit -m "[刚才生成的提交信息]"
     git push origin [当前分支名]
     ```

3. **特别要求**：
   - 如果涉及图像处理或深度学习模型相关的参数调整，请在提交信息中明确标注。