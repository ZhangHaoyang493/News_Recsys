# 预处理数据
.PHONY: preprocess
preprocess:
	python -m src.dataset.FeaturesGenerator.preprocess -c src/dataset/FeaturesGenerator/config_fg.yaml

# 清理临时文件
.PHONY: clean
clean:
	@ echo "是否确认清理临时文件? (src/tmp)"
	@ read -p "输入 'yes' 确认: " confirm; \
	if [ "$$confirm" != "yes" ]; then \
		echo "操作已取消"; \
	else \
		echo "清理临时文件... (src/tmp)"; \
		rm -rf src/tmp; \
	fi

# 特征提取
.PHONY: fe
mode ?= normal
fe:
	python -m src.dataset.FeaturesGenerator.feature_extractor -c src/dataset/FeaturesGenerator/config_fg.yaml --mode $(mode)

# 将提取后的 txt 特征转为 npz（独立步骤）
.PHONY: fe_npz
fe_npz:
	python -m src.dataset.FeaturesGenerator.txt2npz_converter -c src/dataset/FeaturesGenerator/config_fg.yaml --feature_dir src/tmp/extractored_feature

# 训练排序模型，指令示例: make train model=deep
.PHONY: train
stage ?= sort
model_group ?= classic_sort_models
train:
	@if [ -z "$(model)" ]; then \
		echo "Usage: make train model=<model_name> [stage=sort] [model_group=classic_sort_models]"; \
		exit 1; \
	elif [ "$(stage)" != "sort" ]; then \
		echo "当前目录结构仅支持 stage=sort，收到 stage=$(stage)"; \
		exit 1; \
	else \
		python -m src.model.sort_models.$(model_group).$(model).train \
			-c src/model/sort_models/$(model_group)/base_sort_conf.yaml \
			-e src/model/sort_models/$(model_group)/$(model)/$(model)_conf.yaml; \
	fi

# 分析日志，指令示例: make log model=deep
.PHONY: log
log:
	@latest_exp=$$(ls -d experiments/$(model)_20* 2>/dev/null | sort -r | head -n 1); \
	if [ -z "$$latest_exp" ]; then \
		echo "未找到符合条件的实验文件夹 (experiments/$(model)_20*)"; \
	else \
		echo "解析日志文件: $$latest_exp/val_log.log"; \
		python src/scripts/log_analysis.py "$$latest_exp/val_log.log"; \
	fi

.PHONY: visualize_history
visualize_history:
	python -m src.scripts.visiualize_user_history --news Data/MIND/MINDsmall_dev/news.tsv --behaviors Data/MIND/MINDsmall_dev/behaviors.tsv

.PHONY: clean_exper
clean_exper:
	@echo "正在检查experiments文件夹中的实验..."
	@echo ""
	@to_delete=""; \
	for dir in experiments/*/; do \
		if [ -f "$$dir/val_log.log" ]; then \
			line_count=$$(wc -l < "$$dir/val_log.log"); \
			if [ "$$line_count" -lt 25 ]; then \
				to_delete="$$to_delete $$dir"; \
			fi; \
		fi; \
	done; \
	if [ -z "$$to_delete" ]; then \
		echo "没有找到需要删除的实验文件夹 (val_log.log行数 < 25)"; \
	else \
		echo "即将删除以下实验文件夹 (val_log.log行数 < 25):"; \
		for dir in $$to_delete; do \
			echo "  - $$dir"; \
		done; \
		echo ""; \
		read -p "确认删除? (输入 'y' 确认): " confirm; \
		if [ "$$confirm" = "y" ]; then \
			echo "正在删除..."; \
			for dir in $$to_delete; do \
				rm -rf "$$dir"; \
				echo "已删除: $$dir"; \
			done; \
		else \
			echo "操作已取消"; \
		fi; \
	fi

.PHONY: server
server:
	@if [ -z "$(port)" ]; then echo "Usage: make server port=<port>"; exit 1; fi
	@echo "Checking port $(port)..."
	@pids=$$(lsof -i :$(port) | grep "python" | grep "zhy" | awk '{print $$2}'); \
	if [ -n "$$pids" ]; then \
		echo "Found python process(es) by zhy on port $(port): $$pids"; \
		echo "Killing..."; \
		kill -9 $$pids; \
		sleep 1; \
	fi
	python -m http.server $(port) --directory .

.PHONY: vis_recall
vis_recall:
	python -m src.scripts.visualize_recall_html --recall_file $(path)
