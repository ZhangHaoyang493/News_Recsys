# 预处理数据
.PHONY: preprocess
preprocess:
	python -m src.dataset.FeaturesGenerator.preprocess -c src/dataset/FeaturesGenerator/config_fg.yaml

# 清理临时文件
.PHONY: clean
clean:
	@ echo "清理临时文件... (src/tmp)"
	@ rm -rf src/tmp

# 特征提取
.PHONY: fe
fe:
	python -m src.dataset.FeaturesGenerator.feature_extractor -c src/dataset/FeaturesGenerator/config_fg.yaml

# 训练排序模型，指令示例: make train model=deep
.PHONY: train
train:
	python -m src.model.sort.$(model).train -c src/model/sort/$(model)/train_cf_$(model).yaml

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