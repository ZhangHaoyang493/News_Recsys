import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional
from .data_reader import DataReader
from ...Logger.logging import Logger

class MINDDataModule(pl.LightningDataModule):
    """
    全配置驱动的 MIND 数据模块
    所有参数（batch_size, num_workers, file_paths）均从 conf 指向的 YAML 中读取。
    """

    def __init__(self, config):
        """
        Args:
            conf (str): YAML 配置
        """
        super().__init__()
        self.save_hyperparameters() # 自动保存 __init__ 参数 (这里只有 conf)

        
        # 1. 加载配置
        self.conf = config
        
        # 2. 从配置中提取路径
        # 根据之前的 yaml 结构: paths.out_basedir
        self.out_basedir = self.conf.paths.out_basedir
        
        # 3. 从配置中提取数据加载参数 (使用 .get 提供默认值以防 yaml 漏写)
        data_cfg = self.conf.get('dataset', {})
        
        self.batch_size = data_cfg.get('batch_size', 32)
        self.num_workers = data_cfg.get('num_workers', 4)
        self.pin_memory = data_cfg.get('pin_memory', True)
        
        # 4. 组装完整路径
        feature_dir = os.path.join(self.out_basedir, 'extractored_feature')
        self.train_file_path = os.path.join(feature_dir, "train_features.txt")
        self.val_file_path = os.path.join(feature_dir, "dev_features.txt")
        self.news_file_path = os.path.join(feature_dir, "item_features.txt")
        
        # 占位符
        self.train_dataset = None
        self.val_dataset = None

        # 日志
        self.logger = Logger.get_logger("MINDDataModule")

    def setup(self, stage: Optional[str] = None):
        """
        在 fit/test 开始前运行，用于实例化 Dataset
        """
        if stage == "fit" or stage is None:
            # 检查文件是否存在
            if not os.path.exists(self.train_file_path):
                raise FileNotFoundError(f"Training feature file missing: {self.train_file_path}")
            if not os.path.exists(self.val_file_path):
                raise FileNotFoundError(f"Validation feature file missing: {self.val_file_path}")

            self.logger.info(f"Loading Train Data from: {self.train_file_path}")
            self.logger.info(f"Batch Size: {self.batch_size}, Workers: {self.num_workers}")
            
            # 实例化你的 DataReader
            self.train_dataset = DataReader(
                config=self.conf, 
                feature_file_path=self.train_file_path,
                filter_negative=(self.conf.train_stage == "retrieval")
            )

            if self.conf.dataset.only_test_warm_up_user:
                train_user_id_set = self.train_dataset.get_user_id_set()
            
            self.logger.info(f"Loading Val Data from: {self.val_file_path}")
            if self.conf.dataset.only_test_warm_up_user:
                self.logger.info(f"Only test warm up user is True!")
                self.val_dataset = DataReader(
                    config=self.conf, 
                    feature_file_path=self.val_file_path,
                    train_user_id_set=train_user_id_set
                )
            else:
                self.val_dataset = DataReader(
                    config=self.conf, 
                    feature_file_path=self.val_file_path
                )

            self.logger.info("Loading News Data from: {}".format(self.news_file_path))
            self.news_dataset = DataReader(
                config=self.conf,
                feature_file_path=self.news_file_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )

    def val_dataloader(self):
        validation_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        if self.conf.train_stage == "retrieval":
            news_dataloader = DataLoader(
                self.news_dataset,
                batch_size=1024,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False
            )
            return [news_dataloader, validation_dataloader]

        return validation_dataloader