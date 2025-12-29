import sys

from .model import Deep
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from ....dataset.DataReader.pl_dataloader import MINDDataModule
import datetime

L.seed_everything(42, workers=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Training")
    parser.add_argument("--config", "-c", type=str, default="/data2/zhy/Movie_Recsys/src/model/sort/deep/train_cf_deep.yaml", help="Path to config file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data_module = MINDDataModule(args.config)
    data_module.setup()

    model = Deep(args.config)

    # 从配置中获取训练参数
    name = model.config.get('name', 'default_experiment')
    max_epochs = model.train_hparams.get('max_epoch', 10)
    val_freq = model.train_hparams.get('val_freq', 1)

    # 1. 自定义 Logger
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(
        save_dir=".",      # 根目录
        name="experiments",    # 实验名称 (默认是 lightning_logs)
        version=name + '_' + time_str # <--- 这里！设置你想要的名字，代替 version_xxx
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=val_freq,
        accelerator='gpu',
        devices=1,
        logger=logger,
    )

    trainer.fit(model, data_module)
