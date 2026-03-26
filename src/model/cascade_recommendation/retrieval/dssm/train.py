import sys
sys.path.append('/data2/zhy/Movie_Recsys')

from .model import DSSM
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from .....dataset.DataReader.pl_dataloader import MINDDataModule
import datetime
from omegaconf import OmegaConf

L.seed_everything(42, workers=True)

def parse_args():
    parser = argparse.ArgumentParser(description="FM Training")
    parser.add_argument("--config", "-c", type=str, default="fm_conf.yaml", help="Path to config file")
    parser.add_argument("--extra_config", "-e", type=str, default=None, help="Path to extra config file")
    return parser.parse_args()

def load_config(args):
    base_config = OmegaConf.load(args.config)
    if args.extra_config:
        extra_config = OmegaConf.load(args.extra_config)
        combined_config = OmegaConf.merge(base_config, extra_config)
        return combined_config
    return base_config

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    data_module = MINDDataModule(config)
    data_module.setup()

    model = DSSM(config)

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
        val_check_interval=val_freq,
        accelerator='gpu',
        devices=1,
        logger=logger,
    )

    trainer.fit(model, data_module)