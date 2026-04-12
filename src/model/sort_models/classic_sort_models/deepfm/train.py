import os

from .model import DeepFM
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from .....dataset.DataReader.pl_dataloader import MINDDataModule
import datetime
from omegaconf import OmegaConf

L.seed_everything(42, workers=True)


def parse_args():
    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="DeepFM Training")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.path.join(base_dir, "..", "base_sort_conf.yaml"),
        help="Path to base config file",
    )
    parser.add_argument(
        "--extra_config",
        "-e",
        type=str,
        default=os.path.join(base_dir, "deepfm_conf.yaml"),
        help="Path to extra config file",
    )
    return parser.parse_args()


def load_config(args):
    base_config = OmegaConf.load(args.config)
    if args.extra_config:
        extra_config = OmegaConf.load(args.extra_config)
        return OmegaConf.merge(base_config, extra_config)
    return base_config


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    data_module = MINDDataModule(config)
    data_module.setup()

    model = DeepFM(config)

    name = model.config.get("name", "default_experiment")
    max_epochs = model.train_hparams.get("max_epoch", 10)
    val_freq = model.train_hparams.get("val_freq", 1)
    devices = model.train_hparams.get("gpus", [0])

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(
        save_dir=".",
        name="experiments",
        version=name + "_" + time_str,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        val_check_interval=val_freq,
        accelerator="gpu",
        devices=devices,
        logger=logger,
    )

    trainer.fit(model, data_module)
