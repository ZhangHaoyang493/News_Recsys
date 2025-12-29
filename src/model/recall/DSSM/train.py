import sys
sys.path.append('/data2/zhy/Movie_Recsys')

from recall.DSSM.model import DSSM
from torch.utils.data import DataLoader
import torch
import lightning as L
from DataReader.data_reader import DataReader
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DSSM Training")
    parser.add_argument("--config", "-c", type=str, default="/data2/zhy/Movie_Recsys/feature_recall.json", help="Path to config file")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Minimum learning rate")
    parser.add_argument("--lr_milestones", type=int, nargs='+', default=[10000, 60000], help="Learning rate decay milestones")
    parser.add_argument("--negative_sample_rate", type=int, default=3, help="Negative sample rate for training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 配置文件路径
    config_path = args.config
    # 训练数据路径
    train_feature_path = '/data2/zhy/Movie_Recsys/FeatureFiles/train_ratings_features.txt'
    # 验证数据路径
    val_feature_path = '/data2/zhy/Movie_Recsys/FeatureFiles/test_ratings_features.txt'
    # 电影数据路径
    movies_feature_path = "/data2/zhy/Movie_Recsys/FeatureFiles/movie_features.txt"
    
    # 初始化DataReader
    train_dataset = DataReader(config_path, train_feature_path)
    val_dataset = DataReader(config_path, val_feature_path)
    # 构造movies的Dataloader
    movies_dataset = DataReader(config_path, movies_feature_path)
    
    
    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    movies_dataloader = DataLoader(movies_dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=False)
    
    # 初始化DSSM模型
    hparams = {
        'lr': args.lr,
        'min_lr': args.min_lr,
        'lr_milestones': args.lr_milestones,
        'negative_sample_rate': args.negative_sample_rate
    }
    model = DSSM(config_path, {'movies_dataloader': movies_dataloader, 'val_dataloader': val_dataloader}, hparams)

    # 定义检查点回调，只保存模型，不做验证
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        save_top_k=-1,  # 保存所有epoch的模型
        every_n_epochs=1,  # 每个epoch都保存
        dirpath="./checkpoints",  # 保存路径
        filename="dssm-epoch{epoch}",  # 文件名格式
        save_weights_only=True
    )
    
    # 初始化Trainer
    trainer = L.Trainer(
        max_epochs=100,  # 最大训练轮数
        accelerator="gpu", # 使用GPU加速
        devices=[1], # 使用1块GPU
        callbacks=[checkpoint_callback]  # 添加检查点回调
    )

    # 训练模型
    trainer.fit(model, train_dataloader)