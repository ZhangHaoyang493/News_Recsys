import lightning as L
import pyjson5 as json
from src.dataset.DataReader.get_dataloader import get_dataloader

class Trainer:
    def __init__(self):
        self.train_dataloader = None
        self.val_dataloader = None
        self.movies_dataloader = None
        self.model = None
        self.config = None
        self.args = None
        self.train_config = None

    def set_config(self, args):
        self.args = args
        with open(args.config, 'r') as f:
            self.config = json.load(f)
        self.train_config = self.config['train_hparams']

    def set_dataset(self):
        if self.train_config is None:
            raise "You must set train_config first"
        self.train_dataloader, self.val_dataloader, self.movies_dataloader = get_dataloader(self.args)

    def set_model(self, Model: L.LightningModule):
        if self.train_dataloader is None or self.val_dataloader is None or self.movies_dataloader is None:
            raise "You must set dataset first"
        self.model = Model(self.args.config, {'movies_dataloader': self.movies_dataloader, 'val_dataloader': self.val_dataloader})

    def train(self):
        if self.model is None:
            raise "You must set model first"
        
        # 初始化Trainer
        trainer = L.Trainer(
            # max_steps=50000,  # 最大训练轮数
            max_steps=self.train_config['max_step'],
            accelerator=self.train_config['device'], # 使用GPU加速
            devices=self.train_config['gpus'], # 使用1块GPU
            # # callbacks=[checkpoint_callback]  # 添加检查点回调
        )

        # 训练模型
        trainer.fit(self.model, self.train_dataloader)