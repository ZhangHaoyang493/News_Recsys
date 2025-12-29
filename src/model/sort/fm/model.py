import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.BaseModel.base_model import BaseModel
from src.model.model_utils.lr_schedule import CosinDecayLR
from sklearn.metrics import roc_auc_score

class FMModel(nn.Module):
    def __init__(self):
        super(FMModel, self).__init__()

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, w, v):
        first_order = torch.sum(w, dim=1, keepdim=True)  # Bx1
        second_order = 0.5 * torch.sum(
            torch.pow(torch.sum(v, dim=1), 2) - torch.sum(torch.pow(v, 2), dim=1),
            dim=1,
            keepdim=True
        )  # Bx1
        output = self.bias + first_order + second_order  # Bx1
        return torch.sigmoid(output)  # Bx1
    

class FM(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        

        # 定义Deep模型的网络结构
        self.score_fc = FMModel()
        


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        w, v = self.get_inp_embedding(x)  # 获取输入特征向量
        return self.score_fc(w, v)  # 返回预测分数

    
    def get_inp_embedding(self, batch):
        features, dims, _ = self.get_embeddings_from_batch(batch, self.user_feature_names | self.item_feature_names)
        w, v = [], []
        start_idx = 0
        for dim in dims:
            end_idx = start_idx + dim
            w.append(features[:, start_idx:start_idx+1])  # Bx1
            v.append(features[:, start_idx+1:end_idx])  # Bxdim
            start_idx = end_idx
        w = torch.cat(w, dim=1)  # Bxnum_fields
        v = torch.stack(v, dim=1)  # Bxnum_fieldsxdim
        return w, v
    
    def training_step(self, batch, batch_idx):
        scores = self.forward(batch)
        labels = batch['label'][:, 0]  # 获取是否喜欢的标签
        loss = self.bceLoss(scores, labels)  # 计算二元交叉熵损失
        train_auc = roc_auc_score(labels.cpu().numpy(), scores.detach().cpu().numpy())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_auc', train_auc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_hparams.lr, betas=(0.9, 0.999))
        lr_scheduler = CosinDecayLR(optimizer, lrs=[self.train_hparams.lr, self.train_hparams.min_lr], milestones=self.train_hparams.lr_milestones)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',  # 每个训练步骤调用一次
                'frequency': 1
            }

        }
    
    @torch.no_grad()
    def inference(self, batch):
        w, v = self.get_inp_embedding(batch)  # 获取输入特征向量
        return self.score_fc(w, v)  # 返回预测分数


