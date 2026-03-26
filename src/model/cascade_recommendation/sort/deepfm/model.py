import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

from ....BaseModel.base_model_sort import BaseModelSort
from ....model_utils.lr_schedule import CosinDecayLR
from ....model_utils.utils import MLP
from sklearn.metrics import roc_auc_score
    
class DeepFMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32, 1]):
        super(DeepFMModel, self).__init__()
        dims = [input_dim] + hidden_dims
        
        self.deep_network = MLP(dims=dims)
        self.bias = nn.Parameter(torch.zeros(1))

    
    def forward(self, one_order_features, two_order_features, deep_features):
        first_order = torch.sum(one_order_features, dim=1, keepdim=True)  # Bx1
        second_order = 0.5 * torch.sum(
            torch.pow(torch.sum(two_order_features, dim=1), 2) - torch.sum(torch.pow(two_order_features, 2), dim=1),  # Bxdim
            dim=-1,
            keepdim=True
        )  # Bx1

        deep_out = self.deep_network(deep_features)
        return F.sigmoid(first_order + second_order + deep_out + self.bias)

class DeepFM(BaseModelSort):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        # 定义Deep模型的网络结构
        self.score_fc = DeepFMModel(input_dim=self.user_input_dim + self.item_input_dim, hidden_dims=[128, 128, 128, 64, 1])
        self.fm_cfg = self.config.fm_cfg
        self.fm_features = set(self.fm_cfg.fm_feature)
        self.fm_dim = self.fm_cfg.fm_dim


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        one_order_features, two_order_features, deep_features = self.get_inp_embedding(x)  # 获取输入特征向量
        return self.score_fc(one_order_features, two_order_features, deep_features)  # 返回预测分数

    
    def get_inp_embedding(self, batch):
        one_order_features, _, _ = self.get_embeddings_from_batch('one_order_embedding_table', batch, self.fm_features)
        two_order_features, _, _ = self.get_embeddings_from_batch('two_order_embedding_table', batch, self.fm_features)  # Bxall_dim
        deep_features, _, _ = self.get_embeddings_from_batch('base_embedding_table', batch, self.user_feature_names | self.item_feature_names)  # Bxall_dim
        B, _ = two_order_features.shape
        two_order_features = two_order_features.view(B, -1, self.fm_dim)

        return one_order_features, two_order_features, deep_features

    
    def training_step(self, batch, batch_idx):
        scores = self.forward(batch)
        labels = batch['label'][:, 0]  # 获取是否喜欢的标签
        loss = self.bceLoss(scores, labels)  # 计算二元交叉熵损失

        # 添加 L1 正则化 (针对所有 Embedding 表)
        l1_reg = 0
        l1_lambda = getattr(self.config, 'l1_lambda', 1e-5) # 默认 1e-5，可在 config 中配置
        
        # 遍历所有 Parameter，对 Embedding 权重施加 L1
        for name, param in self.named_parameters():
            if 'one_order_embedding_table' in name:
                l1_reg += torch.norm(param, 1)
                
        loss += l1_lambda * l1_reg

        train_auc = roc_auc_score(labels.cpu().numpy(), scores.detach().cpu().numpy())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('l1_reg', l1_reg, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_auc', train_auc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        # 优化器改为 SGD
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_hparams.lr, betas=(0.9, 0.999))
        # 依然使用 Cosine 衰减
        lr_scheduler = CosinDecayLR(optimizer, lrs=[self.train_hparams.lr, self.train_hparams.min_lr], milestones=self.train_hparams.lr_milestones)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    @torch.no_grad()
    def inference(self, batch):
        w, v, deep_f = self.get_inp_embedding(batch)  # 获取输入特征向量
        return self.score_fc(w, v, deep_f)  # 返回预测分数


