import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....BaseModel.base_model_recall import BaseModelRecall
from ....model_utils.lr_schedule import CosinDecayLR

# BaseModel继承于LightningModule
class DSSM(BaseModelRecall):
    def __init__(self, config):
        super().__init__(config)

        
        # 定义DSSM的网络结构
        # 假设我们有两个全连接层，分别用于用户和物品的特征处理
        self.user_fc = nn.Sequential(
            nn.Linear(self.user_input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32)
        )
        
        self.item_fc = nn.Sequential(
            nn.Linear(self.item_input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32)
        )


    def forward(self, x):
        user_vector = self.get_user_embedding(x)  # 获取用户特征向量
        item_vector = self.get_item_embedding(x)  # 获取物品特征向量
        
        user_emb = self.user_fc(user_vector)  # 用户特征通过全连接层  bx16
        item_emb = self.item_fc(item_vector)  # 物品特征通过全连接层  bx16

        # 归一化
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return user_emb, item_emb

    def triplet_loss(self, user_emb, pos_item_emb, margin=1.0, mask=None):
        # user_emb: 用户特征向量，形状为 (batch_size, 16)
        # pos_item_emb: 正样本物品特征向量，形状为 (batch_size, 16)
        # 此时所有 item_emb 都是负样本池的一部分(除了自身的正样本)
        
        # 计算所有的相似度: (B, B)
        scores = torch.matmul(user_emb, pos_item_emb.T)
        
        # 正样本得分: 对角线元素 (B,)
        pos_scores = torch.diag(scores)
        
        # 负样本得分: 所有元素 (B, B)
        # 这里为了简单实现 Triplet，我们要让 (pos - neg) 所有组合都满足要求
        # losses shape: (B, B)
        # (B, 1) - (B, B) -> (B, B)
        # loss[i][j] = margin - pos_scores[i] + scores[i][j]
        losses = F.relu(margin - pos_scores.unsqueeze(1) + scores)
        
        # Mask 掉对角线 (自己不仅仅是正样本，不应该作为负样本贡献 loss，且上面的式子 margin - pos + pos = margin > 0 会导致恒定 loss)
        # 通常 Triplet Loss 不计算 i-i 这一项。
        # 创建一个对角线 mask
        batch_size = user_emb.size(0)
        eye_mask = torch.eye(batch_size, device=user_emb.device).bool()
        
        # 将对角线位置的 loss 置为 0
        losses.masked_fill_(eye_mask, 0)
        
        # 如果有样本级的 mask (例如 label=0 的样本), 它们不应该贡献 active loss
        # 即它们作为 user 时不计算 loss，但它们对应的 item 依然可以作为别人的负样本
        if mask is not None:
            # mask shape (B,) -> (B, 1)
            losses = losses * mask.unsqueeze(1)
            
            # 计算 loss 均值。
            # 分母应该是 (Valid_Users * (Batch_Size - 1))
            num_valid_triplets = mask.sum() * (batch_size - 1)
            return losses.sum() / (num_valid_triplets + 1e-9)
            
        # 如果没有 mask，分母是 B * (B-1)
        return losses.sum() / (batch_size * (batch_size - 1) + 1e-9)

    def infoNCE_loss(self, user_emb, pos_item_emb, temperature=0.1, mask=None):
        # user_emb: 用户特征向量，形状为 (batch_size, 16)
        # pos_item_emb: 正样本物品特征向量，形状为 (batch_size, 16)
        # temperature: 温度参数
        # mask: 可选的mask，用于过滤损失

        batch_size = user_emb.size(0)
        
        # 计算所有 pair 的相似度 (B, B)
        # scores[i][j] 表示 user_i 和 item_j 的相似度
        scores = torch.matmul(user_emb, pos_item_emb.T) / temperature

        # 标签：对于第 i 个用户，正样本即为第 i 个物品
        labels = torch.arange(batch_size, device=user_emb.device, dtype=torch.long)

        # 计算交叉熵损失 (B,)
        # CrossEntropyLoss 包含了 LogSoftmax
        # 它会把 scores[i][target] 当作正类，scores[i][others] 当作负类
        losses = F.cross_entropy(scores, labels, reduction='none')
        
        if mask is not None:
            losses = losses * mask
            return losses.sum() / (mask.sum() + 1e-9)
            
        return losses.mean()
    
        
    def training_step(self, batch, batch_idx):
        user_emb, item_emb = self.forward(batch)

        # 获取mask，将负样本 mask 掉
        mask = batch['label'][:, 0]
        # loss = self.triplet_loss(user_emb, item_emb, mask=mask)
        loss = self.infoNCE_loss(user_emb, item_emb, mask=mask)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        # 记录当前学习率
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
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

    def get_user_embedding(self, batch):
        features, _, _ = self.get_embeddings_from_batch('base_embedding_table', batch, self.user_feature_names)
        return features
    
    def get_item_embedding(self, batch):
        features, _, _ = self.get_embeddings_from_batch('base_embedding_table', batch, self.item_feature_names)
        return features

    def inference_item(self, batch):
        item_vector = self.get_item_embedding(batch)  # 获取物品特征向量
        item_emb = self.item_fc(item_vector)  # 物品特征通过全连接层  bx16
        item_emb = F.normalize(item_emb, p=2, dim=1)
        return item_emb
    
    def inference_user(self, batch):
        user_vector = self.get_user_embedding(batch)  # 获取用户特征向量
        user_emb = self.user_fc(user_vector)  # 用户特征通过全连接层  bx16
        user_emb = F.normalize(user_emb, p=2, dim=1)
        return user_emb
        

