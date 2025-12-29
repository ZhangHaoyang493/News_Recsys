import sys
sys.path.append('/data2/zhy/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np

from DataReader.data_reader import DataReader
from tqdm import tqdm
from model_utils.lr_schedule import CosinDecayLR

# BaseModel继承于LightningModule
class DSSM(BaseModel):
    def __init__(self, config_path, dataloaders={}, hparams={}):
        super(DSSM, self).__init__(config_path)

        # 保存超参数
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams
        
        # 定义DSSM的网络结构
        # 假设我们有两个全连接层，分别用于用户和物品的特征处理
        self.user_fc = nn.Sequential(
            nn.Linear(self.user_input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16)
        )
        
        self.item_fc = nn.Sequential(
            nn.Linear(self.item_input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16)
        )
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)

       

    def forward(self, x):
        user_vector = self.get_user_embedding(x)  # 获取用户特征向量
        item_vector = self.get_item_embedding(x)  # 获取物品特征向量
        
        user_emb = self.user_fc(user_vector)  # 用户特征通过全连接层  bx16
        item_emb = self.item_fc(item_vector)  # 物品特征通过全连接层  bx16

        # 构造一个负采样的item_emb，第i个batch的负样本是随机选取的其他item_emb
        batch_size = item_emb.size(0)
        neg_item_emb = []
        for i in range(self.hparams_['negative_sample_rate']):
            # in-batch负采样 随机打乱item_emb的顺序，作为负样本
            neg_indices = torch.randperm(batch_size)
            neg_item_emb.append(item_emb[neg_indices]) # item_emb[neg_indices]: bx16
        # 将所有负样本拼接在一起，形成一个大的负样本集合
        neg_item_emb = torch.stack(neg_item_emb, dim=1)  # 变为 bxneg_numx16

        # 归一化
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        neg_item_emb = F.normalize(neg_item_emb, p=2, dim=-1)

        return user_emb, item_emb, neg_item_emb

    def triplet_loss(self, user_emb, pos_item_emb, neg_item_emb, margin=1.0, mask=None):
        # user_emb: 用户特征向量，形状为 (batch_size, 16)
        # pos_item_emb: 正样本物品特征向量，形状为 (batch_size, 16)
        # neg_item_emb: 负样本物品特征向量，形状为 (batch_size, neg_num, 16)
        # margin: 三元组损失的边界值
        # mask: 可选的mask，用于过滤损失
        neg_sample_num = neg_item_emb.size(1)  # 获取负样本数量
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        # neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        neg_scores = torch.bmm(user_emb.unsqueeze(1), neg_item_emb.permute(0, 2, 1)).squeeze(1)  # bxneg_num
        neg_scores = torch.sum(neg_scores, dim=1).unsqueeze(1)  # bx1
        pos_scores = pos_scores * neg_sample_num  # 将正样本的分数乘以负样本数量，保持维度一致
        losses = F.relu(margin - pos_scores + neg_scores)
        if mask is not None:
            losses = losses * mask
        return losses.mean()

    def infoNCE_loss(self, user_emb, pos_item_emb, neg_item_emb, temperature=0.1, mask=None):
        # user_emb: 用户特征向量，形状为 (batch_size, 16)
        # pos_item_emb: 正样本物品特征向量，形状为 (batch_size, 16)
        # neg_item_emb: 负样本物品特征向量，形状为 (batch_size, neg_num, 16)
        # temperature: 温度参数
        # mask: 可选的mask，用于过滤损失
        batch_size = user_emb.size(0)
        neg_sample_num = neg_item_emb.size(1)

        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1) / temperature  # bx1
        neg_scores = torch.bmm(user_emb.unsqueeze(1), neg_item_emb.permute(0, 2, 1)).squeeze(1) / temperature  # bxneg_num

        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # bx(1+neg_num)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_emb.device)  # 正样本的索引为0

        losses = F.cross_entropy(logits, labels, reduction='none')
        if mask is not None:
            losses = losses * mask
        return losses.mean()
    
        


    def training_step(self, batch, batch_idx):
        user_emb, item_emb, neg_item_emb = self.forward(batch)

        # 获取mask，将score<4的样本mask掉
        mask = batch['label'][:, 1]
        # loss = self.triplet_loss(user_emb, item_emb, neg_item_emb, mask=mask)
        loss = self.infoNCE_loss(user_emb, item_emb, neg_item_emb, mask=mask)
        
        self.log('train_loss', loss)
        # 记录当前学习率
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss


    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams_['lr'], betas=(0.9, 0.999))
        lr_scheduler = CosinDecayLR(optimizer, lrs=[self.hparams_['lr'], self.hparams_['min_lr']], milestones=self.hparams_['lr_milestones'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',  # 每个训练步骤调用一次
                'frequency': 1
            }

        }

    def get_user_embedding(self, batch):
        user_embeddings = []
        for feature_name in self.user_feature_names:
            emb = self.get_features_embedding(feature_name, batch[feature_name])
            if feature_name in self.sparse_feature_names:
                user_embeddings.append(emb)
            elif feature_name in self.dense_feature_names:
                user_embeddings.append(emb)
            elif feature_name in self.array_feature_names:
                mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                user_embeddings.append(emb)
        user_feature_vector = torch.cat(user_embeddings, dim=1)  # 在特征维度上拼接
        return user_feature_vector
    
    def get_item_embedding(self, batch):
        item_embeddings = []
        for feature_name in self.item_feature_names:
            emb = self.get_features_embedding(feature_name, batch[feature_name])
            if feature_name in self.sparse_feature_names:
                item_embeddings.append(emb)
            elif feature_name in self.dense_feature_names:
                item_embeddings.append(emb)
            elif feature_name in self.array_feature_names:
                mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                item_embeddings.append(emb)
        item_feature_vector = torch.cat(item_embeddings, dim=1)  # 在特征维度上拼接
        return item_feature_vector
    
    @torch.no_grad()
    def hit_rate(self, k=10):
        hits_num = 0
        all_nums = 0
        for batch in tqdm(self.val_dataloader_, desc="Evaluating Hit Rate", ncols=100):
            batch_size = batch['user_id'].size(0)
            if batch_size != 1:
                raise ValueError("Hit rate evaluation only supports batch_size=1 for accurate user history filtering.")

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            user_vector = self.get_user_embedding(batch)  # 获取用户特征向量
            user_emb = self.user_fc(user_vector)  # 用户特征通过全连接层  bx16
            user_emb = F.normalize(user_emb, p=2, dim=1)
            user_emb = user_emb.cpu().numpy()

            

            # 计算命中率
            # targets = batch['movie_id'].cpu().numpy()  # 假设item_id是物品的真实ID
            user_ids = batch['user_id'].cpu().numpy() # 获取用户ID，用于消重
            # for i in range(len(targets)):
            user_true_id = self.emb_idx_2_val_dict['user_id'][str(user_ids[0])]
            user_history = set(self.user_history.get(str(user_true_id), []).keys())


            D, I = self.index.search(user_emb, k + len(user_history))  # 搜索top-k的物品，但是这里需要考虑用户历史交互过的物品，所以多搜索len(user_history)个

            # 将索引映射到item id的hash值
            I = [[self.idx_item_emb_dic[idx] for idx in user_indices] for user_indices in I][0]
            # 过滤掉用户历史交互过的物品
            filtered_I = []
            for item_id in I:
                item_true_id = self.emb_idx_2_val_dict['movie_id'][str(item_id)]
                if item_true_id not in user_history:
                    filtered_I.append(item_id)
                if len(filtered_I) >= k:
                    break
            target_item_id = batch['movie_id'].cpu().numpy()[0]
            if target_item_id in filtered_I:
                hits_num += 1
            all_nums += 1

        hit_rate = (hits_num / all_nums) if all_nums > 0 else 0
        self.log(f'Hit_Rate_{k}', hit_rate)
        print(f"Hit Rate@{k}: {hit_rate}")

    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            all_item_embeddings = []
            self.idx_item_emb_dic = {}
            idx = 0
            for batch in tqdm(self.movies_dataloader, desc="Building Item Embeddings", ncols=100):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                item_vector = self.get_item_embedding(batch)  # 获取物品特征向量
                item_emb = self.item_fc(item_vector)  # 物品特征通过全连接层  bx16
                item_emb = F.normalize(item_emb, p=2, dim=1)
                item_emb = item_emb.cpu().numpy()
                all_item_embeddings.append(item_emb)
                for i, item_id in enumerate(batch['movie_id'].cpu().numpy()):
                    self.idx_item_emb_dic[idx] = item_id  # 记录item_id的hash值对应的embedding在all_item_embeddings中的索引
                    idx += 1

            self.all_item_embeddings = np.concatenate(all_item_embeddings, axis=0)
            self.index = faiss.IndexFlatIP(self.all_item_embeddings.shape[1])  # 内积相似度
            self.index.add(self.all_item_embeddings)  # 添加所有物品向量到索引

            # 计算验证集的命中率
            self.hit_rate(k=10)