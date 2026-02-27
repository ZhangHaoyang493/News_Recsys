import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....BaseModel.base_model_sort import BaseModelSort
from ....model_utils.lr_schedule import CosinDecayLR
from ....model_utils.utils import MLP
from sklearn.metrics import roc_auc_score

class WideDeepModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32, 1]):
        super().__init__()
        dims = [input_dim] + hidden_dims
        
        self.wide_network = torch.sum
        self.deep_network = MLP(dims=dims)
        self.bias = nn.Parameter(torch.zeros(1))

    
    def forward(self, wide_x, deep_x):
        wide_out = self.wide_network(wide_x, dim=1, keepdim=True) + self.bias  # 线性部分
        deep_out = self.deep_network(deep_x)
        return F.sigmoid(wide_out + deep_out), wide_out, deep_out
        # return F.sigmoid(wide_out), wide_out, deep_out
    

class WideDeep(BaseModelSort):
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False

        self.wide_and_deep_config = self.config.wide_and_deep_cfg
        self.wide_feature_names = set(self.wide_and_deep_config.wide_feature_names)

        self.score_fc = WideDeepModel(input_dim=self.user_input_dim + self.item_input_dim, hidden_dims=[128, 128, 128, 64, 1])

    def _build_cross_idx_from_batch(self, batch):
        batch['user_click_category_cross'] = batch['user_click_category'] * self.embedding_table_size['category'] + batch['category']
        batch['user_click_subcategory_cross'] = batch['user_click_category'] * self.embedding_table_size['subcategory'] + batch['subcategory']


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        wide_x, deep_x = self.get_inp_embedding(x)  # 获取输入特征向量
        scores, wide_out, deep_out = self.score_fc(wide_x, deep_x)
        self.log('wide_out', torch.mean(wide_out), prog_bar=True, on_epoch=True)
        self.log('deep_out', torch.mean(deep_out), prog_bar=True, on_epoch=True)
        return scores  # 返回预测分数

            
    def get_inp_embedding(self, batch):
        deep_features, _, _ = self.get_embeddings_from_batch('base_embedding_table', batch, self.user_feature_names | self.item_feature_names)
        wide_features, _, _ = self.get_embeddings_from_batch('wide_embedding_table', batch, self.wide_feature_names)

        return wide_features, deep_features
    
    def training_step(self, batch, batch_idx):
        self._build_cross_idx_from_batch(batch)
            
        opt_wide, opt_deep = self.optimizers()
        
        # Zero grads
        opt_wide.zero_grad()
        opt_deep.zero_grad()

        scores = self.forward(batch)
        labels = batch['label'][:, 0]  # 获取是否喜欢的标签
        loss = self.bceLoss(scores, labels)  # 计算二元交叉熵损失

        # 添加 L1 正则化
        l1_reg = 0
        for param in self.embedding_tables['wide_embedding_table'].parameters():
            l1_reg += torch.norm(param, 1)
        
        # 假设 l1_lambda 是正则化系数，通常在 config 中配置
        # 如果 config 中没有定义，可以给一个默认值，例如 1e-4
        l1_lambda = getattr(self.config.wide_and_deep_cfg, 'l1_lambda', 1e-4)
        loss += l1_lambda * l1_reg

        # Manual backward
        self.manual_backward(loss)
        
        # Step optimizers
        opt_wide.step()
        opt_deep.step()

        # Step schedulers
        sch = self.lr_schedulers()
        if sch is not None:
             sch.step()

        train_auc = roc_auc_score(labels.cpu().numpy(), scores.detach().cpu().numpy())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('l1_reg', l1_reg, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_auc', train_auc, prog_bar=True, on_step=False, on_epoch=True)

    
    def configure_optimizers(self):
        # 分离 wide 部分和 deep 部分的参数
        wide_params = []
        deep_params = []
        
        for name, param in self.named_parameters():
            if 'wide_embedding_table' in name:
                wide_params.append(param)
            else:
                deep_params.append(param)

        # Wide 部分通常使用 FTRL 来保证稀疏性。
        # 由于 PyTorch 标准库不包含 FTRL，这里使用 SGD 作为替代，它是 Wide 侧常用的优化器。
        optimizer_wide = torch.optim.SGD(wide_params, lr=self.wide_and_deep_config.lr)
        
        # Deep 部分使用 AdamW
        optimizer_deep = torch.optim.AdamW(deep_params, lr=self.train_hparams.lr, betas=(0.9, 0.999))
        
        lr_scheduler_deep = CosinDecayLR(optimizer_deep, lrs=[self.train_hparams.lr, self.train_hparams.min_lr], milestones=self.train_hparams.lr_milestones)
        
        # 返回多个优化器
        return (
            {
                "optimizer": optimizer_wide,
            },
            {
                "optimizer": optimizer_deep,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_deep,
                    "interval": "step",
                    "frequency": 1,
                },
            },
        )
    
    @torch.no_grad()
    def inference(self, batch):
        self._build_cross_idx_from_batch(batch)
        wide_x, deep_x = self.get_inp_embedding(batch)  # 获取输入特征向量
        scores, wide_out, deep_out = self.score_fc(wide_x, deep_x)
        return scores  # 返回预测分数