import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....BaseModel.base_model_sort import BaseModelSort
from ....model_utils.lr_schedule import CosinDecayLR
from ....model_utils.utils import MLP, TransformerBlock
from sklearn.metrics import roc_auc_score

class AutoIntModel(nn.Module):
    def __init__(self, feature_num, embed_dim, att_layer_num=3, att_head_num=2, att_res=True, deep_hidden_dims=[32, 32]):
        super().__init__()
        
        # 1. Interacting Layer (Multi-Head Self-Attention)
        self.att_layers = nn.ModuleList()
        for _ in range(att_layer_num):
            # 这里的 TransformerBlock 需要支持传入 input_dim=embed_dim
            # 假设 model_utils.utils.TransformerBlock 接口是 standard transformer block
            self.att_layers.append(
                TransformerBlock(embed_dim=embed_dim, num_heads=att_head_num, ff_dim=embed_dim * 4)
            )
            
        # 2. Deep Layer (MLP)
        # Deep部分的输入通常是所有 Embedding 展平: input_dim = feature_num * embed_dim
        input_dim = feature_num * embed_dim
        deep_dims = [input_dim] + deep_hidden_dims
        self.deep_net = MLP(dims=deep_dims)

        # 3. Output Layer
        # AutoInt 输出 = Deep输出 + Attention输出 (ResNet式拼接或加和，通常 AutoInt 最后一层是 Linear 聚合)
        # 这里我们把 Attention 输出展平后与 Deep 输出拼接，再过一个 Linear
        # Attention Output Flatten Dim: feature_num * embed_dim
        # Deep Output Dim: deep_hidden_dims[-1]
        final_dim = (feature_num * embed_dim) + deep_hidden_dims[-1]
        self.final_linear = nn.Linear(final_dim, 1)

    def forward(self, x):
        # x shape: [Batch, Feature_Num, Embed_Dim]
        # 注意: AutoInt 需要 3D Tensor 输入来做 Attention
        
        # --- Attention Part ---
        att_output = x
        for layer in self.att_layers:
            att_output = layer(att_output)
        
        # Flatten Attention Output: [B, F*E]
        att_flat = att_output.reshape(att_output.size(0), -1)
        
        # --- Deep Part ---
        # Deep Input Flatten: [B, F*E]
        dnn_input = x.reshape(x.size(0), -1)
        deep_output = self.deep_net(dnn_input)
        
        # --- Combination ---
        stack = torch.cat([att_flat, deep_output], dim=1)
        return torch.sigmoid(self.final_linear(stack))


class AutoInt(BaseModelSort):
    def __init__(self, config_path):
        super().__init__(config_path)

        # 定义Deep模型的网络结构
        self.score_fc = DeepModel(input_dim=self.user_input_dim + self.item_input_dim, hidden_dims=[128, 128, 128, 64, 1])


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        inp_feature = self.get_inp_embedding(x)  # 获取输入特征向量
        return self.score_fc(inp_feature)  # 返回预测分数

    
    def get_inp_embedding(self, batch):
        features, _, _ = self.get_embeddings_from_batch('base_embedding_table', batch, self.user_feature_names | self.item_feature_names)
        return features
    
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
        inp_feature = self.get_inp_embedding(batch)  # 获取输入特征向量
        return self.score_fc(inp_feature)  # 返回预测分数

