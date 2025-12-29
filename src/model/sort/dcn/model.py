import sys
sys.path.append('/data2/zhy/Movie_Recsys/')


import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn_arch import DCNNet, DCNv2Net, DCNLayer, DCNv2Layer

from src.model.BaseModel.base_model import BaseModel
from src.model.model_utils.lr_schedule import CosinDecayLR
from src.model.model_utils.utils import MLP
from sklearn.metrics import roc_auc_score

class DCNModel(nn.Module):
    def __init__(self, input_dim, cross_num_layers=3, deep_hidden_dims=[32, 32, 1]):
        super(DCNModel, self).__init__()
        # Cross Network部分
        self.cross_net = DCNNet(input_dim=input_dim, num_layers=cross_num_layers)
        dims = [input_dim * 2] + deep_hidden_dims
        # Deep部分
        self.score_fc = MLP(dims=dims)
        # self.score_fc = nn.Linear(input_dim + deep_hidden_dims[-1], 1)

    def forward(self, x):
        cross_f = self.cross_net(x)
        # deep_f = self.deep_net(x)
        # concat_f = torch.cat([cross_f, deep_f], dim=1)
        return F.sigmoid(self.score_fc(torch.cat([x, cross_f], dim=1)))

class DCN(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        

        self.score_fc = DCNModel(input_dim=self.user_input_dim + self.item_input_dim, cross_num_layers=3, deep_hidden_dims=[128, 128, 128, 64, 1])


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        inp_feature = self.get_inp_embedding(x)  # 获取输入特征向量
        return self.score_fc(inp_feature)  # 返回预测分数
    
    def get_inp_embedding(self, batch):
        features, _, _ = self.get_embeddings_from_batch(batch, self.user_feature_names | self.item_feature_names)
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


    # @torch.no_grad()
    # def on_train_epoch_end(self):
    #     if self.current_epoch % 1 == 0:
    #         self.eval()