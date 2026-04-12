import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from ....BaseModel.base_model_sort import BaseModelSort
from ....model_utils.lr_schedule import CosinDecayLR
from ....model_utils.utils import MLP


class DeepFMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32, 1]

        self.deep_network = MLP(dims=[input_dim] + hidden_dims)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, one_order_features, two_order_features, deep_features):
        first_order = torch.sum(one_order_features, dim=1, keepdim=True)  # [B, 1]
        second_order = 0.5 * torch.sum(
            torch.pow(torch.sum(two_order_features, dim=1), 2)
            - torch.sum(torch.pow(two_order_features, 2), dim=1),
            dim=-1,
            keepdim=True,
        )  # [B, 1]

        deep_out = self.deep_network(deep_features)
        return torch.sigmoid(first_order + second_order + deep_out + self.bias)


class DeepFM(BaseModelSort):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.score_fc = DeepFMModel(
            input_dim=self.user_input_dim + self.item_input_dim,
            hidden_dims=[128, 128, 128, 64, 1],
        )

        fm_cfg = self.config.get("fm_cfg", None)
        if fm_cfg is None:
            raise ValueError("DeepFM config requires `fm_cfg`.")

        self.fm_features = set(fm_cfg.get("fm_feature", []))
        self.fm_dim = int(fm_cfg.get("fm_dim", 0))
        if not self.fm_features:
            raise ValueError("DeepFM config `fm_cfg.fm_feature` must not be empty.")
        if self.fm_dim <= 0:
            raise ValueError("DeepFM config `fm_cfg.fm_dim` must be a positive integer.")

    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction="mean")

    def forward(self, x):
        one_order_features, two_order_features, deep_features = self.get_inp_embedding(x)
        return self.score_fc(one_order_features, two_order_features, deep_features)

    def get_inp_embedding(self, batch):
        one_order_features, _, _ = self.get_embeddings_from_batch(
            "one_order_embedding_table",
            batch,
            self.fm_features,
        )
        two_order_features, _, _ = self.get_embeddings_from_batch(
            "two_order_embedding_table",
            batch,
            self.fm_features,
        )
        deep_features, _, _ = self.get_embeddings_from_batch(
            "base_embedding_table",
            batch,
            self.user_feature_names | self.item_feature_names,
        )

        batch_size, _ = two_order_features.shape
        two_order_features = two_order_features.view(batch_size, -1, self.fm_dim)

        return one_order_features, two_order_features, deep_features

    def training_step(self, batch, batch_idx):
        scores = self.forward(batch)
        labels = batch["label"][:, 0]
        loss = self.bceLoss(scores, labels)

        l1_reg = 0
        l1_lambda = getattr(self.config, "l1_lambda", 1e-5)
        for name, param in self.named_parameters():
            if "one_order_embedding_table" in name:
                l1_reg += torch.norm(param, 1)
        loss += l1_lambda * l1_reg

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("l1_reg", l1_reg, prog_bar=False, on_epoch=True, on_step=False)

        try:
            train_auc = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
            self.log("train_auc", train_auc, prog_bar=True, on_step=False, on_epoch=True)
        except ValueError:
            pass

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_hparams.lr,
            betas=(0.9, 0.999),
        )
        lr_scheduler = CosinDecayLR(
            optimizer,
            lrs=[self.train_hparams.lr, self.train_hparams.min_lr],
            milestones=self.train_hparams.lr_milestones,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def inference(self, batch):
        one_order_features, two_order_features, deep_features = self.get_inp_embedding(batch)
        return self.score_fc(one_order_features, two_order_features, deep_features)
