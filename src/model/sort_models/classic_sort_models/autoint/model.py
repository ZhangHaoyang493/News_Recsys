import torch
import torch.nn as nn
import torch.nn.functional as F

from ....BaseModel.base_model_sort import BaseModelSort
from ....model_utils.lr_schedule import CosinDecayLR
from ....model_utils.utils import MLP, TransformerBlock
from sklearn.metrics import roc_auc_score


class AutoIntModel(nn.Module):
    """
    AutoInt 主体网络。

    输入是 field 级别的 3D 表示 [B, F, E]：
    - B: batch size
    - F: field 数（一个特征对应一个 field）
    - E: 每个 field 的统一 embedding 维度

    结构上分两条分支：
    1) Attention 分支：多层自注意力建模显式高阶特征交互。
    2) Deep 分支（可选）：MLP 对拼平后的输入做非线性拟合，增强表示能力。
    最后将两条分支拼接后映射到 1 维概率。
    """

    def __init__(
        self,
        feature_num,
        embed_dim,
        att_layer_num=3,
        att_head_num=2,
        att_res=True,
        use_deep=True,
        deep_hidden_dims=None,
    ):
        super().__init__()

        if deep_hidden_dims is None:
            deep_hidden_dims = [128, 64]

        # 是否在注意力输出后加输入残差 x + att(x)
        self.att_res = att_res
        # 是否启用 DNN 分支
        self.use_deep = use_deep

        # 交互层：堆叠多层 TransformerBlock。
        # 每层在 field 维度上做 self-attention，学习 field-field 交互。
        self.att_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=att_head_num,
                    ff_dim=embed_dim * 4,
                )
                for _ in range(att_layer_num)
            ]
        )

        # 注意力分支拼平后的维度：F * E
        att_out_dim = feature_num * embed_dim

        # Deep 分支输出维度，默认 0（未启用时）。
        deep_out_dim = 0
        if self.use_deep:
            # Deep 分支输入同样是拼平后的 [B, F*E]
            deep_dims = [att_out_dim] + deep_hidden_dims
            self.deep_net = MLP(dims=deep_dims)
            deep_out_dim = deep_hidden_dims[-1]

        # 最终输出层：输入为 [att_flat, deep_output] 拼接结果。
        self.final_linear = nn.Linear(att_out_dim + deep_out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [B, F, E]

        Returns:
            pred: [B, 1]，点击概率
        """
        # 保存原始输入供残差与 Deep 分支使用。
        att_input = x
        att_output = x

        # 多层自注意力交互。
        for layer in self.att_layers:
            att_output = layer(att_output)

        # 可选残差，缓解深层注意力训练不稳定。
        if self.att_res:
            att_output = att_output + att_input

        # Attention 分支拼平为 [B, F*E]
        att_flat = att_output.reshape(att_output.size(0), -1)

        # Deep 分支（可选）
        if self.use_deep:
            deep_output = self.deep_net(att_input.reshape(att_input.size(0), -1))
            # 拼接后送入最终线性层。
            final_input = torch.cat([att_flat, deep_output], dim=1)
        else:
            final_input = att_flat

        # 统一输出概率
        return torch.sigmoid(self.final_linear(final_input))


class AutoInt(BaseModelSort):
    """
    AutoInt 排序模型包装类。

    该类负责：
    1) 从 batch 构建 field 级输入（3D Tensor）。
    2) 处理不同特征原始 embedding 维度不一致的问题（投影到统一维度）。
    3) 调用 AutoIntModel 完成训练/推理。
    """

    def __init__(self, config_path):
        super().__init__(config_path)

        # AutoInt 专属配置
        self.autoint_cfg = self.config.get("autoint_cfg", {})

        # 参与交互的字段集合：当前按 user+item 全特征进入 AutoInt。
        self.autoint_feature_names = sorted(self.user_feature_names | self.item_feature_names)

        # Attention 统一维度 E。所有 field 最终都会被投影到这个维度。
        self.att_embed_dim = int(self.autoint_cfg.get("att_embed_dim", 16))

        # 不同特征的 embedding 维度可能不一致（如 16/32 混合），
        # AutoInt 要求每个 field 的维度一致，因此为每个 field 建一个投影层。
        self.field_projections = nn.ModuleDict()
        for fname in self.autoint_feature_names:
            field_dim = self._get_field_input_dim(fname)
            if field_dim == self.att_embed_dim:
                # 维度已一致则跳过线性变换。
                self.field_projections[fname] = nn.Identity()
            else:
                self.field_projections[fname] = nn.Linear(field_dim, self.att_embed_dim, bias=False)

        # 组装 AutoInt 主体
        self.score_fc = AutoIntModel(
            feature_num=len(self.autoint_feature_names),
            embed_dim=self.att_embed_dim,
            att_layer_num=int(self.autoint_cfg.get("att_layer_num", 3)),
            att_head_num=int(self.autoint_cfg.get("att_head_num", 2)),
            att_res=bool(self.autoint_cfg.get("att_res", True)),
            use_deep=bool(self.autoint_cfg.get("use_deep", True)),
            deep_hidden_dims=list(self.autoint_cfg.get("deep_hidden_dims", [128, 64])),
        )

    def _get_field_input_dim(self, feature_name):
        """
        获取单个特征在 base embedding table 下的原始向量维度。

        dense 特征返回 dense_feature_dim；
        sparse/array 特征返回对应 embedding dim（考虑 share 映射后）。
        """
        if feature_name in self.dense_feature_names:
            return self.dense_feature_dim

        share_map = self.share_emb_table_features_dict["base_embedding_table"]
        emb_fname = self._get_emb_feature_name(share_map, feature_name)
        dim = self.embedding_tables_cfg["base_embedding_table"]["embedding_dims"].get(emb_fname)
        if dim is None:
            raise ValueError(
                f"Feature '{feature_name}' mapped to '{emb_fname}' has no embedding size config."
            )
        return int(dim)

    def _build_field_embeddings(self, batch):
        """
        从 batch 构建 AutoInt 的输入张量 [B, F, E]。

        流程：
        1) 逐字段取值并做 embedding lookup。
        2) 如果是 array 特征，先用 mask 做 pooling，得到 [B, D]。
        3) 用 field-specific projection 投到统一维度 E。
        4) 在 dim=1 堆叠成 field 序列，得到 [B, F, E]。
        """
        field_embeddings = []

        for fname in self.autoint_feature_names:
            if fname not in batch:
                raise ValueError(f"Feature '{fname}' not found in batch.")

            val = batch[fname]
            mask = None

            # array 特征需要 mask，并将 padding(-1) 映射为 embedding padding_idx(0)。
            if fname in self.array_feature_names:
                mask = batch.get(f"{fname}_mask", None)
                if mask is None:
                    raise ValueError(f"Array feature '{fname}' requires '{fname}_mask' in batch.")
                val = torch.clamp(val, min=0)

            # sparse: [B, D] 或 [B, L, D]
            # dense : [B, D]
            emb = self.get_feature_embedding("base_embedding_table", fname, val)

            # 对序列特征做 pooling，统一为 [B, D]
            if fname in self.array_feature_names:
                emb = self.array_feature_pooling(emb, mask)

            # 防御式 reshape，确保输入 projection 前是二维 [B, D]
            emb = emb.reshape(emb.shape[0], -1)

            # 每个 field 单独投影到统一维度 E
            emb = self.field_projections[fname](emb)

            # [B, D] -> [B, 1, D]，准备在 field 维拼接
            field_embeddings.append(emb.unsqueeze(1))

        # dim=1 拼接后形状为 [B, F, E]（不是 [B, F*E]）
        return torch.cat(field_embeddings, dim=1)

    def bceLoss(self, preds, labels):
        """二分类 BCE 损失。"""
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction="mean")

    def forward(self, x):
        """前向：batch -> [B, F, E] -> AutoInt -> [B, 1]。"""
        field_embeddings = self._build_field_embeddings(x)
        return self.score_fc(field_embeddings)

    def training_step(self, batch, batch_idx):
        """单步训练：计算 loss，并在可计算时记录 batch AUC。"""
        scores = self.forward(batch)
        labels = batch["label"][:, 0]
        loss = self.bceLoss(scores, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        # 当前 batch 若全是正样本或全是负样本，roc_auc_score 会报错。
        # 这里容错跳过，避免中断训练。
        try:
            train_auc = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
            self.log("train_auc", train_auc, prog_bar=True, on_step=False, on_epoch=True)
        except ValueError:
            pass

        return loss

    def configure_optimizers(self):
        """优化器与学习率调度配置，沿用项目通用 AdamW + CosinDecayLR。"""
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
        """推理阶段直接复用 forward。"""
        return self.forward(batch)
