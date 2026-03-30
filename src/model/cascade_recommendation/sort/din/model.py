import torch
import torch.nn as nn
import torch.nn.functional as F

from ....BaseModel.base_model_sort import BaseModelSort
from ....model_utils.lr_schedule import CosinDecayLR
from ....model_utils.utils import MLP
from sklearn.metrics import roc_auc_score
from .....Logger.logging import Logger

logger = Logger.get_logger("DIN")


class DINModel(nn.Module):
    """
    DIN 的最终打分网络。

    输入是已经拼接好的二维向量：
    - 用户侧表示（包含非序列特征 + 序列聚合特征）
    - 物品侧表示

    最后通过 MLP 输出点击概率。
    """

    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64, 1]

        # MLP 维度链: input_dim -> ... -> 1
        dims = [input_dim] + hidden_dims
        self.network = MLP(dims=dims)

    def forward(self, x):
        # x: [B, input_dim]
        # out: [B, 1]
        return torch.sigmoid(self.network(x))


class DIN(BaseModelSort):
    """
    DIN 排序模型。

    关键约束（按你的需求实现）：
    1) 只有 array_item_feature_pairs 声明的用户序列特征使用 DIN 注意力。
    2) 其他用户序列特征保持默认聚合（array_feature_pooling）。
    3) array_item_feature_pairs 的 key 必须是“用户侧序列特征”。

    注意：
    - 同一个 batch 中，DIN 注意力 query 来自“同域 item 特征”。
    - query 与序列 key 通过 [q, k, q-k, q*k] 输入 attention MLP。
    """

    def __init__(self, config_path):
        super().__init__(config_path)

        # 读取 DIN 配置
        self.din_cfg = self.config.get("din_cfg", {})
        self.att_hidden_dims = list(self.din_cfg.get("att_hidden_dims", [64, 32]))
        self.dnn_hidden_dims = list(self.din_cfg.get("dnn_hidden_dims", [128, 128, 64, 1]))

        # 用户侧特征拆分：序列特征 / 非序列特征
        self.user_array_feature_names = sorted(self.user_feature_names & self.array_feature_names)
        self.user_non_array_feature_names = sorted(self.user_feature_names - self.array_feature_names)

        # 配置中的配对关系：用户序列特征 -> 同域 item 特征
        self.array_item_feature_pairs = dict(self.din_cfg.get("array_item_feature_pairs", {}))

        # 强校验：pair 的 key 必须是“用户序列特征”
        invalid_keys = [
            fname
            for fname in self.array_item_feature_pairs.keys()
            if fname not in self.user_feature_names or fname not in self.array_feature_names
        ]
        if invalid_keys:
            raise ValueError(
                "array_item_feature_pairs 的 key 必须是用户侧序列特征，"
                f"非法 key: {invalid_keys}"
            )

        # 走 DIN 注意力的序列特征（仅限配置里声明的）
        self.user_array_feature_names_din = sorted(
            set(self.user_array_feature_names) & set(self.array_item_feature_pairs.keys())
        )

        # 走默认聚合的序列特征（剩余部分）
        self.user_array_feature_names_default = sorted(
            set(self.user_array_feature_names) - set(self.user_array_feature_names_din)
        )

        # 初始化日志（中文）
        logger.info(f"DIN配置: din_cfg={self.din_cfg}")
        logger.info(
            f"DIN配置: 注意力隐藏层={self.att_hidden_dims}, DNN隐藏层={self.dnn_hidden_dims}"
        )
        logger.info(f"DIN特征: 用户序列特征={self.user_array_feature_names}")
        logger.info(f"DIN特征: 用户非序列特征={self.user_non_array_feature_names}")
        logger.info(f"DIN特征: 序列-同域物品配对={self.array_item_feature_pairs}")
        logger.info(f"DIN特征: 使用DIN注意力的序列特征={self.user_array_feature_names_din}")
        logger.info(f"DIN特征: 使用默认聚合的序列特征={self.user_array_feature_names_default}")

        # 每个 DIN 序列特征对应：
        # 1) query 投影层（同域 item 特征维度 -> 序列特征维度）
        # 2) attention MLP（输入 [q, k, q-k, q*k]，输出逐位置打分）
        self.array_query_projs = nn.ModuleDict()
        self.array_attention_nets = nn.ModuleDict()

        for fname in self.user_array_feature_names_din:
            pair_item_feature = self.array_item_feature_pairs[fname]
            if pair_item_feature not in self.item_feature_names:
                raise ValueError(
                    f"序列特征 '{fname}' 配对的物品特征 '{pair_item_feature}' 不在 item_feature_names 中"
                )

            feat_dim = self._get_feature_embedding_dim(fname)
            pair_item_dim = self._get_feature_embedding_dim(pair_item_feature)

            if feat_dim != pair_item_dim:
                self.array_query_projs[fname] = nn.Linear(pair_item_dim, feat_dim, bias=False)
            else:
                self.array_query_projs[fname] = nn.Identity()
            att_dims = [feat_dim * 4] + self.att_hidden_dims + [1]
            self.array_attention_nets[fname] = MLP(dims=att_dims)

        # 最终 CTR 预测头
        self.score_fc = DINModel(
            input_dim=self.user_input_dim + self.item_input_dim,
            hidden_dims=self.dnn_hidden_dims,
        )

    def _get_feature_embedding_dim(self, feature_name):
        """
        获取某个特征在 base embedding table 下的向量维度。

        - dense 特征: dense_feature_dim
        - sparse/array 特征: embedding_dims（考虑 share 映射）
        """
        if feature_name in self.dense_feature_names:
            return self.dense_feature_dim

        share_map = self.share_emb_table_features_dict["base_embedding_table"]
        emb_fname = self._get_emb_feature_name(share_map, feature_name)
        dim = self.embedding_tables_cfg["base_embedding_table"]["embedding_dims"].get(emb_fname)
        if dim is None:
            raise ValueError(f"特征 '{feature_name}' 映射到 '{emb_fname}' 后未找到 embedding 维度配置")
        return int(dim)

    def _get_single_feature_embedding_from_batch(self, batch, feature_name):
        """
        从 batch 提取单个特征的 embedding，并统一返回 [B, D]。

        如果是序列特征，则先用默认 pooling 压成 [B, D]。
        """
        if feature_name not in batch:
            raise ValueError(f"batch 中缺少特征: '{feature_name}'")

        val = batch[feature_name]
        mask = None

        if feature_name in self.array_feature_names:
            mask = batch.get(f"{feature_name}_mask", None)
            if mask is None:
                raise ValueError(f"序列特征 '{feature_name}' 缺少对应 mask: '{feature_name}_mask'")
            # padding 值 -1 映射到 embedding 的 padding_idx=0
            val = torch.clamp(val, min=0)

        emb = self.get_feature_embedding("base_embedding_table", feature_name, val)
        if feature_name in self.array_feature_names:
            emb = self.array_feature_pooling(emb, mask)

        return emb.reshape(emb.shape[0], -1)

    def din_array_attention_pooling(self, embedding, mask=None, context=None):
        """
        DIN 序列注意力聚合函数（通过 BaseModel.get_embeddings_from_batch 回调）。

        Args:
            embedding: [B, L, D]，序列 key
            mask: [B, L]，有效位掩码（1有效，0padding）
            context:
                - feature_name: 当前正在处理的用户序列特征名
                - query_embedding_map: 每个序列特征对应的“同域 item query embedding”

        Returns:
            pooled: [B, D]
        """
        feature_name = context["feature_name"]
        query_embedding_map = context["query_embedding_map"]
        query_embedding = query_embedding_map[feature_name]

        # [B, pair_item_dim] -> [B, D]
        query = self.array_query_projs[feature_name](query_embedding)
        # [B, D] -> [B, L, D]
        query = query.unsqueeze(1).expand(-1, embedding.size(1), -1)

        # DIN 局部激活单元输入: [q, k, q-k, q*k]
        # [B, L, 4D]
        din_inp = torch.cat([query, embedding, query - embedding, query * embedding], dim=-1)

        # [B, L, 4D] -> [B, L]
        att_logits = self.array_attention_nets[feature_name](din_inp).squeeze(-1)

        # mask + softmax
        if mask is not None:
            valid_mask = (mask > 0).float()
            att_logits = att_logits.masked_fill(valid_mask <= 0, -1e9)
            att_weight = torch.softmax(att_logits, dim=1)
            att_weight = att_weight * valid_mask
            att_weight = att_weight / (att_weight.sum(dim=1, keepdim=True) + 1e-8)
        else:
            att_weight = torch.softmax(att_logits, dim=1)

        # [B, L, D] * [B, L, 1] -> [B, D]
        return (embedding * att_weight.unsqueeze(-1)).sum(dim=1)

    def get_inp_embedding(self, batch):
        """
        构造 DIN 最终输入向量。

        流程：
        1) 编码 item 全特征（用于最终拼接）。
        2) 为“走 DIN 的序列特征”构造同域 query embedding。
        3) 用户非序列特征：默认拼接。
        4) 用户序列特征：
           - 在配置中的用 DIN attention
           - 未配置的用默认 pooling
        5) 拼接 [user_features, item_features]。
        """
        item_features, _, _ = self.get_embeddings_from_batch(
            "base_embedding_table",
            batch,
            self.item_feature_names,
        )

        # 只为“走 DIN 的序列特征”准备 query
        query_embedding_map = {}
        for array_feature in self.user_array_feature_names_din:
            item_feature = self.array_item_feature_pairs[array_feature]
            query_embedding_map[array_feature] = self._get_single_feature_embedding_from_batch(batch, item_feature)

        user_parts = []

        if self.user_non_array_feature_names:
            non_array_features, _, _ = self.get_embeddings_from_batch(
                "base_embedding_table",
                batch,
                set(self.user_non_array_feature_names),
            )
            user_parts.append(non_array_features)

        # 仅配置中声明的序列特征使用 DIN 注意力
        if self.user_array_feature_names_din:
            array_features_din, _, _ = self.get_embeddings_from_batch(
                "base_embedding_table",
                batch,
                set(self.user_array_feature_names_din),
                array_pooling_fn=self.din_array_attention_pooling,
                array_pooling_context={"query_embedding_map": query_embedding_map},
            )
            user_parts.append(array_features_din)

        # 其余序列特征走默认聚合
        if self.user_array_feature_names_default:
            array_features_default, _, _ = self.get_embeddings_from_batch(
                "base_embedding_table",
                batch,
                set(self.user_array_feature_names_default),
            )
            user_parts.append(array_features_default)

        if user_parts:
            user_features = torch.cat(user_parts, dim=1)
        else:
            user_features = item_features.new_zeros(item_features.size(0), 0)

        return torch.cat([user_features, item_features], dim=1)

    def bceLoss(self, preds, labels):
        """二分类 BCE 损失。"""
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction="mean")

    def forward(self, x):
        """前向：batch -> 特征拼接 -> CTR 概率。"""
        inp_feature = self.get_inp_embedding(x)
        return self.score_fc(inp_feature)

    def training_step(self, batch, batch_idx):
        """训练单步：计算 loss，并在可计算时记录 batch AUC。"""
        scores = self.forward(batch)
        labels = batch["label"][:, 0]
        loss = self.bceLoss(scores, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        # 如果当前 batch 标签单一，AUC 不可定义，跳过避免中断。
        try:
            train_auc = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
            self.log("train_auc", train_auc, prog_bar=True, on_step=False, on_epoch=True)
        except ValueError:
            pass

        return loss

    def configure_optimizers(self):
        """优化器和学习率调度，保持与项目其他排序模型一致。"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_hparams.lr, betas=(0.9, 0.999))
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
        """推理阶段复用 forward。"""
        return self.forward(batch)
