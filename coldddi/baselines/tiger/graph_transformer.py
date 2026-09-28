"""TIGER GraphTransformer encoder (verbatim from the upstream repo,
import paths and the inner ``BASEDIR/sys.path`` hack stripped).

Source: ``Code-Released/baseline/TIGER/model/GraphTransformer.py``.
"""

from __future__ import annotations

import torch
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing


class GraphTransformerEncode(torch.nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, rel_encoder, spatial_encoder, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim),
        )

        self.multiHeadAttention = MultiheadAttention(
            dim_model=self.in_dim,
            num_heads=self.num_heads,
            rel_encoder=rel_encoder,
            spatial_encoder=spatial_encoder,
        )

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, feature, sp_edge_index, sp_value, edge_rel):
        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multiHeadAttention(
            x_norm, sp_edge_index, sp_value, edge_rel
        )
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + feature

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, attn_weight


class SpatialEncoding(torch.nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU(),
        )

    def forward(self, lap):
        lap_ = torch.unsqueeze(lap, dim=-1)
        return self.fnn(lap_)


class MultiheadAttention(MessagePassing):
    def __init__(self, dim_model, num_heads, rel_encoder, spatial_encoder, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.d_model = dim_model
        self.num_heads = num_heads

        self.rel_embedding = rel_encoder
        self.rel_encoding = Sequential(Linear(dim_model, 1), ReLU())
        self.spatial_encoding = spatial_encoder

        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads

        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)
        self.dense = Linear(dim_model, dim_model)

    def denominator(self, qs, ks):
        all_ones = torch.ones([ks.shape[0]]).to(qs.device)
        ks_sum = torch.einsum("nhm,n->hm", ks, all_ones)
        return torch.einsum("nhm,hm->nh", qs, ks_sum)

    def forward(self, x, sp_edge_index, sp_value, edge_rel):
        rel_embedding = self.rel_embedding(edge_rel)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x).view(x.shape[0], self.num_heads, self.depth)

        row, col = sp_edge_index
        query_end, key_start = q[col], k[row]
        query_end = query_end + rel_embedding
        key_start = key_start + rel_embedding

        query_end = query_end.view(sp_edge_index.shape[1], self.num_heads, self.depth)
        key_start = key_start.view(sp_edge_index.shape[1], self.num_heads, self.depth)

        edge_attn_num = torch.einsum("ehd,ehd->eh", query_end, key_start)
        data_normalizer = 1.0 / torch.sqrt(
            torch.sqrt(torch.tensor(edge_attn_num.shape[-1], dtype=torch.float32))
        )
        edge_attn_num = edge_attn_num * data_normalizer
        edge_attn_bias = self.spatial_encoding(sp_value)
        edge_attn_num = edge_attn_num + edge_attn_bias

        attn_normalizer = self.denominator(
            q.view(x.shape[0], self.num_heads, self.depth),
            k.view(x.shape[0], self.num_heads, self.depth),
        )
        edge_attn_dem = attn_normalizer[col]
        attention_weight = edge_attn_num / edge_attn_dem

        outputs = []
        for i in range(self.num_heads):
            output_per_head = self.propagate(
                edge_index=sp_edge_index,
                x=v[:, i, :],
                edge_weight=attention_weight[:, i],
                size=None,
            )
            outputs.append(output_per_head)

        out = torch.cat(outputs, dim=-1)
        return self.dense(out), attention_weight


class GraphTransformer(torch.nn.Module):
    def __init__(self, layer_num=3, embedding_dim=64, num_heads=4, num_rel=10, dropout=0.2, type="graph"):
        super().__init__()
        self.type = type
        self.rel_encoder = torch.nn.Embedding(num_rel, embedding_dim)
        self.spatial_encoder = SpatialEncoding(embedding_dim)

        self.encoder = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.encoder.append(
                GraphTransformerEncode(
                    num_heads=num_heads,
                    in_dim=embedding_dim,
                    dim_forward=embedding_dim * 2,
                    rel_encoder=self.rel_encoder,
                    spatial_encoder=self.spatial_encoder,
                    dropout=dropout,
                )
            )

    def forward(self, feature, data):
        x = feature
        attn_layer = []
        for graphEncoder in self.encoder:
            x, attn = graphEncoder(x, data.sp_edge_index, data.sp_value, data.sp_edge_rel)
            attn_layer.append(attn)

        if self.type == "graph":
            sub_representation = []
            for index, _ in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = global_mean_pool(x, batch=data.batch)
        else:
            sub_representation = []
            for index, _ in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = x[data.id.nonzero().flatten()]

        return representation, sub_representation, attn_layer
