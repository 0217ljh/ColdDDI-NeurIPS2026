"""HDN-DDI hierarchical drug graph + super-node readout (verbatim).

Source: ``Code-Released/baseline/HDN-DDI-NEW/drugbank_test/models.py``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, SAGPooling, LayerNorm

from coldddi.baselines.hdn_ddi.layers import (
    CoAttentionLayer,
    RESCAL,
    IntraGraphAttention,
    InterGraphAttention,
)


def get_node(node_rep, batch, type_, needed_type):
    """Slice node embeddings whose ``y == needed_type`` (e.g. super-node y=2)."""
    emb_dim = node_rep.shape[-1]
    node_rep = node_rep.masked_select((type_ == needed_type).unsqueeze(-1))
    batch = batch.masked_select(type_ == needed_type)
    node_rep = node_rep.reshape(-1, emb_dim)
    return node_rep, batch


class HDN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total,
                 heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(
            zip(heads_out_feat_params, blocks_params)
        ):
            block = HDN_DDI_Block(
                n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim
            )
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            in_features = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        self.initial_conv = GATConv(
            self.in_features, heads_out_feat_params[0], blocks_params[0]
        )

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        h_data.x = self.initial_conv(h_data.x, h_data.edge_index)
        t_data.x = self.initial_conv(t_data.x, t_data.edge_index)

        repr_h = []
        repr_t = []
        for block in self.blocks:
            out = block(h_data, t_data, b_graph)
            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        attentions = self.co_attention(repr_h, repr_t)
        scores = self.KGE(repr_h, repr_t, rels, attentions)
        return scores


class HDN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads)
        self.pool = GATConv(n_heads * head_out_feats, head_out_feats, n_heads)
        self.norm = LayerNorm(n_heads * head_out_feats)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, h_data, t_data, b_graph):
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = F.elu(self.norm(h_rep, h_data.batch))
        t_data.x = F.elu(self.norm(t_rep, t_data.batch))

        h_data.x, h_weight = self.pool(
            h_data.x, h_data.edge_index, return_attention_weights=True
        )
        t_data.x, t_weight = self.pool(
            t_data.x, t_data.edge_index, return_attention_weights=True
        )
        h_global_graph_emb = get_node(h_data.x, h_data.batch, h_data.y, 2)[0]
        t_global_graph_emb = get_node(t_data.x, t_data.batch, t_data.y, 2)[0]

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb, h_weight, t_weight
