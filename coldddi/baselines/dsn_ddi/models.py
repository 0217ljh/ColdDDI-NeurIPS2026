"""DSN-DDI molecular dual-view GAT.

Source: ``Code-Released/baseline/DSN-DDI/drugbank_test/models.py``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from torch_geometric.nn import (
    GATConv,
    SAGPooling,
    LayerNorm,
    global_add_pool,
)

from coldddi.baselines.dsn_ddi.layers import (
    CoAttentionLayer,
    RESCAL,
    IntraGraphAttention,
    InterGraphAttention,
)


class MVN_DDI(nn.Module):
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
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(
            zip(heads_out_feat_params, blocks_params)
        ):
            block = MVN_DDI_Block(
                n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim
            )
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)
            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)
            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        attentions = self.co_attention(repr_h, repr_t)
        scores = self.KGE(repr_h, repr_t, rels, attentions)
        return scores


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, h_data, t_data, b_graph):
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep

        h_att_x, _, _, h_att_batch, _, _ = self.readout(
            h_data.x, h_data.edge_index, batch=h_data.batch
        )
        t_att_x, _, _, t_att_batch, _, _ = self.readout(
            t_data.x, t_data.edge_index, batch=t_data.batch
        )
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb


class BipartiteData:
    """Marker for (x_s, x_t, edge_index) graphs used by InterGraphAttention.

    :func:`make_bipartite_data` creates the PyG Data with separate node offsets.
    """


def make_bipartite_data(x_s, x_t, edge_index):
    """Build a bipartite ``Data`` whose ``edge_index`` increments use
    independent offsets for ``x_s`` and ``x_t`` when batched."""
    from torch_geometric.data import Data

    class _BData(Data):
        def __inc__(self, key, value, *args, **kwargs):
            if key == "edge_index":
                return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
            return super().__inc__(key, value, *args, **kwargs)

    return _BData(edge_index=edge_index, x_s=x_s, x_t=x_t)
