"""
EmerGNN (pure PyTorch) — no torchdrug / torch_scatter.

Replaces the CUDA kernel `torchdrug.layers.functional.generalized_rspmm` and
`torch_scatter.scatter_add` with portable PyTorch ops:

    * `generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')`
       -> for each edge (h, t, r):  out[t] += relation_input[r] * hiddens[h]
       implemented via `index_select` + `index_add_`.

    * `scatter_add` -> `torch.zeros(...).index_add_(0, index, src)`.

Faithful to Zhang et al. 2023 for architecture: attention-weighted relation
aggregation with L=3 bidirectional message-passing layers; the score head
is a Linear(2*n_dim -> 1) producing a single-logit for binary DDI
(the original code predicts over eval_rel relations; we adapt by switching
the final `Wr` to output 1 for binary ColdDDI).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmerGNN(nn.Module):
    """Flow-based bidirectional message-passing DDI predictor."""

    def __init__(
        self,
        n_ent: int,
        n_base_rel: int,
        n_dim: int = 64,
        length: int = 3,
        feat: str = "M",  # 'M' for Morgan, 'E' for learned embedding
        morgan_features: Optional[np.ndarray] = None,
        morgan_feat_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_ent = n_ent
        self.n_base_rel = n_base_rel
        self.n_dim = n_dim
        self.L = length
        self.feat = feat
        self.all_rel = 2 * n_base_rel + 1  # forward + reverse + self-loop

        if feat == "E":
            self.ent_kg = nn.Embedding(n_ent, n_dim)
            self.Went = None
            self.Wr = nn.Linear(4 * n_dim, 1)  # binary logit
        elif feat == "M":
            if morgan_features is None:
                raise ValueError("feat='M' requires morgan_features (n_ent, 1024) array.")
            assert morgan_features.shape[0] == n_ent
            assert morgan_features.shape[1] == morgan_feat_dim
            # Store as a non-trainable buffer
            self.register_buffer("ent_feat", torch.from_numpy(morgan_features.astype(np.float32)))
            self.Went = nn.Linear(morgan_feat_dim, n_dim)
            self.Wr = nn.Linear(2 * n_dim, 1)  # binary logit
        else:
            raise ValueError(f"Unknown feat={feat!r}; expected 'M' or 'E'.")

        # Per-layer relation embedding tables and linear transforms
        self.rel_kg = nn.ModuleList([nn.Embedding(self.all_rel, n_dim) for _ in range(self.L)])
        self.linear = nn.ModuleList([nn.Linear(n_dim, n_dim) for _ in range(self.L)])
        self.act = nn.ReLU()

        # Attention over relation slots (bottlenecked through a 5-D layer per paper)
        self.relation_linear = nn.ModuleList([nn.Linear(2 * n_dim, 5) for _ in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, self.all_rel) for _ in range(self.L)])

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.data.ndim > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p.data)

    def _entity_embed(self, idx: torch.Tensor) -> torch.Tensor:
        if self.feat == "E":
            return self.ent_kg(idx)
        else:
            return self.Went(self.ent_feat.index_select(0, idx))

    def _propagate(
        self,
        source_idx: torch.Tensor,           # (B,)
        source_embed: torch.Tensor,         # (B, n_dim)
        ht_embed: torch.Tensor,             # (B, 2*n_dim)
        edge_src: torch.Tensor,             # (E,) long, device=same
        edge_dst: torch.Tensor,             # (E,)
        edge_rel: torch.Tensor,             # (E,)
    ) -> torch.Tensor:
        """Run L layers of message passing from `source_idx` outwards.

        Returns: hidden tensor of shape (n_ent, B, n_dim) after L steps.
        """
        B = source_idx.size(0)
        device = source_embed.device

        # Initialize hidden: only source entities have the source embedding, rest are zero
        hiddens = torch.zeros(self.n_ent, B, self.n_dim, device=device)
        batch_arange = torch.arange(B, device=device)
        hiddens[source_idx, batch_arange] = source_embed

        for l in range(self.L):
            # (a) Attention over relation slots, conditioned on (head, tail) query
            rel_w = self.attn_relation[l](F.relu(self.relation_linear[l](ht_embed)))  # (B, all_rel)
            rel_w = torch.sigmoid(rel_w)                                              # (B, all_rel)
            rel_emb = self.rel_kg[l].weight                                           # (all_rel, n_dim)

            # Per-(batch,relation) gated relation embedding: (B, all_rel, n_dim)
            rel_per_batch = rel_w.unsqueeze(-1) * rel_emb.unsqueeze(0)

            # (b) Message passing: out[t, b, :] += rel_per_batch[b, r, :] * hiddens[h, b, :]
            #     for every edge (h, t, r). Processed as one flat gather+index_add.
            #
            # h_feat: (E, B, n_dim) = hiddens[edge_src]
            # r_feat: (E, B, n_dim) = rel_per_batch[:, edge_rel, :]  -> transpose to (E,B,n_dim)
            # msg   : (E, B, n_dim) = h_feat * r_feat
            #
            # Memory note: msg = E * B * n_dim * 4B. For E=50K, B=32, n_dim=64: ~400MB.
            # We chunk over edges if that crosses a threshold.
            hiddens_flat = hiddens  # alias for clarity
            msg = _compute_messages(hiddens_flat, rel_per_batch, edge_src, edge_rel)
            new_hiddens = torch.zeros(self.n_ent, B, self.n_dim, device=device)
            # index_add_ over dim 0 using edge_dst
            new_hiddens.index_add_(0, edge_dst, msg)
            new_hiddens = self.act(self.linear[l](new_hiddens))
            hiddens = new_hiddens
        return hiddens

    def forward(
        self,
        head: torch.Tensor,          # (B,)
        tail: torch.Tensor,          # (B,)
        edge_src: torch.Tensor,      # (E,)
        edge_dst: torch.Tensor,      # (E,)
        edge_rel: torch.Tensor,      # (E,)
    ) -> torch.Tensor:
        """Return logits (B,)."""
        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        # u -> v propagation: source = head, read tail
        hid_uv = self._propagate(head, head_embed, ht_embed, edge_src, edge_dst, edge_rel)
        B = head.size(0)
        b_arange = torch.arange(B, device=head.device)
        tail_hid = hid_uv[tail, b_arange]           # (B, n_dim)

        # v -> u propagation: source = tail, read head
        hid_vu = self._propagate(tail, tail_embed, ht_embed, edge_src, edge_dst, edge_rel)
        head_hid = hid_vu[head, b_arange]           # (B, n_dim)

        if self.feat == "E":
            embed = torch.cat([head_embed, tail_embed, head_hid, tail_hid], dim=-1)
        else:
            embed = torch.cat([head_hid, tail_hid], dim=-1)
        logits = self.Wr(embed).squeeze(-1)
        return logits


def _compute_messages(
    hiddens: torch.Tensor,        # (n_ent, B, n_dim)
    rel_per_batch: torch.Tensor,  # (B, n_rel, n_dim)
    edge_src: torch.Tensor,       # (E,)
    edge_rel: torch.Tensor,       # (E,)
    chunk_size: int = 200_000,
) -> torch.Tensor:
    """Compute per-edge message tensor (E, B, n_dim) = hiddens[src] * rel_per_batch[:, rel, :].

    Chunked along the edge axis to cap peak memory on large KGs.
    """
    E = edge_src.size(0)
    if E <= chunk_size:
        h_feat = hiddens.index_select(0, edge_src)                 # (E, B, n_dim)
        # rel_per_batch is (B, n_rel, n_dim); we want (E, B, n_dim) by taking [:, edge_rel, :]
        # transpose to (n_rel, B, n_dim) then index_select on dim 0
        r_feat = rel_per_batch.transpose(0, 1).index_select(0, edge_rel)  # (E, B, n_dim)
        return h_feat * r_feat
    outs = []
    rel_t = rel_per_batch.transpose(0, 1)  # (n_rel, B, n_dim)
    for start in range(0, E, chunk_size):
        stop = min(start + chunk_size, E)
        h_feat = hiddens.index_select(0, edge_src[start:stop])
        r_feat = rel_t.index_select(0, edge_rel[start:stop])
        outs.append(h_feat * r_feat)
    return torch.cat(outs, dim=0)
