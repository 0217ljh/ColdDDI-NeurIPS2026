"""MKG-FENN four-channel KG-GNN + fusion head.

Verbatim from
``Code-Released/baseline/MKG-FENN-NEW/Code and Datasets/code/modeltask1.py``
(the GPU-efficient NEW version), with:

* ``args`` accessed via attribute lookup (so a plain :class:`SimpleNamespace`
  works in our adapter)
* docstrings preserved
* no other behavioural changes — neighbor sampling is still done offline
  via ``precompute_adj()`` and lives in registered buffers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _sample_adj(
    kg: dict,
    drug_name_id: dict,
    neighbor_sample_size: int,
    n_drug: int,
    ghost_ent_id: int,
    ghost_rel_id: int,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    adj_tail = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    for drug_id_str, idx_drug in drug_name_id.items():
        all_neighbors = kg.get(idx_drug, [])
        n_neighbor = len(all_neighbors)
        if n_neighbor == 0:
            adj_tail[idx_drug] = ghost_ent_id
            adj_relation[idx_drug] = ghost_rel_id
            continue
        replace = n_neighbor < neighbor_sample_size
        sample_indices = rng.choice(n_neighbor, neighbor_sample_size, replace=replace)
        adj_tail[idx_drug] = np.array([all_neighbors[k][0] for k in sample_indices])
        adj_relation[idx_drug] = np.array([all_neighbors[k][1] for k in sample_indices])
    return adj_tail, adj_relation


def _sample_adj_no_ghost(
    kg: dict,
    drug_name_id: dict,
    neighbor_sample_size: int,
    n_drug: int,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    adj_tail = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    for drug_id_str, idx_drug in drug_name_id.items():
        all_neighbors = kg[idx_drug]
        n_neighbor = len(all_neighbors)
        replace = n_neighbor < neighbor_sample_size
        sample_indices = rng.choice(n_neighbor, neighbor_sample_size, replace=replace)
        adj_tail[idx_drug] = np.array([all_neighbors[k][0] for k in sample_indices])
        adj_relation[idx_drug] = np.array([all_neighbors[k][1] for k in sample_indices])
    return adj_tail, adj_relation


class _BaseGNN(nn.Module):
    """Shared body of GNN1..4 (per-drug attention over a sampled neighbourhood)."""

    def __init__(self, kg, dict1, drug_name, embedding_num, neighbor_sample_size,
                 n_relations, n_entities, ent_with_ghost: bool, rel_with_ghost: bool):
        super().__init__()
        self.kg = kg
        self.dict1 = dict1
        self.embedding_num = embedding_num
        self.neighbor_sample_size = neighbor_sample_size
        self.n_drug = len(dict1)

        self.drug_embed = nn.Embedding(self.n_drug, embedding_num)
        rel_total = n_relations + (1 if rel_with_ghost else 0)
        ent_total = n_entities + (1 if ent_with_ghost else 0)
        self.rela_embed = nn.Embedding(rel_total, embedding_num)
        self.ent_embed = nn.Embedding(ent_total, embedding_num)
        self.ghost_rel_id = n_relations if rel_with_ghost else None
        self.ghost_ent_id = n_entities if ent_with_ghost else None
        if rel_with_ghost:
            with torch.no_grad():
                self.rela_embed.weight[n_relations].zero_()
        if ent_with_ghost:
            with torch.no_grad():
                self.ent_embed.weight[n_entities].zero_()

        self.W1 = nn.Parameter(torch.randn(self.n_drug, embedding_num, embedding_num))
        self.b1 = nn.Parameter(torch.randn(neighbor_sample_size, embedding_num))
        self.W2 = nn.Parameter(torch.randn(self.n_drug, embedding_num, embedding_num))
        self.b2 = nn.Parameter(torch.randn(neighbor_sample_size, embedding_num))

        self.Linear1 = nn.Sequential(
            nn.Linear(embedding_num * 2, embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_num),
        )
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.register_buffer("drug_name", torch.LongTensor(drug_name))
        self.register_buffer(
            "adj_tail", torch.zeros(self.n_drug, neighbor_sample_size, dtype=torch.long)
        )
        self.register_buffer(
            "adj_relation",
            torch.zeros(self.n_drug, neighbor_sample_size, dtype=torch.long),
        )

    def _gnn_forward(self):
        n_drug = self.n_drug
        emb = self.embedding_num
        drug_embedding = self.drug_embed(self.drug_name)
        rela_embedding = self.rela_embed(self.adj_relation)
        ent_embedding = self.ent_embed(self.adj_tail)

        drug_rel = drug_embedding.view(n_drug, 1, emb) * rela_embedding
        drug_rel_weigh = torch.bmm(drug_rel, self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = torch.bmm(drug_rel_weigh, self.W2) + self.b2
        drug_rel_score = drug_rel_weigh.sum(dim=-1, keepdim=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.view(n_drug, 1, self.neighbor_sample_size).bmm(
            ent_embedding
        )
        drug_e = torch.cat(
            [weighted_ent.view(n_drug, emb), drug_embedding.view(n_drug, emb)], dim=1
        )
        return self.Linear1(drug_e)


class GNN1(_BaseGNN):
    """Drug → entity (e.g. enzymes/targets/transporters) with ghost handling."""

    def __init__(self, kg, dict1, drug_name, embedding_num, neighbor_sample_size,
                 n_relations, n_entities):
        super().__init__(kg, dict1, drug_name, embedding_num, neighbor_sample_size,
                         n_relations, n_entities,
                         ent_with_ghost=True, rel_with_ghost=True)

    @torch.no_grad()
    def precompute_adj(self, rng=None):
        if rng is None:
            rng = np.random
        adj_tail, adj_relation = _sample_adj(
            self.kg, self.dict1, self.neighbor_sample_size, self.n_drug,
            self.ghost_ent_id, self.ghost_rel_id, rng,
        )
        self.adj_tail.copy_(torch.from_numpy(adj_tail))
        self.adj_relation.copy_(torch.from_numpy(adj_relation))

    def forward(self, idx):
        return self._gnn_forward(), idx


class _NoGhostGNN(_BaseGNN):
    def __init__(self, kg, dict1, drug_name, embedding_num, neighbor_sample_size,
                 n_relations, n_entities):
        super().__init__(kg, dict1, drug_name, embedding_num, neighbor_sample_size,
                         n_relations, n_entities,
                         ent_with_ghost=False, rel_with_ghost=False)

    @torch.no_grad()
    def precompute_adj(self, rng=None):
        if rng is None:
            rng = np.random
        adj_tail, adj_relation = _sample_adj_no_ghost(
            self.kg, self.dict1, self.neighbor_sample_size, self.n_drug, rng,
        )
        self.adj_tail.copy_(torch.from_numpy(adj_tail))
        self.adj_relation.copy_(torch.from_numpy(adj_relation))


class GNN2(_NoGhostGNN):
    def forward(self, arguments):
        gnn1_embedding, idx = arguments
        return self._gnn_forward(), gnn1_embedding, idx


class GNN3(_NoGhostGNN):
    def forward(self, arguments):
        gnn2_embedding, gnn1_embedding, idx = arguments
        return self._gnn_forward(), gnn2_embedding, gnn1_embedding, idx


class GNN4(_NoGhostGNN):
    def forward(self, arguments):
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments
        return self._gnn_forward(), gnn3_embedding, gnn2_embedding, gnn1_embedding, idx


class FusionLayer(nn.Module):
    def __init__(self, embedding_num: int, dropout: float, event_num: int = 2):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(embedding_num * 4 * 2, embedding_num * 4),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_num * 4),
            nn.Dropout(dropout),
            nn.Linear(embedding_num * 4, embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_num * 2),
            nn.Dropout(dropout),
            nn.Linear(embedding_num * 2, event_num),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments, mask_channel: str | None = None):
        """Concatenate the 4 GNN channels' per-drug embeddings and run
        the fusion head.

        Parameters
        ----------
        arguments
            5-tuple from upstream GNN4 forward:
            ``(gnn4_emb, gnn3_emb, gnn2_emb, gnn1_emb, idx)``.
        mask_channel
            Optional channel-ablation knob, mirroring upstream
            ``exps/sec5-3/2_indicators/baseline_mask_predictors/mkgfenn_mask.py``:

            * ``"kg"``  — zero the contribution of GNN1 (drug-entity)
              + GNN3 (drug-DDI) for BOTH drugs in each pair before
              fusion. Used by the paper's KPS-KG channel indicator.
            * ``"mol"`` — zero the contribution of GNN2 (drug-Morgan-FP)
              + GNN4 (drug-property) similarly. Used by KPS-mol.
            * ``None`` (default) — base prediction, no masking.

            The mask is **per-pair**: only the rows for the currently
            evaluated ``(drugA_idx, drugB_idx)`` get zeroed; the rest
            of each channel's embedding table is untouched, so the
            inference is purely a fusion-time ablation (no retraining).
        """
        gnn4_embedding, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx, dtype=torch.long, device=gnn1_embedding.device)
        elif idx.device != gnn1_embedding.device:
            idx = idx.to(gnn1_embedding.device)

        drugA_idx = idx[:, 0]
        drugB_idx = idx[:, 1]
        a1 = gnn1_embedding.index_select(0, drugA_idx)
        a2 = gnn2_embedding.index_select(0, drugA_idx)
        a3 = gnn3_embedding.index_select(0, drugA_idx)
        a4 = gnn4_embedding.index_select(0, drugA_idx)
        b1 = gnn1_embedding.index_select(0, drugB_idx)
        b2 = gnn2_embedding.index_select(0, drugB_idx)
        b3 = gnn3_embedding.index_select(0, drugB_idx)
        b4 = gnn4_embedding.index_select(0, drugB_idx)

        if mask_channel == "kg":
            a1 = torch.zeros_like(a1)
            b1 = torch.zeros_like(b1)
            a3 = torch.zeros_like(a3)
            b3 = torch.zeros_like(b3)
        elif mask_channel == "mol":
            a2 = torch.zeros_like(a2)
            b2 = torch.zeros_like(b2)
            a4 = torch.zeros_like(a4)
            b4 = torch.zeros_like(b4)
        elif mask_channel is not None:
            raise ValueError(
                f"mask_channel must be one of {{None, 'kg', 'mol'}}; "
                f"got {mask_channel!r}"
            )

        Embedding = torch.cat([a1, a2, a3, a4, b1, b2, b3, b4], dim=1).float()
        return self.fullConnectionLayer(Embedding)


class MKGFENN(nn.Module):
    """4-channel KG-GNN + fusion head wrapper. Wraps the GNN1..4 modules
    so :meth:`forward` takes a single ``idx`` tensor and returns logits."""

    def __init__(
        self,
        *,
        kgs: dict,
        tail_len: dict,
        relation_len: dict,
        dict1: dict,
        drug_name: list,
        embedding_num: int,
        neighbor_sample_size: int,
        dropout: float,
        event_num: int = 2,
    ) -> None:
        super().__init__()
        common = dict(
            dict1=dict1,
            drug_name=drug_name,
            embedding_num=embedding_num,
            neighbor_sample_size=neighbor_sample_size,
        )
        self.gnn1 = GNN1(
            kg=kgs["dataset1"],
            n_relations=relation_len["dataset1"],
            n_entities=tail_len["dataset1"],
            **common,
        )
        self.gnn2 = GNN2(
            kg=kgs["dataset2"],
            n_relations=relation_len["dataset2"],
            n_entities=tail_len["dataset2"],
            **common,
        )
        self.gnn3 = GNN3(
            kg=kgs["dataset3"],
            n_relations=relation_len["dataset3"],
            n_entities=tail_len["dataset3"],
            **common,
        )
        self.gnn4 = GNN4(
            kg=kgs["dataset4"],
            n_relations=relation_len["dataset4"],
            n_entities=tail_len["dataset4"],
            **common,
        )
        self.fusion = FusionLayer(embedding_num, dropout, event_num=event_num)

    def precompute_adj(self, rng=None):
        self.gnn1.precompute_adj(rng)
        self.gnn2.precompute_adj(rng)
        self.gnn3.precompute_adj(rng)
        self.gnn4.precompute_adj(rng)

    def forward(
        self,
        idx_pairs: torch.Tensor,
        mask_channel: str | None = None,
    ) -> torch.Tensor:
        """Forward pass; pass ``mask_channel="kg"`` or ``"mol"`` to
        run the paper's KPS-KG / KPS-mol channel ablations
        (zero-out at fusion time, no retraining required).  See
        :class:`FusionLayer.forward` for the semantics."""
        out = self.gnn1(idx_pairs)
        out = self.gnn2(out)
        out = self.gnn3(out)
        out = self.gnn4(out)
        return self.fusion(out, mask_channel=mask_channel)
