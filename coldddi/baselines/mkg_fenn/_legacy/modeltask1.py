"""
MKG-FENN — refactored for GPU efficiency (NEW version, 1900-drug scalable).

Differences vs the original `MKG-FENN/Code and Datasets/code/modeltask1.py`:

  1. **`arrge()` is no longer called inside `forward()`.**
     The original code resampled all `n_drug` neighbors via Python np.random.choice
     in every forward call — at 1900 drugs × 4 GNNs × ~5,000 batches/epoch, this
     became the dominant cost (~38M Python iterations/epoch).

     The new version:
       - exposes `precompute_adj()` on each GNN module, called by the train loop
         once per epoch (or once at init for fully deterministic neighbor sets);
       - stores the resulting `adj_tail` / `adj_relation` as registered buffers,
         so they live on the same device as the model and are visible in forward.

     Semantic change: neighbor sampling cadence changes from per-batch to per-epoch.
     This is a common simplification in KG-GNN training; the loss/gradient
     pipeline is otherwise identical to the original.

  2. **All inputs to forward are pre-positioned on GPU** via `register_buffer`.
     The original `torch.LongTensor(...)` calls inside forward created CPU tensors
     every batch; here `drug_name`, `adj_tail`, `adj_relation` are buffers and
     ride the model to the device on `model.to(device)`.

  3. **Architecture, parameter shapes, init, output shape are all unchanged.**
     - W1, W2 still per-drug `(n_drug, emb, emb)` matrices.
     - Output of each GNN is still `(n_drug, embedding_num)` — the FusionLayer
       gathers the batch's drug pairs from this full tensor (unchanged).
     - Ghost node handling for empty-neighborhood drugs is preserved.

Usage in train script:
    gnn1 = GNN1(...).to(device)
    gnn1.precompute_adj()            # call once per epoch, before forward
    out, idx = gnn1(idx_batch)       # forward is now pure GPU bmm
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Helper: deterministic, GPU-stationary adjacency precomputation
# ──────────────────────────────────────────────────────────────────────────────
def _sample_adj(
    kg: dict,
    drug_name_id: dict,
    neighbor_sample_size: int,
    n_drug: int,
    ghost_ent_id: int,
    ghost_rel_id: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicates the per-drug neighbor sampling of the original `arrge()`.

    For each drug index i in `drug_name_id`:
      - if drug i has no neighbors in kg, fill row i with ghost_ent_id / ghost_rel_id
      - else np.random.choice(n_neighbor, neighbor_sample_size, replace=as_needed)

    Returns adj_tail, adj_relation as int64 arrays of shape (n_drug, neighbor_sample_size).
    """
    adj_tail = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    for i in drug_name_id:
        idx_drug = drug_name_id[i]
        all_neighbors = kg.get(idx_drug, [])
        n_neighbor = len(all_neighbors)
        if n_neighbor == 0:
            adj_tail[idx_drug] = ghost_ent_id
            adj_relation[idx_drug] = ghost_rel_id
            continue
        replace = (n_neighbor < neighbor_sample_size)
        sample_indices = rng.choice(n_neighbor, neighbor_sample_size, replace=replace)
        # all_neighbors[k] is a (entity_id, relation_id) tuple
        adj_tail[idx_drug] = np.array([all_neighbors[k][0] for k in sample_indices])
        adj_relation[idx_drug] = np.array([all_neighbors[k][1] for k in sample_indices])
    return adj_tail, adj_relation


def _sample_adj_no_ghost(
    kg: dict,
    drug_name_id: dict,
    neighbor_sample_size: int,
    n_drug: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Variant for GNN2/GNN4 (original code did not ghost-handle empty neighborhoods)."""
    adj_tail = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros((n_drug, neighbor_sample_size), dtype=np.int64)
    for i in drug_name_id:
        idx_drug = drug_name_id[i]
        all_neighbors = kg[idx_drug]
        n_neighbor = len(all_neighbors)
        replace = (n_neighbor < neighbor_sample_size)
        sample_indices = rng.choice(n_neighbor, neighbor_sample_size, replace=replace)
        adj_tail[idx_drug] = np.array([all_neighbors[k][0] for k in sample_indices])
        adj_relation[idx_drug] = np.array([all_neighbors[k][1] for k in sample_indices])
    return adj_tail, adj_relation


# ──────────────────────────────────────────────────────────────────────────────
# GNN1 — drug-target / drug-substructure / drug-DDI / drug-property channel #1
# ──────────────────────────────────────────────────────────────────────────────
class GNN1(nn.Module):
    """Identical architecture to the original GNN1; only the forward pipeline
    is rewired so neighbor sampling and tensor-creation happen offline."""

    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super().__init__(**kwargs)
        self.kg = dataset["dataset1"]
        self.dict1 = dict1
        self.drug_name_list = drug_name      # list of drug indices used by original
        self.args = args
        self.n_drug = len(dict1)
        self.neighbor_sample_size = args.neighbor_sample_size

        # Embeddings (identical to original)
        self.drug_embed = nn.Embedding(num_embeddings=self.n_drug,
                                        embedding_dim=args.embedding_num)
        self.ghost_ent_id = tail_len["dataset1"]
        self.ghost_rel_id = relation_len["dataset1"]
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset1"] + 1,
                                        embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset1"] + 1,
                                       embedding_dim=args.embedding_num)
        with torch.no_grad():
            self.ent_embed.weight[self.ghost_ent_id].zero_()
            self.rela_embed.weight[self.ghost_rel_id].zero_()

        # Per-drug attention parameters (unchanged shapes)
        self.W1 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b1 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))
        self.W2 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b2 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))

        self.Linear1 = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num),
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

        # Pre-positioned buffers (resampled by precompute_adj, moved with model.to(device))
        self.register_buffer("drug_name", torch.LongTensor(drug_name))
        self.register_buffer(
            "adj_tail",
            torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long),
        )
        self.register_buffer(
            "adj_relation",
            torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long),
        )

    @torch.no_grad()
    def precompute_adj(self, rng: np.random.RandomState | None = None) -> None:
        """Resample per-drug neighbors.

        If `rng` is None, falls back to numpy's GLOBAL RandomState
        (which is seeded by the train script's `setup_seed`),
        NOT a fresh OS-entropy state — this preserves run-level reproducibility.
        """
        if rng is None:
            rng = np.random
        adj_tail, adj_relation = _sample_adj(
            self.kg, self.dict1, self.neighbor_sample_size, self.n_drug,
            self.ghost_ent_id, self.ghost_rel_id, rng,
        )
        # `copy_` handles cross-device transfer itself; skip the redundant .to()
        self.adj_tail.copy_(torch.from_numpy(adj_tail), non_blocking=True)
        self.adj_relation.copy_(torch.from_numpy(adj_relation), non_blocking=True)

    def forward(self, idx):
        n_drug = self.n_drug
        emb = self.args.embedding_num

        drug_embedding = self.drug_embed(self.drug_name)         # (n_drug, emb)
        rela_embedding = self.rela_embed(self.adj_relation)      # (n_drug, k, emb)
        ent_embedding = self.ent_embed(self.adj_tail)            # (n_drug, k, emb)

        drug_rel = drug_embedding.view(n_drug, 1, emb) * rela_embedding   # (n_drug, k, emb)
        drug_rel_weigh = torch.bmm(drug_rel, self.W1) + self.b1            # (n_drug, k, emb)
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = torch.bmm(drug_rel_weigh, self.W2) + self.b2      # (n_drug, k, emb)
        drug_rel_score = drug_rel_weigh.sum(dim=-1, keepdim=True)          # (n_drug, k, 1)
        drug_rel_score = self.soft(drug_rel_score)                          # softmax over k
        weighted_ent = drug_rel_score.view(n_drug, 1, self.neighbor_sample_size).bmm(ent_embedding)
        # weighted_ent: (n_drug, 1, emb)
        drug_e = torch.cat(
            [weighted_ent.view(n_drug, emb), drug_embedding.view(n_drug, emb)], dim=1
        )                                                                   # (n_drug, 2*emb)
        drug_f = self.Linear1(drug_e)                                       # (n_drug, emb)
        return drug_f, idx


# ──────────────────────────────────────────────────────────────────────────────
# GNN2
# ──────────────────────────────────────────────────────────────────────────────
class GNN2(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super().__init__(**kwargs)
        self.kg = dataset["dataset2"]
        self.dict1 = dict1
        self.drug_name_list = drug_name
        self.args = args
        self.n_drug = len(dict1)
        self.neighbor_sample_size = args.neighbor_sample_size

        self.drug_embed = nn.Embedding(num_embeddings=self.n_drug, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset2"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset2"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b1 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))
        self.W2 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b2 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))

        self.Linear1 = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num),
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
        self.register_buffer("adj_tail", torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long))
        self.register_buffer("adj_relation", torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long))

    @torch.no_grad()
    def precompute_adj(self, rng: np.random.RandomState | None = None) -> None:
        """Resample per-drug neighbors. If rng is None, use numpy global state
        (seeded by setup_seed) — preserves run-level reproducibility."""
        if rng is None:
            rng = np.random
        adj_tail, adj_relation = _sample_adj_no_ghost(
            self.kg, self.dict1, self.neighbor_sample_size, self.n_drug, rng,
        )
        self.adj_tail.copy_(torch.from_numpy(adj_tail), non_blocking=True)
        self.adj_relation.copy_(torch.from_numpy(adj_relation), non_blocking=True)

    def forward(self, arguments):
        gnn1_embedding, idx = arguments
        n_drug = self.n_drug
        emb = self.args.embedding_num

        drug_embedding = self.drug_embed(self.drug_name)
        rela_embedding = self.rela_embed(self.adj_relation)
        ent_embedding = self.ent_embed(self.adj_tail)

        drug_rel = drug_embedding.view(n_drug, 1, emb) * rela_embedding
        drug_rel_weigh = torch.bmm(drug_rel, self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = torch.bmm(drug_rel_weigh, self.W2) + self.b2
        drug_rel_score = drug_rel_weigh.sum(dim=-1, keepdim=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.view(n_drug, 1, self.neighbor_sample_size).bmm(ent_embedding)
        drug_e = torch.cat([weighted_ent.view(n_drug, emb), drug_embedding.view(n_drug, emb)], dim=1)
        drug_f = self.Linear1(drug_e)
        return drug_f, gnn1_embedding, idx


# ──────────────────────────────────────────────────────────────────────────────
# GNN3 — has the special "fill missing drugs with ghost node" prep step
# ──────────────────────────────────────────────────────────────────────────────
class GNN3(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super().__init__(**kwargs)
        self.kg = dataset["dataset3"]
        self.dict1 = dict1
        self.drug_name_list = drug_name
        self.args = args
        self.n_drug = len(dict1)
        self.neighbor_sample_size = args.neighbor_sample_size

        self.drug_embed = nn.Embedding(num_embeddings=self.n_drug, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset3"], embedding_dim=args.embedding_num)
        # Original: ent_embed uses len(self.dict1) for num_embeddings (drug-drug KG)
        self.ent_embed = nn.Embedding(num_embeddings=self.n_drug, embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b1 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))
        self.W2 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b2 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))
        self.Linear1 = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num),
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
        self.register_buffer("adj_tail", torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long))
        self.register_buffer("adj_relation", torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long))

        # GNN3 originally appends a ghost neighbor (tails_num+1, relations_num+1) for drugs
        # missing from kg. Replicate that here exactly once at init (kg is mutated).
        self._initial_kg_patch(tail_len["dataset3"], relation_len["dataset3"])

    def _initial_kg_patch(self, tails_num: int, relations_num: int) -> None:
        """Replicates the in-place kg mutation from original GNN3.arrge()."""
        drug_number = list(self.dict1.values())
        drug_list = list(self.kg.keys())
        surplus = set(drug_number).difference(set(drug_list))
        for i in list(surplus):
            self.kg.setdefault(i, []).append((tails_num + 1, relations_num + 1))

    @torch.no_grad()
    def precompute_adj(self, rng: np.random.RandomState | None = None) -> None:
        """Resample per-drug neighbors. If rng is None, use numpy global state
        (seeded by setup_seed) — preserves run-level reproducibility."""
        if rng is None:
            rng = np.random
        adj_tail, adj_relation = _sample_adj_no_ghost(
            self.kg, self.dict1, self.neighbor_sample_size, self.n_drug, rng,
        )
        self.adj_tail.copy_(torch.from_numpy(adj_tail), non_blocking=True)
        self.adj_relation.copy_(torch.from_numpy(adj_relation), non_blocking=True)

    def forward(self, arguments):
        gnn2_embedding, gnn1_embedding, idx = arguments
        n_drug = self.n_drug
        emb = self.args.embedding_num

        drug_embedding = self.drug_embed(self.drug_name)
        rela_embedding = self.rela_embed(self.adj_relation)
        ent_embedding = self.ent_embed(self.adj_tail)

        drug_rel = drug_embedding.view(n_drug, 1, emb) * rela_embedding
        drug_rel_weigh = torch.bmm(drug_rel, self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = torch.bmm(drug_rel_weigh, self.W2) + self.b2
        drug_rel_score = drug_rel_weigh.sum(dim=-1, keepdim=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.view(n_drug, 1, self.neighbor_sample_size).bmm(ent_embedding)
        drug_e = torch.cat([weighted_ent.view(n_drug, emb), drug_embedding.view(n_drug, emb)], dim=1)
        drug_f = self.Linear1(drug_e)
        return drug_f, gnn2_embedding, gnn1_embedding, idx


# ──────────────────────────────────────────────────────────────────────────────
# GNN4
# ──────────────────────────────────────────────────────────────────────────────
class GNN4(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super().__init__(**kwargs)
        self.kg = dataset["dataset4"]
        self.dict1 = dict1
        self.drug_name_list = drug_name
        self.args = args
        self.n_drug = len(dict1)
        self.neighbor_sample_size = args.neighbor_sample_size

        self.drug_embed = nn.Embedding(num_embeddings=self.n_drug, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset4"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset4"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b1 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))
        self.W2 = nn.Parameter(torch.randn(self.n_drug, args.embedding_num, args.embedding_num))
        self.b2 = nn.Parameter(torch.randn(args.neighbor_sample_size, args.embedding_num))
        self.Linear1 = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num),
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
        self.register_buffer("adj_tail", torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long))
        self.register_buffer("adj_relation", torch.zeros(self.n_drug, self.neighbor_sample_size, dtype=torch.long))

    @torch.no_grad()
    def precompute_adj(self, rng: np.random.RandomState | None = None) -> None:
        """Resample per-drug neighbors. If rng is None, use numpy global state
        (seeded by setup_seed) — preserves run-level reproducibility."""
        if rng is None:
            rng = np.random
        adj_tail, adj_relation = _sample_adj_no_ghost(
            self.kg, self.dict1, self.neighbor_sample_size, self.n_drug, rng,
        )
        self.adj_tail.copy_(torch.from_numpy(adj_tail), non_blocking=True)
        self.adj_relation.copy_(torch.from_numpy(adj_relation), non_blocking=True)

    def forward(self, arguments):
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments
        n_drug = self.n_drug
        emb = self.args.embedding_num

        drug_embedding = self.drug_embed(self.drug_name)
        rela_embedding = self.rela_embed(self.adj_relation)
        ent_embedding = self.ent_embed(self.adj_tail)

        drug_rel = drug_embedding.view(n_drug, 1, emb) * rela_embedding
        drug_rel_weigh = torch.bmm(drug_rel, self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = torch.bmm(drug_rel_weigh, self.W2) + self.b2
        drug_rel_score = drug_rel_weigh.sum(dim=-1, keepdim=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.view(n_drug, 1, self.neighbor_sample_size).bmm(ent_embedding)
        drug_e = torch.cat([weighted_ent.view(n_drug, emb), drug_embedding.view(n_drug, emb)], dim=1)
        drug_f = self.Linear1(drug_e)
        return drug_f, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx


# ──────────────────────────────────────────────────────────────────────────────
# FusionLayer — unchanged from original (verified by hand vs source)
# ──────────────────────────────────────────────────────────────────────────────
class FusionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(args.embedding_num * 4 * 2, args.embedding_num * 4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 4, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, args.event_num),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        gnn4_embedding, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments

        # idx is a tensor of shape (batch, 2) on the same device as the embeddings.
        # The original code did `idx = idx.numpy().tolist()` then list-built drugA/drugB —
        # we keep the behaviour but stay on-device with index_select.
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx, dtype=torch.long, device=gnn1_embedding.device)
        elif idx.device != gnn1_embedding.device:
            idx = idx.to(gnn1_embedding.device)

        drugA_idx = idx[:, 0]
        drugB_idx = idx[:, 1]

        Embedding = torch.cat(
            [
                gnn1_embedding.index_select(0, drugA_idx),
                gnn2_embedding.index_select(0, drugA_idx),
                gnn3_embedding.index_select(0, drugA_idx),
                gnn4_embedding.index_select(0, drugA_idx),
                gnn1_embedding.index_select(0, drugB_idx),
                gnn2_embedding.index_select(0, drugB_idx),
                gnn3_embedding.index_select(0, drugB_idx),
                gnn4_embedding.index_select(0, drugB_idx),
            ],
            dim=1,
        ).float()

        return self.fullConnectionLayer(Embedding)
