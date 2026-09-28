"""TIGER thin adapter — implementation of :class:`BaselineModel`.

Wraps :class:`coldddi.baselines.tiger.model.TIGER` in **dual-channel
mode** (mol + KG branches) by default, mirroring the upstream paper
configuration.  Mol-only mode is reachable via ``mol_only=True``
constructor flag for fast inductive comparisons.

Pipeline
--------
1. ``fit`` builds:
   * a SMILES → :class:`torch_geometric.data.Data` map via
     :func:`coldddi.baselines.tiger.mol_features.build_drug_graphs`
     (mol channel input);
   * a Biomedical Knowledge Graph (BKG) edge / relation list via
     :func:`coldddi.baselines.tiger.kg_build.build_bkg` (drug-DDI +
     drug-entity from ``bundle.extra['kb']`` + self-loops);
   * per-drug random-walk subgraphs via
     :func:`coldddi.baselines.tiger.data_process.generate_node_subgraphs`
     (KG channel input).
2. ``predict_proba`` reuses the cached SMILES graphs + subgraphs.
   Cold-start drugs (g2 partition) are flagged via ``unseen_ids`` so
   the model's ``_patch_unseen_center_nodes`` replaces the center
   node's KG embedding with a projection of the mol embedding.

Pairs are scored by a 2-class softmax head, returning the class-1
probability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
from torch_geometric.data import Batch, Data

from coldddi.baselines.base import BaselineModel, register, write_manifest
from coldddi.baselines.tiger import data_process as tiger_dp
from coldddi.baselines.tiger.kg_build import build_bkg
from coldddi.baselines.tiger.mol_features import (
    ATOM_FEATURE_DIM,
    build_drug_graphs,
)
from coldddi.baselines.tiger.model import TIGER

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset
    from coldddi.data.protocols import KnowledgeGraphProtocol


# ── Helpers to convert upstream subgraph dict entries → PyG Data ─────

def _subgraph_dict_to_data(subg_entry, label_val: int) -> Data:
    """Convert one entry from ``data_process.generate_node_subgraphs``
    output (8-tuple) into a PyG :class:`Data` matching what the dual-
    channel TIGER expects on its KG-branch input.

    Upstream tuple layout (`data_process.rwExtractor` /
    `subtreeExtractor`):
      ``(subset, subgraph_edge_index, subgraph_rel,
         mapping_id, s_edge_index, s_value, s_rel, deg_s)``
    """
    subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, _deg = subg_entry
    return Data(
        x=torch.LongTensor(subset),
        edge_index=torch.LongTensor(subgraph_edge_index).transpose(1, 0),
        y=torch.LongTensor([label_val]),
        id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
        rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
        sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
        sp_value=torch.Tensor(np.array(s_value, dtype=int)),
        sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int)),
    )


def _drug_smiles_dict(train: "PairDataset") -> dict[str, str]:
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "TIGER requires `PairDataset.drugs` with a `smiles` column."
        )
    return {
        str(row["drugbank_id"]): "" if pd.isna(row["smiles"]) else str(row["smiles"])
        for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
    }


def _drug_to_idx(train: "PairDataset") -> dict[str, int]:
    """Stable drugbank_id → int index covering every drug in
    ``ds.drugs`` (sorted for determinism)."""
    ids = sorted({str(d) for d in train.drugs["drugbank_id"].astype(str)})
    return {d: i for i, d in enumerate(ids)}


def _g2_indices(train: "PairDataset", drug_to_idx: dict[str, int]) -> set[int]:
    """Indices of cold-start (g2) drugs.  Returns empty set if the
    dataset doesn't expose a g2_drugs attribute (e.g. legacy bundle
    without drug_groups)."""
    g2 = getattr(train.splits, "g2_drugs", None) or []
    return {drug_to_idx[d] for d in (str(x) for x in g2) if d in drug_to_idx}


# ── Adapter ──────────────────────────────────────────────────────────

#: Paper-spec hyperparameters from Appendix C.1 Table 8 (TIGER row).
#: Paper: layer=2, d_dim=64, walk=randomWalk, fixed-num=32, dropout=0.2,
#: LR=1e-3, WD=1e-4, batch=128, epochs=50.
PAPER_HYPERPARAMS: dict[str, object] = {
    "mol_only":           False,         # paper-spec dual-channel
    "max_layer":          2,             # paper "layer=2"
    "output_dim":         64,            # paper "d_dim=64"
    "dropout":            0.2,
    "extractor":          "randomWalk",
    "fixed_num":          32,            # paper "fixed-num=32"
    "learning_rate":      1e-3,
    "weight_decay":       1e-4,
    "batch_size":         128,
    "n_epochs":           50,
}


@register("tiger")
class TIGERBaseline(BaselineModel):
    """TIGER (dual-channel default): SMILES atom GraphTransformer +
    BKG random-walk subgraph GraphTransformer + 2-class head.

    Paper-grade hyperparameters live in :data:`PAPER_HYPERPARAMS`
    (App C.1 Table 8) and are auto-applied by
    ``evaluate.py --preset paper`` (default).  Class ``__init__``
    defaults below are smoke-test values for fast CI.

    Modality: ``"mol+kg"`` (dual-channel mode, the default and the
    only mode that supports the L6 channel-mask indicators). Channel
    mask exposed via ``predict_proba(pairs, mask_channel="mol"|"kg")``.
    L6 dispatch produces KPS-F + KPS-mol + KPS-KG, all three populated.

    Note: ``mol_only=True`` mode disables the KG branch entirely and
    is intentionally NOT a separate modality label — calling
    ``predict_proba(..., mask_channel=...)`` in mol-only mode raises
    ``ValueError`` (no KG branch to mask). The class-level ``modality``
    attribute reflects the dual-channel default, which is the paper-
    spec configuration.
    """

    VERSION = "2.0"  # 2.0 = dual-channel restoration; 1.0 was mol-only
    modality = "mol+kg"

    def __init__(
        self,
        *,
        # ── Architecture ───────────────────────────────────────────
        mol_only: bool = False,
        max_layer: int = 4,
        output_dim: int = 64,
        max_degree_graph: int = 100,
        max_degree_node: int = 100,
        num_relations_mol: int | None = None,
        num_relations_graph: int | None = None,
        num_nodes_kg: int | None = None,
        sub_coeff: float = 0.2,
        mi_coeff: float = 0.5,
        dropout: float = 0.2,
        # ── Subgraph generation (KG branch) ────────────────────────
        extractor: str = "randomWalk",
        graph_fixed_num: int = 8,
        fixed_num: int = 10,
        khop: int = 2,
        # ── Training ──────────────────────────────────────────────
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 64,
        n_epochs: int = 5,
        device: str = "auto",
    ) -> None:
        self.mol_only = mol_only
        # Instance-level modality override: mol_only=True drops the KG
        # branch entirely, so the L6 dispatch must NOT request mol/kg
        # mask passes (predict_proba would raise).  Class-level
        # ``modality`` stays "mol+kg" because that's the paper-spec
        # default; the instance attribute shadows it when the user
        # opts into the mol-only fallback.
        if mol_only:
            self.modality = "mol"
        self.max_layer = max_layer
        self.output_dim = output_dim
        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node
        self.num_relations_mol = num_relations_mol
        self.num_relations_graph = num_relations_graph
        self.num_nodes_kg = num_nodes_kg
        self.sub_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout

        self.extractor = extractor
        self.graph_fixed_num = graph_fixed_num
        self.fixed_num = fixed_num
        self.khop = khop

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = self._resolve_device(device)

        self._model: TIGER | None = None
        self._mol_graphs: dict[str, Data] | None = None  # drugbank_id -> Data
        self._drug_to_idx: dict[str, int] | None = None
        self._subgraphs: dict | None = None              # str(idx) -> 8-tuple
        self._g2_idx: set[int] = set()
        self._missing: list[str] = []
        self._effective_num_rel_mol: int | None = None
        self._effective_num_rel_kg: int | None = None
        self._effective_num_nodes_kg: int | None = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mol_graphs(self, train: "PairDataset") -> int:
        smiles = _drug_smiles_dict(train)
        self._mol_graphs, self._missing, max_rel = build_drug_graphs(smiles)
        if not self._mol_graphs:
            raise ValueError(
                "TIGER built zero molecular graphs — every drug failed to parse."
            )
        if self._missing:
            print(
                f"[tiger] {len(self._missing)} drugs skipped (no parseable SMILES).",
                file=sys.stderr,
            )
        return max_rel + 1

    def _build_subgraphs(
        self,
        train: "PairDataset",
        drug_to_idx: dict[str, int],
    ) -> tuple[dict, int, int]:
        """Build BKG and call upstream subgraph generator.

        Returns ``(subgraphs, num_rel_kg, max_degree_node_observed)``.
        """
        kb = (train.legacy_bundle.extra.get("kb", {})
              if train.legacy_bundle is not None else {}) or {}
        # Training positives (g1×g1 by definition since g2 is held out).
        train_pos = list(zip(
            train.splits.train["drug_a_id"].astype(str),
            train.splits.train["drug_b_id"].astype(str),
        ))
        g2_set = _g2_indices(train, drug_to_idx)
        edge_list, rel_list, num_rel, n_drugs = build_bkg(
            kb=kb,
            drug_to_idx=drug_to_idx,
            train_positive_pairs=train_pos,
            g2_drugs=g2_set,
        )
        self._g2_idx = g2_set

        # Upstream's ``generate_node_subgraphs`` writes a JSON cache
        # to ``data/<dataset>/<extractor>/rw_num_<S>_length_<L>sp.json``
        # next to the script and reuses it across runs.  The cache
        # key only varies by extractor + S + L — NOT by BKG content —
        # so naïve reuse can silently load subgraphs from a previous
        # fit on a different dataset / split / kb.  We sidestep this
        # by deriving the dataset directory name from a content hash
        # of the BKG edge list + relation list + extractor knobs, so
        # different BKGs land in different cache directories.
        _h = hashlib.sha256()
        for u, v in edge_list:
            _h.update(int(u).to_bytes(8, "little", signed=False))
            _h.update(int(v).to_bytes(8, "little", signed=False))
        for r in rel_list:
            _h.update(int(r).to_bytes(4, "little", signed=False))
        # All extractor knobs that affect the saved JSON contents.
        # ``khop`` is only consumed by the ``khop-subtree`` extractor,
        # but include it unconditionally so the hash is robust against
        # silent reuse if the caller switches extractors.
        _h.update(
            f"|{self.extractor}|{self.graph_fixed_num}|{self.fixed_num}|{self.khop}".encode()
        )
        _bkg_hash = _h.hexdigest()[:12]
        _dataset_for_subgraphs = f"coldddi_tiger_bkg_{_bkg_hash}"
        os.makedirs(
            os.path.join("data", _dataset_for_subgraphs, self.extractor),
            exist_ok=True,
        )

        tiger_args = types.SimpleNamespace(
            extractor=self.extractor,
            graph_fixed_num=self.graph_fixed_num,
            fixed_num=self.fixed_num,
            khop=self.khop,
        )
        subgraphs, max_degree, num_rel_update = tiger_dp.generate_node_subgraphs(
            _dataset_for_subgraphs,
            set(str(i) for i in range(n_drugs)),
            edge_list,
            rel_list,
            num_rel,
            tiger_args,
        )
        # ``num_rel_update`` is the max relation id observed in
        # sp_edge_rel after shortest-path expansion (can exceed
        # ``num_rel`` because hop distances 2..khop are appended as
        # extra rel ids).
        effective_num_rel_kg = max(num_rel, num_rel_update + 1)
        return subgraphs, effective_num_rel_kg, max_degree

    def _make_pair_batch(
        self,
        pairs: pd.DataFrame,
        labels: np.ndarray | None = None,
    ):
        """Return ``(h_mol, t_mol, h_sub, t_sub, idx_h, idx_t, mask)``.

        Rows with any missing artefact (parsed mol graph or subgraph)
        are silently filtered, and ``mask`` is a boolean array over the
        original ``pairs`` index indicating which rows survived.
        """
        assert self._drug_to_idx is not None
        keep_idx: list[int] = []
        h_mol_list, t_mol_list = [], []
        h_sub_list, t_sub_list = [], []
        idx_h_list, idx_t_list = [], []

        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            if a not in self._drug_to_idx or b not in self._drug_to_idx:
                continue
            mol_a = self._mol_graphs.get(a) if self._mol_graphs else None
            mol_b = self._mol_graphs.get(b) if self._mol_graphs else None
            if mol_a is None or mol_b is None:
                continue
            idx_a = self._drug_to_idx[a]
            idx_b = self._drug_to_idx[b]
            label_val = int(labels[i]) if labels is not None else 0
            mol_a = mol_a.clone()
            mol_b = mol_b.clone()
            mol_a.y = torch.tensor([label_val], dtype=torch.long)
            mol_b.y = torch.tensor([label_val], dtype=torch.long)
            h_mol_list.append(mol_a)
            t_mol_list.append(mol_b)

            if not self.mol_only:
                sub_a = self._subgraphs.get(str(idx_a)) if self._subgraphs else None
                sub_b = self._subgraphs.get(str(idx_b)) if self._subgraphs else None
                if sub_a is None or sub_b is None:
                    # The upstream pipeline guarantees a subgraph per
                    # drug index (self-loop fallback), so this branch
                    # should rarely fire — but skip the row defensively
                    # so a missing entry doesn't crash the whole batch.
                    h_mol_list.pop()
                    t_mol_list.pop()
                    continue
                h_sub_list.append(_subgraph_dict_to_data(sub_a, label_val))
                t_sub_list.append(_subgraph_dict_to_data(sub_b, label_val))
                idx_h_list.append(idx_a)
                idx_t_list.append(idx_b)

            keep_idx.append(i)

        if not keep_idx:
            mask = np.zeros(len(pairs), dtype=bool)
            return None, None, None, None, None, None, mask

        h_mol_b = Batch.from_data_list(h_mol_list)
        t_mol_b = Batch.from_data_list(t_mol_list)
        if not self.mol_only:
            h_sub_b = Batch.from_data_list(h_sub_list)
            t_sub_b = Batch.from_data_list(t_sub_list)
            idx_h = torch.tensor(idx_h_list, dtype=torch.long)
            idx_t = torch.tensor(idx_t_list, dtype=torch.long)
        else:
            h_sub_b = t_sub_b = idx_h = idx_t = None

        mask = np.zeros(len(pairs), dtype=bool)
        mask[np.asarray(keep_idx)] = True
        return h_mol_b, t_mol_b, h_sub_b, t_sub_b, idx_h, idx_t, mask

    # ------------------------------------------------------------------
    # ABC surface
    # ------------------------------------------------------------------

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        # Step 1 — mol graphs.
        required_mol = self._build_mol_graphs(train)
        if self.num_relations_mol is None:
            effective_mol = max(required_mol + 4, 32)
        else:
            effective_mol = self.num_relations_mol
            if effective_mol < required_mol:
                raise ValueError(
                    f"num_relations_mol={effective_mol} but observed sp_edge_rel "
                    f"requires at least {required_mol} relation slots."
                )
        self._effective_num_rel_mol = effective_mol

        # Step 2 — drug-to-idx + BKG + subgraphs (skipped in mol_only).
        self._drug_to_idx = _drug_to_idx(train)
        n_drugs_total = len(self._drug_to_idx)
        if not self.mol_only:
            subgraphs, observed_num_rel_kg, observed_max_deg = self._build_subgraphs(
                train, self._drug_to_idx,
            )
            self._subgraphs = subgraphs
            self._effective_num_rel_kg = (
                self.num_relations_graph
                if self.num_relations_graph is not None
                else max(observed_num_rel_kg + 4, 16)
            )
            # KG node count: drugs (n_drugs) + entities (variable). The
            # BKG's largest node id determines required Embedding size.
            max_node_id = -1
            for entry in subgraphs.values():
                subset = entry[0]  # x = subset
                if len(subset) > 0:
                    max_node_id = max(max_node_id, int(max(subset)))
            self._effective_num_nodes_kg = (
                self.num_nodes_kg
                if self.num_nodes_kg is not None
                else max(max_node_id + 16, n_drugs_total + 16)
            )
            # ``max_degree_node`` caps the degree-Embedding lookup
            # inside ``NodeFeatures``. If the BKG's actual max degree
            # exceeds the constructor default (100), bump it so degree
            # encoding doesn't index out of range. Also save the
            # effective value so ``load()`` reconstructs the model
            # with a wide-enough embedding.
            required_deg = int(observed_max_deg) + 1
            if required_deg > self.max_degree_node:
                print(
                    f"[tiger] bumping max_degree_node from "
                    f"{self.max_degree_node} to {required_deg} "
                    f"(observed BKG max degree).",
                    file=sys.stderr,
                )
                self.max_degree_node = required_deg

        # Step 3 — instantiate the model.
        self._model = TIGER(
            max_layer=self.max_layer,
            num_features_drug=ATOM_FEATURE_DIM,
            num_nodes=(self._effective_num_nodes_kg or 1),
            num_relations_mol=self._effective_num_rel_mol,
            num_relations_graph=(self._effective_num_rel_kg or 1),
            output_dim=self.output_dim,
            max_degree_graph=self.max_degree_graph,
            max_degree_node=self.max_degree_node,
            sub_coeff=self.sub_coeff,
            mi_coeff=self.mi_coeff,
            dropout=self.dropout,
            device=self.device,
            mol_only=self.mol_only,
        ).to(self.device)

        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Step 4 — training loop.
        pos = train.splits.train.copy()
        best_val_auc = -1.0
        best_state: dict | None = None
        for epoch in range(self.n_epochs):
            neg = train.get_train_negatives(epoch, regenerate=False)
            pairs_df = (
                pd.concat(
                    [
                        pos[["drug_a_id", "drug_b_id"]].assign(label=1),
                        neg[["drug_a_id", "drug_b_id"]].assign(label=0),
                    ],
                    ignore_index=True,
                )
                .sample(frac=1, random_state=epoch)
                .reset_index(drop=True)
            )
            self._model.train()
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start : start + self.batch_size]
                h_mol, t_mol, h_sub, t_sub, idx_h, idx_t, _mask = self._make_pair_batch(
                    batch, labels=batch["label"].to_numpy(),
                )
                if h_mol is None:
                    continue
                h_mol = h_mol.to(self.device)
                t_mol = t_mol.to(self.device)
                if not self.mol_only:
                    h_sub = h_sub.to(self.device)
                    t_sub = t_sub.to(self.device)
                opt.zero_grad(set_to_none=True)
                if self.mol_only:
                    _probs, loss = self._model(h_mol, t_mol)
                else:
                    # NOTE: ``unseen_ids=None`` at training time.  By the
                    # ColdDDI invariant, train pairs are G1-G1 only — no
                    # G2 drug appears in a training batch — so this is a
                    # no-op as long as the invariant holds.  If anything
                    # ever leaks a G2 drug into training, we'd rather see
                    # it via a clean KG-branch forward than silently
                    # activate the cold-start projection during optim.
                    # Cold-start patching is reserved for val/test.
                    _probs, loss = self._model(
                        h_mol, t_mol,
                        drug1_subgraph=h_sub, drug2_subgraph=t_sub,
                        batch_idx1=idx_h, batch_idx2=idx_t,
                        unseen_ids=None,
                    )
                loss.backward()
                opt.step()

            if val is not None:
                auc = self._validate(val)
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_state = copy.deepcopy(self._model.state_dict())

        if best_state is not None:
            self._model.load_state_dict(best_state)

    @torch.no_grad()
    def _validate(self, val: "PairDataset") -> float:
        pos = val.splits.val_s2[["drug_a_id", "drug_b_id"]]
        neg = val.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y_score = np.concatenate(
            [self.predict_proba(pos), self.predict_proba(neg)]
        )
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y_true, y_score))

    @torch.no_grad()
    def predict_proba(
        self,
        pairs: pd.DataFrame,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
        mask_channel: str | None = None,
    ) -> np.ndarray:
        """Score every pair.

        Parameters
        ----------
        mask_channel
            Paper-spec channel ablation knob, forwarded to
            :meth:`TIGER.forward`.  ``"mol"`` zeros the molecular
            GraphTransformer output for BOTH drugs in each pair at
            fusion time (KPS-mol indicator); ``"kg"`` zeros the
            KG GraphTransformer output (KPS-KG); ``None`` (default)
            produces the unmasked base prediction.

            Only valid in dual-channel mode.  Raises ``ValueError``
            if the model was built with ``mol_only=True`` — there is
            no KG branch to mask in mol-only mode, and ``"mol"`` would
            silently no-op (the mol embedding is the only signal).
        """
        if self._model is None or self._mol_graphs is None:
            raise RuntimeError(
                "TIGERBaseline must be fitted (or loaded) before prediction."
            )
        if mask_channel is not None and self.mol_only:
            raise ValueError(
                "mask_channel requires dual-channel TIGER; "
                "this baseline was built with mol_only=True."
            )
        self._model.eval()
        out = np.full(len(pairs), 0.5, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            zero_labels = np.zeros(len(batch), dtype=np.int64)
            h_mol, t_mol, h_sub, t_sub, idx_h, idx_t, mask = self._make_pair_batch(
                batch, labels=zero_labels,
            )
            if h_mol is None:
                continue
            h_mol = h_mol.to(self.device)
            t_mol = t_mol.to(self.device)
            if not self.mol_only:
                h_sub = h_sub.to(self.device)
                t_sub = t_sub.to(self.device)
                probs, _loss = self._model(
                    h_mol, t_mol,
                    drug1_subgraph=h_sub, drug2_subgraph=t_sub,
                    batch_idx1=idx_h, batch_idx2=idx_t,
                    unseen_ids=self._g2_idx,
                    mask_channel=mask_channel,
                )
            else:
                probs, _loss = self._model(h_mol, t_mol)
            probs = probs.detach().cpu().numpy()
            out_slice = np.full(len(batch), 0.5, dtype=np.float32)
            out_slice[mask] = probs
            out[start : start + len(batch)] = out_slice
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._mol_graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "artefacts.pkl").open("wb") as f:
            pickle.dump({
                "mol_graphs": self._mol_graphs,
                "missing": self._missing,
                "drug_to_idx": self._drug_to_idx,
                "subgraphs": self._subgraphs,
                "g2_idx": list(self._g2_idx),
                "effective_num_rel_mol": self._effective_num_rel_mol,
                "effective_num_rel_kg": self._effective_num_rel_kg,
                "effective_num_nodes_kg": self._effective_num_nodes_kg,
            }, f)
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "hyperparameters": {
                    "mol_only": self.mol_only,
                    "max_layer": self.max_layer,
                    "output_dim": self.output_dim,
                    "max_degree_graph": self.max_degree_graph,
                    "max_degree_node": self.max_degree_node,
                    "num_relations_mol": self._effective_num_rel_mol,
                    "num_relations_graph": self._effective_num_rel_kg,
                    "num_nodes_kg": self._effective_num_nodes_kg,
                    "sub_coeff": self.sub_coeff,
                    "mi_coeff": self.mi_coeff,
                    "dropout": self.dropout,
                    "extractor": self.extractor,
                    "graph_fixed_num": self.graph_fixed_num,
                    "fixed_num": self.fixed_num,
                    "khop": self.khop,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "sklearn_version": sklearn.__version__,
                },
                "graph_metadata": {
                    "n_drugs_with_graph": len(self._mol_graphs),
                    "n_drugs_missing": len(self._missing),
                    "n_subgraphs": (len(self._subgraphs)
                                    if self._subgraphs is not None else 0),
                    "n_g2": len(self._g2_idx),
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "TIGERBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        # Pull out effective sizes (must match what the saved model
        # was built with, NOT the constructor defaults).
        eff_num_rel_mol = hparams.pop("num_relations_mol", None)
        eff_num_rel_kg = hparams.pop("num_relations_graph", None)
        eff_num_nodes_kg = hparams.pop("num_nodes_kg", None)
        inst = cls(
            num_relations_mol=eff_num_rel_mol,
            num_relations_graph=eff_num_rel_kg,
            num_nodes_kg=eff_num_nodes_kg,
            **hparams,
        )
        with (p / "artefacts.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._mol_graphs = payload["mol_graphs"]
        inst._missing = payload.get("missing", [])
        inst._drug_to_idx = payload["drug_to_idx"]
        inst._subgraphs = payload.get("subgraphs")
        inst._g2_idx = set(payload.get("g2_idx", []))
        inst._effective_num_rel_mol = payload["effective_num_rel_mol"]
        inst._effective_num_rel_kg = payload.get("effective_num_rel_kg")
        inst._effective_num_nodes_kg = payload.get("effective_num_nodes_kg")
        inst._model = TIGER(
            max_layer=inst.max_layer,
            num_features_drug=ATOM_FEATURE_DIM,
            num_nodes=(inst._effective_num_nodes_kg or 1),
            num_relations_mol=inst._effective_num_rel_mol,
            num_relations_graph=(inst._effective_num_rel_kg or 1),
            output_dim=inst.output_dim,
            max_degree_graph=inst.max_degree_graph,
            max_degree_node=inst.max_degree_node,
            sub_coeff=inst.sub_coeff,
            mi_coeff=inst.mi_coeff,
            dropout=inst.dropout,
            device=inst.device,
            mol_only=inst.mol_only,
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()
        return inst
