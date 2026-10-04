"""TIGER adapter with molecular and KG channels; ``mol_only=True`` disables KG.

``fit`` caches SMILES graphs and per-drug BKG subgraphs from training DDIs,
the legacy bundle KB, and isolated-drug self-loops. Prediction reuses these
graphs and flags G2 drugs for molecular projection of their KG center nodes.
The two-class softmax head returns the class-1 probability.
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


# Subgraph conversion

def _subgraph_dict_to_data(subg_entry, label_val: int) -> Data:
    """Convert a subgraph 8-tuple to PyG :class:`Data` for TIGER's KG channel.

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
    """Map every ID in ``train.drugs`` to a sorted, deterministic integer index."""
    ids = sorted({str(d) for d in train.drugs["drugbank_id"].astype(str)})
    return {d: i for i, d in enumerate(ids)}


def _g2_indices(train: "PairDataset", drug_to_idx: dict[str, int]) -> set[int]:
    """Return G2 drug indices, or an empty set if ``g2_drugs`` is absent or empty."""
    g2 = getattr(train.splits, "g2_drugs", None) or []
    return {drug_to_idx[d] for d in (str(x) for x in g2) if d in drug_to_idx}


# Adapter

#: Paper-spec hyperparameters from Appendix C.1 Table 8 (TIGER row).
PAPER_HYPERPARAMS: dict[str, object] = {
    "mol_only":           False,
    "max_layer":          2,
    "output_dim":         64,
    "dropout":            0.2,
    "extractor":          "randomWalk",
    "fixed_num":          32,
    "learning_rate":      1e-3,
    "weight_decay":       1e-4,
    "batch_size":         128,
    "n_epochs":           50,
}


@register("tiger")
class TIGERBaseline(BaselineModel):
    """TIGER (dual-channel default): SMILES atom GraphTransformer +
    BKG random-walk subgraph GraphTransformer + 2-class head.

    ``evaluate.py --preset paper`` applies :data:`PAPER_HYPERPARAMS`
    (Appendix C.1 Table 8); constructor defaults are for smoke tests.

    Default modality ``"mol+kg"`` supports ``mask_channel="mol"|"kg"``
    and L6's KPS-F, KPS-mol, and KPS-KG indicators. ``mol_only=True`` sets
    the instance modality to ``"mol"`` and rejects channel masks with ValueError.
    """

    VERSION = "2.0"
    modality = "mol+kg"

    def __init__(
        self,
        *,
        # Architecture
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
        # KG subgraph generation
        extractor: str = "randomWalk",
        graph_fixed_num: int = 8,
        fixed_num: int = 10,
        khop: int = 2,
        # Training
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 64,
        n_epochs: int = 5,
        device: str = "auto",
    ) -> None:
        self.mol_only = mol_only
        # Disable L6 channel-mask dispatch when there is no KG branch.
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

    # Internal helpers

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

        # Upstream cache filenames omit BKG content. Hash edges, relations, and
        # extractor settings into the directory name to prevent cross-fit reuse.
        _h = hashlib.sha256()
        for u, v in edge_list:
            _h.update(int(u).to_bytes(8, "little", signed=False))
            _h.update(int(v).to_bytes(8, "little", signed=False))
        for r in rel_list:
            _h.update(int(r).to_bytes(4, "little", signed=False))
        # Include khop even when the current extractor does not use it.
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
        # Shortest-path expansion adds relation IDs beyond the base KG types.
        # num_rel_update is the maximum observed ID, not a count.
        effective_num_rel_kg = max(num_rel, num_rel_update + 1)
        return subgraphs, effective_num_rel_kg, max_degree

    def _make_pair_batch(
        self,
        pairs: pd.DataFrame,
        labels: np.ndarray | None = None,
    ):
        """Return ``(h_mol, t_mol, h_sub, t_sub, idx_h, idx_t, mask)``.

        Skip rows missing a drug ID, molecular graph, or required KG subgraph.
        ``mask`` marks retained positions in ``pairs``. If none remain, return
        six ``None`` values and an all-false mask.
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
                    # Remove the molecular entries too when a KG subgraph is missing.
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

    # ABC surface

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        # Molecular graphs and relation counts.
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

        # Drug indices and optional KG subgraphs.
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
            # Size the KG embedding for both drug and entity node IDs.
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
            # Expand the degree embedding to cover observed IDs. Save this
            # effective size so load() can reconstruct the same embedding.
            required_deg = int(observed_max_deg) + 1
            if required_deg > self.max_degree_node:
                print(
                    f"[tiger] bumping max_degree_node from "
                    f"{self.max_degree_node} to {required_deg} "
                    f"(observed BKG max degree).",
                    file=sys.stderr,
                )
                self.max_degree_node = required_deg

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
                    # Training pairs must be G1-G1; reserve cold-start projection
                    # for validation and testing.
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
            ``"mol"`` or ``"kg"`` zeros that channel for both drugs at fusion
            (KPS-mol or KPS-KG). ``None`` leaves both channels unchanged.
            Any non-None mask raises ValueError in mol-only mode.
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

    # Persistence

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
        # Restore saved embedding sizes, which may differ from constructor defaults.
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
