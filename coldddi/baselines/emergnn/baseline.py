"""EmerGNN thin adapter — implementation of :class:`BaselineModel`.

Wraps the verbatim research-repo modules
:mod:`coldddi.baselines.emergnn.model` (the pure-PyTorch reimplementation
of EmerGNN — no torchdrug / torch_scatter required),
:mod:`coldddi.baselines.emergnn.kg_builder` (KG → triplet table), and
:mod:`coldddi.baselines.emergnn.morgan_features` (Morgan-FP entity feats).

The adapter is responsible for two bridges:

* :class:`coldddi.data.kg.KnowledgeGraph` → the legacy ``my_X_list``
  dict the source kg_builder consumes.
* :class:`coldddi.data.dataset.PairDataset` train/val/test rows → the
  per-pair head/tail entity indices the EmerGNN forward pass expects.

Like :class:`coldddi.baselines.deepddi.DeepDDIBaseline`, this module is
the **template** for any KG-aware baseline: cast the KG once at fit
time, keep entity-id → index lookup as state, batch over
``DataFrame[drug_a_id, drug_b_id]`` for predict.
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn  # noqa: F401  — kept so the manifest version export stays useful
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

from coldddi.baselines.base import BaselineModel, register, write_manifest
from coldddi.baselines.emergnn.kg_builder import (
    N_BASE_REL,
    build_kg_from_kb,
    build_sparse_adj,
    edges_as_dense_lists,
)
from coldddi.baselines.emergnn.model import EmerGNN
from coldddi.baselines.emergnn.morgan_features import compute_morgan_matrix

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset
    from coldddi.data.protocols import KnowledgeGraphProtocol


def _kg_to_kb_dict(kg: "KnowledgeGraphProtocol") -> dict[str, pd.DataFrame]:
    """Convert a :class:`KnowledgeGraph`-shaped object to the legacy
    ``my_X_list`` schema that :func:`build_kg_from_kb` expects."""
    required = ("enzymes", "targets", "transporters", "carriers", "pathways")
    missing = [a for a in required if not hasattr(kg, a)]
    if missing:
        raise TypeError(
            f"EmerGNN requires a knowledge graph with attributes "
            f"{required}; got {type(kg).__name__} (missing {missing})."
        )
    return {
        "my_enzyme_list": kg.enzymes,
        "my_target_list": kg.targets,
        "my_transporter_list": kg.transporters,
        "my_carrier_list": kg.carriers,
        "my_pathway_list": kg.pathways,
    }


def _drug_smiles_dict(train: "PairDataset") -> dict[str, str]:
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "EmerGNN requires `PairDataset.drugs` with a `smiles` column. "
            "Load via `from_release_dir` after Stage 1b, or pass a legacy "
            "bundle whose `extra['kb']` includes `my_drugs_list`."
        )
    return {
        str(row["drugbank_id"]): "" if pd.isna(row["smiles"]) else str(row["smiles"])
        for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
    }


#: Paper-spec hyperparameters from Appendix C.1 Table 8 (EmerGNN row).
#: Paper also lists weight_decay=1e-8 but EmerGNNBaseline's __init__
#: doesn't expose that knob (Adam default is 0); document for audit.
PAPER_HYPERPARAMS: dict[str, object] = {
    "n_dim":          64,
    "length":         3,
    "feat":           "M",          # 'M' = Morgan fingerprint (paper)
    "learning_rate":  1e-3,
    "batch_size":     32,
    "n_epochs":       40,
}


@register("emergnn")
class EmerGNNBaseline(BaselineModel):
    """EmerGNN: bidirectional attention message-passing on the DrugBank KG.

    Modality: ``"mol+kg-fused"`` — consumes both mol-derived node
    features AND KG topology, but the two are fused inside the
    message-passing layer with no separable channel mask interface.
    L6 dispatch produces KPS-F only; KPS-mol / KPS-KG come back as
    NaN rows because there is no architectural channel to ablate
    independently.

    Paper-grade hyperparameters live in :data:`PAPER_HYPERPARAMS`
    (App C.1 Table 8) and are auto-applied by
    ``evaluate.py --preset paper`` (default).  Class ``__init__``
    defaults below are smoke-test values for fast CI.
    """

    VERSION = "1.0"
    modality = "mol+kg-fused"

    def __init__(
        self,
        *,
        n_dim: int = 64,
        length: int = 3,
        feat: str = "M",  # 'M' Morgan / 'E' learned
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        n_epochs: int = 5,
        device: str = "auto",
    ) -> None:
        self.n_dim = n_dim
        self.length = length
        self.feat = feat
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = self._resolve_device(device)
        self._model: EmerGNN | None = None
        self._entity2id: dict[str, int] | None = None
        self._n_ent: int | None = None
        self._edge_src: torch.Tensor | None = None
        self._edge_dst: torch.Tensor | None = None
        self._edge_rel: torch.Tensor | None = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    # ------------------------------------------------------------------
    # KG / feature setup (called by fit/load)
    # ------------------------------------------------------------------

    def _setup_graph(
        self,
        train: "PairDataset",
        kg: "KnowledgeGraphProtocol",
    ) -> tuple[np.ndarray, list[str]]:
        """Build entity vocab + edge tensors from a KG. Returns (morgan_matrix, drug_ids)."""
        kb = _kg_to_kb_dict(kg)

        # The EmerGNN entity vocab must list every drug that appears in any
        # split. We collect them from train + val/test splits.
        drug_ids: set[str] = set()
        for _name, df in train.splits.items():
            drug_ids.update(df["drug_a_id"].astype(str))
            drug_ids.update(df["drug_b_id"].astype(str))
        if hasattr(kg, "drug_ids"):
            drug_ids.update(kg.drug_ids)
        drug_id_list = sorted(drug_ids)

        kg_artifacts = build_kg_from_kb(kb, drug_id_list, keep_only_known_drugs=True)
        self._entity2id = kg_artifacts["entity2id"]
        self._n_ent = kg_artifacts["n_ent"]

        # Edge tensors (forward + reverse + self-loop) — kept on CPU,
        # moved to device at forward time.
        adj = build_sparse_adj(
            kg_artifacts["triplets"], kg_artifacts["n_ent"], N_BASE_REL
        )
        self._edge_src, self._edge_dst, self._edge_rel = edges_as_dense_lists(adj)

        # Morgan features for every entity (zeros for non-drug entities).
        smiles = _drug_smiles_dict(train)
        n_ent = kg_artifacts["n_ent"]
        morgan_mat = np.zeros((n_ent, 1024), dtype=np.float32)
        drug_only_mat, missing = compute_morgan_matrix(drug_id_list, smiles)
        for did, row in zip(drug_id_list, drug_only_mat):
            morgan_mat[self._entity2id[did]] = row
        if missing:
            print(
                f"[emergnn] {len(missing)} drugs lacked parseable SMILES "
                f"and were zero-filled.",
                file=sys.stderr,
            )
        return morgan_mat, drug_id_list

    def _pair_indices(self, pairs: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if self._entity2id is None:
            raise RuntimeError("EmerGNNBaseline.fit() must be called before predict_proba.")
        head = pairs["drug_a_id"].astype(str).map(self._entity2id)
        tail = pairs["drug_b_id"].astype(str).map(self._entity2id)
        if head.isna().any() or tail.isna().any():
            unknown = sorted(set(
                pairs.loc[head.isna(), "drug_a_id"].astype(str)
            ) | set(
                pairs.loc[tail.isna(), "drug_b_id"].astype(str)
            ))[:5]
            raise ValueError(
                f"EmerGNN saw drug ids not in the trained entity vocab: {unknown}…"
            )
        return (
            torch.tensor(head.to_numpy(), dtype=torch.long),
            torch.tensor(tail.to_numpy(), dtype=torch.long),
        )

    def _edges_on_device(self):
        return (
            self._edge_src.to(self.device),
            self._edge_dst.to(self.device),
            self._edge_rel.to(self.device),
        )

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
        if kg is None:
            kg = train.kg
        morgan_mat, _drug_ids = self._setup_graph(train, kg)

        self._model = EmerGNN(
            n_ent=self._n_ent,
            n_base_rel=N_BASE_REL,
            n_dim=self.n_dim,
            length=self.length,
            feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        edge_src, edge_dst, edge_rel = self._edges_on_device()

        pos = train.splits.train.copy()
        best_val_auc = -1.0
        best_state: dict | None = None
        for epoch in range(self.n_epochs):
            neg = train.get_train_negatives(epoch, regenerate=False)
            pairs_df = pd.concat(
                [
                    pos[["drug_a_id", "drug_b_id"]].assign(label=1),
                    neg[["drug_a_id", "drug_b_id"]].assign(label=0),
                ],
                ignore_index=True,
            ).sample(frac=1, random_state=epoch).reset_index(drop=True)

            self._model.train()
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start : start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head = head.to(self.device)
                tail = tail.to(self.device)
                y = torch.tensor(
                    batch["label"].to_numpy(), dtype=torch.float32, device=self.device
                )
                opt.zero_grad(set_to_none=True)
                logits = self._model(head, tail, edge_src, edge_dst, edge_rel)
                loss = binary_cross_entropy_with_logits(logits, y)
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
        y_score = np.concatenate([self.predict_proba(pos), self.predict_proba(neg)])
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y_true, y_score))

    @torch.no_grad()
    def predict_proba(
        self,
        pairs: pd.DataFrame,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EmerGNNBaseline must be fitted (or loaded) before prediction.")
        edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head = head.to(self.device)
            tail = tail.to(self.device)
            logits = self._model(head, tail, edge_src, edge_dst, edge_rel)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            out[start : start + len(batch)] = probs
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._entity2id is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graph.pkl").open("wb") as f:
            pickle.dump(
                {
                    "entity2id": self._entity2id,
                    "n_ent": self._n_ent,
                    "edge_src": self._edge_src,
                    "edge_dst": self._edge_dst,
                    "edge_rel": self._edge_rel,
                    "morgan_features": (
                        self._model.ent_feat.cpu().numpy() if self.feat == "M" else None
                    ),
                },
                f,
            )
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "hyperparameters": {
                    "n_dim": self.n_dim,
                    "length": self.length,
                    "feat": self.feat,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "sklearn_version": sklearn.__version__,
                },
                "graph_metadata": {"n_ent": int(self._n_ent)},
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "EmerGNNBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        inst = cls(**hparams)
        with (p / "graph.pkl").open("rb") as f:
            graph = pickle.load(f)
        inst._entity2id = graph["entity2id"]
        inst._n_ent = graph["n_ent"]
        inst._edge_src = graph["edge_src"]
        inst._edge_dst = graph["edge_dst"]
        inst._edge_rel = graph["edge_rel"]
        morgan = graph.get("morgan_features")
        inst._model = EmerGNN(
            n_ent=inst._n_ent,
            n_base_rel=N_BASE_REL,
            n_dim=inst.n_dim,
            length=inst.length,
            feat=inst.feat,
            morgan_features=morgan,
        ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        return inst
