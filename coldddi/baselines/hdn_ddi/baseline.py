"""HDN-DDI adapter with intra/inter-graph GAT and super-node readout (``y == 2``).

Uses 55-dim atom features and one interaction relation (``rel_total=1``),
trained with binary cross-entropy on positive and negative pairs.
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
import sklearn  # noqa: F401
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.data import Batch

from coldddi.baselines.base import BaselineModel, register, write_manifest
from coldddi.baselines.dsn_ddi.models import make_bipartite_data
from coldddi.baselines.hdn_ddi.mol_features import (
    ATOM_FEATURE_DIM,
    build_drug_graphs,
)
from coldddi.baselines.hdn_ddi.models import HDN_DDI

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset
    from coldddi.data.protocols import KnowledgeGraphProtocol


def _drug_smiles_dict(train: "PairDataset") -> dict[str, str]:
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "HDN-DDI requires `PairDataset.drugs` with a `smiles` column."
        )
    return {
        str(row["drugbank_id"]): "" if pd.isna(row["smiles"]) else str(row["smiles"])
        for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
    }


def _bipartite_edge_index(n_s: int, n_t: int) -> torch.Tensor:
    if n_s == 0 or n_t == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.arange(n_s).repeat_interleave(n_t)
    dst = torch.arange(n_t).repeat(n_s)
    return torch.stack([src, dst], dim=0)


#: Paper-spec hyperparameters from Appendix C.1 Table 8 (HDN-DDI row).
#: Uses the same 128-dim dual-view attention output as DSN-DDI.
PAPER_HYPERPARAMS: dict[str, object] = {
    "in_features":            55,
    "hidd_dim":               128,
    "kge_dim":                128,
    "heads_out_feat_params":  (64, 64),
    "blocks_params":          (2, 2),
    "learning_rate":          1e-3,
    "weight_decay":           5e-4,
    "batch_size":             512,
    "n_epochs":               50,
}


@register("hdn_ddi")
class HDNDDIBaseline(BaselineModel):
    """HDN-DDI: hierarchical drug graph (intra+inter GAT) with super-node readout.

    Modality: ``"mol"`` — hierarchical mol graph with super-node,
    no KG channel. L6 dispatch produces KPS-F; KPS-mol / KPS-KG
    are NaN.

    ``evaluate.py --preset paper`` applies :data:`PAPER_HYPERPARAMS`
    (Appendix C.1 Table 8); constructor defaults are for smoke tests.
    """

    VERSION = "1.0"
    modality = "mol"

    def __init__(
        self,
        *,
        in_features: int = ATOM_FEATURE_DIM,
        hidd_dim: int = 64,
        kge_dim: int = 128,
        heads_out_feat_params: tuple[int, ...] = (64, 64, 64, 64),
        blocks_params: tuple[int, ...] = (2, 2, 2, 2),
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 256,
        n_epochs: int = 5,
        device: str = "auto",
    ) -> None:
        if len(heads_out_feat_params) != len(blocks_params):
            raise ValueError(
                "heads_out_feat_params and blocks_params must have the same length."
            )
        # Intra/inter attention each output 64 dims; their concatenation
        # requires n_heads * head_out_feats == kge_dim == 128.
        for h, n in zip(heads_out_feat_params, blocks_params):
            if h * n != 128:
                raise ValueError(
                    "HDN-DDI's IntraGraphAttention/InterGraphAttention output "
                    "is hardcoded to 64 dims each; every "
                    "(head_out_feats * n_heads) must equal 128. Got "
                    f"{h} * {n} = {h * n}."
                )
        if kge_dim != 128:
            raise ValueError(
                "HDN-DDI requires kge_dim == 128 to match the per-block "
                f"embedding dim. Got {kge_dim}."
            )
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.heads_out_feat_params = tuple(heads_out_feat_params)
        self.blocks_params = tuple(blocks_params)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = self._resolve_device(device)
        self._model: HDN_DDI | None = None
        self._graphs: dict | None = None
        self._missing: list[str] = []

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _build_graphs(self, train: "PairDataset") -> None:
        smiles = _drug_smiles_dict(train)
        self._graphs, self._missing = build_drug_graphs(smiles)
        if not self._graphs:
            raise ValueError(
                "HDN-DDI built zero molecular graphs — every drug failed to parse."
            )
        if self._missing:
            print(
                f"[hdn_ddi] {len(self._missing)} drugs skipped (no parseable SMILES).",
                file=sys.stderr,
            )

    def _make_pair_batch(self, pairs: pd.DataFrame):
        keep_idx: list[int] = []
        h_list, t_list, b_list = [], [], []
        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            ga = self._graphs.get(a)
            gb = self._graphs.get(b)
            if ga is None or gb is None:
                continue
            ga = ga.clone()
            gb = gb.clone()
            edge_b = _bipartite_edge_index(ga.x.size(0), gb.x.size(0))
            b_data = make_bipartite_data(ga.x, gb.x, edge_b)
            keep_idx.append(i)
            h_list.append(ga)
            t_list.append(gb)
            b_list.append(b_data)
        if not keep_idx:
            return None, None, None, None, np.zeros(len(pairs), dtype=bool)

        h_batch = Batch.from_data_list(h_list)
        t_batch = Batch.from_data_list(t_list)
        b_batch = Batch.from_data_list(b_list)
        rels = torch.zeros(len(keep_idx), dtype=torch.long)
        mask = np.zeros(len(pairs), dtype=bool)
        mask[np.asarray(keep_idx)] = True
        return h_batch, t_batch, rels, b_batch, mask

    # ABC surface

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        self._build_graphs(train)

        self._model = HDN_DDI(
            in_features=self.in_features,
            hidd_dim=self.hidd_dim,
            kge_dim=self.kge_dim,
            rel_total=1,
            heads_out_feat_params=list(self.heads_out_feat_params),
            blocks_params=list(self.blocks_params),
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
                h_b, t_b, rels, b_b, mask = self._make_pair_batch(batch)
                if h_b is None:
                    continue
                h_b = h_b.to(self.device)
                t_b = t_b.to(self.device)
                rels = rels.to(self.device)
                b_b = b_b.to(self.device)
                y = torch.tensor(
                    batch.loc[mask, "label"].to_numpy(),
                    dtype=torch.float32,
                    device=self.device,
                )
                opt.zero_grad(set_to_none=True)
                logits = self._model((h_b, t_b, rels, b_b))
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
    ) -> np.ndarray:
        if self._model is None or self._graphs is None:
            raise RuntimeError(
                "HDNDDIBaseline must be fitted (or loaded) before prediction."
            )
        self._model.eval()
        out = np.full(len(pairs), 0.5, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            h_b, t_b, rels, b_b, mask = self._make_pair_batch(batch)
            if h_b is None:
                continue
            h_b = h_b.to(self.device)
            t_b = t_b.to(self.device)
            rels = rels.to(self.device)
            b_b = b_b.to(self.device)
            logits = self._model((h_b, t_b, rels, b_b))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            out_slice = np.full(len(batch), 0.5, dtype=np.float32)
            out_slice[mask] = probs
            out[start : start + len(batch)] = out_slice
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump({"graphs": self._graphs, "missing": self._missing}, f)
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "hyperparameters": {
                    "in_features": self.in_features,
                    "hidd_dim": self.hidd_dim,
                    "kge_dim": self.kge_dim,
                    "heads_out_feat_params": list(self.heads_out_feat_params),
                    "blocks_params": list(self.blocks_params),
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
                    "n_drugs_with_graph": len(self._graphs),
                    "n_drugs_missing": len(self._missing),
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "HDNDDIBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        if "heads_out_feat_params" in hparams:
            hparams["heads_out_feat_params"] = tuple(hparams["heads_out_feat_params"])
        if "blocks_params" in hparams:
            hparams["blocks_params"] = tuple(hparams["blocks_params"])
        inst = cls(**hparams)
        with (p / "graphs.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._graphs = payload["graphs"]
        inst._missing = payload.get("missing", [])
        inst._model = HDN_DDI(
            in_features=inst.in_features,
            hidd_dim=inst.hidd_dim,
            kge_dim=inst.kge_dim,
            rel_total=1,
            heads_out_feat_params=list(inst.heads_out_feat_params),
            blocks_params=list(inst.blocks_params),
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()
        return inst
