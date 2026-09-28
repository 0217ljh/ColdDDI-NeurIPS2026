"""MKG-FENN thin adapter — implementation of :class:`BaselineModel`.

Wraps the four-channel KG-GNN (:class:`coldddi.baselines.mkg_fenn.model.MKGFENN`)
plus :mod:`coldddi.baselines.mkg_fenn.kg_builder` (KG construction from
the modern release schema). The four KGs are built from
:class:`PairDataset.kg`, the drug SMILES table, and the training positive
edges; the FusionLayer outputs 2 logits and we report ``softmax[:, 1]``
as the binary score, mirroring the upstream training recipe.
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.functional import cross_entropy

from coldddi.baselines.base import BaselineModel, register, write_manifest
from coldddi.baselines.mkg_fenn.kg_builder import build_all_kgs
from coldddi.baselines.mkg_fenn.model import MKGFENN

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset
    from coldddi.data.protocols import KnowledgeGraphProtocol


def _drug_smiles_dict(train: "PairDataset") -> dict[str, str]:
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "MKG-FENN requires `PairDataset.drugs` with a `smiles` column."
        )
    return {
        str(row["drugbank_id"]): "" if pd.isna(row["smiles"]) else str(row["smiles"])
        for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
    }


#: Paper-spec hyperparameters from Appendix C.1 Table 8 (MKG-FENN row).
PAPER_HYPERPARAMS: dict[str, object] = {
    "embedding_num":         128,        # paper "embedding=128"
    "neighbor_sample_size":  6,          # paper "neighbor sample=6"
    "dropout":               0.3,
    "learning_rate":         1e-2,
    "weight_decay":          1e-8,
    "batch_size":            256,
    "n_epochs":              50,
}


@register("mkg_fenn")
class MKGFENNBaseline(BaselineModel):
    """MKG-FENN: 4-channel KG-GNN + fusion head, binary cold-start adaptation.

    Modality: ``"mol+kg"`` — channels GNN1/GNN3 carry KG (entity,
    DDI topology); GNN2/GNN4 carry mol (Morgan FP, property).
    Channel mask exposed via ``predict_proba(pairs, mask_channel=
    "mol"|"kg")``. L6 dispatch produces KPS-F + KPS-mol + KPS-KG,
    all three populated.

    Paper-grade hyperparameters live in :data:`PAPER_HYPERPARAMS`
    (App C.1 Table 8) and are auto-applied by
    ``evaluate.py --preset paper`` (default).  Class ``__init__``
    defaults below are smoke-test values for fast CI.
    """

    VERSION = "1.0"
    modality = "mol+kg"

    def __init__(
        self,
        *,
        embedding_num: int = 64,
        neighbor_sample_size: int = 6,
        dropout: float = 0.3,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-8,
        batch_size: int = 256,
        n_epochs: int = 5,
        fp_radius: int = 2,
        fp_nbits: int = 512,
        n_bins: int = 10,
        device: str = "auto",
    ) -> None:
        self.embedding_num = embedding_num
        self.neighbor_sample_size = neighbor_sample_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.n_bins = n_bins
        self.device = self._resolve_device(device)
        self._model: MKGFENN | None = None
        self._dict1: dict[str, int] | None = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _build_dict1(self, train: "PairDataset") -> dict[str, int]:
        # Drug vocab covers every drug appearing in any split + every drug
        # in the drugs table — keeps cold-start drugs reachable at predict time.
        drug_ids: set[str] = set()
        for _name, df in train.splits.items():
            drug_ids.update(df["drug_a_id"].astype(str))
            drug_ids.update(df["drug_b_id"].astype(str))
        if train.drugs is not None and "drugbank_id" in train.drugs.columns:
            drug_ids.update(train.drugs["drugbank_id"].astype(str))
        return {did: idx for idx, did in enumerate(sorted(drug_ids))}

    def _pair_indices(self, pairs: pd.DataFrame) -> tuple[torch.Tensor, np.ndarray]:
        if self._dict1 is None:
            raise RuntimeError("MKGFENNBaseline.fit() must be called before predict_proba.")
        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()
        keep_idx, idx_pairs = [], []
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            ai = self._dict1.get(a)
            bi = self._dict1.get(b)
            if ai is None or bi is None:
                continue
            keep_idx.append(i)
            idx_pairs.append((ai, bi))
        if not keep_idx:
            return None, np.zeros(len(pairs), dtype=bool)
        idx_tensor = torch.tensor(idx_pairs, dtype=torch.long)
        mask = np.zeros(len(pairs), dtype=bool)
        mask[np.asarray(keep_idx)] = True
        return idx_tensor, mask

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
        self._dict1 = self._build_dict1(train)
        smiles = _drug_smiles_dict(train)
        kgs, tail_len, relation_len = build_all_kgs(
            kg=kg,
            drug_id2smiles=smiles,
            dict1=self._dict1,
            train_pos_df=train.splits.train,
            fp_radius=self.fp_radius,
            fp_nbits=self.fp_nbits,
            n_bins=self.n_bins,
        )
        drug_name = list(range(len(self._dict1)))

        self._model = MKGFENN(
            kgs=kgs,
            tail_len=tail_len,
            relation_len=relation_len,
            dict1=self._dict1,
            drug_name=drug_name,
            embedding_num=self.embedding_num,
            neighbor_sample_size=self.neighbor_sample_size,
            dropout=self.dropout,
            event_num=2,
        ).to(self.device)
        self._model.precompute_adj()
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
            self._model.precompute_adj()

            self._model.train()
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start : start + self.batch_size]
                idx_tensor, mask = self._pair_indices(batch)
                if idx_tensor is None:
                    continue
                # FusionLayer's BatchNorm1d cannot consume a size-1 batch
                # in training mode. Skip rather than crash.
                if idx_tensor.size(0) < 2:
                    continue
                idx_tensor = idx_tensor.to(self.device)
                y = torch.tensor(
                    batch.loc[mask, "label"].to_numpy(),
                    dtype=torch.long,
                    device=self.device,
                )
                opt.zero_grad(set_to_none=True)
                logits = self._model(idx_tensor)
                loss = cross_entropy(logits, y)
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
            :meth:`MKGFENN.forward`.  ``"kg"`` zeros GNN1+GNN3 for
            both drugs in each pair at fusion time (KPS-KG indicator);
            ``"mol"`` zeros GNN2+GNN4 (KPS-mol); ``None`` (default)
            produces the unmasked base prediction.
        """
        if self._model is None or self._dict1 is None:
            raise RuntimeError(
                "MKGFENNBaseline must be fitted (or loaded) before prediction."
            )
        self._model.eval()
        out = np.full(len(pairs), 0.5, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            idx_tensor, mask = self._pair_indices(batch)
            if idx_tensor is None:
                continue
            idx_tensor = idx_tensor.to(self.device)
            logits = self._model(idx_tensor, mask_channel=mask_channel)
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            out_slice = np.full(len(batch), 0.5, dtype=np.float32)
            out_slice[mask] = probs
            out[start : start + len(batch)] = out_slice
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._dict1 is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "kg_state.pkl").open("wb") as f:
            pickle.dump(
                {
                    "dict1": self._dict1,
                    "kgs": {
                        "dataset1": self._model.gnn1.kg,
                        "dataset2": self._model.gnn2.kg,
                        "dataset3": self._model.gnn3.kg,
                        "dataset4": self._model.gnn4.kg,
                    },
                    "tail_len": {
                        "dataset1": self._model.gnn1.ent_embed.num_embeddings - 1,
                        "dataset2": self._model.gnn2.ent_embed.num_embeddings,
                        "dataset3": self._model.gnn3.ent_embed.num_embeddings,
                        "dataset4": self._model.gnn4.ent_embed.num_embeddings,
                    },
                    "relation_len": {
                        "dataset1": self._model.gnn1.rela_embed.num_embeddings - 1,
                        "dataset2": self._model.gnn2.rela_embed.num_embeddings,
                        "dataset3": self._model.gnn3.rela_embed.num_embeddings,
                        "dataset4": self._model.gnn4.rela_embed.num_embeddings,
                    },
                },
                f,
            )
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "hyperparameters": {
                    "embedding_num": self.embedding_num,
                    "neighbor_sample_size": self.neighbor_sample_size,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "fp_radius": self.fp_radius,
                    "fp_nbits": self.fp_nbits,
                    "n_bins": self.n_bins,
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "sklearn_version": sklearn.__version__,
                },
                "kg_metadata": {"n_drugs": len(self._dict1)},
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "MKGFENNBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        inst = cls(**hparams)
        with (p / "kg_state.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._dict1 = payload["dict1"]
        kgs = payload["kgs"]
        tail_len = payload["tail_len"]
        relation_len = payload["relation_len"]
        drug_name = list(range(len(inst._dict1)))
        inst._model = MKGFENN(
            kgs=kgs,
            tail_len=tail_len,
            relation_len=relation_len,
            dict1=inst._dict1,
            drug_name=drug_name,
            embedding_num=inst.embedding_num,
            neighbor_sample_size=inst.neighbor_sample_size,
            dropout=inst.dropout,
            event_num=2,
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()
        return inst
