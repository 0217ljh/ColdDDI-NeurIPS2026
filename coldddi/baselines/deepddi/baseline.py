"""DeepDDI thin adapter — implementation of :class:`BaselineModel`.

Wraps :class:`coldddi.baselines.deepddi.model.DeepDDIModel` (the MLP)
and :mod:`coldddi.baselines.deepddi.ssp_features` (the SSP feature
extractor) without re-implementing either. The adapter's job is just
to bridge :class:`PairDataset` → numpy/torch tensors and back.

This module is the **template** for how every other baseline (SSI-DDI,
DSN-DDI, …) gets wrapped: import the model, write a sub-100-line
``fit`` driving its native PyTorch loop, return predictions through a
shared ``ssp_lookup``-style helper.
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
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

from coldddi.baselines.base import BaselineModel, register, write_manifest
from coldddi.baselines.deepddi.model import DeepDDIModel
from coldddi.baselines.deepddi.ssp_features import (
    DEFAULT_SSP_DIM,
    build_ssp_artifacts,
    get_ssp_or_zero,
)

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset
    from coldddi.data.protocols import KnowledgeGraphProtocol


#: Paper-spec hyperparameters from Appendix C.1 Table 8 (the
#: "DeepDDI" row).  Class ``__init__`` defaults are deliberately
#: smoke-test values so unit tests stay fast; the paper-grade
#: configuration is materialised at run time by ``evaluate.py
#: --preset paper`` (default), which constructs the baseline with
#: these kwargs.
PAPER_HYPERPARAMS: dict[str, object] = {
    "ssp_dim":        50,
    "hidden_dim":     2048,
    "n_layers":       9,
    "dropout":        0.3,
    "learning_rate":  1e-3,
    # Paper specifies weight_decay=0; this is the optimizer's default,
    # but documenting it for paper-parity audits.
    "batch_size":     256,
    "n_epochs":       100,
}


@register("deepddi")
class DeepDDIBaseline(BaselineModel):
    """DeepDDI: SSP fingerprint + MLP, single-logit binary head.

    Modality: ``"mol"`` — SSP fingerprint from SMILES only, no KG
    channel. L6 dispatch produces KPS-F; KPS-mol / KPS-KG are NaN.

    Paper-grade hyperparameters live in :data:`PAPER_HYPERPARAMS`
    (App C.1 Table 8) and are auto-applied by
    ``evaluate.py --preset paper`` (default).  Class ``__init__``
    defaults below are smoke-test values for fast CI; pass paper
    kwargs explicitly or use ``--preset paper`` for reproduction.
    """

    VERSION = "1.0"
    modality = "mol"

    def __init__(
        self,
        *,
        ssp_dim: int = DEFAULT_SSP_DIM,
        hidden_dim: int = 2048,
        n_layers: int = 9,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        n_epochs: int = 10,
        device: str = "auto",
    ) -> None:
        self.ssp_dim = ssp_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = self._resolve_device(device)
        # Lazily populated by `fit` / `load`.
        self._model: DeepDDIModel | None = None
        self._ssp_artifacts: dict | None = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    # ------------------------------------------------------------------
    # SSP utilities
    # ------------------------------------------------------------------

    def _drug_smiles_dict(self, train: "PairDataset") -> dict[str, str]:
        if train.drugs is None:
            raise ValueError(
                "DeepDDI requires a `drugs` table on the PairDataset (with a "
                "`smiles` column). Load via `PairDataset.from_release_dir` "
                "after running Stage 1b, or pass a legacy bundle that includes "
                "`my_drugs_list` under `extra['kb']`."
            )
        if "smiles" not in train.drugs.columns:
            raise ValueError("PairDataset.drugs must contain a `smiles` column.")
        return {
            str(row["drugbank_id"]): "" if pd.isna(row["smiles"]) else str(row["smiles"])
            for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
        }

    def _build_ssp(self, train: "PairDataset") -> None:
        """Construct SSP artifacts from G1 training drugs (no G2 leakage)."""
        smiles = self._drug_smiles_dict(train)
        all_drugs = sorted(set(train.drugs["drugbank_id"].astype(str)))
        g1 = sorted(set(map(str, train.splits.g1_drugs)))
        if not g1:
            raise ValueError(
                "PairDataset.splits.g1_drugs is empty — cannot build SSP without a "
                "cold-start G1 reference set. Falling back to `all_drugs` would leak "
                "G2 into the PCA basis. Either rerun Stage 4 (build_splits) or pass a "
                "splits object with a populated G1."
            )
        self._ssp_artifacts = build_ssp_artifacts(
            smiles_dict=smiles,
            reference_drug_ids=g1,
            all_drug_ids=all_drugs,
            train_drug_ids_for_pca_fit=g1,
            n_components=self.ssp_dim,
        )

    def _pair_features(self, pairs: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ssp_artifacts is None:
            raise RuntimeError("DeepDDIBaseline.fit() must be called before predict_proba.")
        a = np.stack(
            [get_ssp_or_zero(self._ssp_artifacts, did) for did in pairs["drug_a_id"].astype(str)]
        )
        b = np.stack(
            [get_ssp_or_zero(self._ssp_artifacts, did) for did in pairs["drug_b_id"].astype(str)]
        )
        return (
            torch.from_numpy(a.astype(np.float32)),
            torch.from_numpy(b.astype(np.float32)),
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
        self._build_ssp(train)
        effective_dim = self._ssp_artifacts["n_components_effective"]
        n_missing_smiles = len(self._ssp_artifacts.get("drugs_without_smiles", []))
        if n_missing_smiles:
            print(
                f"[deepddi] {n_missing_smiles} drugs lacked parseable SMILES "
                f"and were zero-filled.",
                file=sys.stderr,
            )

        self._model = DeepDDIModel(
            ssp_dim=effective_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate)

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
                a, b = self._pair_features(batch)
                a = a.to(self.device)
                b = b.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                logits = self._model(a, b)
                loss = binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                opt.step()

            # Optional best-checkpoint selection on val_s2 AUC.
            if val is not None:
                auc = self._validate(val)
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_state = copy.deepcopy(self._model.state_dict())

        if best_state is not None:
            self._model.load_state_dict(best_state)

    @torch.no_grad()
    def _validate(self, val: "PairDataset") -> float:
        """Return ROC-AUC on val_s2 positives + their static negatives."""
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
            raise RuntimeError("DeepDDIBaseline must be fitted (or loaded) before prediction.")
        self._model.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            a, b = self._pair_features(batch)
            a = a.to(self.device)
            b = b.to(self.device)
            probs = torch.sigmoid(self._model(a, b)).detach().cpu().numpy()
            out[start : start + len(batch)] = probs
        return out

    def save(self, path: "Path | str") -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if self._model is None or self._ssp_artifacts is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "ssp.pkl").open("wb") as f:
            pickle.dump(self._ssp_artifacts, f)
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "hyperparameters": {
                    "ssp_dim": self.ssp_dim,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
                "environment": {
                    "sklearn_version": sklearn.__version__,
                    "torch_version": torch.__version__,
                },
                "ssp_metadata": {
                    "n_components_effective": int(
                        self._ssp_artifacts["n_components_effective"]
                    ),
                    "n_reference_drugs": len(
                        self._ssp_artifacts["reference_drug_ids"]
                    ),
                    "n_drugs_without_smiles": len(
                        self._ssp_artifacts.get("drugs_without_smiles", [])
                    ),
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "DeepDDIBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        inst = cls(**hparams)
        with (p / "ssp.pkl").open("rb") as f:
            inst._ssp_artifacts = pickle.load(f)
        effective_dim = inst._ssp_artifacts["n_components_effective"]
        inst._model = DeepDDIModel(
            ssp_dim=effective_dim,
            hidden_dim=inst.hidden_dim,
            n_layers=inst.n_layers,
            dropout=inst.dropout,
        ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        return inst
