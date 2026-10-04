"""TextDDI adapter for text-encoded binary DDI classification.

Adapted from ``Code-Released/baseline/TextDDI/train_custom_bundle.py``.
The paper preset uses RoBERTa with two labels, as in upstream
``models/roberta_env.py``. Direct construction uses ``SMOKE_BACKBONE``,
a tiny random DistilBert model for smoke tests.

The PPO snippet selector (``models/policy.py`` and ``train_ppo_drugbank.py``)
is not run here. Pass its cached ``DDI_dict_action_roberta.json`` via
``ddi_dict_path``. Otherwise, use up to three entity names per drug across
targets, enzymes, transporters, and carriers from the bundle KB or
``train.kg.name_dict()``. The final fallback is the drug name alone.

Prompt format (verbatim from upstream ``build_prompt_tokens``)::

    {drug1_name}: {drug1_desc_truncated}
    {drug2_name}: {drug2_desc_truncated}
    In the above context, we can predict that the drug-drug interactions
    between {drug1_name} and {drug2_name} is that:

Each description is truncated to ``floor(200 * max_length / 512)`` tokens.
If both are empty, use ``{name1} {name2} {query}`` as in upstream
``train_custom_bundle.py:422-424``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import torch
from sklearn.metrics import roc_auc_score
from torch import optim

from coldddi.baselines.base import BaselineModel, register, write_manifest

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset
    from coldddi.data.protocols import KnowledgeGraphProtocol

#: Paper-preset backbone (Appendix C.1 Table 8), not the constructor default.
#: HuggingFace downloads the model on first use unless it is cached.
DEFAULT_BACKBONE: str = "roberta-base"

#: Constructor default for smoke tests. The paper preset selects RoBERTa-base.
SMOKE_BACKBONE: str = "hf-internal-testing/tiny-random-DistilBertModel"

#: Paper-spec hyperparameters from Appendix C.1 Table 8 (TextDDI row).
#: Applied by ``evaluate.py --preset paper``. Training uses grouped Adam decay.
PAPER_HYPERPARAMS: dict[str, object] = {
    "backbone":       DEFAULT_BACKBONE,
    "max_length":     256,
    "learning_rate":  1e-5,
    "weight_decay":   1e-6,
    "batch_size":     32,
    "n_epochs":       30,
}

#: Description categories in priority order, capped at _KB_MAX_ENTITIES
#: across all categories (upstream ``train_custom_bundle.py:337``).
_KB_CATEGORIES = ("targets", "enzymes", "transporters", "carriers")
_KB_MAX_ENTITIES = 3


# Description cache (three-tier priority)

def _kb_descriptions(kb: dict, all_drug_ids: set[str]) -> dict[str, str]:
    """Return ``{drug_id: text}`` with at most _KB_MAX_ENTITIES names per drug.

    Read categories in _KB_CATEGORIES order. Supported schemas:

    * ``kb["targets"]``: ``{drug_id: [entities]}`` mappings, as in upstream
      ``train_custom_bundle.py:316-338``.
    * ``kb["my_target_list"]``: DataFrames with drug and entity columns.
    * ``kb["dbid_2_targets"]``: legacy ``{drug_id: [entities]}`` mappings.
    """
    if not isinstance(kb, dict):
        return {}
    import pandas as _pd

    kb_desc: dict[str, list[str]] = {}

    def _append_dict(rel_data, _cat):
        if not isinstance(rel_data, dict):
            return
        for drug_id, entities in rel_data.items():
            drug_id = str(drug_id)
            if drug_id not in all_drug_ids:
                continue
            if isinstance(entities, str):
                entities = [entities]
            for ent in (entities or []):
                ent = str(ent).strip()
                if not ent:
                    continue
                kb_desc.setdefault(drug_id, []).append(ent)

    def _append_dataframe(df, cat):
        if not isinstance(df, _pd.DataFrame) or df.empty:
            return
        drug_col = next(
            (c for c in ("drugbank_id", "drug_id", "d1") if c in df.columns),
            None,
        )
        if drug_col is None:
            return
        # Prefer entity names over IDs in descriptions.
        cat_singular = cat[:-1]                       # "targets" → "target"
        preferred = [
            f"{cat_singular}_name", "name",
            f"{cat_singular}_id", "id",
        ]
        ent_col = next(
            (c for c in preferred if c in df.columns and c != drug_col),
            None,
        )
        if ent_col is None:
            ent_col = next(
                (c for c in df.columns if c != drug_col), None,
            )
        if ent_col is None:
            return
        for drug_id, ent_val in zip(df[drug_col].astype(str), df[ent_col]):
            drug_id = str(drug_id)
            if drug_id not in all_drug_ids:
                continue
            if ent_val is None or (isinstance(ent_val, float) and pd.isna(ent_val)):
                continue
            ent = str(ent_val).strip()
            if not ent:
                continue
            kb_desc.setdefault(drug_id, []).append(ent)

    for cat in _KB_CATEGORIES:
        # Schema 1: canonical dict.
        _append_dict(kb.get(cat), cat)
        # Schema 2: my_<cat>_list DataFrame (e.g. ``my_target_list``).
        cat_singular = cat[:-1]                       # "targets" → "target"
        _append_dataframe(kb.get(f"my_{cat_singular}_list"), cat)
        # Schema 3: dbid_2_<cat> dict (e.g. ``dbid_2_targets``).
        _append_dict(kb.get(f"dbid_2_{cat}"), cat)

    return {
        d: " ".join(parts[:_KB_MAX_ENTITIES])
        for d, parts in kb_desc.items()
    }


def _build_description_cache(
    train: "PairDataset",
    ddi_dict_path: str | None,
) -> dict[str, tuple[str, str]]:
    """Build a ``{drug_id: (drug_name, description)}`` cache.

    Priority (matches upstream ``train_custom_bundle.py:289-364``):

    1. ``DDI_dict_action_roberta.json``: cached name and PPO-selected description,
       falling back to ``sent_list`` when the description is empty.
    2. Bundle KB or ``train.kg``: the first three entity names per drug.
    3. Dataset drug name (or ID) with an empty description.

    An explicitly supplied missing or invalid JSON file raises an error.
    """
    if train.drugs is None:
        raise ValueError(
            "TextDDI requires `PairDataset.drugs` to be populated "
            "(needs at least drugbank_id + name columns)."
        )
    drugs_df = train.drugs
    all_drug_ids: set[str] = {str(d) for d in drugs_df["drugbank_id"].astype(str)}
    name_lookup: dict[str, str] = {}
    for _, row in drugs_df.iterrows():
        did = str(row["drugbank_id"])
        name = row.get("name") if "name" in drugs_df.columns else did
        name_lookup[did] = str(name) if not pd.isna(name) else did

    # A supplied cache must load successfully; None skips this tier.
    ddi_dict: dict = {}
    if ddi_dict_path:
        p = Path(ddi_dict_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"ddi_dict_path={ddi_dict_path!r} does not exist. Pass "
                "ddi_dict_path=None to skip this tier."
            )
        with p.open("r", encoding="utf-8") as f:
            try:
                ddi_dict = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse ddi_dict at {ddi_dict_path!r}: {e}. "
                    "If the file is corrupt or you do not have it, pass "
                    "ddi_dict_path=None to fall back to kb / name tiers."
                ) from e

    # Prefer the legacy KB; otherwise adapt release KG names to its schema.
    kb: dict = {}
    if train.legacy_bundle is not None:
        kb = (train.legacy_bundle.extra.get("kb", {}) or {})
    if not kb and getattr(train, "kg", None) is not None:
        kb = {}
        for cat in _KB_CATEGORIES:
            # name_dict expects singular edge types, unlike the legacy keys.
            singular = cat[:-1]
            try:
                kb[cat] = train.kg.name_dict(singular)
            except ValueError:
                # Skip unsupported edge types.
                pass
    kb_desc = _kb_descriptions(kb, all_drug_ids)

    cache: dict[str, tuple[str, str]] = {}
    n_dict = n_kb = n_name = 0
    for did in all_drug_ids:
        if did in ddi_dict:
            entry = ddi_dict[did]
            name = entry.get("name", name_lookup.get(did, did))
            desc = entry.get("description", "")
            if not desc:
                desc = " ".join(entry.get("sent_list", []))
            cache[did] = (name, desc)
            n_dict += 1
        elif did in kb_desc:
            cache[did] = (name_lookup.get(did, did), kb_desc[did])
            n_kb += 1
        else:
            cache[did] = (name_lookup.get(did, did), "")
            n_name += 1
    return cache


# Adapter

@register("textddi")
class TextDDIBaseline(BaselineModel):
    """TextDDI: encode drug-pair descriptions for binary classification.

    Uses the upstream ``train_custom_bundle.py`` prompt format and
    three-tier description cache (DDI dict → kb → name fallback).
    No PPO selector is trained here. Set ``ddi_dict_path`` to upstream
    ``DDI_dict_action_roberta.json`` for policy-selected snippets; the cache
    is distributed separately under the DrugBank license policy.
    """

    VERSION = "2.0"
    # Text has no separable mol/KG masks: L6 gives KPS-F, with NaN channel scores.
    modality = "text"

    def __init__(
        self,
        *,
        backbone: str = SMOKE_BACKBONE,
        max_length: int = 256,
        ddi_dict_path: str | None = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        batch_size: int = 16,
        n_epochs: int = 1,
        device: str = "auto",
    ) -> None:
        self.backbone = backbone
        self.max_length = max_length
        self.ddi_dict_path = ddi_dict_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = self._resolve_device(device)
        self._model = None
        self._tokenizer = None
        self._desc_cache: dict[str, tuple[str, str]] | None = None
        # Per-drug description token budget — upstream formula.
        self._bg_max_length: int = int(200 * max_length / 512)

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _ensure_backbone(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.backbone, num_labels=2,
        ).to(self.device)

    # Prompt construction

    def _truncate_desc(self, desc: str) -> str:
        if not desc:
            return ""
        tok = self._tokenizer(
            desc,
            add_special_tokens=False,
            return_token_type_ids=False,
            truncation=True,
            max_length=self._bg_max_length,
        )
        text = self._tokenizer.decode(tok["input_ids"])
        if text and text[-1] not in ".!?":
            text = text.rstrip() + "."
        return text

    def _build_prompt(self, a_id: str, b_id: str) -> str | None:
        if self._desc_cache is None:
            return None
        # Unknown drugs retain predict_proba's default score of 0.5.
        if a_id not in self._desc_cache or b_id not in self._desc_cache:
            return None
        name_a, desc_a = self._desc_cache[a_id]
        name_b, desc_b = self._desc_cache[b_id]
        desc_a_t = self._truncate_desc(desc_a)
        desc_b_t = self._truncate_desc(desc_b)
        query = (
            f"In the above context, we can predict that the drug-drug interactions "
            f"between {name_a} and {name_b} is that: "
        )
        # Use names alone only when both descriptions are empty.
        if desc_a_t or desc_b_t:
            return f"{name_a}: {desc_a_t} {name_b}: {desc_b_t} {query}"
        return f"{name_a} {name_b} {query}"

    def _encode(self, pairs: pd.DataFrame):
        keep_idx, prompts = [], []
        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            prompt = self._build_prompt(a, b)
            if prompt is None:
                continue
            keep_idx.append(i)
            prompts.append(prompt)
        if not keep_idx:
            return None, np.zeros(len(pairs), dtype=bool)

        enc = self._tokenizer(
            prompts,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        mask = np.zeros(len(pairs), dtype=bool)
        mask[np.asarray(keep_idx)] = True
        return enc, mask

    # ABC surface

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        self._desc_cache = _build_description_cache(train, self.ddi_dict_path)
        self._ensure_backbone()
        # Refresh the per-drug token budget from max_length.
        self._bg_max_length = int(200 * self.max_length / 512)

        # Match upstream Adam grouping: no decay for bias or LayerNorm.weight.
        no_decay = ("bias", "LayerNorm.weight")
        grouped = [
            {
                "params": [
                    p for n, p in self._model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self._model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = optim.Adam(grouped, lr=self.learning_rate)

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
                enc, mask = self._encode(batch)
                if enc is None:
                    continue
                enc = {k: v.to(self.device) for k, v in enc.items()}
                y = torch.tensor(
                    batch.loc[mask, "label"].to_numpy(),
                    dtype=torch.long,
                    device=self.device,
                )
                opt.zero_grad(set_to_none=True)
                out = self._model(**enc, labels=y)
                # Match upstream's gradient-norm limit of 1.0.
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
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
        if self._model is None or self._tokenizer is None or self._desc_cache is None:
            raise RuntimeError(
                "TextDDIBaseline must be fitted (or loaded) before prediction."
            )
        self._model.eval()
        out = np.full(len(pairs), 0.5, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            enc, mask = self._encode(batch)
            if enc is None:
                continue
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            out_slice = np.full(len(batch), 0.5, dtype=np.float32)
            out_slice[mask] = probs
            out[start : start + len(batch)] = out_slice
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(out / "model")
        self._tokenizer.save_pretrained(out / "tokenizer")
        # Save descriptions so prediction does not need the original data sources.
        cache_serialisable = {
            d: list(nd) for d, nd in self._desc_cache.items()
        }
        with (out / "desc_cache.json").open("w", encoding="utf-8") as f:
            json.dump(cache_serialisable, f, ensure_ascii=False)
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "hyperparameters": {
                    "backbone": self.backbone,
                    "max_length": self.max_length,
                    "ddi_dict_path": self.ddi_dict_path,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "sklearn_version": sklearn.__version__,
                },
                "metadata": {
                    "n_drugs_in_cache": len(self._desc_cache),
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "TextDDIBaseline":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        inst = cls(**hparams)
        inst._tokenizer = AutoTokenizer.from_pretrained(p / "tokenizer")
        inst._model = AutoModelForSequenceClassification.from_pretrained(
            p / "model", num_labels=2,
        ).to(inst.device)
        with (p / "desc_cache.json").open("r", encoding="utf-8") as f:
            raw = json.load(f)
        inst._desc_cache = {d: tuple(nd) for d, nd in raw.items()}
        inst._bg_max_length = int(200 * inst.max_length / 512)
        return inst
