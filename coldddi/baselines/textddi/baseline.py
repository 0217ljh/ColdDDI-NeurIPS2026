"""TextDDI adapter — paper-spec text-encoded DDI classifier.

Port of upstream
``Code-Released/baseline/TextDDI/train_custom_bundle.py``
(the exact script that produced ColdDDI's paper Table-6 TextDDI numbers).

Architecture
------------
``RobertaForSequenceClassification`` with ``num_labels=2`` (the same
model class the upstream RL pipeline also uses for its environment
classifier — see ``models/roberta_env.py``).

The paper's PPO-RL **snippet selector** (``models/policy.py`` +
``train_ppo_drugbank.py``) is **not** re-run here.  The bundle script
the paper actually ran consumes the PPO-selected snippets through a
pre-computed JSON cache (``DDI_dict_action_roberta.json``, ~34 MB,
covers 1,710 DrugBank drugs).  At paper-grade fidelity you point
``ddi_dict_path=`` at that file.  Without it, this adapter falls back
to building drug descriptions from the bundle's ``extra['kb']`` (or
the modern ``train.kg.name_dict()`` for release-dir datasets) — taking
at most 3 entity names TOTAL per drug across targets / enzymes /
transporters / carriers — and then to the dataset's ``drugs.name``
column, mirroring upstream's three-tier priority exactly.

Prompt format (verbatim from upstream ``build_prompt_tokens``)::

    {drug1_name}: {drug1_desc_truncated}
    {drug2_name}: {drug2_desc_truncated}
    In the above context, we can predict that the drug-drug interactions
    between {drug1_name} and {drug2_name} is that:

Description truncation is per-drug via the loaded tokenizer with
``max_length = floor(200 * max_length / 512)``.  When neither
description survives truncation, the prompt collapses to
``{name1} {name2} {query}`` (upstream fallback at
``train_custom_bundle.py:422-424``).

Default backbone is ``roberta-base`` per paper Appendix C.1
(:data:`DEFAULT_BACKBONE`).  Test/CI fixtures that don't need a real
pretrained model construct the baseline with
``backbone=SMOKE_BACKBONE`` (a tiny randomly-initialised DistilBert
stub) to keep CI fast and offline-friendly.
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

#: Paper-faithful backbone identifier (Appendix C.1 Table 8:
#: TextDDI uses ``RoBERTa-base``, 124.6M params, max_length=256).
#: This is what :data:`PAPER_HYPERPARAMS` carries through into
#: ``evaluate.py --preset paper`` (the default CLI surface), NOT
#: the class ``__init__`` default — see the SMOKE_BACKBONE comment
#: below for the rationale.  Downloads ~500 MB from HuggingFace
#: on first use by paper-grade runs.
DEFAULT_BACKBONE: str = "roberta-base"

#: Tiny randomly-initialised stub used as the class ``__init__``
#: default so direct Python construction (``TextDDIBaseline()``)
#: stays fast / offline for unit tests.  Paper-grade runs go
#: through ``evaluate.py --preset paper`` which forwards
#: ``PAPER_HYPERPARAMS["backbone"] = DEFAULT_BACKBONE`` ("roberta-base")
#: onto the constructor.  This asymmetry mirrors the other seven
#: baselines (smoke __init__ defaults + paper preset materialisation).
#:
#: Pre-audit the class default WAS this stub already; the C1-TextDDI
#: fix landed RoBERTa-base as the default which violated the smoke-
#: default invariant codex flagged on the broader C1 review.  Now
#: SMOKE_BACKBONE is restored as the class default and
#: PAPER_HYPERPARAMS carries RoBERTa-base through the preset path.
SMOKE_BACKBONE: str = "hf-internal-testing/tiny-random-DistilBertModel"

#: Paper-spec hyperparameters from Appendix C.1 Table 8 (TextDDI row).
#: Materialised at run time by ``evaluate.py --preset paper`` (default).
#: Paper: backbone=RoBERTa-base, max_length=256, AdamW LR=1e-5,
#: WD=1e-6, batch=32, epochs=30.
PAPER_HYPERPARAMS: dict[str, object] = {
    "backbone":       DEFAULT_BACKBONE,   # = "roberta-base"
    "max_length":     256,
    "learning_rate":  1e-5,
    "weight_decay":   1e-6,
    "batch_size":     32,
    "n_epochs":       30,
}

#: kb relation categories to scan for drug descriptions.  Order is the
#: priority within a single drug — we append entities from targets,
#: then enzymes, etc., then cap at ``_KB_MAX_ENTITIES`` TOTAL across all
#: categories (matches upstream ``train_custom_bundle.py:337``).
_KB_CATEGORIES = ("targets", "enzymes", "transporters", "carriers")
_KB_MAX_ENTITIES = 3


# ── Description cache (3-tier priority) ──────────────────────────────

def _kb_descriptions(kb: dict, all_drug_ids: set[str]) -> dict[str, str]:
    """For every drug, return at most ``_KB_MAX_ENTITIES`` entity names
    concatenated (taken across all categories in ``_KB_CATEGORIES``).
    Returns ``{drug_id: text}``.

    Handles all 3 kb layouts the release ships with:

    1. Upstream-canonical ``kb["targets"]`` etc. as ``{drug_id: [entities]}``
       dicts (what ``train_custom_bundle.py:316-338`` expects).
    2. Legacy bundle ``kb["my_target_list"]`` etc. as pandas DataFrames
       with ``drugbank_id`` + ``target_name`` columns.
    3. Even-older ``kb["dbid_2_targets"]`` etc. as flat ``{drug_id: [...]}``
       dicts under a different key name.

    Without this multi-schema support the kb fallback would no-op on
    every release dataset we actually ship, leaving every drug to fall
    through to name-only.
    """
    if not isinstance(kb, dict):
        return {}
    import pandas as _pd  # local — pandas already imported at module top

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
        # Prefer the human-readable name column (``target_name`` /
        # ``enzyme_name`` / ...) over the raw entity id (``target_id``
        # / ``BE000...``).  Legacy tables list both; picking ``_id``
        # by accident would put unhelpful identifiers into the prompt.
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

    1. **DDI_dict_action_roberta.json** — if ``ddi_dict_path`` points
       to a readable file, use its per-drug ``"name"`` + ``"description"``
       (or ``"sent_list"`` fallback).  These are the PPO-policy-selected
       snippets used in the paper.
    2. **bundle.extra['kb']** — concatenate the first 3 target / enzyme /
       transporter / carrier entity names per drug.
    3. **dataset.drugs.name + empty description** — the catch-all so
       every drug at least gets a human-readable name.
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

    # 1) DDI dict — if the caller explicitly supplies a path we treat
    # failures as fatal (silent fallback would mask paper-repro bugs).
    # If the path is None we just skip this tier without complaint.
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

    # 2) kb fallback.  Modern ``from_release_dir`` datasets have
    # ``legacy_bundle is None`` but expose the same per-relation entity
    # tables via ``train.kg.name_dict(...)`` — we materialise a kb-shaped
    # dict from it so ``_kb_descriptions`` can ingest both paths
    # through the same code path.  Legacy bundles still take
    # precedence when both are present.
    kb: dict = {}
    if train.legacy_bundle is not None:
        kb = (train.legacy_bundle.extra.get("kb", {}) or {})
    if not kb and getattr(train, "kg", None) is not None:
        kb = {}
        for cat in _KB_CATEGORIES:
            # The modern ``KnowledgeGraph.name_dict`` API takes the
            # SINGULAR edge type ("target"), while ``_KB_CATEGORIES``
            # uses the legacy plural keys ("targets") so that the same
            # ``_kb_descriptions`` ingestor works for both schemas.
            # Convert here.
            singular = cat[:-1]
            try:
                kb[cat] = train.kg.name_dict(singular)
            except ValueError:
                # ``_table_for`` raises ValueError on unknown edge type;
                # genuinely missing category — skip cleanly.
                pass
    kb_desc = _kb_descriptions(kb, all_drug_ids)

    # 3) merge with priority
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


# ── Adapter ──────────────────────────────────────────────────────────

@register("textddi")
class TextDDIBaseline(BaselineModel):
    """TextDDI: paper-spec text-encoded drug pair → RoBERTa CLS → binary head.

    Uses the upstream ``train_custom_bundle.py`` prompt format and
    three-tier description cache (DDI dict → kb → name fallback).
    The PPO snippet-selector is NOT re-trained at fit time; if you
    want the paper's policy-selected snippets, point ``ddi_dict_path``
    at the upstream ``DDI_dict_action_roberta.json`` (34 MB, ships
    separately per DrugBank-license policy).
    """

    VERSION = "2.0"  # 2.0 = paper-prompt port; 1.0 was BERT-on-strings
    # Modality: ``"text"`` — RoBERTa encodes drug name + description
    # only, no separable mol or KG channel. L6 dispatch produces
    # KPS-F; KPS-mol / KPS-KG come back as NaN rows.
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

    # ── Prompt construction (upstream-equivalent) ──────────────────

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
        # Unknown drugs (not in the dataset's drugs table) → return None
        # so predict_proba's 0.5 default kicks in, matching the rest of
        # the BaselineModel ABC convention.  Upstream's permissive
        # drug_id-as-fallback only applies inside the known cache.
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
        # Symmetric fallback: emit the full template whenever EITHER drug
        # has a non-empty description (so a known kb description for B
        # is still surfaced when A has none).  Only collapse to the
        # name-only template when BOTH descriptions are empty.
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
        self._desc_cache = _build_description_cache(train, self.ddi_dict_path)
        self._ensure_backbone()
        # bg_max_length depends on the tokenizer; recompute now that
        # we've loaded one (the constructor estimate is conservative).
        self._bg_max_length = int(200 * self.max_length / 512)

        # Upstream uses Adam + grouped weight-decay (no decay for bias /
        # LayerNorm.weight).  We mirror that here for paper-grade parity.
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
                # Upstream also clips grad-norm at 1.0; mirror.
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
        # Persist the (name, desc) cache so load() can reproduce
        # predictions without re-running _build_description_cache.
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
