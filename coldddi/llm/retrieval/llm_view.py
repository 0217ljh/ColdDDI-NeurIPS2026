"""``PairDataset`` (+ optional retrieval artifacts) → list of per-pair
sample dicts ready for
:func:`coldddi.llm.prompts.binary_cls.build_binary_prompt`.

The output dict shape mirrors the one the legacy collator constructs::

    {
        "drug_a_id": str, "drug_b_id": str,
        "drug_a_name": str, "drug_b_name": str,
        "drugA_name": str, "drugB_name": str,     # legacy aliases
        "label": int (0 or 1),
        # Optional, present when subgraph_map is supplied:
        "subgraph_1hop": {"neighbors": {entity_type: {"A": [...], "B": [...]}}},
        # Optional, present when fewshot_map is supplied:
        "fewshot_samples":  [(score, ref_a_id, ref_b_id, label_str), ...],
        "fewshot_metadata": [{"shared_QA_CA": [...], ...}, ...],
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset
    from coldddi.llm.retrieval.kg_subgraph import SubgraphMap


def _drug_id2name(ds: "PairDataset") -> dict[str, str]:
    if ds.drugs is None or "name" not in ds.drugs.columns:
        return {}
    out: dict[str, str] = {}
    for did, name in zip(
        ds.drugs["drugbank_id"].astype(str), ds.drugs["name"]
    ):
        if pd.isna(name):
            continue
        out[did] = str(name)
    return out


def to_llm_samples(
    pairs: pd.DataFrame,
    labels: np.ndarray | list[int] | None,
    *,
    ds: "PairDataset",
    subgraph_map: "SubgraphMap | None" = None,
    fewshot_map: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Convert ``(pairs, labels)`` rows into LLM sample dicts.

    Parameters
    ----------
    pairs
        DataFrame with at least the columns ``drug_a_id`` and
        ``drug_b_id``. Extra columns are ignored.
    labels
        Per-row 0/1 labels (positive=1, negative=0). May be ``None``
        for pure inference. When ``None``, the ``label`` field is
        omitted from the sample dict — :func:`build_binary_prompt`
        then renders the assistant slot as a bare leading space
        (``" "``) since ``sample.get("label", "")`` returns the empty
        string. The model is therefore prompted to generate the
        answer token, instead of being trained on a fixed one.
    ds
        The :class:`PairDataset` the pairs came from; used to look up
        drug names from ``ds.drugs``.
    subgraph_map
        Optional :class:`SubgraphMap`. When supplied, each sample
        carries a ``subgraph_1hop`` block for P3 / P4 / R* prompts.
    fewshot_map
        Optional ``{(drug_a_id, drug_b_id): {"fewshot_samples": [...],
        "fewshot_metadata": [...]}}``. When supplied, the relevant keys
        are copied into the sample dict for P2 / P5 prompts.
    """
    id2name = _drug_id2name(ds)
    a_ids = pairs["drug_a_id"].astype(str).to_numpy()
    b_ids = pairs["drug_b_id"].astype(str).to_numpy()

    if labels is not None:
        labels_arr = np.asarray(labels, dtype=int)
        if len(labels_arr) != len(a_ids):
            raise ValueError(
                f"labels length ({len(labels_arr)}) must match pairs "
                f"length ({len(a_ids)})."
            )

    out: list[dict] = []
    for i in range(len(a_ids)):
        a_id, b_id = a_ids[i], b_ids[i]
        a_name = id2name.get(a_id, "")
        b_name = id2name.get(b_id, "")
        sample = {
            "drug_a_id": a_id,
            "drug_b_id": b_id,
            "drug_a_name": a_name,
            "drug_b_name": b_name,
            # Legacy aliases used by the zero-shot branch in the prompt
            # builder (``_format_pair`` reads drugA_name/drugB_name first).
            "drugA_name": a_name,
            "drugB_name": b_name,
        }
        if labels is not None:
            sample["label"] = int(labels_arr[i])
        if subgraph_map is not None:
            sample["subgraph_1hop"] = subgraph_map.get_neighbors_block(
                a_id, b_id
            )
        if fewshot_map is not None:
            fs = fewshot_map.get((a_id, b_id))
            if fs is not None:
                if "fewshot_samples" in fs:
                    sample["fewshot_samples"] = list(fs["fewshot_samples"])
                if "fewshot_metadata" in fs:
                    sample["fewshot_metadata"] = list(fs["fewshot_metadata"])
        out.append(sample)
    return out


__all__ = ["to_llm_samples"]
