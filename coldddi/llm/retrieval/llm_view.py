"""Convert dataset pairs and retrieval results to prompt-builder samples.

Samples contain drug IDs, names (including drugA_name/drugB_name aliases)
and optional binary labels. Retrieval adds ``subgraph_1hop``,
``fewshot_samples`` and ``fewshot_metadata`` when supplied.
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
    """Convert pair rows to LLM samples using drug names from ``ds.drugs``.

    ``pairs`` must contain drug_a_id and drug_b_id; other columns are ignored.
    ``labels`` contains aligned binary labels, or None to omit the label field.
    Without a label, the prompt builder's default assistant content is one space.

    ``subgraph_map`` adds one-hop context. ``fewshot_map`` supplies examples
    and metadata keyed by (drug_a_id, drug_b_id).
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
            # Name aliases used by the zero-shot prompt branch.
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
