"""Retrieval / augmentation modules for the LLM stack.

Each submodule turns a :class:`PairDataset` (and its
:class:`KnowledgeGraph`) into a derived structure that prompt builders
consume:

* :mod:`kg_subgraph` — per-drug one-hop subgraph (Top-k or Full).
* :mod:`llm_view`    — per-pair sample dicts ready for
  :func:`coldddi.llm.prompts.binary_cls.build_binary_prompt`.

P2 / P5 few-shot retrieval lives alongside:

* :mod:`fewshot_smiles` — Morgan-FP similarity few-shot (P2).
* :mod:`fewshot_2hop`   — KG-shared-entity 2-hop few-shot (P5).
"""

from __future__ import annotations

from coldddi.llm.retrieval.fewshot_2hop import build_fewshot_2hop_map
from coldddi.llm.retrieval.fewshot_smiles import build_fewshot_smiles_map
from coldddi.llm.retrieval.kg_subgraph import (
    SubgraphMap,
    SUBGRAPH_ENTITY_TYPES,
    build_subgraph_map,
)
from coldddi.llm.retrieval.llm_view import to_llm_samples

__all__ = [
    "SubgraphMap",
    "SUBGRAPH_ENTITY_TYPES",
    "build_subgraph_map",
    "build_fewshot_smiles_map",
    "build_fewshot_2hop_map",
    "to_llm_samples",
]
