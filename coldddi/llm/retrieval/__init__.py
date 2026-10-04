"""KG subgraphs and P2/P5 few-shot retrieval for binary DDI prompts.

``llm_view`` combines these structures with ``PairDataset`` rows to build
samples consumed by ``build_binary_prompt``.
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
