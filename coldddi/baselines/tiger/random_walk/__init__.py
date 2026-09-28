"""Random-walk subgraph sampling for TIGER.

Byte-equivalent port of upstream ``Code-Released/baseline/TIGER/randomWalk/``
— renamed to ``random_walk`` for snake_case parity with the rest of
``coldddi``.
"""

from coldddi.baselines.tiger.random_walk.node2vec import Node2vec

__all__ = ["Node2vec"]
