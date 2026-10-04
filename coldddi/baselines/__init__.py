"""Baseline implementations for ColdDDI.

Submodules register their models on import. :func:`load_baseline` reads
the checkpoint's ``manifest.json`` to select the loader.
"""

from __future__ import annotations

from coldddi.baselines.base import (
    BASELINE_MANIFEST_FILENAME,
    NAME_TO_MODULE,
    BaselineModel,
    ensure_imported,
    list_baselines,
    load_baseline,
    register,
)

__all__ = [
    "BASELINE_MANIFEST_FILENAME",
    "BaselineModel",
    "NAME_TO_MODULE",
    "ensure_imported",
    "list_baselines",
    "load_baseline",
    "register",
]
