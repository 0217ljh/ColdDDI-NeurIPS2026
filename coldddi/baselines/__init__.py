"""Baseline implementations for ColdDDI.

Each concrete baseline lives in its own submodule (e.g.
``coldddi.baselines.deepddi``) and registers itself on import via
:func:`register`. The :func:`load_baseline` dispatcher reads the
checkpoint's ``manifest.json`` to route to the correct subclass.
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
