"""Common ABC and registry for the eight ColdDDI baselines.

Models manage their own :class:`PairDataset` batching and may ignore ``kg``.
:meth:`BaselineModel.predict_proba` returns a 1-D array of P(positive) for
``DataFrame[drug_a_id, drug_b_id]``. Checkpoints may use any layout but must
include ``manifest.json`` for :func:`load_baseline`. Subclasses register
by name with :func:`register`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from coldddi.data.dataset import PairDataset
    from coldddi.data.protocols import KnowledgeGraphProtocol


#: Checkpoint manifest used by :func:`load_baseline` to select the subclass.
BASELINE_MANIFEST_FILENAME: str = "manifest.json"

#: Baseline name → subclass.
_REGISTRY: dict[str, type["BaselineModel"]] = {}

#: Name → import path for lazy registration without loading all dependencies.
NAME_TO_MODULE: dict[str, str] = {
    "deepddi": "coldddi.baselines.deepddi",
    "emergnn": "coldddi.baselines.emergnn",
    "ssi_ddi": "coldddi.baselines.ssi_ddi",
    "dsn_ddi": "coldddi.baselines.dsn_ddi",
    "hdn_ddi": "coldddi.baselines.hdn_ddi",
    "tiger": "coldddi.baselines.tiger",
    "mkg_fenn": "coldddi.baselines.mkg_fenn",
    "textddi": "coldddi.baselines.textddi",
}


def ensure_imported(name: str) -> None:
    """Lazy-import the submodule that registers ``name`` if not already done."""
    if name in _REGISTRY:
        return
    module_name = NAME_TO_MODULE.get(name)
    if module_name is None:
        return  # The caller handles unknown names.
    try:
        __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Baseline {name!r} is mapped to {module_name!r} but importing "
            f"it failed: {exc}. Install the missing dependencies."
        ) from exc


def register(name: str):
    """Register a :class:`BaselineModel` subclass and set its ``name`` attribute.

    Usage::

        @register("deepddi")
        class DeepDDI(BaselineModel):
            ...

    """
    def decorator(cls: type["BaselineModel"]) -> type["BaselineModel"]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"Baseline name {name!r} is already registered to {_REGISTRY[name]!r}"
            )
        # Reject unsupported modalities at registration, before L6 dispatch.
        declared = getattr(cls, "modality", "mol")
        if declared not in MODALITIES:
            raise ValueError(
                f"Baseline {name!r} declares modality={declared!r}, "
                f"but only {MODALITIES} are recognised."
            )
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


#: Modalities and L6 indicator dispatch in :mod:`coldddi.evaluate`:
#:
#: * ``"mol"``           — molecular graph only (SMILES atom graph).
#:                         L6 dispatch: KPS-F only; channel masks NaN.
#: * ``"text"``          — text descriptions only.
#:                         L6 dispatch: KPS-F only; channel masks NaN.
#: * ``"mol+kg-fused"``  — mol + KG fused at architecture level, no
#:                         separable channel mask.
#:                         L6 dispatch: KPS-F only; channel masks NaN.
#: * ``"mol+kg"``        — mol + KG with separable channel mask via
#:                         ``predict_proba(..., mask_channel="mol"|"kg")``.
#:                         L6 dispatch: KPS-F + KPS-mol + KPS-KG.
#:
#: The unregistered LLM stack uses R0--R3 masks through
#: :func:`coldddi.diagnostics.compute_indicators` instead.
MODALITIES: tuple[str, ...] = (
    "mol",
    "text",
    "mol+kg-fused",
    "mol+kg",
)


class BaselineModel(ABC):
    """Common interface for all ColdDDI baseline models."""

    #: Registered name (set by :func:`register`).
    name: ClassVar[str] = "abstract"

    #: See :data:`MODALITIES` for dispatch semantics. The default gives KPS-F
    #: only, with NaN channel indicators. Instances may override it, as TIGER
    #: does when ``mol_only=True``; this is not a ``ClassVar``.
    modality: str = "mol"

    @abstractmethod
    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        """Train on ``train`` (positives + negatives via the dataset)."""

    @abstractmethod
    def predict_proba(
        self,
        pairs: "pd.DataFrame",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> "np.ndarray":
        """Score ``pairs[["drug_a_id", "drug_b_id"]]`` → 1-D P(positive)."""

    @abstractmethod
    def save(self, path: "Path | str") -> None:
        """Serialize the trained model to ``path``.

        Any layout is allowed, but ``manifest.json`` must include::

            {"baseline_name": cls.name, "version": "<your-version>"}

        :func:`load_baseline` uses it to select this subclass's loader.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: "Path | str") -> "BaselineModel":
        """Reload a baseline saved by :meth:`save`. Subclass-specific."""


def write_manifest(
    out_dir: Path,
    *,
    baseline_name: str,
    extra: dict | None = None,
) -> Path:
    """Write the baseline name and optional metadata to ``out_dir/manifest.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"baseline_name": baseline_name}
    if extra:
        payload.update(extra)
    manifest_path = out_dir / BASELINE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_path


def load_baseline(path: Path | str) -> BaselineModel:
    """Load a checkpoint using the subclass named in ``<path>/manifest.json``."""
    p = Path(path)
    manifest_path = p / BASELINE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Baseline manifest not found at {manifest_path}. "
            "Did the baseline's save() write a manifest.json?"
        )
    payload = json.loads(manifest_path.read_text())
    name = payload.get("baseline_name")
    if not name:
        raise ValueError(
            f"{manifest_path} is missing the required `baseline_name` field."
        )
    ensure_imported(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown baseline {name!r}; registered baselines are "
            f"{sorted(_REGISTRY)} (declared: {sorted(NAME_TO_MODULE)})."
        )
    return _REGISTRY[name].load(p)


def list_baselines() -> list[str]:
    """Return every registered baseline name."""
    return sorted(_REGISTRY.keys())


def get_paper_hyperparams(name: str) -> dict[str, object]:
    """Return the paper-spec hyperparameter dict for ``name``.

    Read the module's ``PAPER_HYPERPARAMS`` for ``evaluate.py --preset paper``
    (Appendix C.1 Table 8). Return an empty dict if none is defined or the
    name has no module mapping.
    """
    if name not in _REGISTRY:
        ensure_imported(name)
    import importlib

    module_name = NAME_TO_MODULE.get(name)
    if module_name is None:
        return {}
    mod = importlib.import_module(module_name)
    return dict(getattr(mod, "PAPER_HYPERPARAMS", {}))
