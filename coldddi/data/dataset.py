"""Dataset combining positive edges, a knowledge graph, splits, and negatives.

``PairDataset.from_release_dir`` loads filtered CSVs and Parquet splits,
reusing cached negatives unless ``regenerate_negatives=True``.
``PairDataset.from_pkl`` loads legacy ``Preprocessor.pipeline.FoldBundle``
pickles, preserving cached ``train_neg_epochs`` for the 800- and 1,900-drug
experiments. ``FoldBundle`` provides the compatible pickle schema.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Iterable, Iterator

import numpy as np
import pandas as pd

# Read NumPy 2.x pickles under 1.x by aliasing numpy._core.* to numpy.core.*.
# Check the version: pandas 1.x may partially populate np._core.
if int(np.__version__.split(".", 1)[0]) < 2:
    import numpy.core
    sys.modules.setdefault("numpy._core", numpy.core)
    for _sub in ("numeric", "multiarray", "umath", "_multiarray_umath",
                 "fromnumeric", "_methods", "arrayprint"):
        try:
            sys.modules.setdefault(
                "numpy._core." + _sub,
                __import__("numpy.core." + _sub, fromlist=[_sub]),
            )
        except ImportError:
            pass

from coldddi.data.kg import KnowledgeGraph
from coldddi.data.negatives import (
    PHASE_OFFSETS,
    TRAIN_NEGATIVES_SEED_BASE,
    UniformNegativeSampler,
    build_static_negatives,
    build_train_negatives,
)
from coldddi.data.protocols import (
    KnowledgeGraphProtocol,
    NegativeSamplerProtocol,
    SplitFoldsProtocol,
)
from coldddi.data.splits import SPLIT_NAMES, SplitFolds, build_splits

DEFAULT_LABEL_COL: str = "label"

#: Training-negative schema. get_train_negatives strips legacy label/name
#: columns; get_legacy_train_negatives preserves them.
PAIR_COLUMNS: tuple[str, ...] = ("drug_a_id", "drug_b_id")


def _narrow_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with a reset index, keeping only pair columns if both exist."""
    if all(c in df.columns for c in PAIR_COLUMNS):
        return df.loc[:, list(PAIR_COLUMNS)].reset_index(drop=True)
    return df.reset_index(drop=True)


def _supports_train_neg_sampling(splits: object) -> bool:
    """Check for a train DataFrame, g1_drugs, and seed without requiring a class."""
    return (
        hasattr(splits, "train")
        and isinstance(getattr(splits, "train", None), pd.DataFrame)
        and hasattr(splits, "g1_drugs")
        and hasattr(splits, "seed")
    )


# Legacy pickle compatibility


@dataclass
class FoldBundle:
    """Load ``Preprocessor.pipeline.FoldBundle`` pickles without that dependency."""

    df: pd.DataFrame
    split: SimpleNamespace
    label_col: str = DEFAULT_LABEL_COL
    target_col: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class _LegacyFoldBundleUnpickler(pickle.Unpickler):
    """Redirect ``Preprocessor.pipeline.FoldBundle`` to our compatible class."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "Preprocessor.pipeline" and name == "FoldBundle":
            return FoldBundle
        return super().find_class(module, name)


def _load_legacy_pickle(path: Path) -> FoldBundle:
    with path.open("rb") as f:
        loaded = _LegacyFoldBundleUnpickler(f).load()
    if isinstance(loaded, FoldBundle):
        return loaded
    if isinstance(loaded, list) and len(loaded) == 1 and isinstance(loaded[0], FoldBundle):
        return loaded[0]
    if hasattr(loaded, "df") and hasattr(loaded, "split"):
        return FoldBundle(
            df=loaded.df,
            split=loaded.split,
            label_col=getattr(loaded, "label_col", DEFAULT_LABEL_COL),
            target_col=getattr(loaded, "target_col", None),
            extra=getattr(loaded, "extra", {}),
        )
    raise TypeError(
        f"Unrecognized pickle payload type: {type(loaded).__name__}; "
        "expected FoldBundle or a SimpleNamespace with .df and .split."
    )


# Dataset composition


@dataclass
class PairDataset:
    """Composed view of a ColdDDI fold.

    Attributes
    ----------
    edges
        The full DDI edge table (positives only).
    splits
        A :class:`SplitFoldsProtocol` implementation.
    kg
        A :class:`KnowledgeGraphProtocol` implementation.
    negatives_by_split
        Cached val/test negatives; :meth:`get_negatives` samples missing splits.
    sampler
        Sampler for missing or explicitly regenerated static negatives.
    legacy_bundle
        Original :meth:`from_pkl` payload, including ``train_neg_epochs``.
    """

    edges: pd.DataFrame
    splits: SplitFoldsProtocol
    kg: KnowledgeGraphProtocol
    drugs: pd.DataFrame | None = None
    negatives_by_split: dict[str, pd.DataFrame] = field(default_factory=dict)
    train_negatives_epochs: list[pd.DataFrame] = field(default_factory=list)
    sampler: NegativeSamplerProtocol | None = None
    annotations: dict[str, pd.DataFrame] | None = None
    legacy_bundle: FoldBundle | None = None
    source: str | None = None
    base_seed: int | None = None

    # Release loader

    @classmethod
    def from_release_dir(
        cls,
        root: Path,
        *,
        seed: int,
        regenerate_negatives: bool = False,
    ) -> "PairDataset":
        """Load filtered tables and one seed's release splits.

        ``root/filtered`` contains Stage-1b KG tables, ``ddi_edges.csv``, and
        optional ``drugs.csv``. ``root/splits/seed{seed}`` contains
        ``manifest.json`` and seven split Parquets.

        Reuse all six static ``negatives/{split}.parquet`` files if present;
        otherwise regenerate the full set with split-specific G1/G2 pools.
        Load ``train_negatives/epoch_{i}.parquet`` consecutively from zero.
        ``regenerate_negatives=True`` skips both caches; training negatives
        are then sampled on demand.
        """
        root = Path(root)
        filtered = root / "filtered"
        splits_dir = root / "splits" / f"seed{seed}"
        if not filtered.is_dir():
            raise FileNotFoundError(f"filtered/ not found under {root}")
        if not splits_dir.is_dir():
            raise FileNotFoundError(f"splits/seed{seed}/ not found under {root}")

        edges = pd.read_csv(filtered / "ddi_edges.csv")
        # SMILES and names for DeepDDI, SSI-DDI, and DSN-DDI features.
        drugs_csv = filtered / "drugs.csv"
        if drugs_csv.is_file():
            drugs_df = pd.read_csv(drugs_csv)
        else:
            import warnings  # Only needed when drugs.csv is missing.
            warnings.warn(
                f"{drugs_csv} not found; PairDataset.drugs is None. "
                "SMILES-based baselines (DeepDDI / SSI-DDI / DSN-DDI) will "
                "fail until you re-run Stage 1b.",
                stacklevel=2,
            )
            drugs_df = None
        kg = KnowledgeGraph.from_filtered_dir(filtered)
        splits = SplitFolds.from_dir(splits_dir)
        sampler = UniformNegativeSampler(
            drug_pool_a=list(splits.g1_drugs),
            drug_pool_b=list(splits.g1_drugs),
        )

        negatives_by_split: dict[str, pd.DataFrame] = {}
        cache_dir = splits_dir / "negatives"
        cache_complete = (
            cache_dir.is_dir()
            and all((cache_dir / f"{name}.parquet").is_file() for name in PHASE_OFFSETS)
        )
        if cache_complete and not regenerate_negatives:
            for name in PHASE_OFFSETS:
                negatives_by_split[name] = pd.read_parquet(cache_dir / f"{name}.parquet")
        else:
            # Rebuild all splits for a missing/partial cache or explicit regeneration.
            negatives_by_split = build_static_negatives(splits, base_seed=seed)

        # Load cached training epochs until the first missing file.
        train_neg_dir = splits_dir / "train_negatives"
        train_neg_epochs: list[pd.DataFrame] = []
        if train_neg_dir.is_dir() and not regenerate_negatives:
            i = 0
            while True:
                pq = train_neg_dir / f"epoch_{i}.parquet"
                if not pq.is_file():
                    break
                train_neg_epochs.append(pd.read_parquet(pq))
                i += 1

        return cls(
            edges=edges,
            splits=splits,
            kg=kg,
            drugs=drugs_df,
            negatives_by_split=negatives_by_split,
            train_negatives_epochs=train_neg_epochs,
            sampler=sampler,
            source=str(root),
            base_seed=seed,
        )

    # Legacy loader

    @classmethod
    def from_pkl(
        cls,
        path: str | Path,
        *,
        kg: KnowledgeGraphProtocol | None = None,
    ) -> "PairDataset":
        """Load a legacy 800- or 1,900-drug ``*.pkl`` fold bundle.

        Preserve ``extra["train_neg_epochs"]`` payloads and pairs referenced
        by ``split.<name>_neg``. Use :meth:`from_release_dir` for release tables.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"PairDataset.from_pkl: file not found: {p}")
        bundle = _load_legacy_pickle(p)

        # Adapt bundle indices to positive-only split DataFrames.
        df = bundle.df.reset_index(drop=True)
        legacy_split = bundle.split
        positives = df[df[bundle.label_col] == 1].copy() if bundle.label_col in df.columns else df

        def _slice(name: str) -> pd.DataFrame:
            attr = "train_idx" if name == "train" else f"{name.split('_')[0]}_idx_{name.split('_')[1]}"
            idx = getattr(legacy_split, attr, None)
            if idx is None:
                return df.iloc[0:0].reset_index(drop=True)
            arr = np.asarray(idx)
            sliced = df.iloc[arr].reset_index(drop=True)
            if bundle.label_col in sliced.columns:
                return sliced[sliced[bundle.label_col] == 1].reset_index(drop=True)
            return sliced

        # cold_start_split_fair_step excludes S0 val/test from G1 x G1.
        # Reuse its single train_idx without repartitioning.
        legacy_splits = SplitFolds(
            train=_slice("train"),
            val_s0=_slice("val_s0"),
            val_s1=_slice("val_s1"),
            val_s2=_slice("val_s2"),
            test_s0=_slice("test_s0"),
            test_s1=_slice("test_s1"),
            test_s2=_slice("test_s2"),
            g1_drugs=sorted(map(str, bundle.extra.get("g1_drugs", []))),
            g2_drugs=sorted(map(str, bundle.extra.get("g2_drugs", []))),
            seed=int(bundle.extra.get("random_seed", 0)),
        )

        # Materialize cached negative indices, e.g. split.test_idx_s0_neg.
        negatives_by_split: dict[str, pd.DataFrame] = {}
        for name in PHASE_OFFSETS:
            base = name.split("_", 1)
            attr = f"{base[0]}_idx_{base[1]}_neg"
            idx = getattr(legacy_split, attr, None)
            if idx is None:
                continue
            arr = np.asarray(idx)
            neg_rows = df.iloc[arr][["drug_a_id", "drug_b_id"]].reset_index(drop=True)
            negatives_by_split[name] = neg_rows

        kg_obj: KnowledgeGraphProtocol
        if kg is not None:
            kg_obj = kg
        else:
            kg_obj = _build_kg_from_legacy_kb(bundle.extra.get("kb", {}) or {})

        # Expose extra['kb']['my_drugs_list'] as ds.drugs for SMILES baselines.
        legacy_drugs_df: pd.DataFrame | None = None
        kb_dict = bundle.extra.get("kb", {}) or {}
        if isinstance(kb_dict, dict):
            cand = kb_dict.get("my_drugs_list")
            if isinstance(cand, pd.DataFrame):
                legacy_drugs_df = cand

        return cls(
            edges=positives.reset_index(drop=True),
            splits=legacy_splits,
            kg=kg_obj,
            drugs=legacy_drugs_df,
            negatives_by_split=negatives_by_split,
            sampler=None,
            legacy_bundle=bundle,
            source=str(p),
        )

    # Convenience APIs

    #: Per-split drug-pool selection that must match
    #: :func:`coldddi.data.negatives.build_static_negatives`.
    _SPLIT_POOL_KEYS: ClassVar[dict[str, tuple[str, str]]] = {
        "test_s0": ("g1_drugs", "g1_drugs"),
        "val_s0": ("g1_drugs", "g1_drugs"),
        "test_s1": ("g1_drugs", "g2_drugs"),
        "val_s1": ("g1_drugs", "g2_drugs"),
        "test_s2": ("g2_drugs", "g2_drugs"),
        "val_s2": ("g2_drugs", "g2_drugs"),
    }

    def get_negatives(
        self,
        split: str,
        *,
        regenerate: bool = False,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Return cached negatives for a static split, or sample fresh ones.

        Use split-specific pools: S0 ``G1×G1``, S1 ``G1×G2``, S2 ``G2×G2``.
        Fresh samples exclude all dataset positives, not just the chosen split.

        Parameters
        ----------
        split
            Name of a val/test split, e.g. ``"test_s2"``.
        regenerate
            Reuse cached negatives unless True or the cache is missing.
        seed
            Sampling seed when needed; defaults to ``PHASE_OFFSETS[split]``.
        """
        if split not in PHASE_OFFSETS:
            raise ValueError(f"split must be one of {sorted(PHASE_OFFSETS)}, got {split!r}")
        if not regenerate and split in self.negatives_by_split:
            return self.negatives_by_split[split]
        positives = getattr(self.splits, split)
        if len(positives) == 0:
            return pd.DataFrame(columns=["drug_a_id", "drug_b_id"])

        pool_a_attr, pool_b_attr = self._SPLIT_POOL_KEYS[split]
        pool_a = list(getattr(self.splits, pool_a_attr))
        pool_b = list(getattr(self.splits, pool_b_attr))

        # Build exclude set across all positives in this dataset.
        excl: set[tuple[str, str]] = set()
        for a, b in zip(self.edges["drug_a_id"].astype(str), self.edges["drug_b_id"].astype(str)):
            if a > b:
                a, b = b, a
            excl.add((a, b))

        sampler = self.sampler if self.sampler is not None else UniformNegativeSampler(
            drug_pool_a=pool_a, drug_pool_b=pool_b
        )
        seed_val = seed if seed is not None else PHASE_OFFSETS[split]
        return sampler.sample(
            drug_pool_a=pool_a,
            drug_pool_b=pool_b,
            n_pairs=len(positives),
            exclude=excl,
            seed=seed_val,
        )

    def get_train_negatives(
        self,
        epoch: int = 0,
        *,
        regenerate: bool = False,
    ) -> pd.DataFrame:
        """Return one epoch of training negatives.

        Return ``[drug_a_id, drug_b_id]`` pairs at the 1:1 training ratio.
        Unless ``regenerate=True``, try ``train_negatives_epochs[epoch]``,
        then legacy ``train_neg_epochs[epoch]``. Out-of-range cache indices
        fall through to sampling rather than raising.

        Fresh samples use ``base_seed + TRAIN_NEGATIVES_SEED_BASE + epoch``,
        falling back to the split's seed when ``base_seed`` is unset.
        """
        if not regenerate:
            if 0 <= epoch < len(self.train_negatives_epochs):
                return _narrow_pair_columns(self.train_negatives_epochs[epoch])
            if self.legacy_bundle is not None:
                legacy_epochs = self.legacy_bundle.extra.get("train_neg_epochs") or []
                if 0 <= epoch < len(legacy_epochs):
                    return _narrow_pair_columns(self.get_legacy_train_negatives(epoch))
                # Out-of-range legacy epochs fall through to fresh sampling.

        # Accept compatible split implementations, not just SplitFolds.
        if not _supports_train_neg_sampling(self.splits):
            raise RuntimeError(
                "get_train_negatives requires a splits object that exposes "
                "`train` (DataFrame), `g1_drugs` (list), and `seed` (int). "
                "Pass kg/splits explicitly when loading from a legacy pkl."
            )
        seed = (
            self.base_seed
            if self.base_seed is not None
            else getattr(self.splits, "seed", 0)
        )
        return _narrow_pair_columns(
            build_train_negatives(self.splits, base_seed=seed, epoch=epoch, n_per_pos=1)
        )

    def get_legacy_train_negatives(self, epoch: int = 0) -> pd.DataFrame:
        """Return one epoch of legacy training negatives (pkl path only).

        Normalize DataFrame, dict-row, or pair-sequence payloads from
        ``Preprocessor.negatives`` to a DataFrame, preserving extra columns.
        Missing bundles/epochs return an empty pair schema; an out-of-range
        index into an existing epoch list raises ``IndexError``.
        """
        if self.legacy_bundle is None:
            return pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        epochs = self.legacy_bundle.extra.get("train_neg_epochs", None)
        if epochs is None:
            return pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        if epoch < 0 or epoch >= len(epochs):
            raise IndexError(
                f"Negatives epoch index out of range: {epoch} (have {len(epochs)})"
            )
        payload = epochs[epoch]
        if isinstance(payload, pd.DataFrame):
            return payload.reset_index(drop=True)
        items = list(payload)
        if not items:
            return pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        # Normalize row payloads with drug_a_id and drug_b_id first.
        first = items[0]
        if isinstance(first, dict):
            df = pd.DataFrame(items)
            if "drug_a_id" not in df.columns or "drug_b_id" not in df.columns:
                raise ValueError(
                    "Legacy train_neg_epochs dict payload missing drug_a_id/drug_b_id."
                )
            cols = ["drug_a_id", "drug_b_id"] + [
                c for c in df.columns if c not in {"drug_a_id", "drug_b_id"}
            ]
            return df[cols].reset_index(drop=True)
        # list[tuple]/list[list]
        df = pd.DataFrame(items, columns=["drug_a_id", "drug_b_id"])
        return df.reset_index(drop=True)

    @property
    def drug_set(self) -> set[str]:
        ids: set[str] = set()
        if "drug_a_id" in self.edges.columns:
            ids |= set(self.edges["drug_a_id"].astype(str))
        if "drug_b_id" in self.edges.columns:
            ids |= set(self.edges["drug_b_id"].astype(str))
        return ids

    def iter_splits(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for name in SPLIT_NAMES:
            yield name, getattr(self.splits, name)

    def __repr__(self) -> str:
        n_neg = sum(len(v) for v in self.negatives_by_split.values())
        return (
            f"PairDataset(edges={len(self.edges):,}, drugs={len(self.drug_set)}, "
            f"splits=SplitFolds(seed={getattr(self.splits, 'seed', '?')}), "
            f"kg={self.kg!r}, "
            f"static_negatives={n_neg:,}, "
            f"source={self.source!r})"
        )


# Legacy extra['kb'] adapter


#: Wrap my_X_list tables as KnowledgeGraph only if all five have these columns.
#: Otherwise use the legacy adapter.
_LEGACY_MY_LIST_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "my_enzyme_list": {"drugbank_id", "enzyme_id", "enzyme_name"},
    "my_target_list": {"drugbank_id", "target_id", "target_name"},
    "my_transporter_list": {"drugbank_id", "transporter_id", "transporter_name"},
    "my_carrier_list": {"drugbank_id", "carrier_id", "carrier_name"},
    "my_pathway_list": {"drugbank_id", "pathway_id", "pathway_name"},
}


def _build_kg_from_legacy_kb(kb: dict[str, Any]) -> KnowledgeGraphProtocol:
    """Wrap legacy ``extra['kb']`` tables or name lookups.

    Use :class:`KnowledgeGraph` when all five ``my_X_list`` DataFrames from
    ``data_pipeline.drugbank_loader`` contain the required columns.
    Otherwise use :class:`_LegacyKGAdapter` for ``dbid_2_X`` mappings of
    ``{drug_id: [name, ...]}``; do not construct a partial KnowledgeGraph.
    """
    if not isinstance(kb, dict):
        return _LegacyKGAdapter({})

    if all(k in kb for k in _LEGACY_MY_LIST_REQUIRED_COLUMNS):
        try:
            for key, required in _LEGACY_MY_LIST_REQUIRED_COLUMNS.items():
                df = kb[key]
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"kb[{key!r}] is {type(df).__name__}, expected DataFrame")
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(
                        f"kb[{key!r}] missing required columns: {sorted(missing)}"
                    )
            return KnowledgeGraph(
                enzymes=kb["my_enzyme_list"],
                targets=kb["my_target_list"],
                transporters=kb["my_transporter_list"],
                carriers=kb["my_carrier_list"],
                pathways=kb["my_pathway_list"],
            )
        except (TypeError, ValueError, KeyError):
            # Incomplete schemas use the legacy adapter.
            pass
    return _LegacyKGAdapter(kb)


@dataclass(repr=False)
class _LegacyKGAdapter:
    """Wrap a legacy ``extra['kb']`` dict as a :class:`KnowledgeGraphProtocol`.

    Read ``dbid_2_enzymes / dbid_2_targets / ...`` name lookups. Entity IDs,
    organisms, and actions are unavailable and returned as empty strings.
    """

    kb: dict[str, Any]

    def __repr__(self) -> str:
        # Summarize keys without printing large tables.
        if not isinstance(self.kb, dict):
            return "_LegacyKGAdapter(kb=<non-dict>)"
        keys = sorted(self.kb.keys())
        if len(keys) > 8:
            return (
                f"_LegacyKGAdapter(n_keys={len(keys)}, "
                f"sample={keys[:5]!r}...)"
            )
        return f"_LegacyKGAdapter(keys={keys!r})"

    @property
    def drug_ids(self) -> set[str]:
        """All drugs that appear anywhere in the legacy kb dict."""
        ids: set[str] = set(map(str, (self.kb.get("dbid_2_drugs", {}) or {}).keys()))
        for et in ("enzyme", "target", "transporter", "carrier", "pathway"):
            key = f"dbid_2_{et}s"
            ids.update(map(str, (self.kb.get(key, {}) or {}).keys()))
        return ids

    def neighbors(self, drug_id: str, *, edge_types: Iterable[str] | None = None) -> pd.DataFrame:
        rows: list[dict] = []
        types = list(edge_types) if edge_types is not None else ["enzyme", "target", "transporter", "carrier", "pathway"]
        for et in types:
            key = f"dbid_2_{et}s"
            entries = (self.kb.get(key, {}) or {}).get(str(drug_id), [])
            for name in entries:
                rows.append(
                    {
                        "drugbank_id": str(drug_id),
                        "edge_type": et,
                        "entity_id": "",
                        "entity_name": str(name),
                        "organism": "",
                        "action": "",
                    }
                )
        return pd.DataFrame(
            rows,
            columns=["drugbank_id", "edge_type", "entity_id", "entity_name", "organism", "action"],
        )

    def shared_entities(self, drug_a: str, drug_b: str) -> pd.DataFrame:
        rows: list[dict] = []
        for et in ["enzyme", "target", "transporter", "carrier", "pathway"]:
            key = f"dbid_2_{et}s"
            a_names = set((self.kb.get(key, {}) or {}).get(str(drug_a), []))
            b_names = set((self.kb.get(key, {}) or {}).get(str(drug_b), []))
            for name in a_names & b_names:
                rows.append({"edge_type": et, "entity_id": "", "entity_name": str(name)})
        return pd.DataFrame(rows, columns=["edge_type", "entity_id", "entity_name"])

    def name_dict(self, edge_type: str) -> dict[str, list[str]]:
        return dict((self.kb.get(f"dbid_2_{edge_type}s", {}) or {}))

    def save(self, out_dir: Path) -> None:
        raise NotImplementedError(
            "Legacy adapter is read-only; export the underlying KG via "
            "coldddi.data.kg.KnowledgeGraph.save instead."
        )


def load_release_dataset(
    root: Path,
    *,
    seed: int,
    regenerate_negatives: bool = False,
) -> PairDataset:
    """Thin wrapper around :meth:`PairDataset.from_release_dir`."""
    return PairDataset.from_release_dir(root, seed=seed, regenerate_negatives=regenerate_negatives)


__all__ = [
    "DEFAULT_LABEL_COL",
    "FoldBundle",
    "PairDataset",
    "load_release_dataset",
]
