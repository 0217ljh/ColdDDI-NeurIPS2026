"""Validate a release-format dataset before running the LLM benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def pair_keys(frame: pd.DataFrame, name: str) -> list[tuple[str, str]]:
    """Canonical pair keys for validation, not for changing prompt direction."""
    require_columns(frame, ("drug_a_id", "drug_b_id"), name)
    pairs = frame[["drug_a_id", "drug_b_id"]]
    if pairs.isna().any().any():
        raise ValueError(f"{name}: null drug IDs")
    keys = [tuple(sorted((str(a), str(b)))) for a, b in pairs.itertuples(index=False, name=None)]
    if any(not a.strip() or not b.strip() or a != a.strip() or b != b.strip() for a, b in keys):
        raise ValueError(f"{name}: empty drug IDs or surrounding whitespace")
    if any(a == b for a, b in keys):
        raise ValueError(f"{name}: self-pairs are not supported")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name}: duplicate pairs (including reversed pairs)")
    return keys


def validate_dataset(root: Path, ab_path: Path, seed: int) -> tuple[PairDataset, pd.DataFrame, dict]:
    """Load and validate IDs, labels, partitions, annotations and used input files.

    Negative pairs may recur across epochs/evaluation folds in the existing
    sampler. Such overlap is reported, not silently removed or re-sampled.
    Positive folds must be disjoint; no negative may be a known positive.
    """
    from coldddi.data.dataset import PairDataset

    root, ab_path = root.resolve(), ab_path.resolve()
    split_names = ("train", "val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2")
    filtered = root / "filtered"
    splits = root / "splits" / f"seed{seed}"
    required = [filtered / "drugs.csv", filtered / "ddi_edges.csv", splits / "manifest.json", ab_path]
    required += [splits / f"{name}.parquet" for name in split_names]
    required += [splits / "negatives" / f"{name}.parquet" for name in split_names[1:]]
    for plural in ("enzymes", "targets", "transporters", "carriers", "pathways"):
        pq = filtered / f"drug_{plural}.parquet"
        required.append(pq if pq.is_file() else filtered / f"drug_{plural}.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing benchmark input files:\n  " + "\n  ".join(missing))
    if ab_path.suffix.lower() != ".parquet":
        raise ValueError("--ab-parquet must be a Parquet file")

    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("seed") != seed:
        raise ValueError(f"manifest.json seed does not match --seed {seed}")
    ds = PairDataset.from_release_dir(root, seed=seed)
    require_columns(ds.drugs, ("drugbank_id", "name", "smiles"), "drugs.csv")
    if ds.drugs[["drugbank_id", "name"]].isna().any().any():
        raise ValueError("drugs.csv: drug IDs and names must be non-null")
    ids = ds.drugs.drugbank_id.astype(str)
    if ids.duplicated().any() or any(not x.strip() or x != x.strip() for x in ids):
        raise ValueError("drugs.csv: drug IDs must be unique, nonempty, and trimmed")
    if any(not str(x).strip() for x in ds.drugs.name):
        raise ValueError("drugs.csv: empty drug names")
    known = set(ids)
    partitions = []
    for name in ("g1_drugs", "g2_drugs"):
        values = manifest.get(name)
        if not isinstance(values, list) or not values or any(x is None for x in values):
            raise ValueError(f"manifest.json: {name} must be a nonempty list of drug IDs")
        group = set(map(str, values))
        if len(group) != len(values) or not group <= known:
            raise ValueError(f"manifest.json: duplicate or unknown drug IDs in {name}")
        partitions.append(group)
    g1, g2 = partitions
    if g1 & g2 or g1 | g2 != known:
        raise ValueError("manifest.json: G1/G2 must be disjoint and cover drugs.csv")

    def checked_pairs(frame: pd.DataFrame, name: str, partition: str | None = None) -> set:
        keys = pair_keys(frame, name)
        if any(a not in known or b not in known for a, b in keys):
            raise ValueError(f"{name}: unknown drug IDs; CSV and Parquet IDs must agree")
        if partition is not None:
            expected_g2 = {"train": 0, "s0": 0, "s1": 1, "s2": 2}[partition]
            if any(int(a in g2) + int(b in g2) != expected_g2 for a, b in keys):
                raise ValueError(f"{name}: drug pair violates the {partition} G1/G2 partition")
        return set(keys)

    positives = checked_pairs(ds.edges, "ddi_edges.csv")
    if not positives:
        raise ValueError("ddi_edges.csv: empty positive edge table")
    seen = set()
    counts = {}
    negative_sets = {}
    for name, frame in ds.splits.items():
        keys = checked_pairs(frame, name, "train" if name == "train" else name[-2:])
        if not keys:
            raise ValueError(f"{name}: empty split; all S0/S1/S2 validation/test splits are required")
        if not keys <= positives:
            raise ValueError(f"{name}: positive pair missing from ddi_edges.csv")
        if keys & seen:
            raise ValueError(f"{name}: positive pairs overlap another train/validation/test split")
        seen.update(keys)
        if "n_pairs" in manifest and manifest["n_pairs"].get(name) != len(frame):
            raise ValueError(f"manifest.json n_pairs does not match {name}")
        negatives = ds.get_train_negatives(0) if name == "train" else ds.get_negatives(name)
        neg = checked_pairs(negatives, f"{name} negatives", "train" if name == "train" else name[-2:])
        if not neg:
            raise ValueError(f"{name}: no negative examples")
        if neg & positives:
            raise ValueError(f"{name}: negative examples conflict with known positive DDI edges")
        negative_sets[name] = neg
        counts[name] = {"positive": len(keys), "negative": len(neg)}

    # Validate the KG tables actually preferred by the existing loader.
    for plural, entity in (("enzymes", "enzyme"), ("targets", "target"),
                           ("transporters", "transporter"), ("carriers", "carrier"),
                           ("pathways", "pathway")):
        table = getattr(ds.kg, plural)
        columns = ("drugbank_id", f"{entity}_id", f"{entity}_name")
        require_columns(table, columns, f"drug_{plural}")
        if table[list(columns)].isna().any().any():
            raise ValueError(f"drug_{plural}: null drug/entity IDs or entity names")
        if not set(table.drugbank_id.astype(str)) <= known:
            raise ValueError(f"drug_{plural}: unknown drug IDs")

    ab = pd.read_parquet(ab_path)
    require_columns(ab, ("drug_a_id", "drug_b_id", "pk_pd_label", "has_key_entity",
                         "key_entity_name", "key_entity_type"), "A/B annotations")
    ab_keys = checked_pairs(ab, "A/B annotations")
    if ab_keys != positives:
        raise ValueError("A/B annotations must cover exactly ddi_edges.csv (wrong dataset or incomplete annotations)")
    if ab.has_key_entity.isna().any() or not all(isinstance(x, (bool, np.bool_)) for x in ab.has_key_entity):
        raise ValueError("A/B annotations: has_key_entity must contain booleans, not strings or nulls")
    if ab.pk_pd_label.isna().any():
        raise ValueError("A/B annotations: null pk_pd_label; use PK, PD, Mixed or Unknown")
    labels = ab.pk_pd_label.astype(str).str.upper()
    if not set(labels) <= {"PK", "PD", "MIXED", "UNKNOWN"}:
        raise ValueError("A/B annotations: pk_pd_label must be PK, PD, Mixed or Unknown")
    for col in ("key_entity_name", "key_entity_type"):
        present = ab.loc[ab.has_key_entity, col]
        if present.isna().any() or present.astype(str).str.strip().eq("").any():
            raise ValueError(f"A/B annotations: Type-A pairs require {col}")

    required += sorted((splits / "train_negatives").glob("epoch_*.parquet"))
    hashes = {str(p.relative_to(root)) if p.is_relative_to(root) else "ab_parquet": file_sha256(p)
              for p in sorted(set(required))}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    overlap = {name: len(negative_sets["train"] & values)
               for name, values in negative_sets.items() if name != "train"}
    warnings = []
    if any(overlap.values()):
        warnings.append("Some training negatives recur in evaluation folds; this is reported, not removed. "
                        "The shipped sampler allows S0 negative overlap.")
    if labels.isin(("MIXED", "UNKNOWN")).any():
        warnings.append("Mixed/Unknown positive pairs contribute to ALL but not the four PK/PD buckets.")
    report = {"data": str(root), "ab_parquet": str(ab_path), "seed": seed,
              "fingerprint": fingerprint, "input_sha256": hashes, "drugs": len(known),
              "splits": counts, "annotation_labels": labels.value_counts().to_dict(),
              "train_negative_overlap": overlap, "warnings": warnings}
    return ds, ab, report
