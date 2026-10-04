"""Seeded drug sampling into a self-contained release-format dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coldddi.benchmark_data import file_sha256, pair_keys


def sample_drug_ids(drug_ids: list[str], *, seed: int, size: int = 800) -> list[str]:
    """Sample without replacement from the given pool order; return sorted IDs."""
    pool = list(drug_ids)
    if len(set(pool)) != len(pool) or any(not isinstance(x, str) or not x.strip() for x in pool):
        raise ValueError("Drug IDs must be unique, nonempty strings")
    if not 0 < size <= len(pool):
        raise ValueError(f"Cannot sample {size} drugs from a pool of {len(pool)}")
    return sorted(np.random.default_rng(seed).choice(pool, size=size, replace=False).tolist())


def build_subset(
    source: Path,
    output: Path,
    *,
    seed: int,
    size: int = 800,
    n_train_negative_epochs: int = 4,
    quiet: bool = False,
) -> Path:
    """Build a release subset with induced DDI edges and matching annotations.

    Use Step-6 degrees with pandas 2.x quicksort tie ordering, as in the
    original sampler. Source is a reconstructed intermediate/ directory;
    output must be empty and separate from source.
    """
    from coldddi.reconstruct import _Logger, _do_stage2c, _do_stage3, _do_stage4
    from coldddi.sanity_check import write_checksums

    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Subset output and source must be separate directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Subset output is not empty: {output}. Choose a new --output.")
    if n_train_negative_epochs < 0:
        raise ValueError("n_train_negative_epochs must be >= 0")

    tables = {name: pd.read_csv(source / "filtered" / name) for name in (
        "drugs.csv", "ddi_edges.csv", "drug_enzymes.csv", "drug_targets.csv",
        "drug_transporters.csv", "drug_carriers.csv", "drug_pathways.csv",
    )}
    candidate_path = source / "filtered/subset_candidates.json"
    if not candidate_path.is_file():
        raise ValueError("Missing subset_candidates.json; rerun reconstruction stage1b to record the original sampling order")
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    if set(candidates) != set(tables["drugs.csv"].drugbank_id):
        raise ValueError("Sampling candidates do not match the filtered drug table")
    selected = sample_drug_ids(candidates, seed=seed, size=size)
    keep = set(selected)
    edges = tables["ddi_edges.csv"]
    edges = edges.loc[edges.drug_a_id.isin(keep) & edges.drug_b_id.isin(keep)].copy()
    active = set(edges.drug_a_id) | set(edges.drug_b_id)
    if active != keep:
        raise ValueError("Sample contains isolated drugs; choose another seed. No resampling was done.")
    annotations = pd.read_csv(source / "enriched/ddi_key_entities.csv")
    annotations = annotations.loc[
        annotations.drug_a_id.isin(keep) & annotations.drug_b_id.isin(keep)
    ].copy()
    if set(pair_keys(annotations, "source annotations")) != set(pair_keys(edges, "subset edges")):
        raise ValueError("Source A/B annotations do not match the subset DDI edges")
    labels = pd.read_csv(source / "enriched/ddi_pk_pd_labels.csv")
    labels = labels.loc[labels.ddi_type.isin(edges.ddi_type.unique())]

    root = output / "intermediate"
    filtered, enriched = root / "filtered", root / "enriched"
    filtered.mkdir(parents=True)
    enriched.mkdir()
    for name, table in tables.items():
        frame = edges if name == "ddi_edges.csv" else table.loc[table.drugbank_id.isin(keep)]
        frame.to_csv(filtered / name, index=False)
    labels.to_csv(enriched / "ddi_pk_pd_labels.csv", index=False)
    annotations.to_csv(enriched / "ddi_key_entities.csv", index=False)
    metadata = {
        "seed": seed, "size": size, "source_drugs": len(tables["drugs.csv"]),
        "sampling": "numpy.default_rng(seed).choice(step6_degree_order_quicksort, size, replace=False)",
        "selected_drugs": selected,
        "source_candidates_sha256": file_sha256(candidate_path),
        "source_edges_sha256": file_sha256(source / "filtered/ddi_edges.csv"),
        "numpy_version": np.__version__,
    }
    (filtered / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log = _Logger(quiet=quiet)
    _do_stage2c(enriched, log)
    _do_stage3(enriched, output, release_mode="full", full_pkpd_csv=None, log=log)
    _do_stage4(filtered, root / "splits", seeds=(seed,),
               n_train_negative_epochs=n_train_negative_epochs, log=log)
    write_checksums(root)
    return root
