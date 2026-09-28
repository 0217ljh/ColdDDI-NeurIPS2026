"""Stage-spanning reconstruction driver - paper @A.6.3 entry point.

A single CLI that runs every published stage of the data pipeline,
from a raw DrugBank XML all the way to the four release Parquet
artifacts and the per-seed train/val/test splits with their negatives.

Stages (each one is implemented in its own module - this driver only
orchestrates):

* ``stage1a`` - :mod:`coldddi.data.extract`              (XML -> 7 raw csvs)
* ``stage1b`` - :mod:`coldddi.data.filter`               (7-step pipeline -> filtered csvs)
* ``stage2a`` - :mod:`coldddi.annotations.pkpd_keywords` (PK/PD type labels)
* ``stage2b`` - :mod:`coldddi.annotations.ab_subdivision` (A/B subdivision per pair)
* ``stage2c`` - :mod:`coldddi.annotations.derive_type_a_tables` (Type-A projections)
* ``stage3``  - :mod:`coldddi.data.release_parquet`      (CSV -> 4 release Parquets)
* ``stage4``  - :mod:`coldddi.data.splits`               (S0/S1/S2 splits + negatives)

Output layout (relative to ``--output``):

::

    <out>/
    ├-- intermediate/
    │   ├-- raw/        (stage1a)
    │   ├-- filtered/   (stage1b)
    │   └-- enriched/   (stage2a/b/c)
    ├-- outputs_full/annotations/      (stage3 - full mode)
    ├-- annotations_sample/             (stage3 - sample mode)
    └-- intermediate/splits/seed{N}/   (stage4 x seed)

CLI
---
``python -m coldddi.reconstruct --drugbank PATH --output DIR --seeds 42 43 44``
``python reconstruct.py ...``  (root-level wrapper at the repo root)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from coldddi.annotations.ab_subdivision import run_ab_subdivision
from coldddi.annotations.derive_type_a_tables import derive_type_a_tables
from coldddi.annotations.pkpd_keywords import label_ddi_types
from coldddi.data.extract import parse_drugbank_xml, write_raw_tables
from coldddi.data.filter import run_filter_pipeline, write_filter_report
from coldddi.data.negatives import build_train_negatives, build_static_negatives
from coldddi.data.release_parquet import dump_release_parquets
from coldddi.data.splits import build_splits

ALL_STAGES: tuple[str, ...] = (
    "stage1a",
    "stage1b",
    "stage2a",
    "stage2b",
    "stage2c",
    "stage3",
    "stage4",
)

#: Repo-relative path to the toy XML. Resolved from the location of this
#: source file, so ``--toy`` works from any cwd.
DEFAULT_TOY_XML: Path = (
    Path(__file__).resolve().parent.parent / "data" / "public" / "drugbank_toy.xml"
)


# ---------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------


class _Logger:
    """Tiny prefix logger that times each stage."""

    def __init__(self, *, quiet: bool) -> None:
        self._quiet = quiet
        self._t0: float = 0.0

    def info(self, msg: str) -> None:
        if not self._quiet:
            print(f"[reconstruct] {msg}", file=sys.stderr, flush=True)

    def stage_begin(self, name: str) -> None:
        self._t0 = time.monotonic()
        self.info(f"== {name} ==")

    def stage_end(self, name: str) -> None:
        elapsed = time.monotonic() - self._t0
        self.info(f"   {name} done in {elapsed:.1f}s")


# ---------------------------------------------------------------------
# Per-stage helpers
# ---------------------------------------------------------------------


def _do_stage1a(xml_path: Path, raw_dir: Path, log: _Logger) -> None:
    log.stage_begin("Stage 1a - XML -> raw csvs")
    raw = parse_drugbank_xml(xml_path)
    log.info(
        f"   parsed: drugs={len(raw.drugs):,}  edges={len(raw.edges):,}  "
        f"enzymes={len(raw.enzymes):,}  targets={len(raw.targets):,}"
    )
    write_raw_tables(raw, raw_dir)
    log.info(f"   wrote 7 csvs -> {raw_dir}")
    log.stage_end("Stage 1a")


def _do_stage1b(raw_dir: Path, filtered_dir: Path, log: _Logger):
    from coldddi.data.extract import load_raw_tables

    log.stage_begin("Stage 1b - 7-step filter pipeline")
    raw = load_raw_tables(raw_dir)
    report = run_filter_pipeline(raw, verbose=False)
    final = report.step_stats[-1]
    log.info(
        f"   final: {final.n_drugs:,} drugs / {final.n_edges:,} edges / "
        f"{final.n_types} types"
    )
    write_filter_report(report, filtered_dir)
    log.info(f"   wrote 7 filtered csvs + stats.json -> {filtered_dir}")
    log.stage_end("Stage 1b")
    return report


def _do_stage2a(filtered_dir: Path, enriched_dir: Path, log: _Logger) -> Path:
    log.stage_begin("Stage 2a - PK/PD keyword labels")
    edges = pd.read_csv(filtered_dir / "ddi_edges.csv", usecols=["ddi_type"])
    pkpd = label_ddi_types(edges["ddi_type"])
    enriched_dir.mkdir(parents=True, exist_ok=True)
    out_path = enriched_dir / "ddi_pk_pd_labels.csv"
    pkpd.to_csv(out_path, index=False)
    counts = pkpd["pk_pd_label"].value_counts().to_dict()
    log.info(f"   {len(pkpd)} types  ({counts})")
    log.stage_end("Stage 2a")
    return out_path


def _do_stage2b(
    filtered_dir: Path, enriched_dir: Path, pkpd_csv: Path, log: _Logger
):
    log.stage_begin("Stage 2b - A/B subdivision")
    edges = pd.read_csv(filtered_dir / "ddi_edges.csv")
    drugs = pd.read_csv(filtered_dir / "drugs.csv", usecols=["drugbank_id", "name"])
    pkpd = pd.read_csv(pkpd_csv)
    id2name = dict(zip(drugs["drugbank_id"], drugs["name"]))
    per_pair, per_type = run_ab_subdivision(
        ddi_edges=edges,
        pk_pd_labels=pkpd,
        enzymes_csv=filtered_dir / "drug_enzymes.csv",
        targets_csv=filtered_dir / "drug_targets.csv",
        transporters_csv=filtered_dir / "drug_transporters.csv",
        carriers_csv=filtered_dir / "drug_carriers.csv",
        drug_id_to_name=id2name,
        verbose=not log._quiet,
    )
    enriched_dir.mkdir(parents=True, exist_ok=True)
    per_pair.to_csv(enriched_dir / "ddi_key_entities.csv", index=False)
    per_type.to_csv(enriched_dir / "ddi_key_entities_type_summary.csv", index=False)
    n_a = int(per_pair["has_key_entity"].sum())
    log.info(
        f"   {len(per_pair):,} pairs  "
        f"({n_a:,} Type-A = {100 * n_a / max(1, len(per_pair)):.1f}%)"
    )
    log.stage_end("Stage 2b")


def _do_stage2c(enriched_dir: Path, log: _Logger) -> None:
    log.stage_begin("Stage 2c - Type-A projections")
    df = pd.read_csv(enriched_dir / "ddi_key_entities.csv")
    mediating, action = derive_type_a_tables(df)
    mediating.to_parquet(enriched_dir / "mediating_entities.parquet", index=False)
    action.to_parquet(enriched_dir / "action_pairs.parquet", index=False)
    mediating.to_csv(enriched_dir / "mediating_entities.csv", index=False)
    action.to_csv(enriched_dir / "action_pairs.csv", index=False)
    log.info(
        f"   {len(mediating):,} mediating_entities + {len(action):,} action_pairs"
    )
    log.stage_end("Stage 2c")


def _do_stage3(
    enriched_dir: Path,
    output_root: Path,
    *,
    release_mode: str,
    full_pkpd_csv: Path | None,
    log: _Logger,
) -> None:
    log.stage_begin(f"Stage 3 - release Parquet ({release_mode} mode)")
    if release_mode == "full":
        out_dir = output_root / "outputs_full" / "annotations"
    else:
        out_dir = output_root / "annotations_sample"
    written = dump_release_parquets(
        enriched_dir=enriched_dir,
        out_dir=out_dir,
        mode=release_mode,  # type: ignore[arg-type]
        full_pkpd_csv=full_pkpd_csv,
    )
    # Don't re-read the parquets just to count rows — on a full run that's
    # several GB of avoidable I/O.
    for name, path in written.items():
        log.info(f"   {name:<20} -> {path.name}")
    log.stage_end("Stage 3")


def _do_stage4(
    filtered_dir: Path,
    splits_root: Path,
    *,
    seeds: Sequence[int],
    n_train_negative_epochs: int,
    log: _Logger,
) -> None:
    log.stage_begin(f"Stage 4 - splits x {len(seeds)} seeds")
    edges = pd.read_csv(filtered_dir / "ddi_edges.csv")
    splits_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed_dir = splits_root / f"seed{seed}"
        splits = build_splits(edges, seed=seed)
        splits.save(seed_dir)
        log.info(
            f"   seed {seed}: |G1|={len(splits.g1_drugs)}, |G2|={len(splits.g2_drugs)}, "
            f"train={len(splits.train):,}, test_s2={len(splits.test_s2):,}"
        )
        # Static (val/test) negatives
        neg_dir = seed_dir / "negatives"
        neg_dir.mkdir(parents=True, exist_ok=True)
        for name, df in build_static_negatives(splits, base_seed=seed).items():
            df.to_parquet(neg_dir / f"{name}.parquet", index=False)
        # Pre-baked train negatives - stream-write for visible progress.
        # Always wipe any stale `epoch_*.parquet` from a previous run so
        # the on-disk count matches the requested n_epochs exactly.
        train_neg_dir = seed_dir / "train_negatives"
        if train_neg_dir.is_dir():
            for stale in train_neg_dir.glob("epoch_*.parquet"):
                stale.unlink()
        if n_train_negative_epochs > 0:
            train_neg_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_train_negative_epochs):
                df = build_train_negatives(splits, base_seed=seed, epoch=i)
                df.to_parquet(train_neg_dir / f"epoch_{i}.parquet", index=False)
                log.info(
                    f"     train_negatives/epoch_{i}: {len(df):,} rows"
                )
    log.stage_end("Stage 4")


# ---------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------


#: Which stage writes which directory under <output>/intermediate/.
_STAGE_PRODUCES: dict[str, str] = {
    "stage1a": "raw/",
    "stage1b": "filtered/",
    "stage2a": "enriched/ddi_pk_pd_labels.csv",
    "stage2b": "enriched/ddi_key_entities.csv",
    "stage2c": "enriched/mediating_entities.parquet",
}


def _producer_of(rel_path: str) -> str:
    """Return which stage writes the given output-relative file."""
    if rel_path.startswith("intermediate/raw/"):
        return "stage1a"
    if rel_path.startswith("intermediate/filtered/"):
        return "stage1b"
    if "ddi_pk_pd_labels.csv" in rel_path:
        return "stage2a"
    if "ddi_key_entities.csv" in rel_path:
        return "stage2b"
    if "mediating_entities" in rel_path or "action_pairs" in rel_path:
        return "stage2c"
    return "stage1a"


#: Files each stage *needs upstream* (relative to the output root).
#: Used to give a clear error when --skip-stages omits a producer
#: but a later stage still runs.
_STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "stage1b": ("intermediate/raw/drugs.csv",),
    "stage2a": ("intermediate/filtered/ddi_edges.csv",),
    "stage2b": (
        "intermediate/filtered/ddi_edges.csv",
        "intermediate/filtered/drugs.csv",
        "intermediate/filtered/drug_enzymes.csv",
        "intermediate/enriched/ddi_pk_pd_labels.csv",
    ),
    "stage2c": ("intermediate/enriched/ddi_key_entities.csv",),
    "stage3":  ("intermediate/enriched/ddi_pk_pd_labels.csv",
                "intermediate/enriched/ddi_key_entities.csv"),
    "stage4":  ("intermediate/filtered/ddi_edges.csv",),
}


def run_reconstruction(
    *,
    drugbank: Path,
    output: Path,
    seeds: Sequence[int] = (42, 43, 44),
    release_mode: str = "full",
    n_train_negative_epochs: int = 4,
    skip_stages: Iterable[str] = (),
    full_pkpd_csv: Path | None = None,
    quiet: bool = False,
) -> None:
    """Run every pipeline stage in sequence.

    Each stage is independently skippable via ``skip_stages`` (e.g. when
    re-running only the splits step on the same Stage-1b output). When a
    skipped stage's downstream artifacts are missing, this function
    raises a clear error before any work begins.
    """
    if n_train_negative_epochs < 0:
        raise ValueError(
            f"--n-train-negative-epochs must be >= 0 (got {n_train_negative_epochs})"
        )
    skip = set(skip_stages)
    unknown = skip - set(ALL_STAGES)
    if unknown:
        raise ValueError(
            f"Unknown stages in --skip-stages: {sorted(unknown)}; "
            f"valid choices are {ALL_STAGES}."
        )

    output = Path(output)
    # Preflight: every stage that will run must have its upstream
    # artifacts present (either produced earlier this run, or already on
    # disk from a previous run). Raise *before* doing any work.
    for stage in ALL_STAGES:
        if stage in skip:
            continue
        # An upstream is satisfied iff its producer stage is either also
        # running this session OR its file already exists on disk.
        for rel in _STAGE_REQUIRES.get(stage, ()):
            producer = _producer_of(rel)
            if producer in skip and not (output / rel).is_file():
                raise FileNotFoundError(
                    f"Stage {stage!r} requires {output / rel}, but its "
                    f"producer {producer!r} is in --skip-stages and the "
                    f"file is missing. Run the producer first or remove "
                    f"it from --skip-stages."
                )

    log = _Logger(quiet=quiet)
    log.info(f"Output root: {output}")
    log.info(f"Stages to run: {[s for s in ALL_STAGES if s not in skip]}")

    raw_dir = output / "intermediate" / "raw"
    filtered_dir = output / "intermediate" / "filtered"
    enriched_dir = output / "intermediate" / "enriched"
    splits_root = output / "intermediate" / "splits"

    if "stage1a" not in skip:
        _do_stage1a(Path(drugbank), raw_dir, log)
    if "stage1b" not in skip:
        _do_stage1b(raw_dir, filtered_dir, log)
    if "stage2a" not in skip:
        _do_stage2a(filtered_dir, enriched_dir, log)
    if "stage2b" not in skip:
        _do_stage2b(
            filtered_dir, enriched_dir, enriched_dir / "ddi_pk_pd_labels.csv", log
        )
    if "stage2c" not in skip:
        _do_stage2c(enriched_dir, log)
    if "stage3" not in skip:
        _do_stage3(
            enriched_dir,
            output,
            release_mode=release_mode,
            full_pkpd_csv=full_pkpd_csv,
            log=log,
        )
    if "stage4" not in skip:
        _do_stage4(
            filtered_dir,
            splits_root,
            seeds=seeds,
            n_train_negative_epochs=n_train_negative_epochs,
            log=log,
        )

    log.info("[OK] Reconstruction complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reconstruct",
        description=(
            "Single-command DrugBank -> ColdDDI reconstruction (Paper @A.6.3). "
            "Runs every pipeline stage in sequence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--drugbank",
        type=Path,
        help="Path to a licensed DrugBank full database.xml.",
    )
    src.add_argument(
        "--toy",
        action="store_true",
        help="Shortcut for `--drugbank data/public/drugbank_toy.xml` (the 100-drug subset).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output root (intermediate / splits / annotations land under here).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Seeds for Stage-4 splits.",
    )
    parser.add_argument(
        "--release-mode",
        choices=["full", "sample"],
        default=None,
        help=(
            "Stage-3 release-Parquet naming convention. Defaults to "
            "'sample' when --toy is set, 'full' otherwise."
        ),
    )
    parser.add_argument(
        "--n-train-negative-epochs",
        type=int,
        default=4,
        help="Stage 4 - pre-bake N training-negative epochs per seed.",
    )
    parser.add_argument(
        "--full-pkpd",
        type=Path,
        default=None,
        help=(
            "Optional override for Stage-3 sample mode. Lets `pkpd.parquet` stay "
            "the full 215-type table while pair-level files come from a smaller "
            "enriched_dir (e.g. the toy)."
        ),
    )
    parser.add_argument(
        "--skip-stages",
        nargs="+",
        default=[],
        choices=list(ALL_STAGES),
        help="Stages to skip (e.g. when reusing Stage-1b output from a previous run).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-stage progress prints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.toy:
        drugbank = DEFAULT_TOY_XML
        if not drugbank.is_file():
            print(f"--toy expected {drugbank} to exist.", file=sys.stderr)
            return 1
    else:
        drugbank = args.drugbank
        if not drugbank.is_file():
            print(f"--drugbank file not found: {drugbank}", file=sys.stderr)
            return 1

    # `--toy` defaults to sample mode (the toy is a 100-drug subset, the
    # full release-mode parquets would be tiny and misleading).
    release_mode = args.release_mode
    if release_mode is None:
        release_mode = "sample" if args.toy else "full"

    run_reconstruction(
        drugbank=drugbank,
        output=args.output,
        seeds=args.seeds,
        release_mode=release_mode,
        n_train_negative_epochs=args.n_train_negative_epochs,
        skip_stages=args.skip_stages,
        full_pkpd_csv=args.full_pkpd,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
