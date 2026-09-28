# `data/` — public + private data root

This directory holds **all** DrugBank-derived artifacts the repository
ever produces or consumes, split by licence sensitivity:

```
data/
├── public/    ← shipped in git, runnable without a DrugBank licence
└── private/   ← user-local only, holds any real DrugBank record
```

The split is purely structural. The `.gitignore` lines

```gitignore
data/private/**
!data/private/
!data/private/**/
!data/private/**/.gitkeep
!data/private/_README.md
```

guarantee that **no real DrugBank file** under `private/` can ever be
committed; only the directory skeleton + the README are tracked.

## Three-stage pipeline (mirrored on both sides)

Every CSV input or output the pipeline produces is one of:

| Stage | Folder name | Producer module |
|---|---|---|
| 1a Raw extraction (XML → 7 raw CSVs)        | `intermediate/raw/`      | `coldddi.data.extract` |
| 1b Seven-step filtering (Appendix A.1)      | `intermediate/filtered/` | `coldddi.data.filter` |
| 2 Annotation (PK/PD + A/B + Type-A views)   | `intermediate/enriched/` | `coldddi.annotations.{pkpd_keywords,ab_subdivision,derive_type_a_tables}` |

Both `data/public/` (the 100-drug toy) and `data/private/` (the licensed
full XML) follow the same `intermediate/{raw,filtered,enriched}/`
sub-tree, so the same code path serves both end to end.

## `public/` — shipped in git

| Path | Purpose |
|---|---|
| `public/drugbank_toy.xml` | A 100-drug random subset of the 1,994 final-pipeline drug pool, drawn with `numpy.random.default_rng(42)`. Lets reviewers run the full pipeline without obtaining a DrugBank licence. |
| `public/intermediate/{raw,filtered,enriched}/` | The Stage-1a / 1b / 2 outputs of running the `coldddi.*` pipeline on `drugbank_toy.xml`. |
| `public/scripts/toy_drug_ids.txt` | The 100 DrugBank IDs that define the toy XML (license-safe — IDs only, no records). |

## `private/` — user-local only

See `private/_README.md` for the full layout. Summary:

| Path | Purpose |
|---|---|
| `private/raw/drugbank_5.1.13.xml` | User-supplied DrugBank XML (academic licence required). |
| `private/subset/` | Optional development-time subset XMLs (e.g. the 160-drug sample chosen by `data-private/scripts/sample_size_experiment.py`). |
| `private/intermediate/{raw,filtered,enriched}/` | Same three-stage layout as `public/`, but produced from the full XML. |
| `private/outputs_full/` | Final reconstruction artifacts (annotations + splits) — Stage 3/4, not yet implemented. |

## How toy and full stay in sync

Both XMLs are processed by the same `coldddi.data.*` and
`coldddi.annotations.*` modules. The only difference between
`public/intermediate/` and `private/intermediate/` is the input scale.
The CSVs in `public/intermediate/` are the smaller-but-real counterparts
of the artifacts that drive `annotations/*_sample.parquet` in the
release package.
