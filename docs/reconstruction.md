# Reconstruction and checks

```bash
python reconstruct.py \
  --drugbank /path/to/drugbank_5.1.13.xml \
  --output data/private \
  --include-subset both \
  --seeds 42 43 44

python sanity_check.py --data data/private --strict
```

`--include-subset` accepts `full`, `800`, or `both`. The default remains
`full` to preserve existing commands. Toy data also uses `full`: an
86-drug dataset cannot supply an 800-drug sample.

Each 800-drug sample uses `numpy.random.default_rng(seed)` without replacement.
The candidate order is recorded before Step 7 removes low-degree drugs:
descending Step 6 degree of canonical `(min-ID, max-ID)` pairs, with explicit `quicksort` tie ordering to preserve
the original pandas 2.x sampling behavior. The existing split and negative
sampling functions are reused. No additional filtering or resampling is done
after selecting the drugs. A sample containing isolated drugs fails explicitly.

| Dataset | Data directory | A/B annotations |
|---|---|---|
| Full | `data/private/intermediate` | `data/private/outputs_full/annotations/ab.parquet` |
| 800, seed 42 | `data/private/subsets/800/seed42/intermediate` | `data/private/subsets/800/seed42/outputs_full/annotations/ab.parquet` |

Use these paths with `scripts/run_benchmark.py`. When built under `data/private`,
`run_llm.py --dataset 800-drug` and `evaluate.py --subset 800` also resolve to
the new release directory; their legacy pickle fallback is retained.
Each sampled dataset is
self-contained. Existing subset output directories are not overwritten;
choose a new `--output` for another reconstruction. In `800` mode, the full
filtered tables and annotations are still built as inputs, but full splits
are not generated. The generated `reconstruction.json` lists datasets to check.

The checks cover all generated seeds, positive split coverage and disjointness,
G1/G2 membership, negative labels and ratios, KG references, and A/B annotations.
`--strict` additionally verifies every file against the saved MD5 manifest,
including cached training-negative epochs. Checking never rewrites data or hashes.
These hashes are recorded at reconstruction time: they detect changed artifacts,
but are not an independent, published paper-reference manifest.

For a reference count check on the full dataset:

```bash
python sanity_check.py \
  --data data/private/intermediate \
  --profile drugbank-5.1.13 \
  --strict
```

This checks 1,900 drugs, 565,731 positive pairs and 215 DDI types, not identity
of every record. Use `--profile toy` for the toy counts. An independent manifest
can be supplied with `--reference-checksums PATH` for a single dataset; paths
inside that manifest are relative to the parent of its `intermediate/` directory,
so the matching annotation Parquet files are covered as well.

Older reconstructions without `filtered/subset_candidates.json` must rerun
stage1b before generating subsets. The candidate list cannot be recovered by
sorting drug IDs or recomputing degrees after Step 7.

For LLM reproduction, use the maintained `scripts/run_benchmark.py` entry point
described in [the runner guide](benchmark.md). Historical appendix commands
using `llm/lora_train.py` or `evaluate.py --method llm-ft` are not valid public
interfaces and are not reintroduced by this reconstruction update.
