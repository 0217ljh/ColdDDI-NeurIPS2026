# `annotation/` — human IAA validation package

Two-annotator pilot study materials supporting Appendix A.4 of the paper
(PK/PD × A/B taxonomy validation). Distinct from `../annotations/`, which
holds the auto-generated release Parquet files.

## Contents

| File | Purpose |
|---|---|
| `annotation_guide.md` | Annotator rubric and decision rules (the exact instructions both annotators worked from) |
| `annotation_blank.csv` | 500-pair sample as presented to annotators (no auto labels visible) |
| `annotation_with_auto.csv` | Same 500 pairs with the automated PK/PD + A/B labels attached |
| `sample_annotation_data.py` | Reproduces the 500-pair stratified sample from the full release |
| `compute_two_annotator_metrics.py` | Computes Cohen's κ per task (PK/PD, A/B) |
| `compute_judged_metrics.py` | Computes automated-label vs consensus precision / recall / F1 |
| `build_release_files.py` | Aggregates per-annotator inputs into `release/final_consensus.csv` |
| `two_annotator_metrics.json` | Aggregated κ output. Paper §A.4.6 numbers (κ = 0.842 / 0.839 / 0.805) |
| `judged_metrics.json` | Aggregated auto-vs-consensus precision / recall / F1 |
| `release/final_consensus.csv` | 500 rows × 11 columns. Pair ids + drug ids + names + ddi_type + auto/judge/consensus labels |

## Privacy

Raw per-annotator labels are **not** released. The three scripts that
operate on the raw inputs (`compute_two_annotator_metrics.py`,
`compute_judged_metrics.py`, `build_release_files.py`) reference the raw
files as `annotator1_raw.xlsx` and `annotator2_raw.csv`, but those files
are intentionally absent from this repository.

The scripts therefore serve as **methodology documentation** rather than
runnable artifacts. The aggregated κ / precision / recall / F1 numbers in
the two `*_metrics.json` files match the numbers reported in the paper,
and `release/final_consensus.csv` carries the consensus labels used in
downstream analysis (the per-annotator label columns have been removed).

## Reproducing the κ numbers

If you have your own two-annotator labels in the expected schema:

```bash
# Place your raw files alongside the scripts as:
#   annotator1_raw.xlsx
#   annotator2_raw.csv
python compute_two_annotator_metrics.py    # writes two_annotator_metrics.json
python compute_judged_metrics.py           # writes judged_metrics.json
python build_release_files.py              # writes release/final_consensus.csv
```

See `annotation_guide.md` for the schema definition.
