# `data/private/` — local-only DrugBank-licensed artifacts

**This entire directory is git-ignored.** Only this `_README.md` and the
empty `.gitkeep` placeholders are tracked, so the folder structure is
visible to users but no actual data ever enters the repository.

DrugBank's academic license (CC-BY-NC 4.0 + ToS) prohibits redistribution
of the underlying records. All real DrugBank-derived files live here and
never leave the user's machine.

## Layout

| Path | Contents | Produced by |
|---|---|---|
| `raw/drugbank_5.1.13.xml` | The full DrugBank XML (and any other licensed XML downloads). | The user, under their own DrugBank academic licence. |
| `subset/drugbank_subset_*drugs.xml` | Smaller XMLs covering a fixed drug list (e.g. the 160-drug development sample, or the toy-XML inputs before they ship to `data/public/`). | `python -m coldddi.data.build_xml_subset --xml data/private/raw/drugbank_5.1.13.xml --drugs-list ... --out ...` |
| `intermediate/raw/` | Stage 1a output for the **full** XML (seven raw CSVs: drugs / ddi_edges / drug_enzymes / drug_targets / drug_transporters / drug_carriers / drug_pathways). | `python -m coldddi.data.extract --xml data/private/raw/drugbank_5.1.13.xml --out data/private/intermediate/raw` |
| `intermediate/filtered/` | Stage 1b output for the full XML (seven filtered CSVs + `stats.json`). | `python -m coldddi.data.filter --raw-dir data/private/intermediate/raw --out data/private/intermediate/filtered` |
| `intermediate/enriched/` | Stage 2 output (PK/PD labels, A/B subdivision, Type-A projections). | The annotation modules under `coldddi.annotations.*`. |
| `outputs_full/annotations/` | Full four release Parquet artifacts (`pkpd / ab / mediating_entities / action_pairs`). | A future `reconstruct.py` driver (Stage 3). |
| `outputs_full/splits/` | Full S0/S1/S2 splits across seeds 42/43/44. | A future `reconstruct.py` driver (Stage 4). |

## What you (the user) need to do

1. Download `drugbank_5.1.13.xml` from DrugBank under your academic licence and place it at `data/private/raw/drugbank_5.1.13.xml`.
2. Run `python -m coldddi.data.extract --xml data/private/raw/drugbank_5.1.13.xml --out data/private/intermediate/raw` to produce the seven raw CSVs.
3. Run `python -m coldddi.data.filter --raw-dir data/private/intermediate/raw --out data/private/intermediate/filtered` to apply the seven-step filtering pipeline (Appendix A.1).
4. Run the Stage-2 annotation modules to produce `intermediate/enriched/`. See the equivalent commands in `data/public/_README.md`; only the input/output paths change.

## What the repository ships (and does not)

- ✅ Ships: schema / column descriptions / sample-only Parquet release artifacts (Stage 3 output) / 100-drug toy XML.
- ❌ Does not ship: any file under this `data/private/` tree.
