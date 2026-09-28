# `data/public/` — git-tracked, license-safe ColdDDI artifacts

Everything here is small, redistributable, and intended to let a fresh
clone reproduce the entire XML → CSV → Parquet pipeline end-to-end
without obtaining a DrugBank academic licence.

## Files

| Path | Description |
|---|---|
| `drugbank_toy.xml` | A **100-drug random subset** of the 1,994-drug post-pipeline pool, drawn with a fixed seed (42). Preserves the full DrugBank 5.1.13 XML schema so it exercises every code path of `coldddi.data.extract` and `coldddi.data.filter`. |
| `intermediate/raw/` | Stage 1a output: seven raw CSVs (drugs / ddi_edges / drug_enzymes / drug_targets / drug_transporters / drug_carriers / drug_pathways). |
| `intermediate/filtered/` | Stage 1b output: the same seven CSVs after the seven-step pipeline, plus `stats.json`. |
| `intermediate/enriched/` | Stage 2 output: `ddi_pk_pd_labels.csv`, `ddi_key_entities.csv`, `ddi_key_entities_type_summary.csv`, plus the Type-A projections `mediating_entities.{csv,parquet}` and `action_pairs.{csv,parquet}`. |
| `scripts/toy_drug_ids.txt` | The 100 DrugBank IDs that define `drugbank_toy.xml` (license-safe — IDs only, no records). |

## Reproducing the contents

```bash
# Step 1 (offline, requires the DrugBank XML in data/private/raw/):
#   regenerate the toy XML from a licensed full DrugBank XML
python -m coldddi.data.build_xml_subset \
  --xml data/private/raw/drugbank_5.1.13.xml \
  --drugs-list data/public/scripts/toy_drug_ids.txt \
  --out data/public/drugbank_toy.xml

# Step 2: Stage 1a — XML → seven raw CSVs (no filtering)
python -m coldddi.data.extract \
  --xml data/public/drugbank_toy.xml \
  --out data/public/intermediate/raw

# Step 3: Stage 1b — apply the seven-step filter
python -m coldddi.data.filter \
  --raw-dir data/public/intermediate/raw \
  --out     data/public/intermediate/filtered

# Step 4: Stage 2a — PK/PD keyword labelling on the 24 retained types
python -m coldddi.annotations.pkpd_keywords \
  --ddi-edges data/public/intermediate/filtered/ddi_edges.csv \
  --out       data/public/intermediate/enriched/ddi_pk_pd_labels.csv

# Step 5: Stage 2b — A/B subdivision for every positive pair
python -m coldddi.annotations.ab_subdivision \
  --ddi-edges        data/public/intermediate/filtered/ddi_edges.csv \
  --pkpd-labels      data/public/intermediate/enriched/ddi_pk_pd_labels.csv \
  --drugs-csv        data/public/intermediate/filtered/drugs.csv \
  --enzymes-csv      data/public/intermediate/filtered/drug_enzymes.csv \
  --targets-csv      data/public/intermediate/filtered/drug_targets.csv \
  --transporters-csv data/public/intermediate/filtered/drug_transporters.csv \
  --carriers-csv     data/public/intermediate/filtered/drug_carriers.csv \
  --out              data/public/intermediate/enriched/ddi_key_entities.csv \
  --summary-out      data/public/intermediate/enriched/ddi_key_entities_type_summary.csv

# Step 6: Stage 2c — derive the two Type-A release tables
python -m coldddi.annotations.derive_type_a_tables \
  --key-entities data/public/intermediate/enriched/ddi_key_entities.csv \
  --out-dir      data/public/intermediate/enriched \
  --also-csv
```

Every step above runs against the toy and against the full DrugBank XML
without modification — only the input/output paths change.
