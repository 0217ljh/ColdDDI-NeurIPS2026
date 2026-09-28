# `annotations/` — auto-generated DDI label data

Machine-generated labels in Parquet format, derived from DrugBank 5.1.13.

| File | Granularity | Release scope |
|---|---|---|
| `pkpd.parquet` | 215 `ddi_type` strings → {PK, PD, Mixed} | Full release (no drug pairs) |
| `ab_sample.parquet` | 10,000-pair subset → {A, B} + mediating-entity ref | Sample only |
| `mediating_entities_sample.parquet` | 10,000-pair subset → CYP / transporter / target / carrier / pathway | Sample only |
| `action_pairs_sample.parquet` | 10,000-pair subset → (substrate, inhibitor, agonist, antagonist, ...) | Sample only |

**Sample-only rationale.** DrugBank's academic license prohibits redistribution
of the underlying records; releasing the full 565,731 (drug_a, drug_b) edge list
would constitute redistribution. The 10,000-pair sample is for pipeline validation
only. Full versions are reconstructed from a licensed DrugBank XML on the
user's machine via `reconstruct.py`.

**Not the same as `../annotation/`**, which is the human IAA validation package.
