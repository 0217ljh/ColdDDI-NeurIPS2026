# Clinical-description generation

The released descriptions are the masked inputs used by the P6/P7 experiments. Generation used drug names and DrugBank IDs, not DrugBank description paragraphs. No API call is needed to use the released JSON.

| Setting | Value |
|---|---|
| Model requested | `gpt-4o` (a dated snapshot was not recorded) |
| Temperature | `0.3` |
| Maximum completion tokens | `400` |
| Requested length | 150–200 words |
| Batch size / concurrent requests | 80 / 20 |

## System prompt

```text
You are a clinical pharmacologist producing structured drug reference entries. Follow the user's format and exclusion rules exactly.
```

## User prompt

Substitute `{name}` and `{db_id}` for each drug. The following is the original generation prompt, reproduced verbatim.

```text
You are writing a clinical pharmacology reference entry for a physician's drug lookup. Describe the drug "{name}" (DrugBank ID: {db_id}) in 150-200 words, covering the following clinical dimensions:

1. Pharmacological class (e.g., "beta-blocker", "SSRI", "statin").
2. Primary and secondary clinical indications (diseases treated).
3. Physiological or organ/tissue-level mechanism of action (describe how the drug affects the body at the physiological system level, e.g., "reduces cardiac output", "increases synaptic serotonin availability"; do NOT describe it at the individual protein level).
4. Route of administration and typical dosing pattern.
5. Common adverse effects.
6. Contraindications and special populations (pregnancy, elderly, renal / hepatic impairment).
7. Clinical monitoring requirements and notable FDA-label warnings.

STRICT EXCLUSIONS -- these MUST NOT appear in the description:
- Any specific enzyme name (do NOT write "CYP3A4", "CYP2D6", "UGT1A1", "monoamine oxidase", or any other named metabolising enzyme).
- Any specific transporter name (do NOT write "P-glycoprotein", "P-gp", "OATP", "BCRP", "OCT2", "MATE", "MRP", etc.).
- Any specific molecular target or receptor by protein name (do NOT write "HMG-CoA reductase", "5-HT2A receptor", "dopamine D2 receptor", "VKORC1", or any other named protein target). Use functional / physiological descriptors instead ("reduces cholesterol synthesis", "increases synaptic serotonin", "antagonises vitamin K-dependent clotting").
- Any specific carrier protein by name (do NOT write "albumin").
- Any specific interacting drug or class of interacting drugs by name (do NOT mention any other drug or drug class that "{name}" is known to interact with, and do NOT list combination therapy partners by name).
- Any general reference to the existence of drug-drug interactions (do NOT write phrases such as "interactions with other medications", "drug-drug interactions", "drug interactions", "has many interactions", "be aware of interactions", or any equivalent phrasing that flags this drug as prone to interactions).
- Chemical structure or synthesis details.

Reason for these exclusions: this description will be paired with a structured knowledge graph that already lists the drug's target / enzyme / transporter / carrier proteins and its known drug-drug interactions. The description must add clinical and physiological information that is complementary to, not redundant with, that structured knowledge.

Tone: neutral, clinical, factual. Length: 150-200 words. Output only the description prose -- no headings, no bullet lists, no preamble, no closing sentence such as "in summary".
```

## Audit and masking

The original audit checked each drug's KG entity names, a global entity-name list, other drug names, and DDI-related phrases. Confirmed per-drug entity matches and the short-form patterns in `scripts/audit_descriptions.py` were replaced by `[MASKED_ENTITY]`, case-insensitively with word boundaries. Longer entity names were processed first; adjacent repeated mask tokens were collapsed. Global matches were review candidates, not automatically treated as confirmed leakage.

The released input contains 800 descriptions, with 95 mask markers across 51 drugs. Its source-file SHA-256 is `329773da102939e537aa5671f8ecfc01adcf93ee1a6edaa8c00b404ac3df9e43`. `annotations/drug800_desc_audit.json` records a fresh verification of this masked artifact, not a reconstruction of the unavailable full historical before/after report.

Use the descriptions as benchmark inputs, not as clinical advice. Generated prose has not been medically validated.
