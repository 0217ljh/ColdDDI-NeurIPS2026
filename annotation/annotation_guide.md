# ColdDDI Taxonomy Validation — Annotation Guide

Welcome, and thank you for helping validate the ColdDDI mechanism taxonomy. Your labels will be used to assess the precision of an automated keyword-based classifier for drug–drug interactions (DDIs). This guide first defines what the labels mean, then explains the workflow, and walks through 4 worked examples from the actual dataset.

**Time budget**: ~6–8 hours total per annotator. No time pressure per item.

---

## What you will do

You receive **one CSV file** named with your annotator ID (e.g., `_A1.csv` / `_A2.csv`):

| File | Rows | Your job |
|---|---|---|
| `annotation_blank.csv` | 500 | For each drug **pair**, fill in two labels: a **PK / PD / Mixed** classification, and an **A / B / ?** judgment on whether the auto-identified mediating entity is the real mechanistic mediator. |

**You work alone.** Do not show your CSV to the other annotator until both finish. Rows are shuffled — consecutive rows do NOT share the same DDI type.

---

# 1. Label definitions

Read this section completely *before* opening the CSV. The procedure in §2 will assume you know all the rubric tables.

## 1.1 PK / PD / Mixed (the mechanism-class label)

| Label | Definition | Trigger words you'll see |
|---|---|---|
| **PK** (Pharmacokinetic) | Interaction acts on absorption, distribution, metabolism, or excretion (ADME). The clinical effect is mediated by *changing how much active drug reaches the target*. | metabolism, serum concentration, clearance, half-life, AUC, Cmax, bioavailability, excretion, absorption, protein binding, P-gp, CYP3A4, UGT, OATP, "increased / decreased serum level", "altered metabolism" |
| **PD** (Pharmacodynamic) | Interaction acts at the receptor / pathway / effector level. Two drugs combine to amplify, oppose, or modify each other's *effect on a clinical endpoint*. | anticoagulant activity, sedative effect, hypoglycemic effect, hypertensive activity, CNS depression, QTc prolongation, serotonin syndrome, "additive", "synergistic", "antagonize", "potentiate", "the risk or severity of …" |
| **Mixed** | Description explicitly invokes BOTH ADME and effector-level changes. **Use sparingly.** When in doubt between PK and PD, pick the dominant mechanism, not Mixed. | both classes of words appear with mechanistic prominence |

### Type-template quick recognition

| `ddi_type` template fragment | Class |
|---|---|
| "the metabolism of … can be decreased / increased" | **PK** (CYP/UGT inhibition or induction) |
| "the serum concentration of …" | **PK** (changed concentration ⇒ ADME) |
| "the excretion rate of …" | **PK** (renal/biliary excretion = ADME) |
| "the absorption of … can be decreased" | **PK** |
| "the protein binding of …" | **PK** (distribution = ADME) |
| "the bioavailability of …" | **PK** |
| "the risk or severity of … can be increased" | **PD** (clinical adverse effect) |
| "the therapeutic efficacy of …" | **PD** |
| "may increase / decrease the … activities of" | **PD** ("activities" = effect-level) |
| "may potentiate / antagonize" | **PD** |

### When to use `Mixed` (rare!)
Only if the description gives **concrete mechanistic detail for BOTH PK and PD**, e.g.:

> "Drug A can both inhibit the metabolism of Drug B (increasing its serum concentration) **and** independently potentiate its anticoagulant effect via …"

If the template is just "the risk or severity of [adverse effect] can be increased" with no ADME word, that is **PD**, not Mixed — even if the underlying biology might involve both mechanisms.

---

## 1.2 A / B / ? (the mediating-entity label)

The automated procedure may have flagged a single biomedical entity (an enzyme, target, transporter, etc.) and proposed each drug's role on that entity (substrate / inhibitor / agonist / antagonist / inducer / modulator). Your job is to judge whether that entity actually mediates the interaction.

#### Label `A` — confirm the auto-identified entity is a real mediator

The auto-flagged entity satisfies BOTH:

1. **The entity is the actual mechanistic responsible party**, as confirmed by the `ddinter_mecddi` or `drugbank_description` text. Either text literally mentions the enzyme / target by name, OR describes a process that requires that entity.
2. **The action pair is pharmacologically coherent**:

| Mechanism class | Valid action pairs (drug_a — drug_b) |
|---|---|
| **PK** | substrate–inhibitor, substrate–inducer, two-substrate competition |
| **PD same-target** | agonist–agonist (additive), antagonist–antagonist (additive), agonist–antagonist (opposing), positive-allosteric-modulator–positive-allosteric-modulator |
| **PD same-pathway** | both drugs acting on the same downstream effect (e.g., both prolong QT, both lower BP, both serotonergic) |

#### Label `B` — confirm absence of a mediator, OR reject the auto entity

- **Confirm auto-B**: `auto_has_key_entity == False`, AND no mediator can be inferred from the mecddi or DrugBank text either (mechanism is multi-hop, downstream-pathway-level, or genuinely undescribed).
- **Reject auto-A**: `auto_has_key_entity == True` but the mecddi text directly names a *different* mediator, or the action pair is incompatible with the mechanism described. (This pattern is uncommon but possible.)

#### Label `?` — undecidable

- Description is a generic placeholder ("Drug A may interact with Drug B") with no mechanism
- DDInter mecddi and DrugBank desc directly contradict each other
- The auto-flagged entity is plausible but the description is too sparse to confirm the action pair

---

# 2. Step-by-step procedure

For each row, follow these steps in sequence. **The PK/PD decision comes first, the A/B decision second** — the second judgment uses the first as input.

> **Step 1 — Read `ddi_type`** (the DrugBank type template). Form a first impression using the template-recognition table in §1.1.

> **Step 2 — Read `ddinter_mecddi`** (the DDInter v2.0 free-text mechanism description). DDInter is usually the *richest* source: it often names the exact mediator (e.g., "via CYP3A4 inhibition") and identifies the mechanism class explicitly ("additive pharmacodynamic effects", "enzyme inhibition", "altered gastric absorption"). **Trust DDInter mecddi as your primary signal whenever it is present.**

> **Step 3 — Cross-check with `drugbank_description`.** DrugBank descriptions are shorter and template-driven; use them to confirm the mechanism class suggested by mecddi. If `ddinter_mecddi` is empty (this happens for the 9 supplementary PK types), skip step 2 and rely on `ddi_type` + `drugbank_description` only — PK templates ("absorption", "protein binding", "bioavailability", "serum concentration", "metabolism", "excretion") are unambiguous on their own.

> **Step 4 — Decide PK / PD / Mixed** (using §1.1). Fill in `your_label_PK_PD_or_Mixed`.

> **Step 5 — Read the auto entity columns** (now that PK/PD is settled).
> &nbsp;&nbsp;• `auto_key_entity_name` (e.g., "Cytochrome P450 3A4")
> &nbsp;&nbsp;• `auto_key_entity_type` (enzyme / target / transporter / pathway / carrier)
> &nbsp;&nbsp;• `auto_action_drug_a` and `auto_action_drug_b` (e.g., substrate / inhibitor / agonist / antagonist)
> &nbsp;&nbsp;• `auto_chain` (e.g., "Drug A —[inhibitor]→ Entity ←[substrate]— Drug B")
>
> Form an A/B hypothesis: "the auto procedure says the mediator is **X** with action pair **Y**", or—if all five auto-entity columns are empty—"the auto procedure proposed no shared mediator."
>
> ⚠️ **Important — empty auto-entity columns are NOT the answer.** When the entity columns are empty, your job is *not* to mechanically copy `B`. Your job is to actively decide: "Is there a mediator the auto procedure missed?" Read `ddinter_mecddi` carefully — if it names a clear single mediator (e.g., "via CYP3A4 inhibition"), the right label might actually be `?` (auto missed it) or even `A` if you're confident the entity is correct and you can write it in `notes`. Only mark `B` when you have *positive evidence* that the mechanism is genuinely multi-pathway, multi-step, or undescribable by a single entity (Examples 3 and 4 below).

> **Step 6 — Re-read `ddinter_mecddi` (and DrugBank desc) with the A/B question in mind.** Two sub-checks:
> &nbsp;&nbsp;**(6a)** Does the mecddi text name the same entity that auto flagged?
> &nbsp;&nbsp;**(6b)** Is the action pair compatible with the PK/PD class you just chose? (See the valid action-pair table in §1.2.)
> If both are *yes* → strong evidence for `A`.
> If mecddi names a *different* mediator → evidence for `B`.
> If mecddi describes a multi-step physiological cascade or pathway-level convergence with no single molecular target → evidence for `B`.

> **Step 7 — Decide A / B / ?** (using §1.2). Fill in `your_label_AorB`.

> **Step 8 — Set `your_confidence_1to5`** (overall confidence in BOTH labels for this row, 1 = guess, 5 = very sure). Add `notes` if useful (required if you marked any label as `?`).

---

# 3. Worked examples — one per bucket (all from the actual dataset)

All four examples below illustrate cases where the human labels *agree with the automated labels*, with one example per mechanism × A/B bucket so you see what each cell of the 2×2 looks like. Most rows in your CSV will be similarly straightforward.

---

**Example 1 — PK-A (enzyme-mediated, A confirmed)**

> **drug_a**: Amprenavir &nbsp;·&nbsp; **drug_b**: Efavirenz
> **ddi_type**: *"The serum concentration of the active metabolites of can be reduced when is used in combination with resulting in a loss in efficacy"*
> **drugbank_description**: "The serum concentration of the active metabolites of Amprenavir can be reduced when Amprenavir is used in combination with Efavirenz resulting in a loss in efficacy."
> **ddinter_mecddi**: "Coadministration with efavirenz may decrease the plasma concentrations of amprenavir. **The mechanism is efavirenz induction of CYP450 3A4, the isoenzyme responsible for the metabolic clearance of amprenavir.**"
> **auto_key_entity**: Cytochrome P450 3A4 (enzyme)
> **auto_action**: Amprenavir = substrate, Efavirenz = inducer

Walk-through using the steps:
1. `ddi_type`: "serum concentration of the active metabolites … reduced" → PK template.
2. mecddi: "**efavirenz induction of CYP450 3A4, the isoenzyme responsible for the metabolic clearance of amprenavir**" → confirms PK + names the mediator.
3. DrugBank desc consistent (decrease in serum concentration via metabolic clearance).
4. **PK/PD decision: `PK`.**
5. Auto flag: CYP3A4 enzyme; Amprenavir = substrate, Efavirenz = inducer.
6. mecddi check: same entity (CYP3A4) ✓ and action pair (substrate–inducer) is a valid PK pair ✓.
7. **A/B decision: `A`.**
8. Confidence: 5.

---

**Example 2 — PD-A (additive on shared receptor, A confirmed)**

> **drug_a**: Midazolam &nbsp;·&nbsp; **drug_b**: Halazepam
> **ddi_type**: *"The risk or severity of sedation and CNS depression can be increased when is combined with"*
> **drugbank_description**: "The risk or severity of sedation and CNS depression can be increased when Midazolam is combined with Halazepam."
> **ddinter_mecddi**: "Central nervous system- and/or respiratory-depressant effects may be **additively or synergistically increased** in patients taking multiple drugs that cause these effects, especially in elderly or debilitated patients."
> **auto_key_entity**: GABA(A) Receptor (target)
> **auto_action**: Midazolam = positive allosteric modulator, Halazepam = positive allosteric modulator

Walk-through:
1. `ddi_type`: "risk or severity of sedation and CNS depression" → PD template.
2. mecddi: "**additively or synergistically increased** CNS / respiratory-depressant effects" → confirms PD with additive flavor.
3. DrugBank desc consistent.
4. **PK/PD decision: `PD`.**
5. Auto flag: GABA(A) receptor; both drugs = positive allosteric modulator.
6. Both midazolam and halazepam are benzodiazepines acting on GABA(A); same target + both PAMs is a valid PD-same-target additive pair.
7. **A/B decision: `A`.**
8. Confidence: 5.

---

**Example 3 — PK-B (multi-step physiological cascade, B confirmed)**

> **drug_a**: Levodopa &nbsp;·&nbsp; **drug_b**: Solifenacin
> **ddi_type**: *"The absorption of can be decreased when combined with"*
> **drugbank_description**: "The absorption of Levodopa can be decreased when combined with Solifenacin."
> **ddinter_mecddi**: "Anticholinergic agents may decrease the absorption and oral bioavailability of levodopa. The proposed mechanism involves **increased gastrointestinal transit time due to reduction of stomach and intestinal motility** by anticholinergic agents, thereby increasing the gastric degradation of levodopa and reducing the amount available for absorption in the small intestine."
> **auto_key_entity**: *(none — auto_has_key_entity = False)*

Walk-through:
1. `ddi_type`: "absorption of … decreased" → PK template.
2. mecddi: "**increased gastrointestinal transit time due to reduction of stomach and intestinal motility**" → still PK (it is an absorption mechanism), but the responsible factor is a multi-step physiological process, not a single molecule.
3. DrugBank desc consistent.
4. **PK/PD decision: `PK`.**
5. Auto flag: no entity (`auto_has_key_entity = False`).
6. mecddi: no single shared molecular target — the cascade is anticholinergic action → reduced GI motility → gastric degradation → reduced absorption. The absence of a mediating entity is correct.
7. **A/B decision: `B`.**
8. Confidence: 5.

---

**Example 4 — PD-B (parallel pathways converging on a clinical endpoint, B confirmed)**

> **drug_a**: Quinapril &nbsp;·&nbsp; **drug_b**: Silodosin
> **ddi_type**: *"The risk or severity of orthostatic hypotension and dizziness can be increased when is combined with"*
> **drugbank_description**: "The risk or severity of orthostatic hypotension and dizziness can be increased when Quinapril is combined with Silodosin."
> **ddinter_mecddi**: "**Additive hypotensive effects may occur when ACE inhibitors are used in combination with alpha-blockers.** In the presence of ACE inhibition, the risk and/or severity of first-dose effects associated with alpha-blockers such as postural hypotension and syncope may be increased."
> **auto_key_entity**: *(none — auto_has_key_entity = False)*

Walk-through:
1. `ddi_type`: "risk or severity of orthostatic hypotension and dizziness" → PD template.
2. mecddi: "**Additive hypotensive effects … ACE inhibitors … alpha-blockers**" → confirms PD with additive flavor on the hypotension endpoint.
3. DrugBank desc consistent.
4. **PK/PD decision: `PD`.**
5. Auto flag: no entity.
6. Quinapril acts via ACE; Silodosin acts via alpha-1A receptor. Two **different** molecular targets converging on the same clinical endpoint (lowered BP). This is the textbook PD-B pattern; auto-B is correct.
7. **A/B decision: `B`.**
8. Confidence: 5.

---

# 4. Common pitfalls

1. **Don't confuse "shared target" with "mediator"**. Two drugs both binding the same protein doesn't always mean the interaction goes through that protein. Always read the mecddi text — the named mediator there is the source of truth.

2. **PK templates with "metabolism / serum concentration"** strongly suggest a CYP/UGT enzyme is the mediator. If `auto_key_entity_type == enzyme`, the action pair is substrate–inhibitor or substrate–inducer, and the mecddi confirms the enzyme by name (Example 1) → almost always `A`.

3. **PD templates ("risk of …", "activities")** can be either `A` (shared target — Example 2) or `B` (parallel pathways — Example 4). The mecddi will tell you which.

4. **Empty `ddinter_mecddi`** — the 9 supplementary PK types and a few PD pairs may have no DDInter text. Use only the `drugbank_description` and reference textbooks. If you cannot decide, mark `?`.

5. **The auto-flagged entity might be plausible but unconfirmed** — if neither mecddi nor DrugBank desc actually mentions it, mark `?` (not `A`). We want *evidence in the text*, not plausibility.

6. **Templates with placeholder names** — judge by template + drug names + auto-entity. If still impossible, mark `?` and explain in notes.

---

# 5. Reference materials

You may consult any of these freely while annotating:

1. **DrugBank** drug pages: <https://go.drugbank.com/drugs/{drugbank_id}>
2. **DDInter v2.0**: <https://ddinter2.scbdd.com/> — search by drug name or `DDInter ID`
3. **Stockley's Drug Interactions**, ed. K. Baxter (Pharmaceutical Press), latest available edition
4. **Goodman & Gilman's *The Pharmacological Basis of Therapeutics***, 13th ed., Brunton et al.
5. **FDA Drug Interaction Guidance for Industry** (2020): <https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-drug-interaction-studies-cytochrome-p450-enzyme-and-transporter-mediated-drug-interactions>

If you cannot find the relevant info in any of these for a specific item, mark `?` and proceed.

---

# 6. Submission

When done, save your filled file as `annotation_<your_id>.csv` (e.g., `annotation_A1.csv` or `annotation_A2.csv`) and email it back.

**Do not modify column order or add/remove columns.** Use UTF-8 encoding (most spreadsheet apps default to this).

Thank you!
