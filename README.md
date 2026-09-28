# ColdDDI

![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8a2be2?style=flat)
![arXiv Preprint](https://img.shields.io/badge/arXiv-Preprint-b31b1b?style=flat)
![OpenReview Paper](https://img.shields.io/badge/OpenReview-Paper-0969da?style=flat)
[![Code License MIT](https://img.shields.io/badge/Code_License-MIT-4c9c2a?style=flat)](LICENSE)

ColdDDI is a diagnostic benchmark for evaluating knowledge utilization in cold-start drug–drug interaction prediction.

## 🔔 News

- [x] **[2026.09.24]** ColdDDI was accepted to **NeurIPS 2026 (Poster)**! 🎉

## 📑 Contents

- [🔍 Overview](#overview)
- [📥 Data Access Notice](#data-access-notice)
- [🗂️ Data Layout](#data-layout)
- [🛠️ Environment](#reference-environment)
- [⚡ Quick Start](#quick-start)
- [🔁 Reproducing](#reproducing-on-the-full-drugbank-xml)
- [🤖 Pipeline](#llm-stack-l1-l6)
  - [🚀 Running the LLM Pipeline](#running-the-llm-pipeline)
  - [📊 Indicator Coverage](#l6-indicator-coverage)
- [📁 Repository Structure](#repository-structure)
- [⚖️ License](#license)
- [✉️ Contact](#contact)
- [📄 Citation](#citation)

<a id="overview"></a>

## 🔍 Overview

ColdDDI is a benchmark for evaluating drug-drug interaction (DDI) prediction methods under **cold-start** settings, where one or both drugs have no recorded interaction history. It comprises:

- **1,900 drugs** and **565,731 positive DDI pairs** derived from DrugBank 5.1.13
- **Drug-wise disjoint splits** S0 (transductive) / S1 (semi-inductive) / S2 (fully inductive)
- **Mechanism-stratified annotations**: each interaction is labeled by pharmacokinetic (PK) vs pharmacodynamic (PD) mechanism, and by whether shared mediating entities are present in the knowledge graph (Type A / Type B)
- **Diagnostic toolkit** for quantifying knowledge utilization across model families

On this benchmark, we evaluate 8 conventional baselines and 13 LLMs across 5 prompt patterns, revealing that the same KG context is exploited in opposite directions across architectures—a failure mode that aggregate metrics cannot surface.

<a id="data-access-notice"></a>

## 📥 Data Access Notice

> ⚠️ **Temporary download pause.** DrugBank has temporarily paused academic downloads while updating its data distribution process. Please check the [official download page](https://go.drugbank.com/releases/5-1-13) for updates on when access will resume. In the meantime, you can try adapting the pipeline to your own DDI dataset.

ColdDDI is derived from **DrugBank version 5.1.13** (released January 2025). DrugBank's academic license **prohibits redistribution** of the underlying records. Consequently, this repository ships only:

- A **100-drug toy XML subset** under `data/public/drugbank_toy.xml` (small enough for reproducibility)
  - The complete **toy-scale pipeline outputs** under `data/public/intermediate/{raw,filtered,enriched}/`
  - The toy-scale **pair-level Parquet artifacts** under `annotations/*_sample.parquet`
- The **full 215-type PK/PD taxonomy** at `annotations/pkpd.parquet` (no pair-level information, hence license-safe)
- A **reconstruction pipeline** that users can run on their own DrugBank XML to produce the full 1,900-drug / 565,731-pair benchmark

To reproduce the full benchmark, users must:

1. Apply for DrugBank academic access at <https://go.drugbank.com/releases/5-1-13>
2. Download the DrugBank 5.1.13 XML release
3. Create the local-only directory: `mkdir -p data/private/raw`
4. Place the XML at `data/private/raw/drugbank_5.1.13.xml`
5. Run the single command under [Reproducing](#reproducing-on-the-full-drugbank-xml) below.

<a id="data-layout"></a>

## 🗂️ Data Layout

```
data/
├── public/                            # ✅ shipped in git, runnable without a DrugBank licence
│   ├── drugbank_toy.xml              # 100-drug random subset (seed=42)
│   ├── intermediate/
│   │   ├── raw/        (Stage 1a)    # 7 csvs: drugs / ddi_edges / drug_{enzymes,targets,transporters,carriers,pathways}
│   │   ├── filtered/   (Stage 1b)    # same 7 csvs after the 7-step filter, plus stats.json + type_to_text.json
│   │   └── enriched/   (Stage 2)     # ddi_pk_pd_labels / ddi_key_entities / mediating_entities / action_pairs (csv + parquet)
│   └── scripts/
│       └── toy_drug_ids.txt          # the 100 DrugBank IDs (license-safe)
└── private/                          # create this by yourself (git-ignored, user-local only)
```

<a id="reference-environment"></a>

## 🛠️ Environment

- **OS**: Linux (also tested under WSL2 / Ubuntu)
- **Python**: 3.10
- **NumPy**: 1.26.x or 2.x
- **PyTorch**: ≥ 2.0 (CUDA)

Full dependency list: `requirements.txt`.

<a id="quick-start"></a>

## ⚡ Quick Start

```bash
# 1. Clone & install (Linux / Python 3.10)
git clone https://github.com/0217ljh/ColdDDI-NeurIPS2026.git
cd ColdDDI-NeurIPS2026
pip install -r requirements.txt

# 2. Smoke tests
python -m pytest tests/ -v
```

Expected toy numbers: **86 drugs / 1,383 edges / 24 ddi_types after Step 7**, **523 Type-A pairs** with mediating-entity coverage.

<a id="reproducing-on-the-full-drugbank-xml"></a>

## 🔁 Reproducing

After placing your licensed `drugbank_5.1.13.xml` at `data/private/raw/`:

```bash
python reconstruct.py \
  --drugbank data/private/raw/drugbank_5.1.13.xml \
  --output   data/private \
  --release-mode full \
  --seeds 42 43 44 \
  --n-train-negative-epochs 4
```

Pass `--skip-stages stageX ...` to resume after a partial run; see
`python reconstruct.py --help` for per-stage controls.

Expected full numbers (paper Table A.1, 100% match):

| Stage | Drugs | DDI edges | Types |
|---|---:|---:|---:|
| Step 0 (XML raw) | 17,430 | 1,428,193 | — |
| Step 7 (final) | **1,900** | **565,731** | **215** |

Type-A coverage: **189,404 / 565,731 = 33.5%** (167,697 PK + 21,686 PD + 21 Mixed).

<a id="llm-stack-l1-l6"></a>

## 🤖 Pipeline

The LLM pipeline is layered so each stage can be tested in isolation and chained end-to-end. Layer responsibilities:

| Layer | Module | Purpose |
|---|---|---|
| L1 prompts | `coldddi.llm.prompts` | Renders P1 zero-shot / P2 few-shot SMILES / P3 one-hop KG triplets / P4 one-hop KG sequence (OHS) / P5 few-shot 2-hop, plus R0-R3 mask variants (name / key-entity / both / neither) |
| L2 inference | `coldddi.llm.inference` | `LLMInferenceRunner` — base + LoRA adapter, Yes/No logit scoring, batched `score_samples()` |
| L3 trainer | `coldddi.llm.trainer` | `LoRATrainer` — HuggingFace `Trainer` wrapper with **multi-split eval** (logs `eval_S0_loss / eval_S1_loss / eval_S2_loss` simultaneously) and a configurable `primary_val_split` |
| L4 collator | `coldddi.llm.collator`, `coldddi.data.dataset` | Causal-LM masked-label collator + `PairDataset` split-aware sample builder |
| L5 select-best | `coldddi.llm.select_best` | `parse_candidate_ckpts → score_candidate_ckpts → select_best` — picks the best LoRA per split via val AUC |
| L6 diagnostics | `coldddi.diagnostics` | KPS / KSAI indicators (byte-exact port of upstream `sec5-3/2_indicators/`), `compute_ab_gap`, drug-replacement swap candidate builder |

<a id="running-the-llm-pipeline"></a>

### 🚀 Running the LLM Pipeline

`run_llm.py` is the canonical runner. Single cells reproduce by varying CLI flags:

```bash
# Qwen2.5-0.5B + P4 + 800-drug + seed 42 (smallest paper-grade cell)
python scripts/run_llm.py --model qwen-0.5b --dataset 800-drug \
    --prompt P4 --seed 42

# Llama-3.2-1B + P1 zero-shot on 1900-drug (no FT, just inference)
python scripts/run_llm.py --model llama-1b --dataset 1900-drug \
    --prompt P1 --seed 42 --skip-ft

# Sweep multiple cells from a shell loop
for seed in 42 43 44; do
  python scripts/run_llm.py --model qwen-0.5b --dataset 800-drug \
      --prompt P4 --seed "$seed"
done
```

Shortcuts:

| Flag | Accepts |
|---|---|
| `--model` | `qwen-0.5b` / `qwen-3b` / `qwen-7b` / `qwen-14b` / `llama-1b` / `llama-3b`, or any HuggingFace id |
| `--dataset` | `toy` / `800-drug` / `1900-drug`, or a directory / `.pkl` path |
| `--prompt` | `P1` / `P2` / `P3` / `P4` / `P5` |
| `--seed` | 42 (default) — paper uses 42 / 43 / 44 for 3-seed reporting |
| `--skip-ft` | Zero-shot path (no LoRA training) |
| `--train-subset` / `--val-subset` / `--test-subset` | Sample caps for fast debug; full splits used by default |
| `--train-bs` / `--eval-bs` | Override the VRAM-aware auto batch-size (default: derived from `AutoConfig`) |

Every cell writes a self-contained `runs/<run_id>/` directory:

```
runs/qwen__qwen2-5-0-5b__800-drug__P4__seed42/
├── ckpts/         # LoRA adapter snapshots (HF Trainer)
    ├──checkpoint-20/
    ├──checkpoint-40/
    ├── fit_info.json               # training contract (yes/no token, prompt, model, max_length)
├── manifest.json               # per-split best ckpt + val AUC (L5 output)
├── test_predictions.parquet    # test_s2 per-row p_yes / pred / prompt
└── result.json                 # headline: test_s2 AUROC for the cell
```

If crashed mid-run, just re-run the same command. Training picks up from the latest checkpoint in `ckpts/`, and test inference skips pairs already in `test_predictions.parquet`. No flags needed.


<a id="l6-indicator-coverage"></a>

### 📊 Indicator Coverage

| Model class | KPS-F | KPS-Name | KPS-KG | KPS-KG-Masked | KPS-Name-KGMasked | KSAI | KPS-mol | KPS-KG (channel) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LLMs (P1-P5) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| DeepDDI / SSI / DSN / HDN / EmerGNN / TextDDI | ✓ | — | — | — | — | — | — | — |
| MKG-FENN / TIGER | ✓ | — | — | — | — | — | ✓ | ✓ |

A dash (`—`) means the model has no separable channel to mask; the indicator returns NaN rows per bucket. Buckets are `PK-A`, `PK-B`, `PD-A`, `PD-B`, plus an `ALL` aggregate over positive base pairs only (matches upstream's `_agg_buckets` convention). The headline A-B gap is `(PK-A + PD-A)/2 - (PK-B + PD-B)/2`.


<a id="repository-structure"></a>

## 📁 Repository Structure

```
coldddi/
├── data/            # Stage 1-4: extract / filter / release Parquet / drug-wise splits
├── annotations/     # Stage 2: PK-PD keywords + A/B subdivision + Type-A
├── baselines/       # 8 conventional baselines (DeepDDI / SSI-DDI / DSN-DDI / HDN-DDI / EmerGNN / TextDDI / TIGER / MKG-FENN)
├── llm/             # LLM stack (L1 prompts / L2 inference / L3 LoRA trainer / L4 collator / L5 select-best)
├── diagnostics/     # L6 KPS / KSAI indicators
└── eval/            # Shared metric helpers

reconstruct.py       # Data-pipeline driver
evaluate.py          # Baseline / LLM eval driver
scripts/run_llm.py   # Paper-grade one-click runner
tests/               # pytest suite
```

**Note:** Several baselines were ported from per-baseline upstream training scripts named `train_custom_bundle.py` (and `train_custom_bundle_neg_edges.py` for TIGER). The release package consolidates them under `coldddi/baselines/<name>/` and exposes a single unified entry point through `evaluate.py`.

<a id="license"></a>

## ⚖️ License

- **Code**: MIT License (see [LICENSE](LICENSE)).
- **Mechanism annotations and split indices**: CC BY-SA 4.0.
- **Underlying DrugBank data**: governed by the DrugBank academic license (CC-BY-NC 4.0); the 100-drug toy subset shipped under `data/public/` and the `_sample.parquet` files under `annotations/` are redistributed under DrugBank's small-scale non-commercial reproducibility clause with attribution.

<a id="contact"></a>

## ✉️ Contact

- **Name:** Jiheng Liang
- **Affiliation:** Data Science, William & Mary
- **Email:** [jliang09@wm.edu](mailto:jliang09@wm.edu)

<a id="citation"></a>

## 📄 Citation

```bibtex
@inproceedings{liang2026coldddi,
  author = {Jiheng Liang and Chen Zhao and Di Wu and Chenyang Bu and Yunpeng Hong and Xingquan Zhu and Yi He},
  title  = {{ColdDDI}: Evaluating Knowledge Utilization in Cold-Start Drug-Drug Interaction Prediction},
  booktitle = {NeurIPS 2026 Evaluations \& Datasets Track},
  year   = {2026}
}
```
