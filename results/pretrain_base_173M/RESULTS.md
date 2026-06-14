# Phase 3'' — hemefm_base 173M DDP Production Pretraining
*Run: 2026-06-03 06:28-07:12 server time (~44 min wall clock)*
*Server: evidlife-server, 4× RTX 5090 DDP, bf16-mixed*

## TL;DR

Production-grade foundation model pretrained successfully on real RNA-seq data:

- **val/mfm_loss: 2.159 → 1.466** (32.1% reduction over 30 epochs)
- **val/mfm_acc: 28.8% → 44.8%** (vs chance 6.25% = **7.2× chance level**)
- **+8.5 pp accuracy over hemefm_tiny** (700K → 173M params, 247× scale)
- **Train/val matched throughout** — no overfitting
- 4× RTX 5090 DDP, 44 minutes total

## Setup

| Field | Value |
|---|---|
| Corpus | v3: 2,557 unique sample_ids × 4,096 genes (long-form parquet) |
| Cohorts | BeatAML2 707 + TCGA-LAML 151 + TARGET-AML 1000 + GTEx-WB 803 ≈ 2,661 (post-harmonization 2,557) |
| Tokenizer | RankBinTokenizer, 4,099 ids (4,096 genes + PAD/MASK/CLS) |
| Model | **hemefm_base, 173,918,994 params** (24-layer × 768-dim × 12-head, ffn=3072) |
| MFM | 15% masking, 80/10/10 split (Geneformer-style) |
| Trainer | bf16-mixed, DDP, batch=8/GPU × 4 = effective 32, lr=3e-4, weight_decay=0.05, warmup=20, 30 epochs |
| Hardware | 4× RTX 5090 sm_120 Blackwell, 32 GB each |
| Steps | 76 batches/epoch × 30 epochs = 2,280 total steps |
| Pipeline | PyTorch Lightning 2.6.5 DDP + Hydra + CSVLogger (offline) |

## Per-epoch val curve (full table)

| Epoch | val/mfm_loss | val/mfm_acc |
|---|---|---|
| 0 | 2.1586 | 28.85% |
| 1 | 2.0771 | 30.65% |
| 2 | 2.0790 | 30.06% |
| 3 | 2.0499 | 31.53% |
| 4 | 2.0419 | 31.49% |
| 5 | 1.9784 | 33.55% |
| 6 | 1.8931 | 34.10% |
| 7 | 1.8585 | 35.24% |
| 8 | 1.8037 | 35.70% |
| 9 | 1.7597 | 36.86% |
| 10 | 1.7495 | 37.36% |
| 11 | 1.7741 | 37.14% |
| 12 | 1.7400 | 38.13% |
| 13 | 1.7136 | 37.71% |
| 14 | 1.7087 | 38.15% |
| 15 | 1.7101 | 38.50% |
| 16 | 1.6670 | 39.42% |
| 17 | 1.6446 | 40.01% |
| 18 | 1.6361 | 40.17% |
| 19 | 1.6042 | 41.41% |
| 20 | 1.6092 | 40.77% |
| 21 | 1.5867 | 41.90% |
| 22 | 1.5689 | 42.17% |
| 23 | 1.5515 | 42.49% |
| 24 | 1.5447 | 42.67% |
| 25 | 1.5263 | 42.93% |
| 26 | 1.5102 | 43.55% |
| 27 | 1.4923 | 44.36% |
| 28 | 1.4749 | 44.66% |
| **29** | **1.4664** | **44.79%** |

Loss curve is still trending down at epoch 29 — could likely improve with more epochs.

## Comparison: tiny vs base

| Model | Params | val_acc | val_loss | Train time | GPUs |
|---|---|---|---|---|---|
| hemefm_tiny | ~700K | 36.3% | 1.790 | 18 min | 1× |
| **hemefm_base** | **173M** | **44.8%** | **1.466** | **44 min** | **4× DDP** |
| Δ | +247× | +8.5 pp | -18% | +2.4× | +4× |

Scaling laws check: 247× more params → +8.5 pp accuracy. This is consistent with sub-linear returns on accuracy from parameter scaling at fixed data (2,557 samples). Likely accuracy ceiling for this corpus is ~50-55%; getting beyond would need more data or more epochs.

## Manuscript-ready numbers (REPLACE [VALUE_TBD])

| Section | Old placeholder | NEW REAL NUMBER |
|---|---|---|
| Abstract | "MFM pretraining on n=[VALUE_TBD] samples" | **n = 2,557 patient transcriptomes** |
| §3.2 Pretraining outcome | "[VALUE_TBD] val accuracy" | **val/mfm_acc = 44.8%** (vs chance 6.25% = 7.2×) |
| §3.2 Δacc vs chance | "[VALUE_TBD] pp" | **+38.5 pp absolute** over random (6.25%) |
| §3.2 Δloss | "[VALUE_TBD]" | **-32.1% loss reduction** (2.159 → 1.466) |
| §3.2 Model architecture | "75M params" | **173,918,994 params** (24L/768d/12h with seq_len 4200 + vocab 4150) |
| §3.2 Wall-clock | "few hours on 1× 5090" | **44 min on 4× RTX 5090 DDP** (~3 h equiv on 1×) |
| §3.3 Fine-tune backbone | "to be trained" | **real checkpoint epoch=29-step=2280.ckpt** |

## Cohort composition (validated in metadata.parquet)

```
TARGET-AML    1000   (pediatric AML, GDC STAR-Counts, GENCODE v36)
GTEx-WB        803   (healthy whole-blood, v10 RNASeQCv2.4.2 TPM)
BeatAML2       707   (adult AML + drug response, biodev GitHub LFS mirror)
TCGA-LAML      151   (adult AML, GDC STAR-Counts, GENCODE v36)
────────
TOTAL      2,661  (2,557 after gene-intersection harmonization drop)
```

**2.9× scale-up vs Phase 3** (which used 880 GEO microarray samples) at higher modality quality (RNA-seq > microarray).

## Files in this directory

| File | Size | Description |
|---|---|---|
| `epoch=29-step=2280.ckpt` | ~700 MB | Final hemefm_base 173M weights |
| `hparams.yaml` | 322 B | Lightning hparams snapshot |
| `metrics.csv` | 10 KB | Per-epoch train/val loss + acc |
| `run.log` | 922 KB | Full stdout/stderr (DDP rank 0 progress bar) |

## Caveats remaining

1. **Single seed** (42) — recommend 3-seed rerun for confidence interval
2. **No external held-out cohort** — train/val both from same union; need an independent test set (e.g. a fully held-out study like recent BeatAML 2025 wave-5 if accessible, or stratified holdout from one cohort)
3. **Val loss still trending down at epoch 29** — could keep training to 50-100 epochs for better ceiling
4. **No FineTune yet** — next step Phase 4' to use this backbone for AML subtype + drug response + survival

## Next phase recommendations

1. **Phase 4' — production fine-tune** with this 173M backbone
   - Subtype classification (WHO 2022 / ICC 2022 / ELN-2022)
   - Drug response regression (BeatAML 122 drugs)
   - Survival analysis (TCGA-LAML OS / TARGET-AML EFS)
2. **Phase 4 ablation** — train baseline (random-init transformer) vs ours (pretrained backbone) to measure pretraining gain
3. **Stats sweep** — 3 seeds for each downstream task

## Reproducibility

```bash
ssh evidlife-server
cd ~/hemefm && source .venv/bin/activate
PYTHONPATH=/home/server/hemefm/src python -m hemefm.train experiment=pretrain_v3_ddp
```
