# HemeFM

**A transductive AML benchmark finds simple per-cohort-z-score PCA competitive with domain-adversarial HemeFM on single-cohort cross-platform risk stratification.**

This repository is the executable companion to the manuscript:

> *A transductive AML benchmark finds simple per-cohort-z-score PCA competitive with domain-adversarial HemeFM on single-cohort cross-platform risk stratification.*
> Cuifang Hu, Ping Yang, Junqing Hu, Qihui Chu.

It hosts the Hydra configs, model code, training pipelines, and analysis scripts behind every executed result reported in the manuscript, so that reviewers and readers can inspect and reproduce the findings.

The prospectively specified internal methods protocol is archived (as retrospective archival documentation, not a pre-training timestamped pre-registration) at the Open Science Framework: **https://doi.org/10.17605/OSF.IO/ER2P5**.

> **Submission snapshot & artifact status.** The tagged release [`v1.1-submission`](https://github.com/Yapan0975/hemefm/releases/tag/v1.1-submission) is the citable snapshot accompanying the manuscript. Source code, Hydra configs, random seeds, the Dockerfile / environment specification, **and the executed per-step `metrics.csv` plus per-patient prediction vectors ([`results/`](results/))** are **public here under the MIT licence at submission**. The trained model **checkpoints** (1.94 GB `hemefm_base`) are the only artefact deferred — they will be archived in a versioned **Zenodo** release **upon manuscript acceptance** (DOI minted from the tagged release). The repository therefore documents an *open code-and-configuration release with metrics and predictions*, not a code-and-weights release.

---

## What this study found (transparent negative result)

This is a pilot/benchmarking study, not a foundation-model success report. The executed evidence shows:

1. **Pretraining produces stable above-chance reconstruction** — Masked Feature Modeling on 2,557 pan-myeloid bulk RNA-seq transcriptomes reaches 44.8 % held-out 16-bin reconstruction accuracy (7.2× chance) over the 30-epoch pilot; convergence is not claimed.
2. **Within-cohort comparisons are inconclusive and subject to model-selection bias** — on BeatAML 2.0 (n = 86 held-out), HemeFM (QWK 0.742) is point-estimate-favoured but inconclusive relative to (overlapping confidence intervals with) both a fair-encoder_lr no-pretraining baseline (best QWK 0.571) and a PCA-64 + logistic-regression baseline (QWK 0.787). The same partition was used for hyperparameter/epoch selection and confidence-interval estimation, so these p-values are descriptive only.
3. **Naive cross-platform transfer fails** — the multimodal fine-tune collapses to QWK 0.000 (class-collapse) on the GSE6891 microarray cohort.
4. **DANN partially recovers but does not reach threshold** — domain-adversarial fine-tuning (lambda = 0.1) lifts external QWK to 0.284, below the prospectively specified >= 0.55 threshold.
5. **A simple per-cohort-z-score PCA baseline outperforms DANN externally** — PCA-64 + per-cohort z-score reaches QWK 0.545 [0.475, 0.610] on GSE6891, above DANN-HemeFM in this exploratory, post-selection comparison (patient-level paired bootstrap gap +0.260 [+0.176, +0.343], p < 0.001). **Important caveats**: both DANN and PCA-zscore are *transductive* (they use the target cohort's distribution); the PCA configuration was selected among three tested (raw / z-score / CORAL), so the bootstrap p-value does not correct for configuration selection; and there is only one external cohort. The PCA > DANN result is reported as an exploratory finding awaiting confirmation on a second held-out target cohort.

See the manuscript (§4-§5) for the full evidentiary (EV1-EV8) and section-level (L1-L4) limitations.

---

## Repository layout

```
src/hemefm/            Model code, Lightning modules, training/eval/baselines/interpret
configs/               Hydra configs (model / data / trainer / experiment)
  experiment/            finetune_v3, finetune_v5_dann, finetune_b3_*, finetune_v4_combat ...
scripts/               Preprocessing, download, analysis (p0_external_pca_v2.py, v25_phase1.py,
                         run_b3_lr_sweep.sh, eln_2022_mapper.py ...)
tests/                 Unit tests
Dockerfile             Pinned CUDA 12.8 + PyTorch 2.10 environment
environment.yml        Conda environment spec
pyproject.toml         Package metadata
```

## Reproducing the key results

All experiments run on a self-managed Ubuntu host with NVIDIA RTX 5090 GPUs (PyTorch 2.10 + CUDA 12.8 + Lightning 2.6.5).

```bash
# install (editable)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# pretraining (hemefm_base 173M; multi-GPU DDP)
PYTHONPATH=src python -m hemefm.train experiment=pretrain_v3_ddp

# within-cohort multi-task fine-tune (pretrained backbone)
PYTHONPATH=src python -m hemefm.train experiment=finetune_v3

# B3 no-pretraining baseline + encoder_lr sweep (1e-5 / 1e-4 / 5e-4 / 1e-3)
bash scripts/run_b3_lr_sweep.sh

# DANN domain-adversarial fine-tune (+ lambda-sweep)
PYTHONPATH=src python -m hemefm.train experiment=finetune_v5_dann

# external PCA-64 baseline on GSE6891 (the key §4.3 bis comparison)
PYTHONPATH=src python scripts/p0_external_pca_v2.py

# patient-level paired bootstrap (PCA-zscore vs DANN-HemeFM)
PYTHONPATH=src python scripts/v25_phase1.py
```

## Data

Datasets are public and **not** redistributed in this repository:

- **BeatAML 2.0** — `biodev/beataml2.0_data` GitHub LFS mirror
- **TCGA-LAML / TARGET-AML** — NCI GDC API (STAR-Counts, open access)
- **GTEx v10 whole-blood** — adult-gtex Google Cloud Storage bucket
- **GSE6891** — NCBI GEO (Affymetrix HG-U133 Plus 2.0)

See manuscript §7 (Data and Code Availability) for exact access URLs, query strings, and access dates.

The executed per-step training metrics (`metrics.csv`) and per-patient prediction vectors are **public at submission** under [`results/`](results/) (also bundled in the `v1.1-submission` release). Only the trained model **checkpoints** (1.94 GB `hemefm_base`) are deferred — they will be archived in a versioned Zenodo release upon manuscript acceptance. Source code, configs, and seeds in this repository are sufficient to re-run every experiment from the public datasets.

## Licence

MIT (see `LICENSE`).

## Citation

If you use this code, please cite the manuscript above and the OSF protocol deposit (DOI 10.17605/OSF.IO/ER2P5).
