# HemeFM — submission reproducibility bundle

This directory holds the **fixed, executed metrics and per-patient prediction vectors** behind the reported §4 outputs of the manuscript — the pretraining curves, the fine-tune and baseline metrics, and the DANN / configuration external predictions — published **at submission** so reviewers can audit and verify those reported outputs without re-running training. The PCA-baseline predictions, the patient-level paired bootstrap, and the B3 learning-rate-sweep trajectories are **not** bundled here but regenerate from the public `scripts/`. The trained model **checkpoints** (1.94 GB `hemefm_base`; 5.6 MB `hemefm_tiny`) are the only artefacts deferred to a **Zenodo** archive at acceptance.

All files are verbatim run outputs; single seed = 42; PyTorch 2.10 + CUDA 12.8 + Lightning 2.6.5 on 4 × RTX 5090.

## Layout

### `pretrain_base_173M/` — Stage 1 Masked Feature Modeling, `hemefm_base` (173.9 M params)
- `metrics.csv` — per-epoch held-out 16-bin reconstruction accuracy + `val/mfm_loss` (§4.1.4; 28.85 % → 44.79 % over the 30-epoch pilot, 7.2× chance).
- `hparams.yaml` — training config. **Note:** `max_seq_len: 4200` here is the **model/checkpoint position capacity**; the tokenizer (`data/processed/tokenizer.json`, `max_seq_len: 4096`) emits **4,096-position sequences = one `[CLS]` token + up to 4,095 gene tokens**. The two values are intentionally different (see manuscript §3.1, §4.1.2).
- `RESULTS.md` — run summary.

### `pretrain_tiny/` — pipeline-validation arm (≈ 700 K params)
- `metrics.csv`, `hparams.yaml`, `RESULTS.md` for `hemefm_tiny`.

### `finetune/` — within-cohort multitask fine-tune + baselines (BeatAML 2.0, n = 86 held-out, seed 42)
- `phase4prime_finetune_metrics.csv` — per-step multitask metrics (`eln_qwk`, `subtype_acc`, `drug_pearson`, survival).
- `phase4prime_v2_metrics.csv` — extended per-step metrics.
- `baselines_results.json` — B1 PCA-64 linear probe + B2 LSC17 (`eln_qwk`, `os_c_index`, `drug_spearman`).

### `external_eval/` — cross-platform transfer to GSE6891 (n = 451) + statistics
- `predictions*.tsv` — per-sample prediction vectors (`sample_id` = public GEO GSM accession; subtype / ELN / OS / drug heads):
  - `predictions.tsv`, `predictions_v2.tsv` — fine-tune configurations
  - `predictions_v4_combat.tsv` — per-cohort z-score arm
  - `predictions_v5_dann.tsv` — **DANN λ = 0.1** (headline external configuration)
  - `predictions_lambda{0.05,0.2,0.5,1.0}.tsv` — DANN λ-sweep (Figure 1)
- `linear_probe_results.json` — frozen-encoder linear probe (BeatAML val + GSE6891).
- `bootstrap_cis.json` — QWK point estimates + 95 % bootstrap CIs for all configs (DANN λ = 0.1 → QWK **0.284 [0.203, 0.363]**, matching Figure 2 / §4.3).

## Provenance
Re-running the `scripts/` and `configs/experiment/` pipelines on the public datasets (manuscript §7) regenerates every file here. The OSF deposit (DOI 10.17605/OSF.IO/ER2P5) archives the analysis plan; a Zenodo deposit will archive the trained checkpoints at acceptance.
