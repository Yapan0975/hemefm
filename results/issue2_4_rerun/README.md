# Issue-2 / Issue-4 re-run (training-matched fair comparison)

Added in response to peer review (Major-revision round).

**Issue 2 — PCA trained on the SAME 347 source patients as DANN.** `scripts/rerun_issue2.py`
reproduces the exact DANN seed-42 split (433 labeled BeatAML -> 347 train / 86 val, via the
data module's own `_derive_labels`), refits the 3 PCA configs on the 347, and re-runs the
patient-level paired bootstrap vs DANN on GSE6891.
- `issue2_pca347_results.json` — PCA raw/z-score/CORAL QWK on 347 + paired-bootstrap summary.
- `pca_zscore_347_external_predictions.tsv` — per-patient PCA-z-score(347) + DANN preds + truth (GSE6891).
- `dann_train_347_ids.txt`, `dann_val_86_ids.txt` — the exact split.

Result: PCA-z-score(347) external QWK **0.467 [0.393, 0.538]** vs DANN **0.284 [0.203, 0.360]**,
paired gap **+0.182 [+0.085, +0.278]**. The headline (simple PCA > DANN) holds under matched training.

**Issue 4 — HemeFM drug response on the same Spearman metric as PCA.** `scripts/drug_spearman.py`
recomputes HemeFM's per-drug-mean Spearman (the metric PCA reports) on the 72 val patients with drug data.
- `hemefm_drug_spearman_val.json` — HemeFM drug Spearman **0.11** (vs PCA 0.276; both < 0.40).

`pca_external_results.json` and `v25_phase1_results.json` are the original (433-patient) outputs, retained for provenance.
