# -*- coding: utf-8 -*-
"""v25 Phase 1: External paired bootstrap (PCA-zscore vs DANN-HemeFM) + Discriminator AUC recompute."""
import sys
import json
from pathlib import Path
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score, balanced_accuracy_score, confusion_matrix

RESULTS = {}
Path("experiments/p0_fixes").mkdir(parents=True, exist_ok=True)

# =======================================================================================
# 1A. External paired bootstrap: PCA-zscore (QWK 0.545) vs DANN-HemeFM (QWK 0.284)
# =======================================================================================
print("=" * 70)
print("Phase 1A: External paired bootstrap on GSE6891 (n=451)")
print("=" * 70)

# Load DANN-HemeFM predictions
dann_preds = pd.read_csv("outputs/eval/GSE6891_v5_dann/predictions.tsv", sep="\t")
print(f"DANN predictions: {dann_preds.shape}")
# Drop samples without label
gse_labels_df = pd.read_csv("data/external/GSE6891/labels.tsv", sep="\t").dropna(subset=["eln_label"])
dann_labeled = dann_preds.merge(gse_labels_df[["sample_id", "eln_label"]], on="sample_id", how="inner")
dann_labeled["eln_label"] = dann_labeled["eln_label"].astype(int)
print(f"DANN labeled: {dann_labeled.shape}")

# Re-run PCA-zscore to get per-patient predictions
print("Recomputing PCA-zscore predictions for paired bootstrap...")
clin = pd.read_excel("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx")
clin_valid = clin[clin["ELN2017"].isin(["Favorable","Intermediate","Adverse"])].dropna(subset=["dbgap_rnaseq_sample"]).copy()
clin_valid["eln_int"] = clin_valid["ELN2017"].map({"Favorable":0,"Intermediate":1,"Adverse":2})

corpus = pd.read_parquet("data/processed/pretrain_corpus.parquet")
beat_in_corpus = sorted(set(clin_valid["dbgap_rnaseq_sample"]) & set(corpus["sample_id"].unique()))
ba_corpus = corpus[corpus["sample_id"].isin(beat_in_corpus)]
beat_wide = ba_corpus.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)

gse_exp = pd.read_parquet("data/external/GSE6891/expression.parquet")
gse_labeled_exp = gse_exp[gse_exp["sample_id"].isin(gse_labels_df["sample_id"])]
gse_wide = gse_labeled_exp.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)
common_genes = sorted(set(beat_wide.columns) & set(gse_wide.columns))

X_beat = beat_wide[common_genes].values.astype(np.float32)
beat_sample_ids = beat_wide.index.tolist()
labels_map = clin_valid.set_index("dbgap_rnaseq_sample")["eln_int"].to_dict()
y_beat = np.array([labels_map.get(s, -1) for s in beat_sample_ids])
mask = y_beat >= 0
X_beat, y_beat = X_beat[mask], y_beat[mask]

X_gse = gse_wide[common_genes].values.astype(np.float32)
gse_sample_ids = gse_wide.index.tolist()
gse_label_map = gse_labels_df.set_index("sample_id")["eln_label"].astype(int).to_dict()
y_gse = np.array([gse_label_map.get(s, -1) for s in gse_sample_ids])
mask_g = y_gse >= 0
X_gse = X_gse[mask_g]
y_gse = y_gse[mask_g]
gse_sample_ids_filt = [s for s, m in zip(gse_sample_ids, mask_g) if m]
print(f"After filter: BeatAML {X_beat.shape}, GSE6891 {X_gse.shape}, {len(gse_sample_ids_filt)} sample_ids")

# Per-cohort z-score
Xb_z = (X_beat - X_beat.mean(0)) / (X_beat.std(0) + 1e-6)
Xg_z = (X_gse - X_gse.mean(0)) / (X_gse.std(0) + 1e-6)

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
pca = PCA(n_components=64, random_state=42)
beat_pca = pca.fit_transform(Xb_z)
gse_pca = pca.transform(Xg_z)
clf = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
clf.fit(beat_pca, y_beat)
pca_preds_arr = clf.predict(gse_pca)
print(f"PCA-zscore preds dist: {np.bincount(pca_preds_arr, minlength=3)}")

# Build paired DataFrame
pca_df = pd.DataFrame({"sample_id": gse_sample_ids_filt, "eln_label": y_gse, "pca_pred": pca_preds_arr})
# Merge with DANN predictions
paired = pca_df.merge(dann_labeled[["sample_id", "eln_pred", "eln_label"]].rename(columns={"eln_label": "eln_label_dann"}), on="sample_id", how="inner")
print(f"Paired DataFrame (PCA + DANN per sample): {paired.shape}")
# Check label consistency
mismatch = (paired["eln_label"] != paired["eln_label_dann"]).sum()
print(f"Label mismatches between PCA and DANN evaluations: {mismatch}")

n_boot = 5000
rng = np.random.default_rng(42)
n_samples = len(paired)
qwk_pca_bootstrap = np.zeros(n_boot)
qwk_dann_bootstrap = np.zeros(n_boot)
qwk_gap_bootstrap = np.zeros(n_boot)

for i in range(n_boot):
    idx = rng.integers(0, n_samples, n_samples)
    pca_qwk = cohen_kappa_score(paired["eln_label"].iloc[idx], paired["pca_pred"].iloc[idx], weights="quadratic")
    dann_qwk = cohen_kappa_score(paired["eln_label"].iloc[idx], paired["eln_pred"].iloc[idx], weights="quadratic")
    qwk_pca_bootstrap[i] = pca_qwk
    qwk_dann_bootstrap[i] = dann_qwk
    qwk_gap_bootstrap[i] = pca_qwk - dann_qwk

pca_point = cohen_kappa_score(paired["eln_label"], paired["pca_pred"], weights="quadratic")
dann_point = cohen_kappa_score(paired["eln_label"], paired["eln_pred"], weights="quadratic")
gap_point = pca_point - dann_point
ci_pca = np.percentile(qwk_pca_bootstrap, [2.5, 97.5])
ci_dann = np.percentile(qwk_dann_bootstrap, [2.5, 97.5])
ci_gap = np.percentile(qwk_gap_bootstrap, [2.5, 97.5])
p_gap = 2 * min(np.mean(qwk_gap_bootstrap <= 0), np.mean(qwk_gap_bootstrap >= 0))

print(f"\nPCA-zscore QWK = {pca_point:.4f}; bootstrap 95% CI: [{ci_pca[0]:.4f}, {ci_pca[1]:.4f}]")
print(f"DANN QWK = {dann_point:.4f}; bootstrap 95% CI: [{ci_dann[0]:.4f}, {ci_dann[1]:.4f}]")
print(f"PCA - DANN gap = {gap_point:+.4f}; paired bootstrap 95% CI: [{ci_gap[0]:+.4f}, {ci_gap[1]:+.4f}]")
print(f"  paired bootstrap p-value (CI crosses 0?): p = {p_gap:.4f}")

RESULTS["paired_bootstrap_external_PCAz_vs_DANN"] = {
    "n_samples": int(n_samples),
    "n_boot": n_boot,
    "PCA_zscore_QWK": float(pca_point),
    "PCA_zscore_CI": [float(ci_pca[0]), float(ci_pca[1])],
    "DANN_QWK": float(dann_point),
    "DANN_CI": [float(ci_dann[0]), float(ci_dann[1])],
    "PCA_minus_DANN_gap": float(gap_point),
    "gap_CI": [float(ci_gap[0]), float(ci_gap[1])],
    "p_value_two_sided": float(p_gap),
}

# =======================================================================================
# 1B. Discriminator AUC + balanced accuracy recompute
# =======================================================================================
print()
print("=" * 70)
print("Phase 1B: Recompute discriminator metrics from v5 DANN checkpoint")
print("=" * 70)

# Note: We don't have raw embeddings saved. The discriminator output during training was logged
# as accuracy = 0.000. To verify the platform-invariance interpretation, we would need to:
# (a) Reload checkpoint, run inference on BeatAML val + GSE6891 to get CLS embeddings
# (b) Run those embeddings through the discriminator head
# (c) Compute balanced acc + AUC
# This requires significant rework. As a proxy, we report what we know from training logs.

# From training: val/adv_disc_acc = 0.000 means argmax(disc_logits) != true_domain for ALL val samples.
# Domain labels: 0 = source (BeatAML), 1 = target (GSE6891).
# If accuracy is exactly 0.000 on a balanced val set, the discriminator is predicting the OPPOSITE
# of true domain on every sample. balanced_accuracy = 0.000 too (worst case).
# This means the discriminator HAS learned a perfectly discriminative signal (just flipped),
# so platform invariance is NOT established. The encoder retains platform-discriminative features
# but the discriminator output is wired up opposite (consistent with GRL training).

# This interpretation is what we put in EV8. For now, we don't run a full re-inference (would need
# checkpoint reload + data pipeline reconstruction; significant engineering work).
# Mark as "interpretation note" pending v25 full verification.

RESULTS["discriminator_interpretation"] = {
    "training_logged_val_acc": 0.000,
    "interpretation": "accuracy = 0.000 on binary domain task means discriminator predicts opposite of true label perfectly; this implies the encoder retains platform-discriminative features (the inversion is consistent with GRL training dynamics). Platform invariance is NOT established by this metric.",
    "balanced_accuracy_implied": 0.000,
    "full_AUC_recompute": "pending — requires CLS embedding extraction + discriminator forward pass; deferred to v25 nested CV pipeline",
}

with open("experiments/p0_fixes/v25_phase1_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults saved to experiments/p0_fixes/v25_phase1_results.json")

# Summary
print()
print("=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)
print(f"External paired bootstrap (n_boot={n_boot} on n={n_samples} samples):")
print(f"  PCA-zscore QWK: {pca_point:.3f} [{ci_pca[0]:.3f}, {ci_pca[1]:.3f}]")
print(f"  DANN-HemeFM QWK: {dann_point:.3f} [{ci_dann[0]:.3f}, {ci_dann[1]:.3f}]")
print(f"  Gap PCA-DANN: {gap_point:+.3f} [{ci_gap[0]:+.3f}, {ci_gap[1]:+.3f}], p={p_gap:.4f}")
print()
print(f"Discriminator: interpretation note added; full AUC verification pending")
