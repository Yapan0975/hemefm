"""P0-5 + P0-6 server experiments: External PCA baseline + Discriminator AUC recompute.

P0-5: Fit PCA-64 + LogReg on BeatAML training, project GSE6891, compute QWK.
       Compare DANN-HemeFM (0.284) vs external PCA on the same external cohort.

P0-6: Reload v5 DANN checkpoint, recompute discriminator AUC + balanced accuracy on
      BeatAML val (source) + GSE6891 (target), to verify whether discriminator_acc=0.000
      reflects platform invariance or label-flip / metric bug.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

RESULTS = {}

# =======================================================================================
# P0-5: External PCA baseline
# =======================================================================================
print("=" * 70)
print("P0-5: External PCA baseline on GSE6891")
print("=" * 70)

# Load BeatAML clinical
clin = pd.read_excel("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx")
print(f"BeatAML clinical: {clin.shape}")

# Keep samples with valid ELN labels (3 classes) and dbgap_rnaseq_sample
valid_labels = ["Favorable", "Intermediate", "Adverse"]
clin_valid = clin[clin["ELN2017"].isin(valid_labels)].dropna(subset=["dbgap_rnaseq_sample"]).copy()
print(f"Valid 3-class samples with RNAseq: {len(clin_valid)}")
clin_valid["eln_int"] = clin_valid["ELN2017"].map({"Favorable": 0, "Intermediate": 1, "Adverse": 2})

# Load BeatAML RNA-seq expression from pretrain corpus (rank-binned, but we need raw or normalized)
# Use the BeatAML waves1to4 norm exp (from biodev)
print("Loading BeatAML normalized expression...")
beat_exp = pd.read_csv("data/raw/beataml_v2_biodev/beataml_waves1to4_norm_exp_dbgap.txt",
                      sep="\t", index_col=0)
print(f"BeatAML expression: {beat_exp.shape} (genes x samples)")

# Match samples
common = list(set(beat_exp.columns) & set(clin_valid["dbgap_rnaseq_sample"]))
print(f"Common samples (expression x clinical): {len(common)}")
X_beat = beat_exp[common].T.values  # samples x genes
labels = clin_valid.set_index("dbgap_rnaseq_sample").loc[common, "eln_int"].values
print(f"X_beat: {X_beat.shape}, labels: {len(labels)}, distribution: {np.bincount(labels)}")

# Load GSE6891 expression
gse6891 = pd.read_parquet("data/external/GSE6891/expression.parquet")
print(f"GSE6891 expression: {gse6891.shape}")
gse_labels = pd.read_csv("data/external/GSE6891/labels.tsv", sep="\t")
print(f"GSE6891 labels: {gse_labels.shape}, eln_label distribution: {gse_labels['eln_label'].value_counts().to_dict()}")

# Gene intersection
beat_genes = set(beat_exp.index)
gse_genes = set(gse6891.columns) if gse6891.shape[0] < gse6891.shape[1] else set(gse6891.index)
print(f"BeatAML genes: {len(beat_genes)}; GSE6891 genes: {len(gse_genes)}; intersection: {len(beat_genes & gse_genes)}")

# Make sure GSE6891 is samples x genes
if "sample_id" in gse_labels.columns:
    gse_samples = gse_labels.dropna(subset=["eln_label"])["sample_id"].tolist()
else:
    gse_samples = []
print(f"GSE6891 labeled samples: {len(gse_samples)}")

# Determine GSE6891 orientation
if gse6891.shape[0] > gse6891.shape[1]:
    # genes x samples
    print("GSE6891 orientation: genes x samples")
    gse_X_full = gse6891.T  # samples x genes
else:
    print("GSE6891 orientation: samples x genes")
    gse_X_full = gse6891

print(f"GSE6891 (samples x genes): {gse_X_full.shape}")

# Subset to labeled samples + common genes
common_genes = sorted(beat_genes & set(gse_X_full.columns))
print(f"Common genes for PCA: {len(common_genes)}")

X_beat_common = beat_exp.loc[common_genes, common].T.values  # n_beat x n_genes
X_gse_common = gse_X_full.loc[gse_samples, common_genes].values  # n_gse x n_genes
print(f"X_beat_common: {X_beat_common.shape}, X_gse_common: {X_gse_common.shape}")

# Run PCA-64 on BeatAML + LogReg + predict on GSE6891
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

print("\n--- (a) Raw PCA + LogReg, no batch correction ---")
scaler = StandardScaler()
X_beat_z = scaler.fit_transform(X_beat_common)
X_gse_z = scaler.transform(X_gse_common)  # use BeatAML stats (transduction-free)

pca = PCA(n_components=64, random_state=42)
beat_pca = pca.fit_transform(X_beat_z)
gse_pca = pca.transform(X_gse_z)

clf = LogisticRegression(C=1.0, max_iter=10000, multi_class="multinomial", random_state=42)
clf.fit(beat_pca, labels)

gse_labels_filt = gse_labels.dropna(subset=["eln_label"]).set_index("sample_id").loc[gse_samples, "eln_label"].astype(int).values
gse_preds = clf.predict(gse_pca)
qwk_raw = cohen_kappa_score(gse_labels_filt, gse_preds, weights="quadratic")
cm_raw = confusion_matrix(gse_labels_filt, gse_preds)
print(f"  External PCA QWK (raw, no batch correction): {qwk_raw:.4f}")
print(f"  Predicted distribution: {np.bincount(gse_preds, minlength=3)}")
print(f"  Confusion matrix:\n{cm_raw}")
RESULTS["pca_external_raw_qwk"] = float(qwk_raw)
RESULTS["pca_external_raw_predicted_dist"] = np.bincount(gse_preds, minlength=3).tolist()
RESULTS["pca_external_raw_confusion"] = cm_raw.tolist()

print("\n--- (b) Per-cohort z-score (cohort-wise normalization), then PCA + LogReg ---")
# Z-score per cohort separately
scaler_beat = StandardScaler()
scaler_gse = StandardScaler()
X_beat_zb = scaler_beat.fit_transform(X_beat_common)
X_gse_zg = scaler_gse.fit_transform(X_gse_common)

pca_z = PCA(n_components=64, random_state=42)
beat_pca_z = pca_z.fit_transform(X_beat_zb)
gse_pca_z = pca_z.transform(X_gse_zg)

clf_z = LogisticRegression(C=1.0, max_iter=10000, multi_class="multinomial", random_state=42)
clf_z.fit(beat_pca_z, labels)
gse_preds_z = clf_z.predict(gse_pca_z)
qwk_z = cohen_kappa_score(gse_labels_filt, gse_preds_z, weights="quadratic")
cm_z = confusion_matrix(gse_labels_filt, gse_preds_z)
print(f"  External PCA QWK (per-cohort z-score): {qwk_z:.4f}")
print(f"  Predicted distribution: {np.bincount(gse_preds_z, minlength=3)}")
print(f"  Confusion matrix:\n{cm_z}")
RESULTS["pca_external_zscore_qwk"] = float(qwk_z)
RESULTS["pca_external_zscore_predicted_dist"] = np.bincount(gse_preds_z, minlength=3).tolist()
RESULTS["pca_external_zscore_confusion"] = cm_z.tolist()

print("\n--- (c) CORAL alignment + PCA + LogReg ---")
# CORAL: align source covariance to target covariance
def coral(Xs, Xt):
    """Align Xs to have covariance matching Xt."""
    Cs = np.cov(Xs.T) + np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + np.eye(Xt.shape[1])
    Cs_inv_half = np.linalg.cholesky(np.linalg.inv(Cs))
    Ct_half = np.linalg.cholesky(Ct)
    A = Cs_inv_half @ Ct_half
    return Xs @ A
try:
    X_beat_coral = coral(X_beat_zb, X_gse_zg)
    pca_c = PCA(n_components=64, random_state=42)
    beat_pca_c = pca_c.fit_transform(X_beat_coral)
    gse_pca_c = pca_c.transform(X_gse_zg)
    clf_c = LogisticRegression(C=1.0, max_iter=10000, multi_class="multinomial", random_state=42)
    clf_c.fit(beat_pca_c, labels)
    gse_preds_c = clf_c.predict(gse_pca_c)
    qwk_c = cohen_kappa_score(gse_labels_filt, gse_preds_c, weights="quadratic")
    cm_c = confusion_matrix(gse_labels_filt, gse_preds_c)
    print(f"  External PCA QWK (z-score + CORAL): {qwk_c:.4f}")
    print(f"  Predicted distribution: {np.bincount(gse_preds_c, minlength=3)}")
    print(f"  Confusion matrix:\n{cm_c}")
    RESULTS["pca_external_coral_qwk"] = float(qwk_c)
    RESULTS["pca_external_coral_predicted_dist"] = np.bincount(gse_preds_c, minlength=3).tolist()
    RESULTS["pca_external_coral_confusion"] = cm_c.tolist()
except Exception as e:
    print(f"  CORAL failed: {e}")
    RESULTS["pca_external_coral_qwk"] = None

# Summary line
print("\n--- SUMMARY: External-cohort comparison (GSE6891 n=451 labeled) ---")
print(f"  DANN HemeFM (lr=0.1):  QWK = 0.284 (reported)")
print(f"  PCA-64 + LogReg raw:                  QWK = {qwk_raw:.4f}")
print(f"  PCA-64 + LogReg + per-cohort z-score: QWK = {qwk_z:.4f}")
print(f"  PCA-64 + LogReg + z-score + CORAL:    QWK = {RESULTS.get('pca_external_coral_qwk', float('nan')):.4f}")

# Save results
Path("experiments/p0_fixes").mkdir(parents=True, exist_ok=True)
with open("experiments/p0_fixes/pca_external_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults saved to experiments/p0_fixes/pca_external_results.json")
