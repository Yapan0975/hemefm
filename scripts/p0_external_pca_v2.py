"""P0-5: External PCA baseline on GSE6891 using rank-binned representation (apples-to-apples with HemeFM input)."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

print("=" * 70)
print("P0-5: External PCA baseline on GSE6891 (rank-binned, apples-to-apples)")
print("=" * 70)

# Load BeatAML clinical
clin = pd.read_excel("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx")
valid_labels = ["Favorable", "Intermediate", "Adverse"]
clin_valid = clin[clin["ELN2017"].isin(valid_labels)].dropna(subset=["dbgap_rnaseq_sample"]).copy()
clin_valid["eln_int"] = clin_valid["ELN2017"].map({"Favorable": 0, "Intermediate": 1, "Adverse": 2})
print(f"BeatAML valid 3-class samples: {len(clin_valid)}")

# Load rank-binned BeatAML from pretrain corpus
corpus = pd.read_parquet("data/processed/pretrain_corpus.parquet")
print(f"Pretrain corpus rows: {len(corpus):,}, unique samples: {corpus['sample_id'].nunique()}")

# Identify BeatAML samples in corpus (samples starting with "BA" or matching dbgap_rnaseq_sample)
beat_samples = set(clin_valid["dbgap_rnaseq_sample"])
beat_in_corpus = sorted(beat_samples & set(corpus["sample_id"].unique()))
print(f"BeatAML samples in pretrain corpus: {len(beat_in_corpus)}")

if len(beat_in_corpus) < 100:
    # Try alternative: corpus has BA-prefix samples instead of dbgap IDs?
    beat_corpus_samples = corpus[corpus["sample_id"].str.startswith("BA")]["sample_id"].unique()
    print(f"  BA-prefix samples in corpus: {len(beat_corpus_samples)}")
    # Match via cross-walk if exists. Else just use all BA samples
    if len(beat_corpus_samples) > 100:
        # Use the BA samples — try matching to dbgap via sample mapping
        sample_mapping = pd.read_excel("data/raw/beataml_v2_biodev/beataml_waves1to4_sample_mapping.xlsx")
        print(f"  Sample mapping shape: {sample_mapping.shape}, cols: {sample_mapping.columns.tolist()[:10]}")
        # Look for column that has 'BA' samples (corpus) and column that maps to dbgap_rnaseq_sample (clinical)
        # Common BeatAML mapping has 'sample_id' (BA-prefix) and 'dbgap_rnaseq_sample'
        if "dbgap_rnaseq_sample" in sample_mapping.columns:
            # Build mapping BA -> dbgap_rnaseq
            sm_cols = sample_mapping.columns.tolist()
            ba_col = next((c for c in sm_cols if "BA" in str(sample_mapping[c].dropna().astype(str).head(3).tolist()[0] if len(sample_mapping[c].dropna()) else "")), None)
            print(f"  Looking for BA-prefix col: {ba_col}")

# Take 433 BeatAML samples filtered by clinical (or use 707 = full cohort)
# Filter corpus to BeatAML samples with valid ELN
ba_corpus = corpus[corpus["sample_id"].isin(beat_in_corpus)]
print(f"BeatAML corpus rows: {len(ba_corpus):,}; samples: {ba_corpus['sample_id'].nunique()}")

# Pivot to wide
print("Pivoting BeatAML to samples x genes ...")
beat_wide = ba_corpus.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)
print(f"BeatAML wide: {beat_wide.shape}")

# Load GSE6891 rank-binned + labels
gse = pd.read_parquet("data/external/GSE6891/expression.parquet")
gse_labels_df = pd.read_csv("data/external/GSE6891/labels.tsv", sep="\t").dropna(subset=["eln_label"])
gse_labeled = gse[gse["sample_id"].isin(gse_labels_df["sample_id"])]
print(f"GSE6891 labeled rows: {len(gse_labeled):,}; samples: {gse_labeled['sample_id'].nunique()}")

print("Pivoting GSE6891 to samples x genes ...")
gse_wide = gse_labeled.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)
print(f"GSE6891 wide: {gse_wide.shape}")

# Common genes
common_genes = sorted(set(beat_wide.columns) & set(gse_wide.columns))
print(f"Common genes (rank-binned vocab): {len(common_genes)}")

# Restrict
X_beat = beat_wide[common_genes].values.astype(np.float32)
beat_sample_ids = beat_wide.index.tolist()
labels_map = clin_valid.set_index("dbgap_rnaseq_sample")["eln_int"].to_dict()
y_beat = np.array([labels_map.get(s, -1) for s in beat_sample_ids])
mask = y_beat >= 0
X_beat, y_beat = X_beat[mask], y_beat[mask]
print(f"BeatAML X: {X_beat.shape}, y: {y_beat.shape}, label dist: {np.bincount(y_beat)}")

X_gse = gse_wide[common_genes].values.astype(np.float32)
gse_sample_ids = gse_wide.index.tolist()
gse_label_map = gse_labels_df.set_index("sample_id")["eln_label"].astype(int).to_dict()
y_gse = np.array([gse_label_map.get(s, -1) for s in gse_sample_ids])
mask_g = y_gse >= 0
X_gse, y_gse = X_gse[mask_g], y_gse[mask_g]
print(f"GSE6891 X: {X_gse.shape}, y: {y_gse.shape}, label dist: {np.bincount(y_gse)}")

# Run 3 PCA variants
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, confusion_matrix

RESULTS = {}

def run_pca_baseline(X_train, y_train, X_test, y_test, label):
    pca = PCA(n_components=64, random_state=42)
    Xt = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    clf = LogisticRegression(C=1.0, max_iter=10000, multi_class="multinomial", random_state=42)
    clf.fit(Xt, y_train)
    preds = clf.predict(Xte)
    qwk = cohen_kappa_score(y_test, preds, weights="quadratic")
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    pred_dist = np.bincount(preds, minlength=3)
    per_class_recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print(f"\n--- {label} ---")
    print(f"  QWK = {qwk:.4f}")
    print(f"  Predicted distribution (F/I/A): {pred_dist.tolist()}")
    print(f"  Per-class recall (F/I/A): {per_class_recall.round(3).tolist()}")
    print(f"  Confusion matrix:\n{cm}")
    return qwk, pred_dist.tolist(), per_class_recall.tolist(), cm.tolist()

# (a) raw rank-bin, no normalization
print("\n--- (a) Raw rank-bin (apples-to-apples with HemeFM input) ---")
qwk_a, pd_a, pcr_a, cm_a = run_pca_baseline(X_beat, y_beat, X_gse, y_gse, "(a) raw rank-bin PCA + LogReg")
RESULTS["pca_raw"] = {"qwk": float(qwk_a), "pred_dist": pd_a, "per_class_recall": pcr_a, "confusion": cm_a}

# (b) per-cohort z-score on rank-bin values
print("\n--- (b) Per-cohort z-score on rank-bin ---")
Xb_z = (X_beat - X_beat.mean(0)) / (X_beat.std(0) + 1e-6)
Xg_z = (X_gse - X_gse.mean(0)) / (X_gse.std(0) + 1e-6)
qwk_b, pd_b, pcr_b, cm_b = run_pca_baseline(Xb_z, y_beat, Xg_z, y_gse, "(b) z-score PCA + LogReg")
RESULTS["pca_zscore"] = {"qwk": float(qwk_b), "pred_dist": pd_b, "per_class_recall": pcr_b, "confusion": cm_b}

# (c) CORAL
print("\n--- (c) CORAL feature alignment ---")
def coral_align(Xs, Xt):
    """Align Xs covariance to Xt covariance."""
    Cs = np.cov(Xs.T) + np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + np.eye(Xt.shape[1])
    # Whitening
    Us, Ss, _ = np.linalg.svd(Cs, full_matrices=False)
    Ws = Us @ np.diag(1.0 / np.sqrt(Ss + 1e-6)) @ Us.T
    Ut, St, _ = np.linalg.svd(Ct, full_matrices=False)
    Wt = Ut @ np.diag(np.sqrt(St)) @ Ut.T
    return Xs @ Ws @ Wt
try:
    Xb_coral = coral_align(Xb_z, Xg_z)
    qwk_c, pd_c, pcr_c, cm_c = run_pca_baseline(Xb_coral, y_beat, Xg_z, y_gse, "(c) z-score + CORAL PCA + LogReg")
    RESULTS["pca_coral"] = {"qwk": float(qwk_c), "pred_dist": pd_c, "per_class_recall": pcr_c, "confusion": cm_c}
except Exception as e:
    print(f"  CORAL failed: {e}")
    RESULTS["pca_coral"] = None

# (d) DANN HemeFM reference for comparison
print("\n" + "=" * 70)
print("COMPARISON SUMMARY: External GSE6891 (n=451 labeled, F/I/A)")
print("=" * 70)
print(f"  DANN HemeFM (lr=0.1) reported:        QWK = 0.284")
print(f"  PCA-64 + LogReg (raw rank-bin):       QWK = {RESULTS['pca_raw']['qwk']:.4f}")
print(f"  PCA-64 + LogReg (per-cohort z-score): QWK = {RESULTS['pca_zscore']['qwk']:.4f}")
if RESULTS["pca_coral"]:
    print(f"  PCA-64 + LogReg (z-score + CORAL):    QWK = {RESULTS['pca_coral']['qwk']:.4f}")

Path("experiments/p0_fixes").mkdir(parents=True, exist_ok=True)
with open("experiments/p0_fixes/pca_external_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults saved to experiments/p0_fixes/pca_external_results.json")
