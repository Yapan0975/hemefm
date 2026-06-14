# -*- coding: utf-8 -*-
"""Issue-2 fix: re-run the 3 external PCA configs + paired bootstrap, TRAINING PCA on the
EXACT same 347 BeatAML source-training patients the DANN fine-tune used (matched comparison)."""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Reproduce the EXACT DANN source split (beataml_adversarial.setup, lines ~195-203)
from hemefm.data.beataml_multimodal import _derive_labels
COHORT, SEED, VAL_FRAC = "BeatAML2", 42, 0.20
meta = pd.read_parquet("data/processed/pretrain_corpus_metadata.parquet")
beataml_sids = set(meta.loc[meta["cohort"] == COHORT, "sample_id"])
cl = pd.read_excel("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx")
labels = _derive_labels(cl)
all_sids = sorted(beataml_sids & set(labels.index))
all_sids = [s for s in all_sids if pd.notna(labels.loc[s, "eln_label"])]
rng = np.random.default_rng(SEED); rng.shuffle(all_sids)
n_val = max(1, int(len(all_sids) * VAL_FRAC))
val_sids = set(all_sids[:n_val]); train_sids = set(all_sids[n_val:])
print(f"[split] labeled total = {len(all_sids)}  ->  train = {len(train_sids)}  val = {len(val_sids)}")

# Build rank-binned BeatAML wide (same as p0_external_pca_v2.py)
corpus = pd.read_parquet("data/processed/pretrain_corpus.parquet")
beat_in_corpus = sorted(beataml_sids & set(corpus["sample_id"].unique()))
ba = corpus[corpus["sample_id"].isin(beat_in_corpus)]
beat_wide = ba.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)

gse = pd.read_parquet("data/external/GSE6891/expression.parquet")
gse_lab = pd.read_csv("data/external/GSE6891/labels.tsv", sep="\t").dropna(subset=["eln_label"])
gse_l = gse[gse["sample_id"].isin(gse_lab["sample_id"])]
gse_wide = gse_l.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)
common = sorted(set(beat_wide.columns) & set(gse_wide.columns))
print(f"[data] common rank-bin genes = {len(common)}")

lab_map = {s: int(v) for s, v in labels["eln_label"].dropna().items()}
beat_ids_all = beat_wide.index.tolist()
Xb_all = beat_wide[common].values.astype(np.float32)
yb_all = np.array([lab_map.get(s, -1) for s in beat_ids_all])

# === ISSUE-2 RESTRICTION: keep ONLY the 347 source-training patients DANN used ===
keep = np.array([(beat_ids_all[i] in train_sids and yb_all[i] >= 0) for i in range(len(beat_ids_all))])
X_beat, y_beat = Xb_all[keep], yb_all[keep]
print(f"[issue-2] PCA now trains on {X_beat.shape[0]} patients (matched to DANN's 347)")

gse_ids = gse_wide.index.tolist()
X_gse = gse_wide[common].values.astype(np.float32)
gmap = {s: int(v) for s, v in gse_lab.set_index("sample_id")["eln_label"].items()}
y_gse = np.array([gmap.get(s, -1) for s in gse_ids])
mg = y_gse >= 0
X_gse, y_gse = X_gse[mg], y_gse[mg]
gse_ids = [s for s, m in zip(gse_ids, mg) if m]
print(f"[data] GSE6891 external test n = {len(y_gse)}, dist = {np.bincount(y_gse).tolist()}")

def run_pca(Xtr, ytr, Xte, label):
    pca = PCA(n_components=64, random_state=42)
    clf = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
    clf.fit(pca.fit_transform(Xtr), ytr)
    return clf.predict(pca.transform(Xte))

RES = {"n_train": int(X_beat.shape[0]), "n_external": int(len(y_gse))}
# (a) raw
p_raw = run_pca(X_beat, y_beat, X_gse, "raw")
RES["pca_raw_QWK"] = float(cohen_kappa_score(y_gse, p_raw, weights="quadratic"))
# (b) per-cohort z-score
Xbz = (X_beat - X_beat.mean(0)) / (X_beat.std(0) + 1e-6)
Xgz = (X_gse - X_gse.mean(0)) / (X_gse.std(0) + 1e-6)
p_z = run_pca(Xbz, y_beat, Xgz, "zscore")
RES["pca_zscore_QWK"] = float(cohen_kappa_score(y_gse, p_z, weights="quadratic"))
# (c) z-score + CORAL
def coral(Xs, Xt):
    Cs = np.cov(Xs.T) + np.eye(Xs.shape[1]); Ct = np.cov(Xt.T) + np.eye(Xt.shape[1])
    Us, Ss, _ = np.linalg.svd(Cs); Ut, St, _ = np.linalg.svd(Ct)
    return Xs @ (Us @ np.diag(1/np.sqrt(Ss+1e-6)) @ Us.T) @ (Ut @ np.diag(np.sqrt(St)) @ Ut.T)
try:
    p_c = run_pca(coral(Xbz, Xgz), y_beat, Xgz, "coral")
    RES["pca_coral_QWK"] = float(cohen_kappa_score(y_gse, p_c, weights="quadratic"))
except Exception as e:
    RES["pca_coral_QWK"] = None; print("coral failed:", e)

print("\n=== PCA on matched 347 — external GSE6891 QWK ===")
print(f"  raw      : {RES['pca_raw_QWK']:.4f}")
print(f"  z-score  : {RES['pca_zscore_QWK']:.4f}   (headline config)")
print(f"  z+CORAL  : {RES['pca_coral_QWK']}")
print(f"  DANN(ref): 0.284")

# === Paired bootstrap: PCA-zscore(347) vs DANN, per-patient on GSE6891 ===
dann = pd.read_csv("outputs/eval/GSE6891_v5_dann/predictions.tsv", sep="\t")
dann = dann.merge(gse_lab[["sample_id", "eln_label"]], on="sample_id", how="inner")
dann["eln_label"] = dann["eln_label"].astype(int)
pca_df = pd.DataFrame({"sample_id": gse_ids, "y": y_gse, "pca": p_z})
paired = pca_df.merge(dann[["sample_id", "eln_pred"]], on="sample_id", how="inner")
print(f"\n[paired] n = {len(paired)} (PCA-zscore-347 + DANN per GSE6891 patient)")
nb = 5000; rb = np.random.default_rng(42); n = len(paired)
gap = np.zeros(nb); pca_b = np.zeros(nb); dann_b = np.zeros(nb)
yv = paired["y"].values; pv = paired["pca"].values; dv = paired["eln_pred"].values
for i in range(nb):
    idx = rb.integers(0, n, n)
    a = cohen_kappa_score(yv[idx], pv[idx], weights="quadratic")
    b = cohen_kappa_score(yv[idx], dv[idx], weights="quadratic")
    pca_b[i], dann_b[i], gap[i] = a, b, a - b
pca_pt = cohen_kappa_score(yv, pv, weights="quadratic"); dann_pt = cohen_kappa_score(yv, dv, weights="quadratic")
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
p_gap = float(2 * min((gap <= 0).mean(), (gap >= 0).mean()))
RES["paired_bootstrap_347"] = {
    "n": int(n), "PCA_zscore_QWK": float(pca_pt), "PCA_CI": ci(pca_b),
    "DANN_QWK": float(dann_pt), "DANN_CI": ci(dann_b),
    "gap_PCA_minus_DANN": float(pca_pt - dann_pt), "gap_CI": ci(gap), "p_two_sided": p_gap,
}
print(f"  PCA-zscore(347) QWK = {pca_pt:.4f}  CI {ci(pca_b)}")
print(f"  DANN QWK            = {dann_pt:.4f}  CI {ci(dann_b)}")
print(f"  gap PCA-DANN        = {pca_pt - dann_pt:+.4f}  CI {ci(gap)}  p = {p_gap:.4f}")

# Save outputs (per-patient PCA preds + summary) for results/
Path("experiments/p0_fixes").mkdir(parents=True, exist_ok=True)
paired.to_csv("experiments/p0_fixes/pca_zscore_347_external_predictions.tsv", sep="\t", index=False)
with open("experiments/p0_fixes/issue2_pca347_results.json", "w") as f:
    json.dump(RES, f, indent=2)
with open("experiments/p0_fixes/dann_train_347_ids.txt", "w") as f:
    f.write("\n".join(sorted(train_sids)))
with open("experiments/p0_fixes/dann_val_86_ids.txt", "w") as f:
    f.write("\n".join(sorted(val_sids)))
print("\nSaved -> experiments/p0_fixes/issue2_pca347_results.json (+ predictions + 347/86 id lists)")
