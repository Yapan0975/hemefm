# -*- coding: utf-8 -*-
"""Final fix: CORAL target covariance also estimated over ALL 537 GSE6891 samples (label-blind),
to match the z-score normalization. Recompute all 3 PCA configs; z-score/raw unchanged, CORAL updated."""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from hemefm.data.beataml_multimodal import _derive_labels

COHORT, SEED, VAL_FRAC = "BeatAML2", 42, 0.20
meta = pd.read_parquet("data/processed/pretrain_corpus_metadata.parquet")
beataml_sids = set(meta.loc[meta["cohort"] == COHORT, "sample_id"])
cl = pd.read_excel("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx")
labels = _derive_labels(cl)
all_sids = sorted(beataml_sids & set(labels.index))
all_sids = [s for s in all_sids if pd.notna(labels.loc[s, "eln_label"])]
rng = np.random.default_rng(SEED); rng.shuffle(all_sids)
train_sids = set(all_sids[max(1, int(len(all_sids)*VAL_FRAC)):])

corpus = pd.read_parquet("data/processed/pretrain_corpus.parquet")
bw = corpus[corpus["sample_id"].isin(sorted(beataml_sids & set(corpus["sample_id"].unique())))].pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)
gse = pd.read_parquet("data/external/GSE6891/expression.parquet")
gse_lab = pd.read_csv("data/external/GSE6891/labels.tsv", sep="\t").dropna(subset=["eln_label"])
gw_all = gse.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)
common = sorted(set(bw.columns) & set(gw_all.columns))

lab_map = {s: int(v) for s, v in labels["eln_label"].dropna().items()}
beat_ids = bw.index.tolist(); Xb_all = bw[common].values.astype(np.float32)
yb_all = np.array([lab_map.get(s, -1) for s in beat_ids])
keep = np.array([(beat_ids[i] in train_sids and yb_all[i] >= 0) for i in range(len(beat_ids))])
X_beat, y_beat = Xb_all[keep], yb_all[keep]

X_all_t = gw_all[common].values.astype(np.float32)              # 537
tmean, tstd = X_all_t.mean(0), X_all_t.std(0)
gmap = {s: int(v) for s, v in gse_lab.set_index("sample_id")["eln_label"].items()}
lab_ids = [s for s in gw_all.index if s in gmap]
X_gse = gw_all.loc[lab_ids, common].values.astype(np.float32)   # 451
y_gse = np.array([gmap[s] for s in lab_ids])

def fit_pred(Xtr, ytr, Xte):
    pca = PCA(n_components=64, random_state=42); clf = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
    clf.fit(pca.fit_transform(Xtr), ytr); return clf.predict(pca.transform(Xte))
def st(p): cm = confusion_matrix(y_gse, p, labels=[0,1,2]); return round(float(cohen_kappa_score(y_gse, p, weights="quadratic")),4), (cm.diagonal()/cm.sum(1).clip(min=1)*100).round(0).astype(int).tolist()

Xbz = (X_beat - X_beat.mean(0)) / (X_beat.std(0) + 1e-6)
Xgz_451 = (X_gse - tmean) / (tstd + 1e-6)
Xgz_all = (X_all_t - tmean) / (tstd + 1e-6)                     # 537 z-scored (for CORAL target cov)

R = {}
R["raw"]    = st(fit_pred(X_beat, y_beat, X_gse))
R["zscore"] = st(fit_pred(Xbz, y_beat, Xgz_451))
def coral537(Xs, Xt_full):                                      # target covariance from ALL 537
    Cs = np.cov(Xs.T) + np.eye(Xs.shape[1]); Ct = np.cov(Xt_full.T) + np.eye(Xt_full.shape[1])
    Us,Ss,_ = np.linalg.svd(Cs); Ut,St,_ = np.linalg.svd(Ct)
    return Xs @ (Us@np.diag(1/np.sqrt(Ss+1e-6))@Us.T) @ (Ut@np.diag(np.sqrt(St))@Ut.T)
R["coral_537cov"] = st(fit_pred(coral537(Xbz, Xgz_all), y_beat, Xgz_451))
print("raw    ", R["raw"])
print("zscore ", R["zscore"], "  <- headline (unchanged)")
print("CORAL  (537 target cov):", R["coral_537cov"], "  (was 0.406 with 451 cov)")
json.dump({"n_train": int(X_beat.shape[0]), "results": R}, open("experiments/p0_fixes/coral_537cov_results.json","w"), indent=2)
print("saved coral_537cov_results.json")
