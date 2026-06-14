# -*- coding: utf-8 -*-
"""Issue-2 v2: PCA target z-score computed over ALL 537 GSE6891 samples (label-blind),
matching the data DANN's adaptation sees; PCA still trained on the SAME 347 BeatAML source
patients as DANN. Paired NONPARAMETRIC bootstrap (relabelled). Outputs recall too."""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from hemefm.data.beataml_multimodal import _derive_labels

# --- exact DANN split (347 train / 86 val) ---
COHORT, SEED, VAL_FRAC = "BeatAML2", 42, 0.20
meta = pd.read_parquet("data/processed/pretrain_corpus_metadata.parquet")
beataml_sids = set(meta.loc[meta["cohort"] == COHORT, "sample_id"])
cl = pd.read_excel("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx")
labels = _derive_labels(cl)
all_sids = sorted(beataml_sids & set(labels.index))
all_sids = [s for s in all_sids if pd.notna(labels.loc[s, "eln_label"])]
rng = np.random.default_rng(SEED); rng.shuffle(all_sids)
n_val = max(1, int(len(all_sids) * VAL_FRAC))
train_sids = set(all_sids[n_val:])
print(f"[split] train={len(train_sids)} val={len(all_sids)-len(train_sids)}")

# --- BeatAML source (347) ---
corpus = pd.read_parquet("data/processed/pretrain_corpus.parquet")
beat_in = sorted(beataml_sids & set(corpus["sample_id"].unique()))
bw = corpus[corpus["sample_id"].isin(beat_in)].pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)

# --- GSE6891: ALL 537 for target stats; 451 labeled for evaluation ---
gse = pd.read_parquet("data/external/GSE6891/expression.parquet")
gse_lab = pd.read_csv("data/external/GSE6891/labels.tsv", sep="\t").dropna(subset=["eln_label"])
gw_all = gse.pivot(index="sample_id", columns="gene", values="rank_bin").fillna(0).astype(np.int8)
print(f"[gse] all samples for target stats = {gw_all.shape[0]}")
common = sorted(set(bw.columns) & set(gw_all.columns))
print(f"[data] common genes = {len(common)}")

lab_map = {s: int(v) for s, v in labels["eln_label"].dropna().items()}
beat_ids = bw.index.tolist()
Xb_all = bw[common].values.astype(np.float32)
yb_all = np.array([lab_map.get(s, -1) for s in beat_ids])
keep = np.array([(beat_ids[i] in train_sids and yb_all[i] >= 0) for i in range(len(beat_ids))])
X_beat, y_beat = Xb_all[keep], yb_all[keep]
print(f"[issue-2] PCA trains on {X_beat.shape[0]} patients")

X_all_target = gw_all[common].values.astype(np.float32)          # 537 x G
tmean, tstd = X_all_target.mean(0), X_all_target.std(0)          # <-- label-blind 537 stats
gmap = {s: int(v) for s, v in gse_lab.set_index("sample_id")["eln_label"].items()}
lab_ids = [s for s in gw_all.index if s in gmap]                 # 451 labeled
X_gse = gw_all.loc[lab_ids, common].values.astype(np.float32)
y_gse = np.array([gmap[s] for s in lab_ids])
print(f"[gse] labeled eval n = {len(y_gse)}, dist = {np.bincount(y_gse).tolist()}")

def fit_pred(Xtr, ytr, Xte):
    pca = PCA(n_components=64, random_state=42); clf = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
    clf.fit(pca.fit_transform(Xtr), ytr); return clf.predict(pca.transform(Xte))
def stats(y, p):
    cm = confusion_matrix(y, p, labels=[0,1,2]); rec = (cm.diagonal()/cm.sum(1).clip(min=1)*100).round(0).astype(int)
    return float(cohen_kappa_score(y, p, weights="quadratic")), rec.tolist()

Xbz = (X_beat - X_beat.mean(0)) / (X_beat.std(0) + 1e-6)
Xgz = (X_gse - tmean) / (tstd + 1e-6)                            # 451 normalized by 537 stats
RES = {"n_train": int(X_beat.shape[0]), "n_external": int(len(y_gse)), "target_norm_n": int(X_all_target.shape[0])}
p_raw = fit_pred(X_beat, y_beat, X_gse); RES["pca_raw"] = stats(y_gse, p_raw)
p_z = fit_pred(Xbz, y_beat, Xgz);       RES["pca_zscore"] = stats(y_gse, p_z)
def coral(Xs, Xt):
    Cs=np.cov(Xs.T)+np.eye(Xs.shape[1]); Ct=np.cov(Xt.T)+np.eye(Xt.shape[1])
    Us,Ss,_=np.linalg.svd(Cs); Ut,St,_=np.linalg.svd(Ct)
    return Xs@(Us@np.diag(1/np.sqrt(Ss+1e-6))@Us.T)@(Ut@np.diag(np.sqrt(St))@Ut.T)
p_c = fit_pred(coral(Xbz, Xgz), y_beat, Xgz); RES["pca_coral"] = stats(y_gse, p_c)
print(f"\n=== target z-score over ALL {X_all_target.shape[0]} GSE6891 samples ===")
print(f"  raw     QWK {RES['pca_raw'][0]:.4f}  recall {RES['pca_raw'][1]}")
print(f"  z-score QWK {RES['pca_zscore'][0]:.4f}  recall {RES['pca_zscore'][1]}")
print(f"  CORAL   QWK {RES['pca_coral'][0]:.4f}  recall {RES['pca_coral'][1]}")

# paired NONPARAMETRIC bootstrap, z-score vs DANN
dann = pd.read_csv("outputs/eval/GSE6891_v5_dann/predictions.tsv", sep="\t").merge(gse_lab[["sample_id","eln_label"]], on="sample_id")
pca_df = pd.DataFrame({"sample_id": lab_ids, "y": y_gse, "pca": p_z})
pair = pca_df.merge(dann[["sample_id","eln_pred"]], on="sample_id")
yv, pv, dv = pair["y"].values, pair["pca"].values, pair["eln_pred"].values
nb = 5000; rb = np.random.default_rng(42); n = len(pair); gap = np.zeros(nb); pb = np.zeros(nb); db = np.zeros(nb)
for i in range(nb):
    idx = rb.integers(0, n, n)
    pb[i] = cohen_kappa_score(yv[idx], pv[idx], weights="quadratic"); db[i] = cohen_kappa_score(yv[idx], dv[idx], weights="quadratic"); gap[i] = pb[i]-db[i]
ci = lambda x:[round(float(np.percentile(x,2.5)),4), round(float(np.percentile(x,97.5)),4)]
pca_pt = cohen_kappa_score(yv, pv, weights="quadratic"); dann_pt = cohen_kappa_score(yv, dv, weights="quadratic")
RES["paired_nonparam_bootstrap_347_537norm"] = {"n":int(n),"PCA_QWK":round(float(pca_pt),4),"PCA_CI":ci(pb),"DANN_QWK":round(float(dann_pt),4),"DANN_CI":ci(db),"gap":round(float(pca_pt-dann_pt),4),"gap_CI":ci(gap)}
print(f"\n  PCA-zscore QWK {pca_pt:.4f} {ci(pb)}; DANN {dann_pt:.4f} {ci(db)}; gap {pca_pt-dann_pt:+.4f} {ci(gap)}")
pair.to_csv("experiments/p0_fixes/pca_zscore_347_537norm_predictions.tsv", sep="\t", index=False)
json.dump(RES, open("experiments/p0_fixes/issue2_pca347_537norm_results.json","w"), indent=2)
print("saved issue2_pca347_537norm_results.json")
