# -*- coding: utf-8 -*-
"""Issue-4: recompute HemeFM drug-response SPEARMAN (per-drug mean) on the 86 val patients,
matching the metric PCA reports (baselines drug_spearman_mean = 0.276)."""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from hemefm.data.beataml_multimodal import _build_drug_matrix

preds = pd.read_csv("outputs/eval/BeatAML2_drugcheck/predictions.tsv", sep="\t")
preds["drug_response"] = preds["drug_response"].apply(json.loads)
drug_matrix, _ = _build_drug_matrix(
    Path("data/raw/beataml_v2_biodev/beataml_probit_curve_fits_v4_dbgap.txt"), top_n=50)
print(f"drug_matrix: {drug_matrix.shape}; preds: {preds.shape}")

val_set = set(open("experiments/p0_fixes/dann_val_86_ids.txt").read().split())
pv = preds[preds["sample_id"].isin(val_set)].set_index("sample_id")
common = [s for s in pv.index if s in drug_matrix.index]
print(f"val patients: with preds {len(pv)}, with drug truth {len(common)}")

P = np.array([pv.loc[s, "drug_response"] for s in common], dtype=float)
Y = drug_matrix.loc[common].values.astype(float)
print(f"P {P.shape}  Y {Y.shape}")

rhos, pears = [], []
for d in range(P.shape[1]):
    m = np.isfinite(P[:, d]) & np.isfinite(Y[:, d])
    if m.sum() >= 5:
        r = spearmanr(P[m, d], Y[m, d]).statistic
        p = pearsonr(P[m, d], Y[m, d])[0]
        if np.isfinite(r): rhos.append(r)
        if np.isfinite(p): pears.append(p)

# global (flatten) too, for reference
mm = np.isfinite(P) & np.isfinite(Y)
g_rho = spearmanr(P[mm], Y[mm]).statistic
g_pear = pearsonr(P[mm], Y[mm])[0]

print("\n=== HemeFM drug-response on n=%d val patients ===" % len(common))
print(f"  per-drug mean SPEARMAN = {np.nanmean(rhos):.4f}  (over {len(rhos)} drugs)   <-- same metric as PCA 0.276")
print(f"  per-drug mean Pearson  = {np.nanmean(pears):.4f}  (sanity vs reported val pearson 0.221)")
print(f"  global flatten Spearman = {g_rho:.4f}; global Pearson = {g_pear:.4f}")

with open("experiments/p0_fixes/hemefm_drug_spearman_val.json", "w") as f:
    json.dump({"n_val": len(common),
               "hemefm_drug_spearman_mean": float(np.nanmean(rhos)),
               "n_drugs": len(rhos),
               "hemefm_drug_pearson_mean": float(np.nanmean(pears)),
               "global_spearman": float(g_rho), "global_pearson": float(g_pear)}, f, indent=2)
print("saved experiments/p0_fixes/hemefm_drug_spearman_val.json")
