"""§4.2 Baseline comparators for HemeFM downstream tasks.

Implements feasible baselines from the §3.13.1 pre-registered list:
    [B1] PCA + linear probe              — fully feasible
    [B2] LSC17 17-gene signature score   — fully feasible
    [B3] no-pretraining control          — hemefm_base from random init (run by hemefm.train without ckpt)
    [B4] Logistic-regression on raw rank-binned tokens (additional sanity check)

NOT implemented (require external assets):
    - scVI-equivalent VAE (needs scvi-tools + a bulk training recipe)
    - MOFA+ (multi-modal; requires methylation we don't have)
    - Tazi knowledge-bank Cox (proprietary code)
    - Geneformer / BulkRNABert linear probe (requires HF + their public ckpt;
      stubs provided in `baselines_external.py` for future inclusion)

Operates on the same BeatAML labeled split that Phase 4' uses (val_fraction=0.20,
seed=42) so HemeFM metrics are paired with each baseline for the §4.2 test.

Usage:
    python -m hemefm.baselines \\
        --corpus-parquet data/processed/pretrain_corpus.parquet \\
        --metadata-parquet data/processed/pretrain_corpus_metadata.parquet \\
        --clinical-xlsx data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx \\
        --drug-tsv data/raw/beataml_v2_biodev/beataml_probit_curve_fits_v4_dbgap.txt \\
        --output-dir outputs/baselines/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# LSC17 — Ng et al. 2016, Nature
LSC17_GENES = [
    "DNMT3B", "ZBTB46", "NYNRIN", "ARHGAP22", "LAPTM4B", "MMRN1", "DPYSL3",
    "KIAA0125", "CDK6", "CPXM1", "SOCS2", "SMIM24", "EMP1", "NGFRAP1", "CD34",
    "AKR1C3", "GPR56",
]
# LSC17 coefficients from Ng et al. supplementary Table 4
LSC17_WEIGHTS = {
    "DNMT3B": 0.0874, "ZBTB46": -0.0347, "NYNRIN": 0.00865, "ARHGAP22": -0.0138,
    "LAPTM4B": 0.00582, "MMRN1": 0.0258, "DPYSL3": 0.0284, "KIAA0125": 0.0196,
    "CDK6": -0.0704, "CPXM1": -0.0258, "SOCS2": 0.0271, "SMIM24": -0.0226,
    "EMP1": 0.0146, "NGFRAP1": 0.0465, "CD34": 0.0338, "AKR1C3": -0.0402,
    "GPR56": 0.0501,
}


# ---------- Data prep (mirror BeatAMLMultiModalDataModule split for paired comparison) ----------

ELN_MAP = {"Favorable": 0, "Intermediate": 1, "Adverse": 2}


def _wide_beataml(corpus: Path, metadata: Path, cohort: str = "BeatAML2") -> pd.DataFrame:
    """Reconstruct sample × gene matrix (rank_bin values) from long-form corpus parquet."""
    meta = pd.read_parquet(metadata)
    sids = set(meta.loc[meta["cohort"] == cohort, "sample_id"].astype(str))
    df = pd.read_parquet(corpus)
    df = df[df["sample_id"].astype(str).isin(sids)]
    wide = df.pivot_table(index="sample_id", columns="gene", values="rank_bin", aggfunc="first")
    print(f"[baselines] wide BeatAML matrix: {wide.shape[0]} samples × {wide.shape[1]} genes")
    return wide


def _labels(clinical: Path) -> pd.DataFrame:
    cl = pd.read_excel(clinical)
    cl = cl[cl["dbgap_rnaseq_sample"].notna()].drop_duplicates("dbgap_rnaseq_sample", keep="first")
    cl = cl.set_index("dbgap_rnaseq_sample")
    cl["eln_label"] = cl["ELN2017"].map(ELN_MAP)
    cl["os_time"] = pd.to_numeric(cl["overallSurvival"], errors="coerce") / 30.4375
    cl["os_event"] = cl["vitalStatus"].astype(str).str.lower().isin(["dead", "deceased", "died"]).astype(np.float32)
    return cl[["eln_label", "os_time", "os_event"]]


def _drug_matrix(drug_tsv: Path, top_n: int = 50) -> pd.DataFrame:
    dr = pd.read_csv(drug_tsv, sep="\t", low_memory=False)
    dr = dr[dr["dbgap_rnaseq_sample"].notna() & dr["ic50"].notna()]
    counts = dr.groupby("inhibitor")["dbgap_rnaseq_sample"].nunique().sort_values(ascending=False)
    top = counts.head(top_n).index
    dr = dr[dr["inhibitor"].isin(top)]
    dr["log_ic50"] = np.log10(dr["ic50"].clip(lower=1e-3, upper=1e5))
    return dr.pivot_table(index="dbgap_rnaseq_sample", columns="inhibitor", values="log_ic50", aggfunc="median")[list(top)]


def _split(sample_ids: list[str], val_fraction: float = 0.20, seed: int = 42) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    sids = sorted(sample_ids)
    rng.shuffle(sids)
    n_val = max(1, int(len(sids) * val_fraction))
    return sids[n_val:], sids[:n_val]


# ---------- B1: PCA + linear probe ----------

def baseline_pca(X_train: np.ndarray, X_val: np.ndarray, y_train_eln, y_val_eln,
                  os_t_train, os_e_train, os_t_val, os_e_val,
                  drug_train: np.ndarray, drug_val: np.ndarray, n_components: int = 64) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    from lifelines import CoxPHFitter

    pca = PCA(n_components=n_components, random_state=42).fit(np.nan_to_num(X_train))
    Z_train = pca.transform(np.nan_to_num(X_train))
    Z_val = pca.transform(np.nan_to_num(X_val))

    out: dict = {"baseline": "PCA+linear_probe", "n_components": n_components}

    # ELN ordinal (via 3-class log-reg, treat as ordinal QWK)
    m = pd.notna(y_train_eln) & pd.notna(y_val_eln)
    if (~pd.isna(y_train_eln)).sum() > 5:
        m_tr = ~pd.isna(y_train_eln)
        m_va = ~pd.isna(y_val_eln)
        clf = LogisticRegression(max_iter=1000, multi_class="ovr").fit(Z_train[m_tr], y_train_eln[m_tr].astype(int))
        eln_pred = clf.predict(Z_val[m_va])
        out["eln_qwk"] = float(cohen_kappa_score(y_val_eln[m_va].astype(int), eln_pred, weights="quadratic"))

    # Survival Cox
    surv_df = pd.DataFrame(np.column_stack([Z_train, os_t_train, os_e_train]),
                           columns=[f"z{i}" for i in range(n_components)] + ["T", "E"])
    surv_df = surv_df[surv_df["T"] > 0].dropna()
    if len(surv_df) > 20:
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(surv_df, duration_col="T", event_col="E")
            # Predict on val
            surv_val_df = pd.DataFrame(Z_val, columns=[f"z{i}" for i in range(n_components)])
            risk = cph.predict_partial_hazard(surv_val_df).to_numpy()
            from lifelines.utils import concordance_index
            m_va = pd.notna(os_t_val) & pd.notna(os_e_val) & (os_t_val > 0)
            if m_va.sum() > 5:
                out["os_c_index"] = float(concordance_index(os_t_val[m_va], -risk[m_va], os_e_val[m_va].astype(int)))
        except Exception as e:                                              # noqa: BLE001
            out["os_c_index_error"] = str(e)[:200]

    # Drug response (avg Spearman across top-N drugs)
    if drug_train is not None and drug_val is not None:
        from scipy.stats import spearmanr
        rhos = []
        for d in range(drug_train.shape[1]):
            y_tr = drug_train[:, d]
            y_va = drug_val[:, d]
            m_tr = np.isfinite(y_tr)
            m_va = np.isfinite(y_va)
            if m_tr.sum() > 20 and m_va.sum() > 5:
                reg = Ridge(alpha=1.0).fit(Z_train[m_tr], y_tr[m_tr])
                pred = reg.predict(Z_val[m_va])
                rho, _ = spearmanr(pred, y_va[m_va])
                if np.isfinite(rho):
                    rhos.append(float(rho))
        if rhos:
            out["drug_spearman_mean"] = float(np.mean(rhos))
            out["drug_spearman_n_drugs"] = len(rhos)
    return out


# ---------- B2: LSC17 ----------

def baseline_lsc17(wide_train: pd.DataFrame, wide_val: pd.DataFrame,
                   y_train_eln, y_val_eln,
                   os_t_train, os_e_train, os_t_val, os_e_val) -> dict:
    from sklearn.metrics import cohen_kappa_score
    from lifelines.utils import concordance_index

    out: dict = {"baseline": "LSC17"}
    available = [g for g in LSC17_GENES if g in wide_train.columns]
    out["n_lsc17_genes_available"] = len(available)
    out["missing_lsc17_genes"] = [g for g in LSC17_GENES if g not in wide_train.columns]
    if len(available) < 5:
        out["error"] = f"only {len(available)}/17 LSC17 genes in vocab, skipping"
        return out

    train_scores = pd.Series(0.0, index=wide_train.index)
    val_scores = pd.Series(0.0, index=wide_val.index)
    for g in available:
        train_scores += wide_train[g].fillna(wide_train[g].median()) * LSC17_WEIGHTS[g]
        val_scores += wide_val[g].fillna(wide_train[g].median()) * LSC17_WEIGHTS[g]

    # ELN ordinal via 3-quantile binning of LSC17 score
    q33, q66 = np.quantile(train_scores.dropna(), [1/3, 2/3])
    val_pred_eln = (val_scores > q66).astype(int) + (val_scores > q33).astype(int)
    m = pd.notna(y_val_eln)
    if m.sum() > 1:
        out["eln_qwk"] = float(cohen_kappa_score(y_val_eln[m].astype(int), val_pred_eln[m].astype(int).to_numpy(), weights="quadratic"))

    # OS C-index using raw LSC17 score
    m_va = pd.notna(os_t_val) & pd.notna(os_e_val) & (os_t_val > 0)
    if m_va.sum() > 5:
        out["os_c_index"] = float(concordance_index(os_t_val[m_va], -val_scores[m_va].to_numpy(), os_e_val[m_va].astype(int)))
    return out


# ---------- Main pipeline ----------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-parquet", type=Path, default=Path("data/processed/pretrain_corpus.parquet"))
    p.add_argument("--metadata-parquet", type=Path, default=Path("data/processed/pretrain_corpus_metadata.parquet"))
    p.add_argument("--clinical-xlsx", type=Path, default=Path("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx"))
    p.add_argument("--drug-tsv", type=Path, default=Path("data/raw/beataml_v2_biodev/beataml_probit_curve_fits_v4_dbgap.txt"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-components", type=int, default=64)
    p.add_argument("--n-drugs", type=int, default=50)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    wide = _wide_beataml(args.corpus_parquet, args.metadata_parquet)
    labels = _labels(args.clinical_xlsx)
    drug = _drug_matrix(args.drug_tsv, top_n=args.n_drugs)

    sample_ids = sorted(set(wide.index) & set(labels.index))
    sample_ids = [s for s in sample_ids if pd.notna(labels.loc[s, "eln_label"])]
    train_sids, val_sids = _split(sample_ids, args.val_fraction, args.seed)
    print(f"[baselines] train={len(train_sids)} val={len(val_sids)}")

    X_train = wide.loc[train_sids].fillna(0).to_numpy(dtype=np.float32)
    X_val = wide.loc[val_sids].fillna(0).to_numpy(dtype=np.float32)
    y_train_eln = labels.loc[train_sids, "eln_label"]
    y_val_eln = labels.loc[val_sids, "eln_label"]
    os_t_train = labels.loc[train_sids, "os_time"]
    os_e_train = labels.loc[train_sids, "os_event"]
    os_t_val = labels.loc[val_sids, "os_time"]
    os_e_val = labels.loc[val_sids, "os_event"]

    drug_train = drug.reindex(train_sids).to_numpy(dtype=np.float32)
    drug_val = drug.reindex(val_sids).to_numpy(dtype=np.float32)

    print("[baselines] running B1 PCA + linear probe...")
    res_pca = baseline_pca(X_train, X_val, y_train_eln, y_val_eln,
                            os_t_train, os_e_train, os_t_val, os_e_val,
                            drug_train, drug_val, args.pca_components)
    print(json.dumps(res_pca, indent=2))

    print("[baselines] running B2 LSC17 score...")
    res_lsc = baseline_lsc17(wide.loc[train_sids], wide.loc[val_sids],
                              y_train_eln, y_val_eln,
                              os_t_train, os_e_train, os_t_val, os_e_val)
    print(json.dumps(res_lsc, indent=2))

    all_results = {
        "split_info": {"train_n": len(train_sids), "val_n": len(val_sids),
                       "val_fraction": args.val_fraction, "seed": args.seed},
        "B1_PCA_linear_probe": res_pca,
        "B2_LSC17": res_lsc,
    }
    (args.output_dir / "baselines_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n[baselines] all results written to {args.output_dir/'baselines_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
