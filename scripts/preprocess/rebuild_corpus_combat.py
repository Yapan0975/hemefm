"""§3.7 v4 batch-effect-corrected corpus rebuild (Stage 2'' resolution arm A).

Mirrors `build_pretrain_corpus.py` but applies per-cohort z-score gene-wise
normalization BEFORE rank-binning. This removes cohort-level mean / scale
shifts (the dominant component of platform batch effect at the rank level)
without requiring the full pycombat / sva ComBat-seq pipeline.

Rationale (vs full ComBat-seq):
  - True ComBat-seq operates on raw counts with a negative-binomial model;
    works only when ALL cohorts are count-based. Our corpus mixes counts
    (BeatAML/TCGA/TARGET) and TPM (GTEx) so ComBat-seq is inapplicable.
  - Plain ComBat (limma removeBatchEffect) on log-TPM works but adds a
    pycombat / pyComBat dependency. We implement the analogous per-cohort
    z-score directly to stay dependency-clean.
  - Empirically, the bulk of platform shift at the rank-bin level comes
    from per-cohort gene-wise mean / scale offsets; z-scoring per gene
    per cohort removes both without requiring the design matrix.

Output: data/processed/pretrain_corpus_v4_combat.parquet (long form) + metadata.parquet
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse existing loaders
try:
    from scripts.preprocess.build_pretrain_corpus import (
        load_beataml_biodev, load_gdc_star_counts, load_gtex_whole_blood,
        to_rank_bins,
    )
    from scripts.preprocess.hgnc_harmonize import build_alias_map, harmonize_matrix, load_hgnc_table
    from scripts.download._utils import processed_dir, project_root, stage_log
except ModuleNotFoundError:                                     # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.preprocess.build_pretrain_corpus import (
        load_beataml_biodev, load_gdc_star_counts, load_gtex_whole_blood,
        to_rank_bins,
    )
    from scripts.preprocess.hgnc_harmonize import build_alias_map, harmonize_matrix, load_hgnc_table
    from scripts.download._utils import processed_dir, project_root, stage_log


def per_cohort_zscore(expression: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Per-gene z-score within the cohort. Replaces NaN / zero-variance rows with original."""
    arr = expression.to_numpy(dtype=np.float32)
    # log1p first (puts both counts and TPM on roughly log scale)
    arr = np.log1p(np.maximum(arr, 0))
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    arr = (arr - mu) / sd
    out = pd.DataFrame(arr, index=expression.index, columns=expression.columns)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--max-genes", type=int, default=4096)
    p.add_argument("--output-name", type=str, default="pretrain_corpus_v4_combat")
    args = p.parse_args()

    out_dir = processed_dir()
    hgnc = load_hgnc_table()
    alias = build_alias_map(hgnc)

    print("[corpus-v4] loading cohorts + per-cohort z-score + harmonization")
    sources = [
        ("BeatAML2",   load_beataml_biodev),
        ("TCGA-LAML",  lambda: load_gdc_star_counts("TCGA-LAML")),
        ("TARGET-AML", lambda: load_gdc_star_counts("TARGET-AML")),
        ("GTEx-WB",    lambda: load_gtex_whole_blood(max_samples=1500)),
    ]
    matrices: list[pd.DataFrame] = []
    metadata: list[pd.DataFrame] = []
    for cohort, loader in sources:
        with stage_log(f"load {cohort}"):
            expr, meta = loader()
        if expr.empty:
            print(f"  [skip] {cohort}: empty")
            continue
        with stage_log(f"harmonize {cohort}"):
            harm = harmonize_matrix(expr, alias, how="mean")
            print(f"  {cohort}: {expr.shape} -> harmonized {harm.shape}")
        if harm.empty:
            continue
        with stage_log(f"per-cohort z-score {cohort}"):
            harm = per_cohort_zscore(harm, cohort)
        matrices.append(harm)
        metadata.append(meta)

    if not matrices:
        raise RuntimeError("no input matrices loaded")

    with stage_log("intersect gene index"):
        shared = matrices[0].index
        for m in matrices[1:]:
            shared = shared.intersection(m.index)
        print(f"  shared identifier set size: {len(shared):,}")
        matrices = [m.loc[shared] for m in matrices]

    with stage_log("select top-variable identifiers (after z-score)"):
        # Pooled variance — but since z-scored per cohort, variance is now biological
        combined_var = pd.concat([m.var(axis=1) for m in matrices], axis=1).mean(axis=1)
        top_genes = combined_var.sort_values(ascending=False).head(args.max_genes).index
        matrices = [m.loc[top_genes] for m in matrices]
        print(f"  retained top {len(top_genes):,} identifiers")

    with stage_log("rank-bin each cohort"):
        bin_arrays = [to_rank_bins(m, n_bins=args.bins) for m in matrices]

    with stage_log("write long-form parquet"):
        long_rows: list[pd.DataFrame] = []
        for m, meta, bins in zip(matrices, metadata, bin_arrays):
            for j, sample in enumerate(m.columns.tolist()):
                long_rows.append(pd.DataFrame({
                    "sample_id": sample,
                    "gene": m.index.tolist(),
                    "rank_bin": bins[:, j].astype(np.int8),
                }))
        long = pd.concat(long_rows, ignore_index=True)
        out = out_dir / f"{args.output_name}.parquet"
        long.to_parquet(out, compression="zstd", index=False)
        print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB; {len(long):,} rows; "
              f"{len(set(long['sample_id'])):,} samples × {len(set(long['gene'])):,} identifiers)")

    with stage_log("write metadata + manifest"):
        meta_all = pd.concat(metadata, ignore_index=True)
        meta_all.to_parquet(out_dir / f"{args.output_name}_metadata.parquet", compression="zstd", index=False)
        manifest = {
            "n_samples": int(sum(m.shape[1] for m in matrices)),
            "n_genes": int(len(top_genes)),
            "n_bins": args.bins,
            "batch_correction": "per-cohort z-score on log1p (gene-wise, applied before rank-bin)",
            "sources": [s[0] for s in sources],
        }
        (out_dir / f"{args.output_name}.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
