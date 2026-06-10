"""Build the unified pan-myeloid pretraining corpus (parquet).

Inputs (one or more, controlled by --include):
    data/raw/beataml_v2_cbioportal/data_mrna_seq_v2_rsem.txt
    data/raw/xena_tcga_laml/TCGA-LAML.htseq_counts.tsv
    data/raw/xena_target_aml/TARGET-AML.htseq_counts.tsv
    data/raw/hemap/hemap_expression_matrix.tsv.gz
    data/raw/gtex_v10/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz
    data/raw/geo_gse6891/GSE6891_series_matrix.txt
    data/raw/geo_gse37642/GSE37642_series_matrix.txt

Output:
    data/processed/pretrain_corpus.parquet         (sample × gene matrix in long form)
    data/processed/pretrain_corpus_metadata.parquet (sample-level metadata)
    data/processed/pretrain_corpus.json            (provenance manifest)

Layout of the main parquet (long-form for memory-efficient streaming):
    columns:  sample_id (str) | gene (str, HGNC symbol) | rank_bin (uint8) | bin_count
    optional shard split: --shards 32 -> writes pretrain_corpus_*.parquet

The rank-binning step happens AFTER batch correction. The pipeline:
    1. Load each raw matrix.
    2. Harmonize gene symbols to HGNC.
    3. Restrict to the intersection of high-variance genes across cohorts
       (or use a fixed gene dictionary).
    4. Optionally apply ComBat-seq on raw counts where source allows.
    5. Per-sample log1p, then per-sample rank → 16 bins.
    6. Concatenate, shuffle, write parquet.

This script is the ORCHESTRATION skeleton. Specific source-parsing helpers are
defined inline below as TODO sites — adapt each as the matching raw file becomes
available.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

try:
    from scripts.download._utils import processed_dir, project_root, raw_dir, stage_log
    from scripts.preprocess.hgnc_harmonize import build_alias_map, harmonize_matrix, load_hgnc_table
except ModuleNotFoundError:                                     # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.download._utils import processed_dir, project_root, raw_dir, stage_log
    from scripts.preprocess.hgnc_harmonize import build_alias_map, harmonize_matrix, load_hgnc_table


# -------------------- source loaders ------------------------------------------

def load_beataml_rsem() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (expression genes × samples, sample_metadata)."""
    src = raw_dir("beataml_v2_cbioportal") / "data_mrna_seq_v2_rsem.txt"
    if not src.exists():
        return pd.DataFrame(), pd.DataFrame()
    expr = pd.read_csv(src, sep="\t", index_col=0, low_memory=False)
    if "Entrez_Gene_Id" in expr.columns:
        expr = expr.drop(columns=["Entrez_Gene_Id"])
    meta = pd.DataFrame({
        "sample_id": expr.columns,
        "cohort": "BeatAML",
        "platform": "RNA-seq (RSEM)",
    })
    return expr, meta


def load_xena_htseq(cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = f"xena_{cohort.lower().replace('-', '_')}"
    candidates = [
        raw_dir(dataset) / f"{cohort}.htseq_counts.tsv",
        raw_dir(dataset) / f"{cohort}.htseq_counts.tsv.gz",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return pd.DataFrame(), pd.DataFrame()
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0, low_memory=False)
    # Xena htseq_counts are log2(count+1) — we treat them as already-log-normalized.
    meta = pd.DataFrame({
        "sample_id": expr.columns,
        "cohort": cohort,
        "platform": "RNA-seq (HTSeq, Xena log2(count+1))",
    })
    return expr, meta


def load_beataml_biodev() -> tuple[pd.DataFrame, pd.DataFrame]:
    """BeatAML 2.0 from biodev GitHub LFS mirror.

    File layout (wide):
        stable_id | display_label | description | biotype | BA2392R | BA2611R | ...

    Use `display_label` (HGNC symbol) as the gene index to feed straight into the
    HGNC harmonizer; values are RAW counts. The waves1to4 counts file covers all
    cohorts (Wave 1 - 4 combined).
    """
    src = raw_dir("beataml_v2_biodev") / "beataml_waves1to4_counts_dbgap.txt"
    if not src.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(src, sep="\t", low_memory=False)
    # Drop non-sample metadata columns
    drop_cols = [c for c in ("stable_id", "description", "biotype") if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.set_index("display_label")
    # Some HGNC symbols are duplicated due to readthroughs / locus collapse -> mean over duplicates
    if df.index.duplicated().any():
        df = df.groupby(df.index).mean()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    meta = pd.DataFrame({
        "sample_id": df.columns,
        "cohort": "BeatAML2",
        "platform": "RNA-seq (raw counts, biodev mirror)",
    })
    return df, meta


def load_gdc_star_counts(project_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GDC STAR-Counts: 1 TSV per case, with header `# gene-model: GENCODE v36`.

    Iterates `data/raw/gdc_{project}/*.rna_seq.star_counts.tsv`, uses the
    `unstranded` raw-count column, and indexes by `gene_name` (HGNC). Sample IDs
    come from the `manifest.json` (uuid → submitter_id mapping).
    """
    dataset = f"gdc_{project_id.lower().replace('-', '_')}"
    src_dir = raw_dir(dataset)
    if not src_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    manifest_path = src_dir / "manifest.json"
    uuid_to_sid: dict[str, str] = {}
    if manifest_path.exists():
        try:
            mani = json.loads(manifest_path.read_text())
            for f in mani.get("files", []):
                uuid_to_sid[f["file_id"]] = f.get("submitter_id") or f.get("case_id") or f["file_id"]
        except Exception:                                       # noqa: BLE001
            pass

    series_list: list[pd.Series] = []
    for tsv in sorted(src_dir.glob("*.rna_seq.star_counts.tsv")):
        uuid = tsv.name.split(".")[0]
        sid = uuid_to_sid.get(uuid, uuid)
        df = pd.read_csv(tsv, sep="\t", comment="#", low_memory=False)
        if "gene_name" not in df.columns or "unstranded" not in df.columns:
            continue
        # Drop the leading 4 N_* summary rows (no ENSG gene_id), keep real genes
        df = df[df["gene_id"].astype(str).str.startswith("ENSG")]
        s = df.set_index("gene_name")["unstranded"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.index.duplicated().any():
            s = s.groupby(s.index).sum()                        # readthroughs → sum
        s.name = sid
        series_list.append(s)
    if not series_list:
        return pd.DataFrame(), pd.DataFrame()
    expr = pd.concat(series_list, axis=1).fillna(0).astype("float32")
    meta = pd.DataFrame({
        "sample_id": expr.columns,
        "cohort": project_id,
        "platform": "RNA-seq STAR-Counts unstranded (GDC)",
    })
    return expr, meta


def load_gtex_whole_blood(max_samples: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GTEx v10 — whole-blood subset as healthy normal control.

    Reads SampleAttributesDS to identify SAMPIDs with SMTSD == 'Whole Blood',
    then streams the gct.gz file loading ONLY those sample columns (memory-efficient).

    Index: HGNC symbol from the `Description` column (already harmonized in GTEx).
    Values: TPM (log1p-transform happens downstream in the rank-bin stage).

    Args:
        max_samples: optionally cap whole-blood sample count for testing (None = all 4369).
    """
    src_dir = raw_dir("gtex_v10")
    # Source layout: either data/raw/gtex_v10/ or D:\_7_sci\hu\manu01\data_dl\gtex_v10\
    candidates = [
        src_dir,
        project_root().parent / "data_dl" / "gtex_v10",
    ]
    src_dir = next((d for d in candidates if (d / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz").exists()), None)
    if src_dir is None:
        return pd.DataFrame(), pd.DataFrame()

    attr_path = src_dir / "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
    gct_path = src_dir / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
    if not attr_path.exists() or not gct_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    attrs = pd.read_csv(attr_path, sep="\t", low_memory=False)
    wb_ids = set(attrs.loc[attrs["SMTSD"] == "Whole Blood", "SAMPID"].astype(str))

    # Read GCT header (3rd line) to know which columns to load
    with gzip.open(gct_path, "rt") as f:
        next(f)                                                  # #1.2
        next(f)                                                  # n_rows n_cols
        header_cols = next(f).rstrip("\n").split("\t")
    keep_cols = ["Name", "Description"] + [c for c in header_cols if c in wb_ids]
    if max_samples is not None:
        keep_cols = keep_cols[:2 + max_samples]
    print(f"  GTEx: loading {len(keep_cols) - 2} whole-blood samples (of {len(wb_ids)} annotated)")

    # Stream-extract: line-by-line, write whole-blood cols only — avoids OOM on 2.1 GB gzip
    keep_idx = [i for i, c in enumerate(header_cols) if c in keep_cols]
    rows_descr: list[str] = []
    rows_vals: list[list[float]] = []
    with gzip.open(gct_path, "rt") as f:
        next(f); next(f); next(f)                                # skip 3 header lines
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            descr = parts[1]                                     # HGNC symbol column
            try:
                vals = [float(parts[i]) for i in keep_idx[2:]]   # skip Name+Description idx
            except (IndexError, ValueError):
                continue
            rows_descr.append(descr)
            rows_vals.append(vals)
    expr = pd.DataFrame(
        rows_vals,
        index=pd.Index(rows_descr, name="Description"),
        columns=[c for c in keep_cols if c not in ("Name", "Description")],
        dtype="float32",
    )
    # Some HGNC symbols repeat (paralogs / readthroughs) — mean across duplicates
    if expr.index.duplicated().any():
        expr = expr.groupby(expr.index).mean()
    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(how="any").astype("float32")

    meta = pd.DataFrame({
        "sample_id": expr.columns,
        "cohort": "GTEx-WB",
        "platform": "RNA-seq TPM (GTEx v10 RNASeQCv2.4.2)",
    })
    return expr, meta


def load_geo_matrix(gse: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a GEO series_matrix.txt into (expression, metadata).

    For multi-platform GSEs (e.g. GSE37642 with GPL96/GPL97/GPL570) we pick the
    file with the highest sample count, which is usually the latest platform.
    """
    dataset = f"geo_{gse.lower()}"
    candidates = sorted(
        raw_dir(dataset).glob("*_series_matrix.txt*"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    src = candidates[0] if candidates else None
    if src is None or not src.exists():
        return pd.DataFrame(), pd.DataFrame()
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt", errors="replace") as f:
        lines = f.readlines()

    sample_titles: list[str] = []
    data_lines: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("!Sample_title"):
            sample_titles = line.strip().split("\t")[1:]
        elif line.startswith("!series_matrix_table_begin"):
            in_table = True
        elif line.startswith("!series_matrix_table_end"):
            break
        elif in_table:
            data_lines.append(line)

    if not data_lines:
        return pd.DataFrame(), pd.DataFrame()

    from io import StringIO
    expr = pd.read_csv(StringIO("".join(data_lines)), sep="\t", index_col=0, low_memory=False)
    expr.index.name = "probe_id"
    meta = pd.DataFrame({
        "sample_id": expr.columns,
        "title": sample_titles if len(sample_titles) == len(expr.columns) else [""] * len(expr.columns),
        "cohort": gse,
        "platform": "microarray (GEO)",
    })
    return expr, meta


# -------------------- rank-binning -----------------------------------------

def to_rank_bins(expression: pd.DataFrame, n_bins: int = 16) -> np.ndarray:
    """Per-sample rank → discrete bin index (0..n_bins-1).

    `expression` is genes × samples. Returns a uint8 array of the same shape.
    Zero-expression genes share the lowest rank.
    """
    arr = expression.to_numpy(dtype=np.float32)
    # rank along the gene axis per sample
    ranks = arr.argsort(axis=0).argsort(axis=0).astype(np.float32)
    norm = ranks / max(arr.shape[0] - 1, 1)
    bins = np.minimum((norm * n_bins).astype(np.uint8), n_bins - 1)
    return bins


# -------------------- main pipeline ----------------------------------------

LOADERS = {
    # Legacy cBioPortal / UCSC Xena paths (kept for back-compat; data not downloaded)
    "beataml_rsem":    lambda: load_beataml_rsem(),
    "xena_tcga_laml":  lambda: load_xena_htseq("TCGA-LAML"),
    "xena_target_aml": lambda: load_xena_htseq("TARGET-AML"),
    # New canonical sources (2026-06)
    "beataml_biodev":  lambda: load_beataml_biodev(),
    "gdc_tcga_laml":   lambda: load_gdc_star_counts("TCGA-LAML"),
    "gdc_target_aml":  lambda: load_gdc_star_counts("TARGET-AML"),
    "gtex_wb":         lambda: load_gtex_whole_blood(max_samples=1500),
    # GEO microarray
    "gse6891":         lambda: load_geo_matrix("GSE6891"),
    "gse37642":        lambda: load_geo_matrix("GSE37642"),
    # GTEx + Hemap loaders are deliberately omitted here — they need cohort-
    # specific parsing (very large files, header-driven sample annotation).
    # Add them once you've inspected the raw layout once.
}


def iter_sources(include: list[str]) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    for name in include:
        loader = LOADERS.get(name)
        if loader is None:
            print(f"[warn] no loader for source {name!r}; skipping")
            continue
        with stage_log(f"load {name}"):
            expr, meta = loader()
        if expr.empty:
            print(f"  [skip] {name}: raw file not found")
            continue
        yield name, expr, meta


def build(include: list[str], n_bins: int = 16, max_genes: int = 4096, harmonize: bool = True) -> Path:
    out_dir = processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if harmonize:
        hgnc = load_hgnc_table()
        alias = build_alias_map(hgnc)
    else:
        alias = None

    matrices: list[pd.DataFrame] = []
    metadata: list[pd.DataFrame] = []

    for name, expr, meta in iter_sources(include):
        if harmonize:
            with stage_log(f"harmonize {name}"):
                harm = harmonize_matrix(expr, alias, how="mean")
                print(f"  {name}: {expr.shape} -> harmonized {harm.shape}")
        else:
            # No HGNC harmonization — use source identifiers as-is (probe ids, etc.).
            print(f"  {name}: {expr.shape}  (using raw identifiers, no HGNC mapping)")
            harm = expr.copy()
            # Coerce to numeric, dropping rows that cannot be parsed.
            harm = harm.apply(pd.to_numeric, errors="coerce")
            harm = harm.dropna(how="any")
        matrices.append(harm)
        metadata.append(meta)

    if not matrices:
        raise RuntimeError("no input matrices loaded — check data/raw/*")

    with stage_log("intersect gene index"):
        shared = matrices[0].index
        for m in matrices[1:]:
            shared = shared.intersection(m.index)
        print(f"  shared identifier set size: {len(shared):,}")
        if len(shared) == 0:
            raise RuntimeError(
                "Zero shared identifiers across cohorts. If sources use different ID schemes "
                "(probe IDs vs Ensembl vs HGNC), run with --harmonize and a per-cohort probe→gene mapping, "
                "or restrict --include to one cohort at a time for a single-source smoke test.",
            )
        matrices = [m.loc[shared] for m in matrices]

    with stage_log("select top-variable identifiers"):
        if len(matrices) > 1:
            combined_var = pd.concat([m.var(axis=1) for m in matrices], axis=1).mean(axis=1)
        else:
            combined_var = matrices[0].var(axis=1)
        top_genes = combined_var.sort_values(ascending=False).head(max_genes).index
        matrices = [m.loc[top_genes] for m in matrices]
        print(f"  retained top {len(top_genes):,} identifiers")

    with stage_log("rank-bin each cohort"):
        bin_arrays = [to_rank_bins(m, n_bins=n_bins) for m in matrices]

    with stage_log("write long-form parquet"):
        long_rows: list[pd.DataFrame] = []
        for m, meta, bins in zip(matrices, metadata, bin_arrays):
            samples = m.columns.tolist()
            genes = m.index.tolist()
            for j, sample in enumerate(samples):
                long_rows.append(pd.DataFrame({
                    "sample_id": sample,
                    "gene": genes,
                    "rank_bin": bins[:, j].astype(np.int8),
                }))
        long = pd.concat(long_rows, ignore_index=True)
        out = out_dir / "pretrain_corpus.parquet"
        long.to_parquet(out, compression="zstd", index=False)
        print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB; {len(long):,} rows; "
              f"{len(set(long['sample_id'])):,} samples × {len(set(long['gene'])):,} identifiers)")

    with stage_log("write metadata + manifest"):
        meta_all = pd.concat(metadata, ignore_index=True)
        meta_out = out_dir / "pretrain_corpus_metadata.parquet"
        meta_all.to_parquet(meta_out, compression="zstd", index=False)
        manifest = {
            "n_samples": int(sum(m.shape[1] for m in matrices)),
            "n_genes_in_corpus": int(len(top_genes)),
            "n_bins": n_bins,
            "sources": include,
            "per_source_shapes": {n: m.shape for n, m in zip(include, matrices)},
        }
        (out_dir / "pretrain_corpus.json").write_text(json.dumps(manifest, indent=2))
        print(f"  manifest -> {out_dir/'pretrain_corpus.json'}")

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Build HemeFM pretraining corpus parquet.")
    p.add_argument("--include", nargs="+", default=list(LOADERS),
                   help="Source loaders to include.")
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--max-genes", type=int, default=4096)
    p.add_argument("--no-harmonize", action="store_true",
                   help="Skip HGNC harmonization (use raw probe/ID names; "
                        "only useful when restricting --include to a single cohort).")
    args = p.parse_args()

    out = build(args.include, n_bins=args.bins, max_genes=args.max_genes,
                harmonize=not args.no_harmonize)
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
