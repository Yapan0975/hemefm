"""Harmonize gene symbols across heterogeneous source matrices.

Different cohorts use different gene-id conventions:
    BeatAML:    HGNC symbols (current)
    TCGA Xena:  Ensembl gene IDs (versioned, ENSG00000000003.13)
    TARGET:    Ensembl IDs
    Hemap:      HGNC symbols (older alias set)
    GTEx:       Ensembl IDs (versioned)
    GEO arrays: probeset IDs (e.g., Affymetrix HG-U133 Plus 2.0)

We standardize on the **HGNC approved symbol** as the canonical identifier and
keep a mapping table on disk. The reference comes from HGNC's official download
(https://www.genenames.org/download/statistics-and-files/) — bundle a snapshot
under data/references/hgnc_complete_set.tsv.gz.

Probeset → gene mapping uses Bioconductor annotation packages (hgu133plus2.db)
processed once offline and shipped as a parquet under data/references/.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

# Allow execution either as `python -m scripts.preprocess.hgnc_harmonize` or directly.
try:
    from scripts.download._utils import processed_dir, project_root
except ModuleNotFoundError:                                     # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.download._utils import processed_dir, project_root


def reference_dir() -> Path:
    d = project_root() / "data" / "references"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_hgnc_table(path: Path | None = None) -> pd.DataFrame:
    """Load HGNC complete-set TSV (snapshot kept under data/references/)."""
    path = path or (reference_dir() / "hgnc_complete_set.tsv.gz")
    if not path.exists():
        raise FileNotFoundError(
            f"HGNC reference not found at {path}.\n"
            "Download once from https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt\n"
            "and gzip it into data/references/.",
        )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", low_memory=False)
    return df


def build_alias_map(hgnc: pd.DataFrame) -> dict[str, str]:
    """Construct {alias_or_id -> approved_symbol}.

    Sources of mapping (in priority):
        1. Approved symbol (self-map)
        2. Previous symbols
        3. Alias symbols
        4. Ensembl gene id (versioned and unversioned)
        5. NCBI / Entrez id
    """
    m: dict[str, str] = {}
    for _, row in hgnc.iterrows():
        approved = str(row.get("symbol", "")).strip()
        if not approved or approved == "nan":
            continue
        m[approved] = approved
        for col in ("alias_symbol", "prev_symbol"):
            raw = row.get(col)
            if pd.isna(raw):
                continue
            for sym in str(raw).split("|"):
                sym = sym.strip()
                if sym and sym not in m:
                    m[sym] = approved
        ensembl = row.get("ensembl_gene_id")
        if pd.notna(ensembl):
            m[str(ensembl)] = approved
        entrez = row.get("entrez_id")
        if pd.notna(entrez):
            m[str(int(entrez))] = approved
    return m


def strip_ensembl_version(eid: str) -> str:
    """ENSG00000000003.13 -> ENSG00000000003"""
    return eid.split(".", 1)[0] if eid.startswith("ENSG") else eid


def harmonize_index(index: pd.Index, alias_map: dict[str, str]) -> pd.Series:
    """Map a pandas Index of gene identifiers to canonical HGNC symbols.

    Returns a Series with NaN for unmapped entries; caller decides whether to drop.
    """
    raw = pd.Series(index, dtype="object")
    # Strip Ensembl versions first
    stripped = raw.apply(strip_ensembl_version)
    mapped = stripped.map(alias_map)
    return mapped.where(mapped.notna(), other=None)


def harmonize_matrix(expression: pd.DataFrame, alias_map: dict[str, str], how: str = "mean") -> pd.DataFrame:
    """Re-index an (genes × samples) matrix on harmonized HGNC symbols.

    Collisions (two source IDs mapping to the same canonical symbol) are
    aggregated with `how`:  'mean' (default), 'sum', or 'first'.
    """
    mapped = harmonize_index(expression.index, alias_map)
    keep_mask = mapped.notna()
    expression = expression.loc[keep_mask.values].copy()
    expression.index = mapped[keep_mask].values
    expression.index.name = "hgnc_symbol"

    if expression.index.has_duplicates:
        if how == "mean":
            expression = expression.groupby(level=0).mean()
        elif how == "sum":
            expression = expression.groupby(level=0).sum()
        elif how == "first":
            expression = expression[~expression.index.duplicated(keep="first")]
        else:
            raise ValueError(f"unknown collision strategy: {how}")

    return expression


def example_usage() -> None:
    """Documentation-only: show the typical pipeline call site."""
    hgnc = load_hgnc_table()
    alias_map = build_alias_map(hgnc)
    print(f"HGNC alias map: {len(alias_map):,} entries")
    # In a downstream call:
    #   matrix_canonical = harmonize_matrix(beataml_matrix, alias_map, how="mean")


if __name__ == "__main__":
    example_usage()
