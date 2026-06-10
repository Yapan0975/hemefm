"""Generic NCBI GEO series fetcher.

Used for:
    GSE6891       (Verhaak et al., AML microarray, ~461 samples)
    GSE37642      (Herold et al., AML microarray, ~562 samples)
    GSE116256     (van Galen et al., AML scRNA-seq, ~40k cells)

NCBI exposes per-series TAR archives at predictable URLs:
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE6nnn/GSE6891/suppl/GSE6891_RAW.tar
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116256/suppl/GSE116256_RAW.tar

The series matrix (already-normalized expression) is at:
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE6nnn/GSE6891/matrix/GSE6891_series_matrix.txt.gz
"""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from scripts.download._utils import (
    assert_disk_space,
    env_proxy_hint,
    http_download,
    raw_dir,
    record_checksum,
    stage_log,
    write_manifest,
)


def _series_dir_url(gse: str) -> str:
    """Return the GEO 'GSExnnn' directory URL for a given GSE accession."""
    n = gse[3:]
    prefix = n[:-3] if len(n) > 3 else "0"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}nnn/{gse}"


def fetch_series(
    gse: str,
    *,
    matrix: bool = True,
    raw_tar: bool = False,
    decompress_matrix: bool = True,
) -> Path:
    """Download GEO series files.

    Args:
        gse: e.g. "GSE6891"
        matrix: fetch the series_matrix.txt.gz (always cheap, ~5-50 MB)
        raw_tar: fetch the *_RAW.tar (large — for scRNA datasets only when needed)
        decompress_matrix: gunzip the matrix file in place after download
    """
    dataset = f"geo_{gse.lower()}"
    assert_disk_space(min_gb=5.0 + (10.0 if raw_tar else 0.0))
    env_proxy_hint()
    out = raw_dir(dataset)
    base = _series_dir_url(gse)

    urls: dict[str, str] = {}
    if matrix:
        urls[f"{gse}_series_matrix.txt.gz"] = f"{base}/matrix/{gse}_series_matrix.txt.gz"
    if raw_tar:
        urls[f"{gse}_RAW.tar"] = f"{base}/suppl/{gse}_RAW.tar"

    with stage_log(f"download GEO {gse}"):
        for name, url in urls.items():
            dest = out / name
            http_download(url, dest)
            record_checksum(dataset, dest)

    if matrix and decompress_matrix:
        gz = out / f"{gse}_series_matrix.txt.gz"
        plain = gz.with_suffix("")
        if not plain.exists():
            with gzip.open(gz, "rb") as src, plain.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            print(f"  [extracted] {plain.name}")

    write_manifest(
        dataset,
        {
            "gse": gse,
            "matrix_fetched": matrix,
            "raw_tar_fetched": raw_tar,
            "source": "NCBI GEO FTP",
        },
    )
    print(f"[ok] {gse} -> {out}")
    return out


SERIES = {
    "GSE6891": {"description": "Verhaak AML microarray (n=461)", "raw_tar": False},
    "GSE37642": {"description": "Herold AML microarray (n=562)", "raw_tar": False},
    "GSE116256": {"description": "van Galen AML scRNA-seq (n=40k cells)", "raw_tar": True},
}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gse", choices=[*SERIES.keys(), "all"], default="all")
    p.add_argument("--raw-tar", action="store_true", help="Also pull *_RAW.tar (large)")
    args = p.parse_args()

    targets = list(SERIES) if args.gse == "all" else [args.gse]
    for gse in targets:
        info = SERIES[gse]
        fetch_series(gse, raw_tar=args.raw_tar and info["raw_tar"])
