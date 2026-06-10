"""Download Hemap pan-myeloid transcriptome corpus.

Hemap is hosted at https://hemap.uta.fi — the curated expression matrix
(~9,500 samples × ~17,000 genes, RMA-normalized) is exposed under
/files/ with a structured form. **The host requires a manual click-through
acknowledgement** for first-time access, after which a direct link works.

This script:
    1. Prints the URL the user needs to visit once.
    2. After the user drops the downloaded matrix into data/raw/hemap/,
       re-running the script verifies presence and writes a manifest.

Expected files (place manually):
    hemap_expression_matrix.tsv.gz         (~700 MB compressed)
    hemap_sample_annotation.tsv            (~5 MB)
    hemap_gene_annotation.tsv              (~1 MB)
"""
from __future__ import annotations

import sys
from pathlib import Path

from scripts.download._utils import (
    raw_dir,
    record_checksum,
    sha256,
    stage_log,
    write_manifest,
)

DATASET = "hemap"

REQUIRED = [
    "hemap_expression_matrix.tsv.gz",
    "hemap_sample_annotation.tsv",
    "hemap_gene_annotation.tsv",
]

LANDING = "https://hemap.uta.fi/hemap_files.php"


def fetch() -> Path:
    out = raw_dir(DATASET)
    missing = [f for f in REQUIRED if not (out / f).exists()]
    if missing:
        print(f"[hemap] missing files in {out}:")
        for m in missing:
            print(f"        {m}")
        print()
        print(f"Hemap requires a manual click-through. Open in a browser:")
        print(f"    {LANDING}")
        print()
        print("Place the three files into:")
        print(f"    {out}")
        print()
        print("Then re-run this script.")
        sys.exit(2)

    with stage_log(f"hash {DATASET}"):
        for name in REQUIRED:
            p = out / name
            print(f"  {name:50s} {sha256(p)[:16]}…  {p.stat().st_size/1e6:.1f} MB")
            record_checksum(DATASET, p)

    write_manifest(
        DATASET,
        {
            "source": LANDING,
            "citation": "Pölönen P et al. The Hemap online resource. Cancer Cell (2019).",
            "manual_step": "Click-through agreement at the landing URL required.",
            "files": REQUIRED,
        },
    )
    print(f"[ok] {DATASET} -> {out}")
    return out


if __name__ == "__main__":
    fetch()
