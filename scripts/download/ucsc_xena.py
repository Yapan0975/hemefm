"""Download TCGA-LAML and TARGET-AML transcriptomes from UCSC Xena.

Xena hosts harmonized, log2(TPM+0.001) expression matrices plus aligned
clinical/mutation/CNV tables. URLs are stable.

TCGA-LAML hub: https://gdc.xenahubs.net/datapages/?dataset=TCGA-LAML.htseq_fpkm.tsv
TARGET-AML hub: https://tcga.xenahubs.net/datapages/?dataset=TARGET-AML
"""
from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
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


@dataclass(frozen=True)
class XenaFile:
    name: str
    url: str
    description: str


TCGA_LAML_FILES: list[XenaFile] = [
    XenaFile(
        "TCGA-LAML.htseq_fpkm-uq.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LAML.htseq_fpkm-uq.tsv.gz",
        "GDC harmonized HTSeq FPKM-UQ (log2(value+1)).",
    ),
    XenaFile(
        "TCGA-LAML.htseq_counts.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LAML.htseq_counts.tsv.gz",
        "GDC harmonized HTSeq raw counts (log2(count+1)) — preferred input to ComBat-seq.",
    ),
    XenaFile(
        "TCGA-LAML.mutect2_snv.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LAML.mutect2_snv.tsv.gz",
        "Mutect2 SNV calls.",
    ),
    XenaFile(
        "TCGA-LAML.gistic.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LAML.gistic.tsv.gz",
        "GISTIC copy-number focal calls.",
    ),
    XenaFile(
        "TCGA-LAML.methylation450.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LAML.methylation450.tsv.gz",
        "Illumina 450K beta-values.",
    ),
    XenaFile(
        "TCGA-LAML.GDC_phenotype.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LAML.GDC_phenotype.tsv.gz",
        "Clinical phenotype data.",
    ),
    XenaFile(
        "TCGA-LAML.survival.tsv",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LAML.survival.tsv",
        "OS / DFS / DSS / PFI survival.",
    ),
]

# TARGET-AML files (pediatric AML, on a different Xena hub)
TARGET_AML_FILES: list[XenaFile] = [
    XenaFile(
        "TARGET-AML.htseq_counts.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TARGET-AML.htseq_counts.tsv.gz",
        "HTSeq raw counts (log2(count+1)).",
    ),
    XenaFile(
        "TARGET-AML.htseq_fpkm-uq.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TARGET-AML.htseq_fpkm-uq.tsv.gz",
        "FPKM-UQ.",
    ),
    XenaFile(
        "TARGET-AML.mutect2_snv.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TARGET-AML.mutect2_snv.tsv.gz",
        "Mutect2 SNV calls.",
    ),
    XenaFile(
        "TARGET-AML.GDC_phenotype.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TARGET-AML.GDC_phenotype.tsv.gz",
        "Phenotype.",
    ),
    XenaFile(
        "TARGET-AML.survival.tsv",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TARGET-AML.survival.tsv",
        "OS / EFS.",
    ),
]


def _decompress_gz(gz_path: Path) -> Path:
    out = gz_path.with_suffix("")          # strip .gz
    if out.exists():
        return out
    with gzip.open(gz_path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    return out


def fetch(cohort: str = "TCGA-LAML", decompress: bool = False) -> Path:
    files = {"TCGA-LAML": TCGA_LAML_FILES, "TARGET-AML": TARGET_AML_FILES}[cohort]
    dataset = f"xena_{cohort.lower().replace('-', '_')}"
    assert_disk_space(min_gb=5.0)
    env_proxy_hint()
    out = raw_dir(dataset)

    with stage_log(f"download {cohort} from UCSC Xena"):
        for f in files:
            dest = out / f.name
            http_download(f.url, dest)
            record_checksum(dataset, dest)
            if decompress and dest.suffix == ".gz":
                _decompress_gz(dest)

    write_manifest(
        dataset,
        {
            "cohort": cohort,
            "source": "UCSC Xena (GDC hub for TCGA-LAML and TARGET-AML)",
            "decompressed": decompress,
            "files": [{"name": f.name, "url": f.url, "description": f.description} for f in files],
        },
    )
    print(f"[ok] {cohort} -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=["TCGA-LAML", "TARGET-AML", "both"], default="both")
    p.add_argument("--decompress", action="store_true")
    args = p.parse_args()
    if args.cohort == "both":
        fetch("TCGA-LAML", decompress=args.decompress)
        fetch("TARGET-AML", decompress=args.decompress)
    else:
        fetch(args.cohort, decompress=args.decompress)
