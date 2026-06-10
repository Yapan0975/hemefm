"""Download GTEx whole-blood + bone-marrow transcriptomes.

GTEx Portal v10 (Jan 2025) public bulk RNA-seq:
    https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/

Files (~700 whole-blood, ~150 bone-marrow samples by v10):
    GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz  (all tissues; ~9 GB)
    GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt  (tissue type mapping)
    GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt

For HemeFM we subset whole-blood (SMTS == "Blood") and bone-marrow (SMTSD == "Cells - Cultured fibroblasts" is NOT this; the canonical hematopoietic tissue in GTEx is "Whole Blood" — bone-marrow is not collected by GTEx). We use whole blood as healthy myeloid reference instead.
"""
from __future__ import annotations

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

DATASET = "gtex_v10"
BASE = "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq"
ANNOT_BASE = "https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files"

FILES = {
    # The full TPM matrix is large (~9 GB). Prefer the gene-level filtered version
    # when v10 publishes one; otherwise we filter to whole-blood after download.
    "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz": f"{BASE}/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz",
    "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz": f"{BASE}/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz",
    "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt": f"{ANNOT_BASE}/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt",
    "GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt": f"{ANNOT_BASE}/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt",
}


def fetch(skip_large: bool = False) -> Path:
    """Download GTEx v10 files. Set skip_large=True to skip the full TPM matrix."""
    assert_disk_space(min_gb=25.0)
    env_proxy_hint()
    out = raw_dir(DATASET)

    with stage_log(f"download {DATASET}"):
        for name, url in FILES.items():
            if skip_large and "gene_tpm" in name:
                print(f"  [skip] {name} (skip_large=True)")
                continue
            dest = out / name
            http_download(url, dest)
            record_checksum(dataset=DATASET, file_path=dest)

    write_manifest(
        DATASET,
        {
            "source": "GTEx Portal v10 (https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression)",
            "v10_release_date": "2025-01",
            "subset_target": "whole-blood samples only (bone marrow not collected by GTEx)",
            "filter_logic": "SMTSD == 'Whole Blood' from SampleAttributesDS",
        },
    )
    print(f"[ok] {DATASET} -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-large", action="store_true", help="skip 9 GB TPM matrix")
    args = p.parse_args()
    fetch(skip_large=args.skip_large)
