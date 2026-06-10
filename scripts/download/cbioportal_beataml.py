"""Download BeatAML 2.0 from cBioPortal.

Reference:
    Bottomly D et al. (2022). Integrative analysis of drug response and clinical
    outcome in AML. Cancer Cell 40(8):850-864. DOI: 10.1016/j.ccell.2022.07.002

Public bundle:  https://cbioportal-datahub.s3.amazonaws.com/aml_ohsu_2022.tar.gz
Contents (≈ 350 MB tar.gz expanding to ~1.2 GB):
    data_clinical_patient.txt
    data_clinical_sample.txt
    data_cna.txt
    data_mutations.txt
    data_mrna_seq_v2_rsem.txt              (~5 GB if uncompressed)
    data_mrna_seq_v2_rsem_zscores_ref_*.txt
    case_lists/*

Drug sensitivity files (separate, hosted at Vizome — fetched in a separate script).
"""
from __future__ import annotations

import sys
import tarfile
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

DATASET = "beataml_v2_cbioportal"
URL = "https://cbioportal-datahub.s3.amazonaws.com/aml_ohsu_2022.tar.gz"
# No published checksum from cBioPortal; we record the local one after first fetch.

WANTED_FILES = [
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
    "data_mutations.txt",
    "data_cna.txt",
    "data_mrna_seq_v2_rsem.txt",
    "data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt",
    "meta_study.txt",
]


def fetch() -> Path:
    assert_disk_space(min_gb=10.0)
    env_proxy_hint()
    out = raw_dir(DATASET)
    archive = out / "aml_ohsu_2022.tar.gz"

    with stage_log(f"download {DATASET}"):
        http_download(URL, archive)
        record_checksum(DATASET, archive)

    with stage_log(f"extract {DATASET}"):
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if Path(m.name).name in WANTED_FILES]
            if not members:
                # extract everything if the canonical filenames have shifted
                members = tar.getmembers()
            # `filter="data"` (Python 3.12+) blocks unsafe paths; fall back gracefully.
            extract_kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
            tar.extractall(out, members=members, **extract_kwargs)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    write_manifest(
        DATASET,
        {
            "source_url": URL,
            "citation": "Bottomly et al., Cancer Cell 40:850-864 (2022). DOI 10.1016/j.ccell.2022.07.002",
            "n_files": len(files),
            "files": [str(p.relative_to(out)) for p in files],
        },
    )
    print(f"[ok] {DATASET} -> {out} ({len(files)} files)")
    return out


if __name__ == "__main__":
    fetch()
