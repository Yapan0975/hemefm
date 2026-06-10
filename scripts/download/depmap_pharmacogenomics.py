"""Download DepMap CCLE expression + GDSC2 + CTRP v2 drug response.

These are used for the leave-platform-out cross-platform drug-response evaluation
in §3.8 / RQ4 of the HemeFM manuscript.

DepMap (release rolls forward; pin a release in the URL below):
    https://depmap.org/portal/api/downloads/release/  -> resolves release ID
    https://depmap.org/portal/api/downloads/file?file_name=OmicsExpressionProteinCodingGenesTPMLogp1.csv&release=...

GDSC2:
    https://www.cancerrxgene.org/api/cell_line_drug_response?screening_set=GDSC2

CTRP v2:
    Broad: https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip
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

# Pin a DepMap release here. Update when newer.
DEPMAP_RELEASE = "25Q2"

DEPMAP_FILES: dict[str, str] = {
    # Expression (protein-coding TPM, log2(TPM+1)) for ~1900 cell lines
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv":
        f"https://depmap.org/portal/api/downloads/file?file_name=OmicsExpressionProteinCodingGenesTPMLogp1.csv&release={DEPMAP_RELEASE}&download=public",
    # Model metadata (lineage, primary disease, etc.)
    "Model.csv":
        f"https://depmap.org/portal/api/downloads/file?file_name=Model.csv&release={DEPMAP_RELEASE}&download=public",
    # GDSC2 IC50 (released under PRISM/Sanger collaboration; primary on cancerrxgene.org)
    "sanger-dose-response.csv":
        f"https://depmap.org/portal/api/downloads/file?file_name=sanger-dose-response.csv&release={DEPMAP_RELEASE}&download=public",
    # CTRP v2 secondary screen
    "CTRP_v2.0_2015_ctd2_ExpandedDataset.zip":
        "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip",
    # PRISM Repurposing Secondary Screen
    "secondary-screen-dose-response-curve-parameters.csv":
        f"https://depmap.org/portal/api/downloads/file?file_name=secondary-screen-dose-response-curve-parameters.csv&release={DEPMAP_RELEASE}&download=public",
}


def fetch() -> Path:
    dataset = f"depmap_{DEPMAP_RELEASE.lower()}_plus_ctrp"
    assert_disk_space(min_gb=5.0)
    env_proxy_hint()
    out = raw_dir(dataset)

    with stage_log(f"download {dataset}"):
        for name, url in DEPMAP_FILES.items():
            dest = out / name
            try:
                http_download(url, dest)
                record_checksum(dataset, dest)
            except Exception as exc:                      # noqa: BLE001
                print(f"  [warn] failed to fetch {name}: {exc}")
                # DepMap URLs sometimes need a redirect chase; keep going.

    write_manifest(
        dataset,
        {
            "depmap_release": DEPMAP_RELEASE,
            "files": list(DEPMAP_FILES),
            "notes": "AML cell lines: filter Model.csv with OncotreeCode == 'AML' or PrimaryDisease == 'Acute Myeloid Leukemia'.",
        },
    )
    print(f"[ok] {dataset} -> {out}")
    return out


if __name__ == "__main__":
    fetch()
