"""Pediatric TARGET-AML — primary route is UCSC Xena (see ucsc_xena.py).

This stub remains for a future GDC API-based fetcher (more granular metadata
than UCSC Xena) if needed. For HemeFM's pretraining + downstream tasks, the
Xena-harmonized TARGET-AML matrix is sufficient.

To switch to the GDC route later:
    https://api.gdc.cancer.gov/
    Project: TARGET-AML  (case_filter: "cases.project.project_id": "TARGET-AML")
    Filtered RNA-seq STAR counts → STAR FPKM (or HTSeq legacy).

The Xena copy contains the same data with GDC's harmonization pipeline applied.
"""
from __future__ import annotations

from scripts.download.ucsc_xena import fetch as fetch_xena


def fetch() -> None:
    """Wrapper: pull TARGET-AML through UCSC Xena."""
    fetch_xena("TARGET-AML")


if __name__ == "__main__":
    fetch()
