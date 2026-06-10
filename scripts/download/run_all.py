"""Run every dataset fetcher in sequence.

Usage:
    python -m scripts.download.run_all                  # everything except large
    python -m scripts.download.run_all --include-large  # incl. GTEx 9 GB + scRNA RAW
    python -m scripts.download.run_all --only beataml tcga_laml
"""
from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable

from scripts.download import cbioportal_beataml, depmap_pharmacogenomics, geo_series, gtex, hemap, ucsc_xena

TARGETS: dict[str, Callable[..., object]] = {
    "beataml":    lambda **kw: cbioportal_beataml.fetch(),
    "tcga_laml":  lambda **kw: ucsc_xena.fetch("TCGA-LAML"),
    "target_aml": lambda **kw: ucsc_xena.fetch("TARGET-AML"),
    "gse6891":    lambda **kw: geo_series.fetch_series("GSE6891"),
    "gse37642":   lambda **kw: geo_series.fetch_series("GSE37642"),
    "gse116256":  lambda **kw: geo_series.fetch_series("GSE116256", raw_tar=kw.get("include_large", False)),
    "gtex":       lambda **kw: gtex.fetch(skip_large=not kw.get("include_large", False)),
    "depmap":     lambda **kw: depmap_pharmacogenomics.fetch(),
    "hemap":      lambda **kw: hemap.fetch(),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=list(TARGETS), default=None)
    p.add_argument("--skip", nargs="+", choices=list(TARGETS), default=[])
    p.add_argument("--include-large", action="store_true", help="Include 9 GB GTEx + scRNA RAW tar")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    selected = args.only or list(TARGETS)
    selected = [s for s in selected if s not in args.skip]

    results: dict[str, str] = {}
    for name in selected:
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        try:
            TARGETS[name](include_large=args.include_large)
            results[name] = "ok"
        except SystemExit as e:
            results[name] = f"SystemExit:{e.code}"
        except Exception as e:                       # noqa: BLE001
            results[name] = f"FAILED: {type(e).__name__}: {e}"
            traceback.print_exc()
            if not args.continue_on_error:
                break

    print(f"\n\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    width = max(len(k) for k in results)
    for k, v in results.items():
        marker = "✓" if v == "ok" else "✗"
        print(f"  {marker} {k:<{width}}  {v}")

    failed = [k for k, v in results.items() if v != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
