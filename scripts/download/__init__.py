"""HemeFM data fetchers (one module per dataset).

All fetchers share the conventions in `_utils.py`:
    - destination = `<repo>/data/raw/<dataset>/`
    - checksums recorded in `SHA256SUMS`
    - manifest written to `manifest.json`

Run a single fetcher with `python -m scripts.download.<name>`.
Run all with                `python -m scripts.download.run_all`.
"""
