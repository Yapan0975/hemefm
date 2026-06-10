"""Shared HTTP / cache / checksum utilities for dataset fetchers.

Designed for an air-gapped server: every fetch is resumable, idempotent, and
records a SHA-256 in `data/raw/<dataset>/SHA256SUMS` for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

CHUNK = 1 << 20            # 1 MiB streaming chunks
USER_AGENT = "HemeFM-fetcher/0.1 (research; contact: placeholder@example.com)"


def project_root() -> Path:
    """Return the hemefm/ repo root regardless of where the script is invoked."""
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def raw_dir(dataset: str) -> Path:
    d = project_root() / "data" / "raw" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def interim_dir(dataset: str) -> Path:
    d = project_root() / "data" / "interim" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir() -> Path:
    d = project_root() / "data" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path: Path, buf_size: int = CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def record_checksum(dataset: str, file_path: Path, digest: str | None = None) -> None:
    """Append `<digest>  <relpath>` to data/raw/<dataset>/SHA256SUMS."""
    digest = digest or sha256(file_path)
    rel = file_path.relative_to(raw_dir(dataset))
    sums = raw_dir(dataset) / "SHA256SUMS"
    existing = {}
    if sums.exists():
        for line in sums.read_text().splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                existing[parts[1]] = parts[0]
    existing[str(rel)] = digest
    with sums.open("w") as f:
        for k in sorted(existing):
            f.write(f"{existing[k]}  {k}\n")


def http_download(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    max_retries: int = 4,
    retry_backoff: float = 5.0,
    overwrite: bool = False,
) -> Path:
    """Resumable HTTP/HTTPS GET with retry-on-failure + optional SHA-256 check.

    Returns the destination path. If `expected_sha256` is given and the existing
    file already matches, the network is not touched.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        if expected_sha256 is None:
            return dest
        if sha256(dest) == expected_sha256:
            return dest
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    if not overwrite and tmp.exists():
        start = tmp.stat().st_size
    else:
        if tmp.exists():
            tmp.unlink()
        start = 0

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                **({"Range": f"bytes={start}-"} if start > 0 else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (research script)
                mode = "ab" if start > 0 and resp.status == 206 else "wb"
                if mode == "wb":
                    start = 0
                with tmp.open(mode) as out:
                    total = int(resp.headers.get("Content-Length", "0")) + start
                    done = start
                    next_log = time.monotonic()
                    while chunk := resp.read(CHUNK):
                        out.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - next_log > 5:
                            pct = (100 * done / total) if total else 0
                            print(f"  [{dest.name}] {done/1e6:8.1f} MB ({pct:5.1f}%)", flush=True)
                            next_log = time.monotonic()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc = exc
            print(f"  [warn] attempt {attempt}/{max_retries} failed: {exc}", flush=True)
            if attempt < max_retries:
                time.sleep(retry_backoff * attempt)
                if tmp.exists():
                    start = tmp.stat().st_size
    else:
        raise RuntimeError(f"download failed after {max_retries} attempts: {url}") from last_exc

    tmp.rename(dest)
    if expected_sha256 is not None:
        actual = sha256(dest)
        if actual != expected_sha256:
            dest.unlink()
            raise ValueError(f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}")
    return dest


@contextmanager
def stage_log(stage: str) -> Iterator[None]:
    print(f"\n[stage] {stage}", flush=True)
    t0 = time.monotonic()
    yield
    print(f"[stage] {stage} done in {time.monotonic() - t0:.1f}s", flush=True)


def write_manifest(dataset: str, payload: dict) -> Path:
    """Write a JSON manifest summarising what was fetched."""
    payload = {**payload, "_dataset": dataset, "_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = raw_dir(dataset) / "manifest.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def assert_disk_space(min_gb: float = 5.0) -> None:
    free = shutil.disk_usage(project_root()).free / 1e9
    if free < min_gb:
        raise RuntimeError(f"insufficient disk: {free:.1f} GB free, need >= {min_gb} GB")


def env_proxy_hint() -> None:
    """If running on a network-constrained host, print the proxy/env hint once."""
    if any(os.environ.get(k) for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")):
        return
    print("[hint] no http_proxy / https_proxy set; if downloads fail with DNS errors,")
    print("       see hemefm/docs/server_dns_fix.md")
