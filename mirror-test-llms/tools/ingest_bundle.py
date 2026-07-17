"""
ingest_bundle.py — absorb a mirror_bundle.zip downloaded from Kaggle into
this repository, then tell you (or Claude) what to run next.

    python tools/ingest_bundle.py                 # newest bundle in Desktop/Mirror
    python tools/ingest_bundle.py path/to/mirror_bundle.zip

Merge rule (matches the Kaggle-side seeding): a file is copied in only if it
is missing locally or the bundle's copy is LARGER — pipeline outputs are
append-only JSONL, so larger means strictly newer. Regenerable outputs
(results/tables, results/figures) are skipped; they get rebuilt locally.
The bundle's SESSION_REPORT.md is archived under results/session_reports/.
"""

from __future__ import annotations

import datetime
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SKIP_PREFIXES = ("results/tables", "results/figures")


def find_default_bundle() -> Path:
    """Newest Kaggle download in Desktop/Mirror or Desktop/Mirror/Output.

    Accepts both the bare bundle (mirror_bundle*.zip) and Kaggle's
    'Download All' wrapper (results*.zip / <notebook-name>*.zip), which
    contains mirror_bundle.zip inside — unwrap() handles that case."""
    hits = []
    for folder in (REPO.parent, REPO.parent / "Output"):
        for pat in ("mirror_bundle*.zip", "results*.zip", "*.zip"):
            hits += list(folder.glob(pat))
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        sys.exit(f"[ingest] no bundle zip found in {REPO.parent} or "
                 f"{REPO.parent / 'Output'} - download it from the Kaggle "
                 "version's Output tab first, or pass a path explicitly")
    return hits[0]


def unwrap(bundle: Path, td: str) -> Path:
    """If `bundle` is Kaggle's wrapper zip, extract the inner
    mirror_bundle.zip it contains and return that; else return `bundle`."""
    with zipfile.ZipFile(bundle) as zf:
        inner = [n for n in zf.namelist()
                 if Path(n).name == "mirror_bundle.zip"
                 and "_seed_" not in n]          # ignore re-packed seed copies
        if not inner:
            return bundle
        inner.sort(key=len)                      # shallowest = current session
        out = Path(td) / "inner_mirror_bundle.zip"
        out.write_bytes(zf.read(inner[0]))
        print(f"[ingest] wrapper zip detected - using inner {inner[0]}")
        return out


def main() -> None:
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else find_default_bundle()
    if not bundle.exists():
        sys.exit(f"[ingest] not found: {bundle}")
    print(f"[ingest] reading {bundle}")

    copied = kept = skipped = 0
    with tempfile.TemporaryDirectory() as td:
        bundle = unwrap(bundle, td)
        extract_dir = Path(td) / "x"
        with zipfile.ZipFile(bundle) as zf:   # close before temp-dir cleanup (Windows locks)
            zf.extractall(extract_dir)
        base = extract_dir
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(base).as_posix()
            if rel == "SESSION_REPORT.md":
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
                dst = REPO / "results" / "session_reports" / f"SESSION_REPORT_{stamp}.md"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                print(f"[ingest] report archived -> {dst.relative_to(REPO)}")
                continue
            if any(rel.startswith(p) for p in SKIP_PREFIXES):
                skipped += 1
                continue
            dst = REPO / rel
            if dst.exists() and dst.stat().st_size >= f.stat().st_size:
                kept += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied += 1

    print(f"[ingest] copied {copied} files, kept {kept} newer/equal local, "
          f"skipped {skipped} regenerable")
    print("[ingest] next steps:")
    print("  python src/05_baselines.py stylometric   # CPU baseline (if pairs exist)")
    print("  python src/06_stats.py                   # rebuild all tables + figures")
    print("  git add -A && git commit -m 'ingest kaggle session results'")


if __name__ == "__main__":
    main()
