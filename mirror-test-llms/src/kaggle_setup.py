"""
kaggle_setup.py — one-command environment bootstrap for Kaggle (and Colab).

Run this FIRST in every GPU session:

    python src/kaggle_setup.py                      # normal (repo cloned into
                                                    #  a writable directory)
    python src/kaggle_setup.py --from-input mirror-run-3
                                                    # additionally seed data/
                                                    #  + results/ from a prior
                                                    #  session saved as a
                                                    #  Kaggle dataset/output

WHAT IT DOES (idempotent — safe to run every session)
=====================================================
1. Detects the platform (Kaggle / Colab / local) and reports the GPUs.
2. Configures Hugging Face authentication non-interactively:
   env var HF_TOKEN, or on Kaggle the notebook secret named "HF_TOKEN"
   (Add-ons -> Secrets). Needed only for the gated Llama/Gemma foils.
3. Verifies the data/results tree is WRITABLE. If the repo itself sits on a
   read-only disk (e.g. attached as a Kaggle *dataset* under /kaggle/input),
   it redirects all data/results I/O to /kaggle/working by setting the
   MIRROR_ROOT environment variable convention (and tells you how to make
   that stick for subsequent commands).
4. Optionally (--from-input) copies data/ and results/ from a previous
   session's saved output into the working tree WITHOUT overwriting anything
   newer you already have — this is the resume-across-sessions mechanism if
   you don't want to push to GitHub from Kaggle.
5. Prints exactly what to run next.

KAGGLE SESSION RECIPE (see notebooks/kaggle_pipeline.ipynb for the wired-up
notebook version):
    Settings: Accelerator = GPU T4 x2, Internet = ON, add secret HF_TOKEN.
    !git clone https://github.com/<you>/mirror-test-llms
    %cd mirror-test-llms
    !pip -q install -U transformers accelerate bitsandbytes datasets sentence-transformers pyyaml
    !python src/kaggle_setup.py
    !python src/01_generate.py --models qwen2.5-14b-instruct --placebo
    ... at session end: Save Version (its output preserves data/ + results/),
        or git push (see the notebook's final cell).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402  (import as module so we can report its resolved paths)


def detect_platform() -> str:
    if Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        return "colab"
    return "local"


def report_gpus() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            print("[gpu] NO CUDA GPU VISIBLE - on Kaggle: Settings -> Accelerator "
                  "-> GPU T4 x2; on Colab: Runtime -> Change runtime type -> T4.")
            return
        n = torch.cuda.device_count()
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"[gpu] cuda:{i} {p.name} ({p.total_memory / 2**30:.1f} GB)")
        if n >= 2:
            print("[gpu] 2+ GPUs: device_map='auto' will shard big models "
                  "(the 14B judge fits on Kaggle's 2xT4).")
    except ImportError:
        print("[gpu] torch not installed yet - run the pip install cell first.")


def ensure_writable_root() -> Path:
    """Make sure the data/results tree is writable; redirect to
    /kaggle/working if the repo is on a read-only mount."""
    root = utils.IO_ROOT
    try:
        root.joinpath("data").mkdir(parents=True, exist_ok=True)
        probe = root / "data" / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print(f"[io] data/results root: {root} (writable)")
        return root
    except OSError:
        pass
    fallback = Path("/kaggle/working") if detect_platform() == "kaggle" else Path.cwd()
    os.environ["MIRROR_ROOT"] = str(fallback)
    print(f"[io] {root} is READ-ONLY -> redirecting data/results to {fallback}")
    print("[io] IMPORTANT: subsequent commands in OTHER cells must also see "
          "MIRROR_ROOT. In a notebook run:  %env MIRROR_ROOT=" + str(fallback))
    # Note: this process's utils paths were computed at import; scripts run
    # AFTER this one inherit the env var (via %env / os.environ in the same
    # kernel) and resolve correctly.
    return fallback


def seed_from_input(source: str, dest_root: Path) -> None:
    """Copy data/ and results/ from a previous session's saved output.

    `source` is a Kaggle input name (looked up under /kaggle/input/<name>)
    or a direct path. Only copies files that do NOT already exist at the
    destination — never clobbers newer local work. Handles both layouts:
    <src>/data,<src>/results and <src>/mirror-test-llms/data,... .
    """
    src = Path(source)
    if not src.exists():
        src = Path("/kaggle/input") / source
    if not src.exists():
        sys.exit(f"[seed] source not found: {source} (looked in /kaggle/input/)")
    candidates = [src, src / "mirror-test-llms"]
    base = next((c for c in candidates if (c / "data").exists() or (c / "results").exists()), None)
    if base is None:
        sys.exit(f"[seed] no data/ or results/ found under {src}")
    copied = skipped = 0
    for sub in ("data", "results"):
        sdir = base / sub
        if not sdir.exists():
            continue
        for f in sdir.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(base)
            dst = dest_root / rel
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied += 1
    print(f"[seed] copied {copied} files from {base} ({skipped} already present, kept local)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap a Kaggle/Colab GPU session.")
    ap.add_argument("--from-input", default=None, metavar="NAME_OR_PATH",
                    help="seed data/ + results/ from a previous session's saved "
                         "output (Kaggle dataset name under /kaggle/input, or a path)")
    args = ap.parse_args()

    platform = detect_platform()
    print(f"[env] platform: {platform}")
    report_gpus()
    utils.setup_hf_auth()
    root = ensure_writable_root()
    if args.from_input:
        seed_from_input(args.from_input, root)

    print("\n[ready] next commands (see ROADMAP.md for the full session plan):")
    print("  python src/01_generate.py --models <key> [--placebo]")
    print("  python src/03_judge_ppp.py --judge <key> --include-placebo")
    print("At session end on Kaggle: 'Save Version' preserves data/ + results/ "
          "as this notebook's output; next session, attach it as input and run "
          "this script with --from-input <that-output-name>.")


if __name__ == "__main__":
    main()
