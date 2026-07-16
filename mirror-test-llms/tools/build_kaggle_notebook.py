"""
build_kaggle_notebook.py — regenerate the self-contained Kaggle runner.

The runner notebook (kaggle_run_all.ipynb) embeds the ENTIRE pipeline
(src/ + configs/ + tests/ + requirements.txt) as a base64 zip, so the user
uploads ONE file to Kaggle and never touches git. Whenever the pipeline
code changes, re-run this builder:

    python tools/build_kaggle_notebook.py

Outputs:
    <repo>/kaggle_run_all.ipynb          (canonical)
    <repo>/../kaggle_run_all.ipynb       (convenience copy next to mirrortest.md)

Cell sources live in tools/kaggle_notebook_src/ — edit those, not the .ipynb.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "tools" / "kaggle_notebook_src"

EMBED = ["src", "configs", "tests"]          # directories shipped to Kaggle
EMBED_FILES = ["requirements.txt", "PREREGISTRATION.md"]


def build_payload() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in EMBED:
            for f in sorted((REPO / d).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    zf.write(f, f.relative_to(REPO))
        for name in EMBED_FILES:
            if (REPO / name).exists():
                zf.write(REPO / name, name)
    return buf.getvalue()


def code_cell(cid: str, source: str) -> dict:
    return {"id": cid, "cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}


def md_cell(cid: str, source: str) -> dict:
    return {"id": cid, "cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def main() -> None:
    payload = build_payload()
    b64 = base64.b64encode(payload).decode("ascii")
    stamp = (datetime.date.today().isoformat() + "-"
             + hashlib.sha256(payload).hexdigest()[:8])

    intro = (SRC / "01_intro.md").read_text(encoding="utf-8")
    config = (SRC / "02_config.py").read_text(encoding="utf-8") \
        .replace("__BUILD_STAMP__", stamp)
    bootstrap = (SRC / "03_bootstrap.py").read_text(encoding="utf-8") \
        .replace("__PAYLOAD_B64__", b64)
    orchestrator = (SRC / "04_orchestrator.py").read_text(encoding="utf-8")
    finalize = (SRC / "05_finalize.py").read_text(encoding="utf-8")

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
        },
        "cells": [
            md_cell("intro", intro),
            code_cell("config", config),
            code_cell("bootstrap", bootstrap),
            code_cell("orchestrator", orchestrator),
            code_cell("finalize", finalize),
        ],
    }

    out = REPO / "kaggle_run_all.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    convenience = REPO.parent / "kaggle_run_all.ipynb"
    shutil.copy2(out, convenience)
    print(f"[build] payload: {len(payload)//1024} KB zipped "
          f"({len(b64)//1024} KB embedded), build={stamp}")
    print(f"[build] wrote {out}")
    print(f"[build] copied to {convenience}  <- upload THIS file to Kaggle")


if __name__ == "__main__":
    main()
