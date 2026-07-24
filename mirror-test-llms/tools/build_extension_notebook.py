"""
build_extension_notebook.py — stitch the cross-family extension notebook.

This mirrors the convention of tools/build_kaggle_notebook.py: the cell
sources live as real, individually-lintable files under
tools/extension_notebook_src/ (edit THOSE, not the .ipynb), and this script
assembles them into

    <repo>/kaggle_extend_families.ipynb          (canonical, versioned)
    <repo>/../kaggle_extend_families.ipynb        (convenience copy to upload)

UNLIKE kaggle_run_all.ipynb, this notebook does NOT embed the pipeline as a
base64 zip. It reads the repo + frozen data from a READ-ONLY Kaggle Dataset
attached under /kaggle/input/ and writes only to /kaggle/working/. That is
the model the task specifies and keeps the notebook small and auditable.

    python tools/build_extension_notebook.py

Cells (in order):
    01_intro.md          what this is + Kaggle setup checklist
    02_config.py         RUN SETTINGS — every path/model/seed/pool knob
    03_bootstrap.py      fail-loud dataset discovery, patched config, imports
    04_orchestrator.py   resumable task DAG driving the published src/ scripts
    05_stats_outputs.py  extended_table1.csv + dissociation_summary.csv + raw
    06_dissociation.py   dissociation.png (implicit vs explicit, per recipe)
    07_finalize.py       run_report.txt + closing summary
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "tools" / "extension_notebook_src"

# (filename, cell_type). Order is the notebook order.
CELLS = [
    ("01_intro.md", "markdown"),
    ("02_config.py", "code"),
    ("03_bootstrap.py", "code"),
    ("04_orchestrator.py", "code"),
    ("05_stats_outputs.py", "code"),
    ("06_dissociation.py", "code"),
    ("07_finalize.py", "code"),
]


def code_cell(cid: str, source: str) -> dict:
    return {"id": cid, "cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}


def md_cell(cid: str, source: str) -> dict:
    return {"id": cid, "cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def main() -> None:
    h = hashlib.sha256()
    cells = []
    for fname, ctype in CELLS:
        path = SRC / fname
        text = path.read_text(encoding="utf-8")
        h.update(path.name.encode("utf-8"))
        h.update(text.encode("utf-8"))
        cid = fname.split(".")[0]
        if ctype == "markdown":
            text = text.replace("__BUILD_STAMP__",
                                datetime.date.today().isoformat() + "-" + h.hexdigest()[:8])
            cells.append(md_cell(cid, text))
        else:
            cells.append(code_cell(cid, text))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                       "isGpuEnabled": True},
        },
        "cells": cells,
    }

    out = REPO / "kaggle_extend_families.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    convenience = REPO.parent / "kaggle_extend_families.ipynb"
    try:
        shutil.copy2(out, convenience)
        conv_msg = f"[build] copied to {convenience}"
    except OSError as e:
        conv_msg = f"[build] (skipped convenience copy: {e})"
    print(f"[build] {len(cells)} cells, build={datetime.date.today().isoformat()}-{h.hexdigest()[:8]}")
    print(f"[build] wrote {out}")
    print(conv_msg)
    print("[build] upload kaggle_extend_families.ipynb to Kaggle; attach the repo "
          "as a read-only Dataset under /kaggle/input/.")


if __name__ == "__main__":
    main()
