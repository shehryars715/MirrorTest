# paper/ — the LaTeX manuscript

| File | What |
|---|---|
| `PAPER_DRAFT.md` | **THE CURRENT DRAFT (v1, 2026-07-20, all numbers final)** — read and review this one; it matches main.tex word-for-word |
| `main.tex` | the same complete draft in ACL LaTeX, ready for Overleaf (needs acl.sty + the two figure PDFs uploaded) |
| `references.bib` | all references incl. dataset citations — every entry carries a VERIFY note: check each on arXiv/ACL Anthology before submission |
| `TEMPLATE_PAPER.md` | the original fill-in template (kept for reference; superseded by PAPER_DRAFT.md) |

## Compiling

This file set expects the **official ACL style** (`acl.sty`), which is not
redistributed here. Easiest path (full steps in
[`../docs/06_writing_guide.md`](../docs/06_writing_guide.md)):

1. On [Overleaf](https://overleaf.com), create a project from the official
   ACL template (search the gallery, or upload the ZIP from the
   `acl-org/acl-style-files` GitHub repository).
2. Replace its `main.tex` with this one; upload `references.bib`.
3. Upload the figures: `results/figures/fig1_scale_curve.pdf` and
   `fig2_paraphrase.pdf` (then fix the `\includegraphics` paths — on
   Overleaf they sit next to main.tex, so drop the `../results/figures/`
   prefix).
4. Recompile until zero errors, fixing from the FIRST error downward.

## Workflow

* Write Method → Results → Intro → Related → Discussion → Abstract
  (in that order; reasons in the writing guide).
* Table 1's body is auto-generated: paste from
  `results/tables/table1_main.tex` after running `06_stats.py`.
* Replace every `<angle-bracket>` placeholder — search for `<`.
* Keep `\usepackage[review]{acl}` until camera-ready.
