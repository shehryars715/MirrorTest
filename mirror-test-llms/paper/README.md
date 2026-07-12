# paper/ — the LaTeX manuscript

| File | What |
|---|---|
| `TEMPLATE_PAPER.md` | **start here** — the paper written out in full prose with 〈placeholders〉 and per-outcome variant sentences; read it to see what "done" looks like, draft into it, then port section-by-section into main.tex |
| `main.tex` | full ACL short-paper skeleton: draft abstract, section scaffolds with page budgets, Limitations pre-filled, appendix wired to `results/` |
| `references.bib` | the 11 protocol references (+ dataset-citation notes) — every entry marked for verification |

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
