# results/ — what lands where

| Directory | Contents | Produced by |
|---|---|---|
| `judgments/` | raw per-run judge decisions: `ppp__{judge}.jsonl`, `ipp__{judge}.jsonl` | `03_judge_ppp.py`, `04_judge_ipp.py` |
| `baselines/` | `stylo__*.json` + per-pair `stylo_pairs__*.jsonl`, `transfer__*.json`, `ppl__{judge}.jsonl` | `05_baselines.py` |
| `tables/` | every table as CSV (+ MD for reading, + `table1_main.tex` for the paper) and `all_stats.json` | `06_stats.py` |
| `figures/` | `fig1_scale_curve`, `fig2_paraphrase`, `fig3_placebo` as PNG (viewing) + PDF (LaTeX) | `06_stats.py` |

Read first, in order: `tables/main_ppp.md` → `tables/placebo.csv`
(**must be ≈50%** — if not, stop and debug before interpreting anything)
→ `tables/paraphrase.csv` → `tables/ipp.csv`.

The whole directory is regenerable: `python src/06_stats.py` rebuilds every
table and figure from `judgments/` + `baselines/` in about a minute — that
command is also the paper's "reproduce Figure 1 in one command" claim.

Raw judgments are released with the repo (protocol §10 "All raw judgments
released as JSONL").
