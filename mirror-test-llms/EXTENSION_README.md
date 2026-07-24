# Cross-family extension — `kaggle_extend_families.ipynb`

A single, resumable Kaggle notebook (**GPU T4 ×2**) that extends the single-family
(Qwen2.5) mirror-test study's **implicit/explicit dissociation** to more open training
recipes **without changing any published number**. It answers the reviewer question the
paper invites: *is the dissociation a Qwen artifact, or does it hold across training
recipes?*

It reuses the repository's own scripts and functions verbatim, so every new cell is
directly comparable to the released ones. The extended table was checked to reproduce
`results/tables/main_ppp.csv` **exactly to 4 decimals** on the Qwen cells (acc, Wilson CI,
AUROC, stylometry, perplexity-rule acc, κ, Holm p and reject flag), and the margin–margin
ρ reproduces `tools/margin_correlation.py` exactly.

## What it does

Promote three existing *foil* families to *judges* at their current size: `gemma-2-9b-it`,
`mistral-7b-instruct-v0.3`, `llama-3.2-3b-instruct`. Their generations already exist; the
only new generation is each one's **placebo second sample (`s2`)**. Then PPP + IPP judging,
per-token NLL for the perplexity rule, placebo, and stylometry. → the dissociation
(near-perfect implicit self-info, chance-level explicit choice) replicates on **4 recipes**
(Qwen + these three). Every judge is ≤ 9B and loads on a **single T4 in 4-bit** — no
sharding, no large downloads.

**Deliberately not extended** (kept Qwen-only, as in the paper): the **scale-trend** result
(no second-family size sweep), the **paraphrase** attack, and the **base-vs-instruct**
ablation. All compute goes to judging, perplexity, and stylometry for the three new judges.

### Common foil pool (family exclusion)
Pool = `{llama-3.2-3b-instruct, gemma-2-9b-it, mistral-7b-instruct-v0.3, human}`; each
judge's foils = the pool families **not in its own family**, plus `human`. All foil text
pre-exists — foils are never regenerated.

| Judge | Foils |
|---|---|
| gemma-2-9b-it | llama-3.2-3b-instruct, mistral-7b-instruct-v0.3, human |
| mistral-7b-instruct-v0.3 | llama-3.2-3b-instruct, gemma-2-9b-it, human |
| llama-3.2-3b-instruct | gemma-2-9b-it, mistral-7b-instruct-v0.3, human |
| qwen2.5-* (reused, unchanged) | llama, gemma-2-9b, mistral, human |

## Prerequisites (once)

Upload-and-run, like `kaggle_run_all.ipynb`: the notebook **clones the repo itself**, so
there is no Dataset to attach and nothing to unzip.

1. **Push the repo (public)** to the URL in the config cell's `REPO_GIT_URL`
   (default `https://github.com/shehryars715/MirrorTest.git`); a private repo just needs a
   `GITHUB_TOKEN` secret.
2. On Kaggle: **Accelerator = GPU T4 ×2**, **Internet = ON** (already required for the model
   weights, so the clone adds no new dependency).
3. **Add secret `HF_TOKEN`** (Add-ons → Secrets), and with that same HF account **accept
   the licenses** for `google/gemma-2-9b-it` and `meta-llama/Llama-3.2-3B-Instruct`
   (Mistral-v0.3 is not gated).

## How to run

**Upload `kaggle_extend_families.ipynb` and press Run All.** The `02_config` cell has every
knob (`REPO_GIT_URL`/`REPO_GIT_REF`, budget hours, foil pool). The bootstrap clones the
repo, redirects all I/O to `/kaggle/working/`, verifies the frozen inputs (checksums), and
never regenerates them. (If you prefer, attach the repo as a Dataset instead — it is
auto-detected and used without cloning.)

**Robust 4-bit.** The bootstrap does not pin the whole stack (pinning `bitsandbytes` on top
of Kaggle's torch is what broke it before); it refreshes only the fragile pieces and then
**verifies 4-bit NF4 works**. If 4-bit is unavailable it **skips GPU work** rather than
silently run in float16 (which would break comparability with the paper's 4-bit judges).

**Resumable.** Each pipeline stage is internally resumable (skips finished items) and the
orchestrator checkpoints to `/kaggle/working/checkpoints/`. Three ≤9B judges fit comfortably
in one ~12 h session; if a session ends early, *Save Version*, start a fresh session, attach
this run's saved output as a Dataset, and Run All — the repo re-clones and completed units
are skipped. No model is ever loaded in the notebook kernel; every model runs in an isolated
subprocess, so a per-model failure is contained (recorded, others continue) rather than
crashing the run.

The HF weight cache is placed on `/kaggle/temp` (large ephemeral scratch), **not**
`/kaggle/working` (which has a ~20 GB persisted cap).

## Outputs (`/kaggle/working/`)

| File | What it is |
|---|---|
| `extended_table1.csv` | Per (judge, foil): n, PPP acc + Wilson CI, AUROC + CI, stylometry acc, perplexity-rule acc, κ + CI, exact-binomial p, **Holm p + reject recomputed over ALL cells present** (5 Qwen judges + 3 new judges, not the old 20). Base ablation excluded, as in the paper's Table 1. |
| `dissociation_summary.csv` | Per judge: implicit (PPL-rule) vs explicit (PPP) accuracy pooled over the **same** LLM-foil universe, the **implicit−explicit accuracy gap** (primary dissociation evidence), an `_allfoils` implicit column (reproduces the paper's prose headline), κ (median/abs-max), the **margin–margin Spearman ρ + CI** (non-saturating companion to κ), and human-vs-machine AUROC + CI. |
| `dissociation.png` | A dumbbell per judge: implicit (perplexity-rule) vs explicit (verbal PPP) accuracy, colored by family, chance line — the gap is visibly the same across all four recipes. |
| `extended_outputs/raw/` | Raw per-(judge,foil) PPP judgments (JSONL + parquet if pandas available) plus each new judge's `ppl__*` and `ipp__*`. |
| `run_report.txt` | Timings, GPU-hours, reused-vs-generated counts, model revisions used, Holm family size, per-cell n + power note, library versions. |

The new raw pipeline artifacts also land in the usual repo layout under
`/kaggle/working/results/` and `/kaggle/working/data/` so `src/06_stats.py` can be re-run
on the union if desired.

## Methodological identity — how comparability is guaranteed

* New generations use `src/01_generate.py` unchanged (same decoding: temp 0.8, top-p 0.95,
  ≤160 tokens; seeds `1000+idx` / placebo `2000+idx`; pinned revisions from the frozen config).
* Pairs, judging, IPP, perplexity, stylometry run the published `src/02–05` scripts via a
  patched config that only **adds** the new judges (`foils: [human]` so the promoted
  families keep `is_judge=True` for their `s2` sample; foil text is supplied explicitly and
  read off disk).
* Statistics reuse `src/06_stats.py` and `src/stats_utils.py` functions unchanged
  (`cell_stats`, `judge_vs_ppl_kappa`, `holm_bonferroni`, `auroc`, `wilson_ci`,
  `bootstrap_ci`). AUROC CI = 500 resamples (repo default) and κ CI = 1000 (frozen config),
  matching the released numbers exactly.
* Gemma-2's chat template (no system role) is handled by the repo's `build_chat_text`
  (folds system into user). Sharded loading is unused (all judges fit on one T4 in 4-bit).

## Assumptions / decisions (also printed at runtime)

1. **Placebo needs `s2`.** This is not literally pure inference — placebo pairs require the
   second seeded sample, which the foil families lack. It is generated additively via the
   identical seeded path; frozen `s1` foil text is never touched.
2. **Nothing new to pin.** All three new judges are promoted foils, so they reuse the
   revisions already pinned in the frozen `configs/models.yaml` (bootstrap fails loud on
   drift); `run_report.txt` echoes the revisions used.
3. **Dissociation pooling.** `dissociation_summary` pools implicit and explicit over the
   **same** LLM-foil universe (human excluded) so the gap is apples-to-apples; the
   `_allfoils` column additionally reproduces the paper's prose headline. Per-cell values
   (incl. human) live in `extended_table1` and match the paper exactly.
4. **margin-ρ is a companion, not a claim of ρ≈0.** In Qwen it is small and *rises* with
   scale (≈−0.02 → +0.34); the dissociation is carried by the accuracy gap, which does not
   saturate. Reported honestly per judge with a CI.

## Limitations / scope honesty (carried into the outputs)

* Three new families is a **targeted generalization of the dissociation**, not a universal
  law, and it does **not** extend the scale-trend claim (which stays Qwen-only). Scope claims
  to *open instruction-tuned models*; every accuracy ships with a CI so the reader judges
  consistency, not just significance.
* The paraphrase mechanism result (implicit signal is surface-lexical) stays **Qwen-only** —
  it is not claimed to generalize.
* Human-foil cells are smaller (length-matching removes verbose-model-vs-concise-human
  pairs), so their CIs are wider; the per-cell n and a power note are in `run_report.txt`.

## Files added by this work

```
kaggle_extend_families.ipynb              # the runnable notebook (built)
EXTENSION_README.md                       # this file
tools/build_extension_notebook.py         # regenerates the .ipynb from cell sources
tools/extension_notebook_src/*.{md,py}    # the cell sources (edit THESE, then rebuild)
```
Rebuild after editing a cell source: `python tools/build_extension_notebook.py`.
