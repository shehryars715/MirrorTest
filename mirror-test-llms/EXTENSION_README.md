# Cross-family extension — `kaggle_extend_families.ipynb`

A single, resumable Kaggle notebook (**GPU T4 ×2**) that extends the single-family
(Qwen2.5) mirror-test study to more open training recipes **without changing any
published number**. It exists to answer the reviewer question the paper invites: *is the
implicit/explicit dissociation and the scale trend a Qwen artifact, or does it generalize?*

It reuses the repository's own scripts and functions verbatim, so every new cell is
directly comparable to the released ones. The extended table was checked to reproduce
`results/tables/main_ppp.csv` **exactly to 4 decimals** on the Qwen cells (acc, Wilson CI,
AUROC, stylometry, perplexity-rule acc, κ, Holm p and reject flag), and the margin–margin
ρ reproduces `tools/margin_correlation.py` exactly.

## What it does

* **Tier 1 — generalize the DISSOCIATION** (≈ pure inference). Promote three existing
  *foil* families to *judges*: `gemma-2-9b-it`, `mistral-7b-instruct-v0.3`,
  `llama-3.2-3b-instruct`. Their generations already exist; the only new generation is
  each one's **placebo second sample (`s2`)**. Then PPP + IPP judging, per-token NLL for
  the perplexity rule, placebo, and stylometry. → the dissociation replicates on **4
  recipes** (Qwen + these three).
* **Tier 2 — generalize the SCALE CURVE**. Full Gemma-2 size sweep — `gemma-2-2b-it`
  (generate), `gemma-2-9b-it` (reuse), `gemma-2-27b-it` (generate; sharded across both
  T4s) — judged and plotted beside Qwen's curve.
* **Not re-run** (already established on Qwen, per instruction): the paraphrase attack and
  the base-vs-instruct ablation.

### Common foil pool (family exclusion)
Pool = `{llama-3.2-3b-instruct, gemma-2-9b-it, mistral-7b-instruct-v0.3, human}`; each
judge's foils = the pool families **not in its own family**, plus `human`. All foil text
pre-exists — foils are never regenerated.

| Judge | Foils |
|---|---|
| gemma-2-{2b,9b,27b}-it | llama-3.2-3b-instruct, mistral-7b-instruct-v0.3, human |
| mistral-7b-instruct-v0.3 | llama-3.2-3b-instruct, gemma-2-9b-it, human |
| llama-3.2-3b-instruct | gemma-2-9b-it, mistral-7b-instruct-v0.3, human |
| qwen2.5-* (reused, unchanged) | llama, gemma-2-9b, mistral, human |

## Prerequisites (once)

Upload-and-run, like `kaggle_run_all.ipynb`: the notebook **clones the repo itself**, so
there is no Dataset to attach and nothing to unzip.

1. **Push the repo (public)** to the URL in the config cell's `REPO_GIT_URL`
   (default `https://github.com/shehryars715/MirrorTest.git`); a private repo just needs a
   `GITHUB_TOKEN` secret. The push must include the `src/utils.py` `MIRROR_MAX_MEMORY`
   sharding path — the 27B judge needs it.
2. On Kaggle: **Accelerator = GPU T4 ×2**, **Internet = ON** (already required for the model
   weights, so the clone adds no new dependency).
3. **Add secret `HF_TOKEN`** (Add-ons → Secrets), and with that same HF account **accept
   the licenses** for `google/gemma-2-2b-it`, `google/gemma-2-9b-it`,
   `google/gemma-2-27b-it`, `meta-llama/Llama-3.2-3B-Instruct`.

## How to run

**Upload `kaggle_extend_families.ipynb` and press Run All.** The `02_config` cell has every
knob (`REPO_GIT_URL`/`REPO_GIT_REF`, budget hours, `ENABLE_27B`, foil pool, revisions). The
bootstrap clones the repo, redirects all I/O to `/kaggle/working/`, verifies the frozen
inputs (checksums), and never regenerates them. (If you prefer, attach the repo as a
Dataset instead — it is auto-detected and used without cloning.)

**Resumable across sessions.** Each pipeline stage is internally resumable (skips finished
items) and the orchestrator checkpoints to `/kaggle/working/checkpoints/`. Tier 1 + Tier 2
may exceed one 12 h session; 27B is scheduled **last** so the rest ships even if it slips.
To continue: *Save Version*, start a fresh session, attach **this run's saved output** as a
Dataset, and Run All — the repo re-clones automatically and completed units are skipped. Set
`ENABLE_27B = False` to cap the Gemma-2 curve at 2B/9B and guarantee single-session
completion (the protocol, §18, sanctions a truncated scale axis).

The HF weight cache is placed on `/kaggle/temp` (large ephemeral scratch), **not**
`/kaggle/working` — the latter has a ~20 GB persisted cap and gemma-2-27b-it alone is ~54 GB.

## Outputs (`/kaggle/working/`)

| File | What it is |
|---|---|
| `extended_table1.csv` | Per (judge, foil): n, PPP acc + Wilson CI, AUROC + CI, stylometry acc, perplexity-rule acc, κ + CI, exact-binomial p, **Holm p + reject recomputed over ALL cells present** (not the old 20). All instruct judges (base ablation excluded, as in the paper's Table 1). |
| `dissociation_summary.csv` | Per judge: implicit (PPL-rule) vs explicit (PPP) accuracy pooled over the **same** LLM-foil universe, the **implicit−explicit accuracy gap** (primary dissociation evidence), an `_allfoils` implicit column (reproduces the paper's prose headline), κ (median/abs-max), the **margin–margin Spearman ρ + CI** (non-saturating companion to κ), and human-vs-machine AUROC + CI. |
| `scale_curves.png` | Qwen2.5 vs Gemma-2 for PPP accuracy, PPL-rule accuracy, and human-vs-machine AUROC, vs log judge size (CI whiskers, chance line). |
| `extended_outputs/raw/` | Raw per-(judge,foil) PPP judgments (JSONL + parquet if pandas available) plus each new judge's `ppl__*` and `ipp__*`. |
| `run_report.txt` | Timings, GPU-hours, reused-vs-generated counts, **resolved revisions to pin**, 27B sharded pre-flight result, Holm family size, per-cell n + power note, library versions. |

The new raw pipeline artifacts also land in the usual repo layout under
`/kaggle/working/results/` and `/kaggle/working/data/` so `src/06_stats.py` can be re-run
on the union if desired.

## Methodological identity — how comparability is guaranteed

* New generations use `src/01_generate.py` unchanged (same decoding: temp 0.8, top-p 0.95,
  ≤160 tokens; seeds `1000+idx` / placebo `2000+idx`; pinned revisions).
* Pairs, judging, IPP, perplexity, stylometry run the published `src/02–05` scripts via a
  patched config that only **adds** the new judges (`foils: [human]` so the promoted
  families keep `is_judge=True` for their `s2` sample; foil text is supplied explicitly and
  read off disk).
* Statistics reuse `src/06_stats.py` and `src/stats_utils.py` functions unchanged
  (`cell_stats`, `judge_vs_ppl_kappa`, `holm_bonferroni`, `auroc`, `wilson_ci`,
  `bootstrap_ci`). AUROC CI = 500 resamples (repo default) and κ CI = 1000 (frozen config),
  matching the released numbers exactly.
* Gemma-2's chat template (no system role) is handled by the repo's `build_chat_text`
  (folds system into user); the 27B sharded path is verified with a generation + per-token
  NLL sanity check before the heavy stages.

## Assumptions / decisions (also printed at runtime)

1. **Placebo needs `s2`.** Tier 1 is not literally pure inference — placebo pairs require
   the second seeded sample, which the foil families lack. It is generated additively via
   the identical seeded path; frozen `s1` foil text is never touched.
2. **Two revisions are not yet pinned** (`gemma-2-2b-it`, `gemma-2-27b-it`). They load
   `latest`; the **resolved commit is recorded in `run_report.txt`** — pin it in
   `configs/models.yaml` for camera-ready (as the repo did for its other models).
3. **Dissociation pooling.** `dissociation_summary` pools implicit and explicit over the
   **same** LLM-foil universe (human excluded) so the gap is apples-to-apples; the
   `_allfoils` column additionally reproduces the paper's prose headline. Per-cell values
   (incl. human) live in `extended_table1` and match the paper exactly.
4. **margin-ρ is a companion, not a claim of ρ≈0.** In Qwen it is small and *rises* with
   scale (≈−0.02 → +0.34); the dissociation is carried by the accuracy gap, which does not
   saturate. Reported honestly per judge with a CI.
5. **27B risk** (≈54 GB download + slow sharded forward). Isolated final stage with
   graceful fallback: on OOM/disk/time failure the Gemma-2 curve still ships at 2B/9B and
   `run_report.txt` records it skipped.

## Limitations / scope honesty (carried into the outputs)

* Two new families + a one-family scale sweep is a **targeted generalization**, not a
  universal law. Scope claims to *open instruction-tuned models*; every accuracy ships with
  a CI so the reader judges consistency, not just significance.
* The paraphrase mechanism result (implicit signal is surface-lexical) stays **Qwen-only** —
  it is not claimed to generalize.
* Human-foil cells are smaller (length-matching removes verbose-model-vs-concise-human
  pairs), so their CIs are wider; the per-cell n and a power note are in `run_report.txt`.

## Files added by this work

```
kaggle_extend_families.ipynb              # the runnable notebook (built)
../kaggle_extend_families.ipynb           # convenience copy to upload
EXTENSION_README.md                       # this file
tools/build_extension_notebook.py         # regenerates the .ipynb from cell sources
tools/extension_notebook_src/*.{md,py}    # the cell sources (edit THESE, then rebuild)
```
Rebuild after editing a cell source: `python tools/build_extension_notebook.py`.
