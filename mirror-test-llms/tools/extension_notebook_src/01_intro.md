# Mirror Test — cross-family extension (Gemma-2, Mistral, Llama)

**Build `__BUILD_STAMP__`.** One notebook, runnable end-to-end on **Kaggle GPU T4 ×2**,
that extends the single-family (Qwen2.5) study's **implicit/explicit dissociation** to more
open training recipes **without touching any published number**. Every new measurement
reuses the repository's own functions, config, seeds, and templates, so new cells are
directly comparable to the released ones.

## Why this exists
The paper's central dissociation — *near-perfect implicit self-information (a
lower-perplexity rule) coexisting with chance-level explicit verbal self-recognition* — is
currently shown on one family (Qwen2.5). A reviewer's first move is "that's a Qwen case
study, not a general result." This notebook answers that directly:

* **Generalize the DISSOCIATION** (≈ pure inference). Promote three existing *foil*
  families to *judges* at their current size — `gemma-2-9b-it`, `mistral-7b-instruct-v0.3`,
  `llama-3.2-3b-instruct`. Their generations already exist, so the only new generation is
  each one's **placebo second sample** (`s2`). We then run them as PPP + IPP judges, compute
  per-token NLL for the perplexity rule, and add placebo + stylometry. Target: *near-perfect
  implicit self-info, chance-level explicit choice, κ≈0* replicates on **4 recipes** (Qwen +
  these three).

Every judge is ≤ 9B and loads on a **single T4 in 4-bit** — no sharding, no large downloads.

**Scope, honestly:** this extends the *dissociation* across recipes. The **scale-trend**
result stays **Qwen-only**, exactly as in the paper (we do not add a second-family size
sweep). The paraphrase attack and base-vs-instruct ablation also stay Qwen-only. Compute
goes entirely to judging, perplexity, and stylometry for the three new judges.

## Common foil pool (comparability)
Pool = `{llama-3.2-3b-instruct, gemma-2-9b-it, mistral-7b-instruct-v0.3, human}`. Each
judge's foils = the pool families that are **not its own family**, plus `human`. All foil
text already exists — **foils never need new generation.**

| Judge | Foils |
|---|---|
| gemma-2-9b-it | llama-3.2-3b-instruct, mistral-7b-instruct-v0.3, human |
| mistral-7b-instruct-v0.3 | llama-3.2-3b-instruct, gemma-2-9b-it, human |
| llama-3.2-3b-instruct | gemma-2-9b-it, mistral-7b-instruct-v0.3, human |
| qwen2.5-* (unchanged, reused) | llama, gemma-2-9b, mistral, human |

## Kaggle setup — just upload this one file and run
No dataset to attach and nothing to unzip: the notebook **clones the repo itself** (code +
frozen data + published results) and writes everything to `/kaggle/working/`.

1. **Push the repo first** (public) to the URL in the config cell's `REPO_GIT_URL` — the
   notebook clones it. (Private repo? add a `GITHUB_TOKEN` secret.)
2. New Kaggle notebook → **Accelerator = GPU T4 ×2**, **Internet = ON**.
3. **Add secret `HF_TOKEN`** (Add-ons → Secrets), and with that same Hugging Face account
   **accept the licenses** for `google/gemma-2-9b-it` and `meta-llama/Llama-3.2-3B-Instruct`
   (Mistral-v0.3 is not gated) — otherwise the gated downloads 403.
4. **Upload `kaggle_extend_families.ipynb`** and **Run All**. That's it.

The notebook is **checkpointed and resumable**: rerun in a fresh session and it skips every
unit already complete. Three ≤9B judges fit comfortably in one ~12 h session, but if a
session ends early just rerun — completed units are skipped. (You can also attach the repo as
a Dataset instead of cloning — it is auto-detected — but nothing requires it.)

## Outputs (to `/kaggle/working/`)
* `extended_table1.csv` — per (judge, foil): n, PPP acc + Wilson CI, AUROC + CI, stylometry
  acc, perplexity-rule acc, κ, Holm-adjusted significance (recomputed over **all** cells:
  the 5 Qwen judges + the 3 new judges).
* `dissociation_summary.csv` — per judge: implicit (PPL-rule) vs explicit (PPP) accuracy
  (both pooled over the same LLM-foil universe), the **implicit−explicit accuracy gap**
  (the primary, non-saturating dissociation evidence), κ, and the margin–margin Spearman ρ
  (a companion to κ that does not collapse under marginal saturation).
* `dissociation.png` — per judge, a dumbbell of implicit (perplexity-rule) vs explicit
  (verbal PPP) accuracy, colored by family, so the gap is visibly the same across all four
  recipes.
* `extended_outputs/raw/…` — raw per-cell judgments (JSONL + parquet if available).
* `run_report.txt` — timings, GPU-hours, reused vs generated, skipped/failed, model
  revisions used, Holm family size, per-cell n + power note, library versions.

> **Scope honesty (kept in the outputs).** Three new families is a targeted generalization
> of the *dissociation*, not a universal law, and it does not touch the scale-trend claim
> (which stays Qwen-only). Claims are scoped to *open instruction-tuned models*; every
> accuracy ships with a CI so the reader judges consistency, not just significance.
