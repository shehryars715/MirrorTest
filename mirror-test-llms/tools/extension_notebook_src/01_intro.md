# Mirror Test — cross-family extension (Gemma-2, Mistral, Llama)

**Build `__BUILD_STAMP__`.** One notebook, runnable end-to-end on **Kaggle GPU T4 ×2**,
that extends the single-family (Qwen2.5) study to more open training recipes **without
touching any published number**. Every new measurement reuses the repository's own
functions, config, seeds, and templates, so new cells are directly comparable to the
released ones.

## Why this exists
The paper's central claims — the **implicit/explicit dissociation** and the **scale
trend** — are currently supported on one family (Qwen2.5). A reviewer's first move is
"that's a Qwen case study, not a general result." This notebook answers that:

* **Tier 1 — generalize the DISSOCIATION** (≈ pure inference). Promote three existing
  *foil* families to *judges* at their current size — `gemma-2-9b-it`,
  `mistral-7b-instruct-v0.3`, `llama-3.2-3b-instruct`. Their generations already exist,
  so the only new generation is each one's **placebo second sample** (`s2`). We then run
  them as PPP + IPP judges, compute per-token NLL for the perplexity rule, and add
  placebo + stylometry. Target: *near-perfect implicit self-info, chance-level explicit
  choice, κ≈0* replicates on **4 recipes** (Qwen + these three).
* **Tier 2 — generalize the SCALE CURVE** (moderate new generation). A full Gemma-2 size
  sweep — `gemma-2-2b-it` (generate), `gemma-2-9b-it` (reuse from Tier 1),
  `gemma-2-27b-it` (generate; sharded across both T4s) — judged and plotted **beside**
  Qwen's curve.

We deliberately **do not** re-run the paraphrase attack or the base-vs-instruct
ablation (already established on Qwen); compute goes to judging, perplexity, and
stylometry for the new judges.

## Common foil pool (comparability)
Pool = `{llama-3.2-3b-instruct, gemma-2-9b-it, mistral-7b-instruct-v0.3, human}`. Each
judge's foils = the pool families that are **not its own family**, plus `human`. All foil
text already exists — **foils never need new generation.**

| Judge | Foils |
|---|---|
| gemma-2-{2b,9b,27b}-it | llama-3.2-3b-instruct, mistral-7b-instruct-v0.3, human |
| mistral-7b-instruct-v0.3 | llama-3.2-3b-instruct, gemma-2-9b-it, human |
| llama-3.2-3b-instruct | gemma-2-9b-it, mistral-7b-instruct-v0.3, human |
| qwen2.5-* (unchanged, reused) | llama, gemma-2-9b, mistral, human |

## Kaggle setup — just upload this one file and run
No dataset to attach and nothing to unzip: the notebook **clones the repo itself** (code +
frozen data + published results) and writes everything to `/kaggle/working/`.

1. **Push the repo first** (public) to the URL in the config cell's `REPO_GIT_URL` — the
   notebook clones it. (Private repo? add a `GITHUB_TOKEN` secret.) Make sure the push
   includes the `src/utils.py` `MIRROR_MAX_MEMORY` sharding path — the 27B judge needs it.
2. New Kaggle notebook → **Accelerator = GPU T4 ×2**, **Internet = ON**.
3. **Add secret `HF_TOKEN`** (Add-ons → Secrets), and with that same Hugging Face account
   **accept the licenses** for `google/gemma-2-2b-it`, `google/gemma-2-9b-it`,
   `google/gemma-2-27b-it`, `meta-llama/Llama-3.2-3B-Instruct` — otherwise the gated
   downloads 403.
4. **Upload `kaggle_extend_families.ipynb`** and **Run All**. That's it.

The notebook is **checkpointed and resumable**: rerun in a fresh session and it skips every
unit already complete. Tier 1 + Tier 2 may span more than one 12 h session; 27B is scheduled
**last** so the rest ships even if it slips. (You can still attach the repo as a Dataset
instead of cloning — it is auto-detected — but nothing requires it.)

## Outputs (to `/kaggle/working/`)
* `extended_table1.csv` — per (judge, foil): n, PPP acc + Wilson CI, AUROC + CI, stylometry
  acc, perplexity-rule acc, κ, Holm-adjusted significance (recomputed over **all** cells).
* `dissociation_summary.csv` — per judge: implicit (PPL-rule) vs explicit (PPP) accuracy
  (both pooled over the same LLM-foil universe), the **implicit−explicit accuracy gap**
  (the primary, non-saturating dissociation evidence), κ, and the margin–margin Spearman ρ
  (a companion to κ that does not collapse under marginal saturation).
* `scale_curves.png` — Qwen2.5 vs Gemma-2 for PPP accuracy, PPL-rule accuracy, and
  human-vs-machine AUROC, vs log judge size.
* `extended_outputs/raw/…` — raw per-cell judgments (JSONL + parquet if available).
* `run_report.txt` — timings, GPU-hours, reused vs generated, skipped/failed, resolved
  model revisions, Holm family size, per-cell n + power note, library versions.

> **Scope honesty (kept in the outputs).** Two new families + a one-family scale sweep is
> a targeted generalization, not a universal law: claims are scoped to *open
> instruction-tuned models*, the paraphrase mechanism result stays Qwen-only, and every
> accuracy ships with a CI so the reader judges consistency, not just significance.
