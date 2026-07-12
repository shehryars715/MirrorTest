# Troubleshooting & FAQ — when things break (they will; it's normal)

Organized by symptom. If your problem isn't here, read the error message
bottom-up (the last line is usually the real cause), and search it verbatim.

---

## Installation / environment

**`bitsandbytes` fails to install or import on my Windows laptop.**
Expected. 4-bit quantization needs an NVIDIA GPU and works best on
Linux/Colab. You don't need it locally: every GPU script is meant for
Colab/Kaggle, and `utils.load_model_and_tokenizer` falls back to
fp16/fp32 automatically when bitsandbytes is missing. Locally you only run
the CPU stages (00, 02, 05-stylometric, 06, tests).

**`ModuleNotFoundError: yaml`** → `pip install pyyaml`. Same pattern for
`datasets`, `sklearn` (`pip install scikit-learn`), `matplotlib`.

**Colab: `NVIDIA-SMI has failed` / torch says no CUDA.**
Runtime → Change runtime type → T4 GPU, then restart the runtime and re-run
your install cell.

## Hugging Face access

**`401 Unauthorized` / `GatedRepoError` / "awaiting a review of your request"
when loading Llama or Gemma.**
You must (1) accept the license on the model page while logged in, (2)
authenticate in the session: `from huggingface_hub import login; login()`.
If you requested access seconds ago, wait — Meta approvals can take minutes
to hours. Until then, run the non-gated models (Qwen, Mistral, Phi).

**`RepositoryNotFoundError` for a dataset/model.**
The hub id changed (it happens). Search the hub for the dataset name and
update `configs/models.yaml`. Record the change in PREREGISTRATION.md's
deviations log if it's after your freeze.

**XSum won't load (`trust_remote_code` / script errors).**
Known issue with new `datasets` versions — one reason this repo defaults to
CNN/DailyMail for news. If you insist on XSum, you'll need an older
`datasets` or a parquet mirror; easier to stay with the default.

## GPU runs

**`CUDA out of memory`.**
In order of preference: (1) you're on the 14B → use Kaggle 2×T4;
(2) another model is still in memory from the same session → Runtime →
Restart, run only one model per session; (3) drop the 14B row and cap the
scale axis at 7B — explicitly allowed (protocol §18).

**Colab disconnected mid-run. Did I lose everything?**
No. Every script writes each finished item immediately. Reconnect, re-run
the SAME command, and it resumes (watch for "already done" counts in the
log). This is by design — see `utils.append_jsonl` / `existing_ids`.

**Generation is absurdly slow.**
Check `!nvidia-smi` — if the process shows no GPU memory, you're on CPU
(wrong runtime, or torch installed without CUDA). Also: first-ever run
includes the model download (up to ~9 GB for 14B) — that's bandwidth, not
compute.

**`ValueError: No single-token spelling among [...]`** (from
`candidate_token_ids`).
The judge's tokenizer splits "A"/" A" into multiple tokens — extremely rare
with the configured models. If you swapped in an exotic model, add a
single-token variant of the answer word for that tokenizer to
`configs/models.yaml → judging`.

**Gemma crashes with "System role not supported".**
Handled: `build_chat_text` folds the system text into the user message and
logs a warning. If you see the crash anyway, you're calling the tokenizer
directly somewhere — use the helper.

## Data / results look wrong

**Placebo rate is far from 50% (e.g. 65%).**
This is the control doing its job — do NOT proceed to interpretation.
It means position bias (fine — counterbalancing absorbs it: check that both
orders exist per pair in the judgments) or a pipeline bug (bad: check that
`text_self`/`text_foil` aren't swapped anywhere, and that cleaning didn't
leave an artifact in one sample stream). The per-run records contain
everything needed to audit an individual pair by hand — actually read a few.

**Stylometric baseline hits ~99%+ on real data.**
Boilerplate is leaking authorship (protocol §18). Look at 20 random cleaned
texts per author (`data/pairs/...`); you will spot the tell (a signature
opener, markdown habits, "As an AI..."). Extend the cleaning patterns in
`utils.py` (they're unit-tested — add a test first), re-run 02 onward, and
report the change in the deviations log.

**A (judge, foil, domain) cell has far fewer than 150 pairs.**
Check `data/pairs/pairs_report.json`: if `len_mismatch` dominates, the two
models write very different lengths for that domain (common with Gemma).
Options: accept the smaller n (report it — the stats handle it), or raise
`max_new_tokens` is NOT an option after freezing. Never loosen the filter
for one cell only.

**`06_stats.py` says "no judgments found".**
Results live in `results/judgments/ppp__*.jsonl` — did 03 actually run on
this machine, or are the files on Drive/another clone? Pull them first.

**Figures look empty / one line missing.**
A foil line appears only if that judge×foil cell has judgments. Check
`results/tables/main_ppp.csv` for which cells exist.

## Statistics questions

**Why is my accuracy 0.74 but the binomial test not significant?**
Small decisive-item count: if most items are ties (score 0.5 — judge picked
the same POSITION both times), accuracy is inflated by ties but evidence is
thin. Look at the `consistency` column — low consistency is itself a finding
about the fragility of the behaviour (protocol §8.3).

**κ is negative — bug?**
No: negative κ = the judge and the perplexity rule agree LESS than chance
given their base rates. Small |κ| either sign ≈ "unrelated decision rules."

**Wilson CI crosses 0.5 but p < .05 (or vice versa).**
They answer slightly different questions (CI on the tie-inclusive mean vs
test on decisive items). Report both; if they disagree, say the evidence is
borderline — that's the honest reading.

## Process

**I changed something after the pre-registration commit. Is the study dead?**
No — log it in PREREGISTRATION.md §6 (date + what + why) and disclose it in
the paper. Undisclosed changes are the sin, not changes.

**How do I know I'm not fooling myself somewhere?**
The pipeline's built-in tripwires: placebo ≈ 50%, cleaning identical for all
authors, counterbalanced orders, frozen prompts with checksums, decisive-item
testing, Holm correction, AUROC alongside accuracy. If all of those are
green and reported, you are doing better than most published work.
