# Pipeline Walkthrough — every script, every file, every command

This is the operating manual. For each stage: what it does, where it runs,
the exact command, what comes out, and how long it takes. Data schemas are
at the bottom — every JSONL field defined once.

**The golden rule of every GPU script here:** output is appended line by
line and synced to disk, and re-running the same command skips finished
items. A Colab disconnect costs you nothing. Never babysit a run.

---

## Stage 0 — smoke test (week 3; do this before anything real)

Prove the whole pipe works end-to-end at toy scale, on the smallest model:

```bash
pytest tests/ -q                                          # CPU, 1 second
python src/00_build_prompts.py --n 8                      # tiny prompt sets
python src/01_generate.py --models qwen2.5-0.5b-instruct \
       --placebo --max-prompts 8                          # GPU (or slow CPU)
python src/01_generate.py --models llama-3.2-3b-instruct --max-prompts 8
python src/02_build_pairs.py --judges qwen2.5-0.5b-instruct
python src/03_judge_ppp.py --judge qwen2.5-0.5b-instruct --limit 8 --include-placebo
python src/04_judge_ipp.py --judge qwen2.5-0.5b-instruct --limit 20
python src/05_baselines.py stylometric --judge qwen2.5-0.5b-instruct
python src/05_baselines.py perplexity --judge qwen2.5-0.5b-instruct --limit 8
python src/06_stats.py
```

If tables appear in `results/tables/`, the pipeline is healthy. **Then
delete the toy data** (`data/prompts/*.jsonl`, `data/generations/*`,
`data/pairs/*`, `results/judgments/*`, `results/baselines/*`,
`results/tables/*`) and rebuild prompts at full size for the real freeze.

## Stage 1 — freeze prompts (week 3, CPU + internet, ~10 min)

```bash
python src/00_build_prompts.py
git add data/prompts && git commit -m "Freeze prompt sets (seed 42)"
```

* Downloads CNN/DailyMail, Dolly-15k, WritingPrompts from Hugging Face.
* Keeps the first 200 items per domain passing the §6 filters (fixed seed).
* Writes `data/prompts/{news,dolly,wp}.jsonl` + SHA-256 `CHECKSUMS.txt` +
  a rejection report.
* **Refuses to overwrite existing files** without `--force` — the freeze is
  the credibility of the whole study. Commit before generating anything.

## Stage 2 — generation (week 4, GPU, ~4–8 h total, split freely)

Every model answers every prompt once (plus a second sample for judges,
which becomes the placebo control). One model per session is a good rhythm:

```bash
# judges (the --placebo flag adds the second seed for SELF-vs-SELF pairs)
python src/01_generate.py --models qwen2.5-0.5b-instruct --placebo
python src/01_generate.py --models qwen2.5-1.5b-instruct --placebo
python src/01_generate.py --models qwen2.5-3b-instruct   --placebo
python src/01_generate.py --models qwen2.5-7b-instruct   --placebo
python src/01_generate.py --models qwen2.5-14b-instruct  --placebo   # Kaggle 2xT4

# foils (no placebo needed)
python src/01_generate.py --models llama-3.2-3b-instruct
python src/01_generate.py --models gemma-2-9b-it
python src/01_generate.py --models mistral-7b-instruct-v0.3
```

Output: `data/generations/{model}__{domain}.jsonl` with the RAW text
(cleaning happens in stage 3, so it stays auditable). Times on a T4,
4-bit, 600 prompts + 450 placebo samples: 0.5B ≈ 20–30 min … 7B ≈ 1.5–2 h.

## Stage 3 — cleaning + pair building (week 5, CPU, ~2 min)

```bash
python src/02_build_pairs.py
```

Applies the §7-step-2 cleaner identically to every author (incl. HUMAN),
then builds:

* `data/pairs/ppp__{judge}__{foil}__{domain}.jsonl` — the mirror-test pairs
  (length-matched ≤25%, ROUGE-L ≤ 0.7 dedup). Watch the console: any cell
  `BELOW TARGET 150` gets reported in the paper with its exact n.
* `data/pairs/placebo__{judge}__{domain}.jsonl` — SELF vs SELF.
* `data/pairs/ipp__{judge}.jsonl` — 300 balanced Yes/No items.
* `data/pairs/pairs_report.json` — all counts + cleaning stats (appendix!).

## Stage 4 — paraphrase materials (week 8 prep, GPU, ~1 h)

```bash
python src/02b_paraphrase.py            # uses config: largest judge vs F1
# if you dropped the 14B judge:  --judge qwen2.5-7b-instruct
```

Phi-3.5-mini rewrites both texts of each pair; MiniLM cosine ≥ 0.80 +
length gate decides which rewrites count. Output:
`data/pairs/para__{judge}__{foil}__{domain}.jsonl`.
**If you change the paraphrase judge here, change `paraphrase.judge` in the
config too** — `06_stats.py` reads it to find the cells.

## Stage 5 — judging (weeks 5–6, GPU, the core measurement)

```bash
# full grid, one judge per session (phrasing 0):
python src/03_judge_ppp.py --judge qwen2.5-0.5b-instruct --include-placebo
python src/03_judge_ppp.py --judge qwen2.5-1.5b-instruct --include-placebo
python src/03_judge_ppp.py --judge qwen2.5-3b-instruct   --include-placebo
python src/03_judge_ppp.py --judge qwen2.5-7b-instruct   --include-placebo
python src/03_judge_ppp.py --judge qwen2.5-14b-instruct  --include-placebo

# the MAIN CELL additionally runs all 3 instruction phrasings + paraphrase:
python src/03_judge_ppp.py --judge qwen2.5-14b-instruct \
    --foils llama-3.2-3b-instruct --phrasings 0 1 2 --include-paraphrase

# IPP for every judge (~10 min each):
python src/04_judge_ipp.py --judge qwen2.5-0.5b-instruct
# ... repeat per judge

# base-vs-instruct ablation (§9.5):
python src/03_judge_ppp.py --judge qwen2.5-7b-base
python src/03_judge_ppp.py --judge qwen2.5-14b-base
```

Output: `results/judgments/ppp__{judge}.jsonl` and `ipp__{judge}.jsonl`.
Each PPP pair costs 2 forward passes (both orders); expect ~30–90 min per
judge for its full grid, scaling with model size.

## Stage 6 — baselines (week 7)

```bash
python src/05_baselines.py stylometric                        # CPU, ~5 min
python src/05_baselines.py perplexity --judge qwen2.5-0.5b-instruct   # GPU
# ... perplexity for every judge; add --include-paraphrase on the para judge
python src/05_baselines.py perplexity --judge qwen2.5-7b-base          # §9.5 proxy
```

Output: `results/baselines/stylo__*.json`, `stylo_pairs__*.jsonl`,
`transfer__*.json`, `ppl__{judge}.jsonl`.

## Stage 7 — statistics + figures (week 8, CPU, ~1 min)

```bash
python src/06_stats.py
```

Everything lands in `results/tables/` (CSV + Markdown + `table1_main.tex`)
and `results/figures/` (PNG for reading + PDF for LaTeX). Read
`main_ppp.md` first, then `placebo.csv` (should be ≈50%! if not, stop and
investigate before writing anything), then `paraphrase.csv`, `ipp.csv`.

---

## Data schemas (single source of truth)

**data/prompts/{domain}.jsonl**
```json
{"prompt_id": "news_0007", "prompt_idx": 7, "domain": "news",
 "task_prompt": "Summarize the following article in 3-4 sentences: ...",
 "human_reference": "...", "source_dataset": "abisee/cnn_dailymail",
 "source_id": "...", "n_words_prompt": 123, "n_words_reference": 58}
```

**data/generations/{model}__{domain}.jsonl**
```json
{"gen_id": "news_0042__qwen2.5-7b-instruct__s1", "prompt_id": "news_0042",
 "prompt_idx": 42, "domain": "news", "model": "qwen2.5-7b-instruct",
 "hf_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "<commit>", "sample": "s1",
 "seed": 1042, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 160,
 "text": "<RAW model output>", "n_words": 97, "created_at": "..."}
```
`sample` is `s1` (main) or `s2` (placebo twin, judges only).

**data/pairs/ppp__{judge}__{foil}__{domain}.jsonl**
```json
{"pair_id": "qwen2.5-7b-instruct__human__news__news_0042",
 "judge": "...", "foil": "...", "domain": "news", "prompt_id": "news_0042",
 "task_prompt": "...", "text_self": "<cleaned>", "text_foil": "<cleaned>",
 "n_words_self": 96, "n_words_foil": 88, "rouge_l": 0.31}
```
`placebo__*` adds `"is_placebo": true` (text_self = s1, text_foil = s2).
`para__*` adds `condition/orig_text_*/cos_*/passed_gate`.

**data/pairs/ipp__{judge}.jsonl**
```json
{"item_id": "ipp__<judge>__<author>__<domain>__<prompt_id>", "judge": "...",
 "domain": "...", "prompt_id": "...", "task_prompt": "...", "text": "...",
 "author": "self" , "is_self": true}
```

**results/judgments/ppp__{judge}.jsonl** — one record per (pair, order, phrasing)
```json
{"run_id": "<pair_id>__self_A__ph0", "pair_id": "...", "judge": "...",
 "pairs_of": null, "foil": "...", "domain": "...", "prompt_id": "...",
 "condition": "core", "phrasing": 0, "order": "self_A", "self_position": "A",
 "logp_A": -0.31, "logp_B": -1.42, "choice": "A", "correct": true,
 "freetext": "A", "freetext_status": "ok", "template": "chat",
 "created_at": "..."}
```
`condition` ∈ core | placebo | paraphrase. `correct` is null for placebo.
`pairs_of` is set for base judges (whose pairs they judged).

**results/judgments/ipp__{judge}.jsonl**
```json
{"item_id": "...", "judge": "...", "author": "self", "is_self": true,
 "domain": "...", "prompt_id": "...", "logp_yes": -0.4, "logp_no": -1.6,
 "choice": "Yes", "correct": true, "confidence": null,
 "freetext": null, "freetext_status": null, "created_at": "..."}
```

**results/baselines/ppl__{judge}.jsonl**
```json
{"row_id": "<pair_id>__core", "pair_id": "...", "judge": "...",
 "foil": "...", "domain": "...", "condition": "core",
 "nll_self": 2.31, "nll_foil": 2.87, "n_tok_self": 118, "n_tok_foil": 121,
 "rule_chose_self": true, "created_at": "..."}
```

**results/baselines/stylo__{judge}__{foil}__{domain}.json**
```json
{"judge": "...", "foil": "...", "domain": "...", "n_pairs": 172,
 "n_texts": 344, "acc_singletext_char_ngram": 0.83,
 "acc_singletext_surface": 0.64, "acc_pairwise_char_ngram": 0.87}
```

## Command cheat sheet for a single Colab session

```bash
!git clone https://github.com/<you>/mirror-test-llms && cd mirror-test-llms
!pip -q install -U transformers accelerate bitsandbytes datasets sentence-transformers pyyaml
from huggingface_hub import login; login()      # paste your token
!cd mirror-test-llms && python src/01_generate.py --models qwen2.5-3b-instruct --placebo
# ... commit results back, or copy to Drive (see 04_colab_kaggle_guide.md)
```
