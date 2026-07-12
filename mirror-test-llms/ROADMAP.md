# ROADMAP — every step from today to a submitted paper

This is the single operational checklist. Each task says **where it runs**
(💻 laptop / ☁️ Colab / 🟦 Kaggle), the exact command or file, and what
"done" looks like. Tick the boxes in order; phases match the protocol's
12-week plan (§16). If a week slips, cut Phase 8 (optional extensions),
never Phases 4–7 (controls).

Legend: 💻 = your Windows laptop (CPU) · ☁️ = Colab T4 · 🟦 = Kaggle 2×T4

---

## Phase 0 — Setup (today, ~2 hours, 💻)

- [ ] Read `docs/00_START_HERE.md`, then skim `README.md`.
- [ ] Install the CPU toolchain:
      `pip install pyyaml datasets scikit-learn matplotlib pytest`
- [ ] Verify: `pytest tests/ -q` → **27 passed**.
- [ ] Create a GitHub account (if needed) and an empty **public** repo named
      `mirror-test-llms`; then, inside this folder:
      ```
      git init
      git add -A && git commit -m "Initial pipeline"
      git remote add origin https://github.com/<you>/mirror-test-llms.git
      git branch -M main && git push -u origin main
      ```
- [ ] Create a Hugging Face account; make a **read** access token
      (Settings → Access Tokens); store it somewhere safe.
- [ ] While logged in on huggingface.co, open these model pages and accept
      the licenses (needed later): `meta-llama/Llama-3.2-3B-Instruct`,
      `google/gemma-2-9b-it`.
- [ ] Create a Google Scholar alert for **"self-recognition language
      models"** (scoop insurance, protocol §18).
- [ ] Find the next **ACL/EMNLP/NAACL Student Research Workshop CFP**
      (search "<conference> SRW <year> call for papers"), note its deadline,
      and write it at the top of this file: **DEADLINE: ______**. Plan
      backwards from it.

**Done when:** tests pass, repo is on GitHub, licenses accepted, deadline known.

## Phase 1 — Reading + pre-registration (weeks 1–2, 💻)

Work through `docs/05_reading_guide.md` (1–2 papers/day, 5-sentence summary
each, same day — template in that file):

- [ ] Gallup 1970 (mirror test) — summary written
- [ ] Panickssery et al. 2024 ⭐ anchor — full 3-pass read — summary written
- [ ] Davidson et al. 2024 — summary written
- [ ] Laine et al. 2024 (SAD) — summary written
- [ ] Zhou et al. 2025 (+ chase the Ackerman ref) — summary written
- [ ] Mitchell et al. 2023 (DetectGPT) — summary written
- [ ] Zheng et al. 2023 (LLM-as-judge biases) — summary written
- [ ] Schaeffer et al. 2023 (emergence mirage) — summary written
- [ ] Butlin et al. 2023 + Chalmers 2023 — one-line notes each
- [ ] Read `docs/01_llm_primer.md` and `docs/02_stats_primer.md` alongside.
- [ ] Fill `PREREGISTRATION.md` §2 (your predicted outcome row) and the
      header (name, date).
- [ ] **THE COMMIT THAT MATTERS:**
      `git add PREREGISTRATION.md configs/ && git commit -m "Pre-registration: hypotheses + analysis plan (frozen)" && git push`

**Done when:** 10 summaries exist; the pre-registration commit is on GitHub
*before any real data exists*.

## Phase 2 — Freeze prompts + smoke test (week 3)

- [ ] 💻 Pin model revisions: for each model in `configs/models.yaml`, open
      its HF page → *Files and versions* → copy the commit hash into
      `revision:` (replaces the `TODO(pin)` nulls). Commit.
- [ ] 💻 Build the real prompt sets: `python src/00_build_prompts.py`
      (downloads a few hundred MB; ~10 min).
- [ ] 💻 Inspect 10 random lines of each `data/prompts/*.jsonl` — do the
      prompts/references look sane?
- [ ] 💻 Commit the freeze:
      `git add data/prompts && git commit -m "Freeze prompt sets (seed 42)" && git push`
- [ ] ☁️ First Colab session — full smoke test at toy scale, following
      "Stage 0" in `docs/03_pipeline_walkthrough.md` **but keep the real
      frozen prompts** (use `--max-prompts 8` / `--limit 8` flags only).
      Open `notebooks/colab_pipeline.ipynb`, edit the repo URL, run SETUP,
      then the smoke commands.
- [ ] ☁️ Confirm `results/tables/main_ppp.csv` appears; then delete the toy
      generations/judgments (`data/generations/*`, `data/pairs/*`,
      `results/judgments/*`, `results/baselines/*`, `results/tables/*`) —
      NOT `data/prompts/`. Commit.

**Done when:** frozen prompts are committed and one tiny end-to-end run has
produced a results table.

## Phase 3 — Full generation (week 4, ~8 sessions)

One model per GPU session; after each, push (☁️) or Save Version (🟦).
Each session: open the notebook → SETUP cells → the one generation line.

- [ ] ☁️ `python src/01_generate.py --models qwen2.5-0.5b-instruct --placebo` (~30 min)
- [ ] ☁️ `... --models qwen2.5-1.5b-instruct --placebo` (~45 min)
- [ ] ☁️ `... --models qwen2.5-3b-instruct --placebo` (~1 h)
- [ ] ☁️ `... --models qwen2.5-7b-instruct --placebo` (~2 h)
- [ ] 🟦 `... --models qwen2.5-14b-instruct --placebo` (~2.5 h; 2×T4)
- [ ] ☁️ `... --models llama-3.2-3b-instruct` (~1 h)
- [ ] 🟦 `... --models gemma-2-9b-it` (~2 h)
- [ ] ☁️ `... --models mistral-7b-instruct-v0.3` (~1.5 h)
- [ ] 💻 Sanity-read 5 generations per model (`data/generations/…`): any
      empty texts? refusals? weird formats? Note observations for the paper.

**Done when:** 8 generation files × 3 domains exist and are pushed.

## Phase 4 — Pairs + first judging (week 5)

- [ ] 💻 `python src/02_build_pairs.py` — read the console: any cell
      `BELOW TARGET 150`? Record the counts (they go in the appendix).
      Commit `data/pairs/`.
- [ ] ☁️ Judge the two smallest models first (fast feedback):
      `python src/03_judge_ppp.py --judge qwen2.5-0.5b-instruct --include-placebo`
      `python src/03_judge_ppp.py --judge qwen2.5-1.5b-instruct --include-placebo`
- [ ] 💻 Early stats: `python src/06_stats.py --skip-figures` → open
      `results/tables/placebo.csv`. **GATE: placebo must be ≈50%** (CI
      covering 0.5). If not → `docs/07_troubleshooting_faq.md` before
      continuing. This gate protects everything downstream.

**Done when:** pairs committed; 2 judges judged; placebo gate passed.

## Phase 5 — Full grid + phrasings + IPP (week 6)

- [ ] ☁️ `python src/03_judge_ppp.py --judge qwen2.5-3b-instruct --include-placebo`
- [ ] ☁️ `python src/03_judge_ppp.py --judge qwen2.5-7b-instruct --include-placebo`
- [ ] 🟦 `python src/03_judge_ppp.py --judge qwen2.5-14b-instruct --include-placebo`
- [ ] 🟦 Main-cell phrasing robustness (§8.5):
      `python src/03_judge_ppp.py --judge qwen2.5-14b-instruct --foils llama-3.2-3b-instruct --phrasings 0 1 2`
- [ ] ☁️ IPP for all five judges (one session):
      `python src/04_judge_ipp.py --judge <each>`
- [ ] 💻 `python src/06_stats.py` → this is **main results table v1**. Look
      at it. Which §11 outcome row is it shaping into?

**Done when:** `main_ppp.csv` has all 5 judges × 4 foils.

## Phase 6 — Baselines (week 7)

- [ ] 💻 `python src/05_baselines.py stylometric` (~5 min). If any cell hits
      ~99%+, STOP: boilerplate is leaking (see FAQ) — fix cleaning, rebuild
      pairs, re-judge affected cells, log the deviation.
- [ ] ☁️ Perplexity rule, all judges (2 sessions):
      `python src/05_baselines.py perplexity --judge <each>`
- [ ] 💻 `python src/06_stats.py` → ablation columns (stylo, PPL, κ) now
      filled = **ablation table v1**.

## Phase 7 — Paraphrase + base-vs-instruct + final stats (week 8)

- [ ] ☁️ Build paraphrase materials: `python src/02b_paraphrase.py`
      (if you dropped 14B: `--judge qwen2.5-7b-instruct` AND update
      `paraphrase.judge` in the config; commit the config change + log it).
- [ ] 🟦 Judge the paraphrased pairs:
      `python src/03_judge_ppp.py --judge qwen2.5-14b-instruct --foils llama-3.2-3b-instruct --include-paraphrase`
- [ ] 🟦 Base-vs-instruct (§9.5):
      `python src/03_judge_ppp.py --judge qwen2.5-7b-base`
      `python src/03_judge_ppp.py --judge qwen2.5-14b-base`
      `python src/05_baselines.py perplexity --judge qwen2.5-7b-base`
- [ ] 💻 Final stats pass: `python src/06_stats.py` → all tables + figures.
- [ ] 💻 Verify against `README.md`'s honest-work checklist (every box).
- [ ] 💻 Commit everything; tag it: `git tag results-v1 && git push --tags`.

**Done when:** every figure/table the paper needs exists and is reproducible
by one command.

## Phase 8 — OPTIONAL extensions (week 9 — only if on schedule)

- [ ] Urdu (§9.8): follow `docs/09_extensions.md` §A (100 prompts, 2 fluency
      checkers, one foil, PPP only), or
- [ ] LoRA (§9.9): follow `docs/09_extensions.md` §B, or
- [ ] Neither — use the week as buffer. That is a fine choice.

## Phase 9 — Write the paper (weeks 10–11, 💻)

Use `paper/TEMPLATE_PAPER.md` (full prose with placeholders) +
`paper/main.tex` (LaTeX vehicle) + `docs/06_writing_guide.md` (process).

- [ ] Set up Overleaf with the official ACL template; drop in `main.tex`,
      `references.bib`, and the two PDF figures.
- [ ] Week 10: **Method** (from config + pairs_report.json) then **Results**
      (paste `results/tables/table1_main.tex`; one paragraph per finding,
      numbers from `main_ppp.csv`, `paraphrase.csv`, `ipp.csv`).
- [ ] Week 11: **Introduction** → **Related Work** (stitch your 10
      summaries) → **Discussion** (pick the §13 conclusion matching your
      outcome) → **Limitations** (from §14) → **Abstract** (fill brackets).
- [ ] Verify EVERY reference on arXiv/Scholar (the .bib is full of VERIFY
      notes — resolve each).
- [ ] Fill all appendices (prompts verbatim, per-cell counts, placebo/
      refusals, per-domain, reproducibility, power).
- [ ] Grep the PDF for "conscious" — must appear ≤2 times, hedged.

## Phase 10 — Review + submit (week 12)

- [ ] Supervisor/faculty member reads the full draft (book them in week 10!).
- [ ] Two peers read it; fix what confused them.
- [ ] Clean the repo: README quickstart works from a fresh clone;
      `python src/06_stats.py` reproduces Fig. 1.
- [ ] Double-blind prep: anonymized repo mirror (anonymous.4open.science),
      no names in the PDF, `\usepackage[review]{acl}`.
- [ ] Check the venue's preprint policy → post to arXiv (cs.CL) if allowed.
- [ ] Submit to the SRW/workshop. 🎉
- [ ] Afterwards: present at NUST's undergrad research showcase for feedback;
      if reviews are strong, extend (Urdu + LoRA) toward Findings next cycle.

---

## Standing rules (apply to every phase)

1. **Commit at the end of every session** — laptop or GPU, no exceptions.
2. **Placebo gate:** any time placebo drifts from ~50%, stop and debug.
3. **Deviations log:** any post-freeze change → dated entry in
   `PREREGISTRATION.md` §6 + a sentence in the paper.
4. **One model per GPU session**; re-running a dead session is free.
5. **When behind schedule:** cut Phase 8, then shrink the 14B to 7B
   (protocol §18) — never cut controls or error bars.
