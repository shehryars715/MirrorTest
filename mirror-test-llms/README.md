# The Mirror Test for Language Models

**Do small open language models recognize their own writing — and if they do,
what signal are they actually using?**

This repository is a complete, runnable implementation of the research
protocol in `../mirrortest.md` ("Do Language Models Recognize Their Own
Reflection? Scale, Style, and the Limits of Self-Recognition in Small Open
LLMs"): a behavioural analogue of the animal mirror test, run across five
sizes of one model family (0.5B → 14B), with the controls that make the
result mean something — placebo pairs, position counterbalancing, paraphrase
attacks, a stylometric classifier, and a perplexity-preference rule.

> **New here? Read [`docs/00_START_HERE.md`](docs/00_START_HERE.md) first.**
> It maps every file in this repo to the protocol and tells you what to read
> in what order, assuming no prior background. When you're ready to WORK,
> open [`ROADMAP.md`](ROADMAP.md) — the step-by-step checklist from today to
> submission.

---

## The experiment in one picture

```
                          THE PIPELINE (what runs where)
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ LAPTOP (CPU)              GPU (free Colab / Kaggle)         LAPTOP (CPU)│
 │                                                                         │
 │ 00_build_prompts.py  ──►  01_generate.py        ──►  02_build_pairs.py  │
 │ freeze 600 prompts        every model answers        clean identically, │
 │ (news / dolly / wp)       every prompt               build PPP pairs,   │
 │                           (temp 0.8, seeded)         placebo pairs, IPP │
 │                                                             │           │
 │                           02b_paraphrase.py  ◄──────────────┘           │
 │                           rewrite both texts                            │
 │                           (paraphrase attack)                           │
 │                                  │                                      │
 │                                  ▼                                      │
 │                           03_judge_ppp.py       "Which did YOU write,   │
 │                           04_judge_ipp.py        A or B?" scored by     │
 │                           05_baselines.py        first-token log-probs  │
 │                            (perplexity)                 │               │
 │                                                         ▼               │
 │ 05_baselines.py (stylometric, CPU)  ──►  06_stats.py (CPU)              │
 │ TF-IDF char n-grams + LogReg             every table, test, and figure  │
 └─────────────────────────────────────────────────────────────────────────┘
```

## Research questions (frozen in PREREGISTRATION.md)

* **RQ1 (existence):** can instruction-tuned open models ≤14B identify their
  own generations above chance in a forced-choice setting?
* **RQ2 (scale):** how does that ability change with parameter count within
  one model family?
* **RQ3 (mechanism — paraphrase):** does it survive when both candidate texts
  are rewritten by a third model?
* **RQ4 (mechanism — cheap baselines):** can a character-n-gram classifier or
  a "pick the lower-perplexity text" rule reproduce it?
* **RQ5 (optional — language):** does the pattern replicate in Urdu?

Every possible outcome is publishable (protocol §11) — that is the point of
the design.

## Quickstart

```bash
# 0. install the CPU-side requirements (see requirements.txt for profiles)
pip install pyyaml datasets scikit-learn matplotlib pytest

# 1. sanity-check your install (no GPU needed, <1 s)
pytest tests/ -q

# 2. build + freeze the prompt sets (CPU + internet)
python src/00_build_prompts.py

# 3. GPU phases: run on Colab (notebooks/colab_pipeline.ipynb) or Kaggle
#    (notebooks/kaggle_pipeline.ipynb — run src/kaggle_setup.py first;
#    it handles auth + paths). Guide: docs/04_colab_kaggle_guide.md.

# 4. after judging: analysis (CPU)
python src/05_baselines.py stylometric
python src/06_stats.py          # -> results/tables/*, results/figures/*
```

Reproducing Figure 1 from raw judgments is one command: `python src/06_stats.py`.

## Repository map

| Path | What it is | Protocol § |
|---|---|---|
| `ROADMAP.md` | **the operational checklist** — every step from today to submission, with commands | §16 |
| `configs/models.yaml` | ALL experimental parameters, models, prompts, templates | §5–§10 |
| `PREREGISTRATION.md` | hypotheses + analysis plan, committed **before** running | §2, §10 |
| `src/00_build_prompts.py` | freeze 200 prompts × 3 domains | §6 |
| `src/01_generate.py` | all models answer all prompts (seeded) | §7.1 |
| `src/02_build_pairs.py` | clean + PPP pairs + placebo + IPP sets | §7.2–4, §8.2 |
| `src/02b_paraphrase.py` | paraphrase attack materials + quality gate | §7.5 |
| `src/03_judge_ppp.py` | the mirror test, log-prob scored, counterbalanced | §8.1–8.5 |
| `src/04_judge_ipp.py` | Yes/No protocol for signal detection | §8.2, §9.7 |
| `src/05_baselines.py` | stylometric classifier + perplexity rule | §9.2, §9.3 |
| `src/06_stats.py` | Wilson CIs, binomial/Holm/McNemar/κ/d′/AUROC, figures | §10 |
| `src/stats_utils.py` | every statistic implemented + explained from scratch | §10 |
| `src/utils.py` | cleaning, ROUGE-L, JSONL checkpointing, log-prob helpers | §7, §8.4 |
| `src/kaggle_setup.py` | Kaggle/Colab bootstrap: GPUs, HF auth from secrets, writable paths, session resume | §15 |
| `tests/` | 27 unit tests incl. reproduction of the protocol's power claims | — |
| `kaggle_run_all.ipynb` | **self-driving Kaggle runner** — carries the whole pipeline inside itself, auto-resumes across sessions, outputs `mirror_bundle.zip` (regenerate with `tools/build_kaggle_notebook.py` after code changes) | §15 |
| `tools/ingest_bundle.py` | absorb a downloaded `mirror_bundle.zip` back into this repo | §15 |
| `notebooks/colab_pipeline.ipynb` | manual Colab driver for the GPU phases | §15 |
| `notebooks/kaggle_pipeline.ipynb` | manual Kaggle (2×T4) driver — secrets, save-version checkpointing | §15 |
| `paper/` | full-prose template paper (`TEMPLATE_PAPER.md`) + ACL LaTeX skeleton + references.bib | §12 |
| `docs/` | beginner → advanced guides (start at `00_START_HERE.md`) | all |

## Documented deviations from the protocol

Honesty rule: wherever this implementation had to interpret or adjust the
protocol, the choice is recorded here and explained in
[`docs/08_design_rationale.md`](docs/08_design_rationale.md):

1. **News dataset defaults to CNN/DailyMail, not XSum** — XSum gold summaries
   are single ~21-word sentences and would fail the protocol's own 40–160-word
   reference filter; CNN/DM highlights match the "3–4 sentences" instruction.
2. **Prompt sampling keeps the first 200 items that pass filters** (protocol's
   "sample 200 → filter" would leave fewer than 200).
3. **Per-domain prompt-length bounds** — the 50–400-word bound is applied to
   news articles; Dolly questions and WP premises get their own bounds
   (they are naturally short).
4. **Base-model judging (§9.5)** — base checkpoints judge their instruct
   sibling's pairs via a completion template ("The response I wrote is
   Response ___"), since base models never generated anything themselves.
5. **Binary coding for McNemar** — an item counts "correct" only if it was
   correct under *both* presentation orders; applied identically to both
   conditions being compared.
6. **PPP AUROC definition** — over runs, score = logp(A)−logp(B), positive
   class = SELF-shown-as-A; threshold-free and immune to position bias.

## The honest-work checklist (from protocol §19)

- [ ] `PREREGISTRATION.md` committed **before** main runs (dated commit)
- [ ] every accuracy reported with n + 95% Wilson CI; Holm correction applied
- [ ] placebo ≈ 50% reported; refusal rates reported
- [ ] both A/B orders run; position-consistency reported
- [ ] stylometric + perplexity baselines in the main table
- [ ] paraphrase attack with McNemar p-value
- [ ] AUROC alongside accuracy for the scale curve
- [ ] prompts, seeds, model revisions in the appendix; repo public
- [ ] "conscious(ness)" appears ≤ 2 times in the paper, both hedged
- [ ] Limitations section present
- [ ] a supervisor has read the full draft

## Cost

Everything is free (protocol §15): open-weight models, free Colab/Kaggle
GPUs (~15–25 GPU-hours total), Hugging Face datasets, Overleaf. Every GPU
script writes incrementally and resumes automatically, so Colab disconnects
cost you nothing but time.

## License

MIT — see `LICENSE`. If you use this pipeline, please cite the paper (once
it exists) and the prior work in `paper/references.bib`.
