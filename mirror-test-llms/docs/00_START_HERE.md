# START HERE — a map of everything

You have a research protocol (`../mirrortest.md`), and this repository turns
it into working code. This page tells you what to read in what order,
depending on how much you already know. Nothing here assumes prior research
experience.

---

## What is this project, in plain words?

When a chimpanzee touches a paint mark on its own forehead after seeing it
in a mirror, we say it "recognizes itself." Researchers found something that
looks similar in large language models: show GPT-4 two texts — one it wrote,
one written by someone else — and it picks out its own more often than
chance. Spooky? Maybe. Or maybe totally boring: maybe the model just prefers
text that *statistically resembles* its own habits, the way you might
recognize your own handwriting without any deep self-awareness being
involved.

Your study settles which explanation fits, for small open models you can run
for free:

1. **Scale:** test five sizes of the same model family (0.5B → 14B
   parameters). Does the ability grow with size? Appear suddenly? Never appear?
2. **Mechanism:** attack the ability three ways —
   * **Paraphrase attack:** rewrite both texts in different words. If
     "self-recognition" dies, it was surface wording all along.
   * **Stylometric baseline:** a 1990s-style classifier (character n-grams +
     logistic regression). If IT can tell the authors apart as well as the
     model can, no "self-access" is needed to explain the behaviour.
   * **Perplexity rule:** "pick the text the model finds more probable." If
     this dumb rule agrees with the model's choices, the model is plausibly
     just expressing "this sounds like me."

Every outcome is publishable. That is deliberate — see protocol §11.

## Reading order

**Level 0 — orientation (today, ~1 hour):**
1. This page.
2. `README.md` — the pipeline picture and repo map.
3. Protocol §1–§4 in `../mirrortest.md` (the .md you already have).

**Level 1 — the concepts you'll need (week 1, alongside paper-reading):**
4. [`01_llm_primer.md`](01_llm_primer.md) — tokens, log-probabilities,
   temperature, chat templates, quantization. *Everything the code does with
   models is explained here.*
5. [`02_stats_primer.md`](02_stats_primer.md) — every statistic in the
   analysis, with worked examples. *Everything `06_stats.py` computes is
   explained here.*
6. [`GLOSSARY.md`](GLOSSARY.md) — look things up as you go.

**Level 2 — doing the work (weeks 1–9):**
7. [`05_reading_guide.md`](05_reading_guide.md) — the 10 papers, what to
   extract from each (weeks 1–2).
8. [`03_pipeline_walkthrough.md`](03_pipeline_walkthrough.md) — script by
   script: what it does, how to run it, what comes out (weeks 3–8).
9. [`04_colab_kaggle_guide.md`](04_colab_kaggle_guide.md) — free GPUs,
   Hugging Face tokens, checkpointing discipline.
10. [`07_troubleshooting_faq.md`](07_troubleshooting_faq.md) — when things
    break (they will; it's normal).

**Level 3 — understanding WHY (read before writing the paper):**
11. [`08_design_rationale.md`](08_design_rationale.md) — why every control
    exists, and what breaks without it. Reviewers ask exactly these
    questions.
12. [`06_writing_guide.md`](06_writing_guide.md) — LaTeX/Overleaf, the ACL
    template, section-by-section instructions, venue strategy (weeks 10–12).
13. [`09_extensions.md`](09_extensions.md) — optional Urdu and LoRA
    extensions, only after the core is done.

## The 12-week plan (protocol §16, condensed)

| Weeks | You are doing | Repo output |
|---|---|---|
| 1–2 | read 10 papers, 5-sentence summaries each | `PREREGISTRATION.md` committed |
| 3 | freeze prompts, smoke-test pipeline on 0.5B | `data/prompts/` locked |
| 4 | full generation run (all models) | `data/generations/` complete |
| 5 | pairs + placebo; first PPP judging | core results for 2 judges |
| 6 | full PPP grid + phrasings + IPP | main results table v1 |
| 7 | stylometric + perplexity baselines | ablation table v1 |
| 8 | paraphrase attack; base-vs-instruct; full stats pass | figures/tables final-ish |
| 9 | buffer / optional Urdu or LoRA | slack absorbed |
| 10 | write Method + Results | half draft |
| 11 | write Intro/Related/Discussion/Limitations | full draft |
| 12 | feedback, revise, clean repo | submit arXiv + venue 🎉 |

**Rule of thumb (protocol §16):** if a week slips, cut the optional
extensions (§9.8/§9.9), never the controls (§9.1–§9.7).

## Three habits that make this succeed

1. **Commit early, commit often.** `git add -A && git commit -m "..."` at the
   end of every work session. The pre-registration only counts if it is a
   *dated commit before the runs*.
2. **Never edit frozen files.** Once `data/prompts/*.jsonl` exists and is
   committed, it never changes (the code enforces this — see `--force`).
3. **Write while you work.** The 5-sentence paper summaries become your
   Related Work section. The cleaning statistics become your appendix. The
   deviations log becomes your Limitations section. Nothing is wasted.

## Set up git right now (5 minutes)

```bash
cd mirror-test-llms
git init
git add -A
git commit -m "Initial pipeline + pre-registration skeleton"
# create an empty repo on github.com named mirror-test-llms, then:
git remote add origin https://github.com/<your-username>/mirror-test-llms.git
git branch -M main
git push -u origin main
```

Also set a Google Scholar alert for **"self-recognition language models"**
today (protocol §18 — scoop insurance).
