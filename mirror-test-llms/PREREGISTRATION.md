# Pre-registration — The Mirror Test for Language Models

> **What this file is.** An informal pre-registration (protocol §2, §10):
> the hypotheses, design, and analysis plan, frozen **before** the main
> experiments run. Commit this file (and `configs/models.yaml`) to git and
> push it BEFORE running `03_judge_ppp.py` on real data. The dated commit is
> the evidence that analyses were planned, not fished for. Being wrong
> publicly is good science (§11).

**Author:** Shehryar
**Institution:** NUST, BS Data Science
**Date frozen:** 2026-07-16 (hypotheses, design, and analysis plan; commit
`68dfb3d`). Prediction in §2 recorded 2026-07-17 — see deviations log.
**Config hash:** `358a530de8b2a67ba62271f102985d936395ed46` (last commit
touching `configs/models.yaml` at freeze time; model revisions were pinned
into the config on 2026-07-17 from the recorded generation runs — see
deviations log).

---

## 1. Research questions and hypotheses (verbatim commitments)

**RQ1 (existence, replication).** Can instruction-tuned open models of modest
size (≤14B) identify their own generations above chance in a forced-choice
setting?
**H1:** Yes for the largest sizes, consistent with prior findings on larger
models; unknown for <3B.

**RQ2 (scale).** How does self-recognition accuracy change with parameter
count within a single model family trained on the same recipe?
**H2:** Accuracy increases monotonically with log(parameters). A sudden jump
would be an "emergence" claim, which requires the continuous-metric (AUROC)
check before it can be asserted.

**RQ3 (mechanism — paraphrase).** Does accuracy survive when both candidate
texts are paraphrased by a third model (content preserved, wording changed)?
**H3:** Accuracy drops substantially toward chance, indicating reliance on
surface lexical/stylistic features.

**RQ4 (mechanism — cheap baselines).** Can (a) a character-n-gram
logistic-regression classifier and (b) a "choose the candidate with lower
perplexity under the judge" rule match or exceed the judge's own accuracy?
**H4:** Yes for (a); (b) will correlate strongly (Cohen's κ > 0.4) with the
judge's choices, suggesting likelihood preference as the underlying signal.

**RQ5 (optional — language).** Does the pattern replicate in Urdu?
No directional hypothesis; exploratory.

## 2. My prediction (recorded 2026-07-17, before any judging runs)

> I predict **outcome #2** (deflationary): accuracy rises with scale, but
> collapses under paraphrasing; the character-n-gram classifier matches or
> exceeds the judges' accuracy on the same items; and the lower-perplexity
> rule agrees substantially with the judges' choices (κ > 0.4) —
> "self-recognition" in this regime is style/likelihood matching.
>
> Reasoning: (1) the signal most plausibly available to a stateless
> next-token predictor is its own stylistic fingerprint — exactly what the
> paraphrase attack removes and the stylometric classifier replicates;
> (2) the protocol itself notes this outcome is statistically the most
> likely; (3) the stylometric baseline, already computed at registration
> time, shows the information needed to solve the task is largely present
> in surface statistics (pairwise accuracy > 0.95 in many cells), so
> privileged self-access is not required to explain a positive result.
>
> **Disclosure:** this prediction was recorded after generation and the
> stylometric baseline were complete, but BEFORE any judge was run on any
> pair (the primary endpoint) — the judges' accuracies, the paraphrase
> attack, the perplexity-rule agreement, and all significance tests were
> unseen at registration time.

## 3. Design (fixed)

* **Judges:** Qwen2.5-Instruct at 0.5B / 1.5B / 3B / 7B / 14B (exact HF ids
  and pinned revisions in `configs/models.yaml`).
* **Foils:** Llama-3.2-3B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3,
  HUMAN reference texts.
* **Domains:** news summarization (CNN/DailyMail), open QA (Dolly-15k),
  creative writing (WritingPrompts); 200 frozen prompts each (seed 42),
  SHA-256 checksums in `data/prompts/CHECKSUMS.txt`.
* **Generation:** temperature 0.8, top-p 0.95, ≤160 new tokens, seed
  = 1000 + prompt_idx (placebo second sample: 2000 + prompt_idx).
* **Protocols:** PPP (primary; both A/B orders, greedy judging, first-token
  log-prob scoring) and IPP (secondary; signal detection analysis).
* **Controls:** identical cleaning for all authors incl. human; ≤25%
  length-ratio filter; ROUGE-L ≤ 0.7 near-duplicate filter; SELF-vs-SELF
  placebo pairs; 3 instruction phrasings on the main cell.

## 4. Analysis plan (frozen; implemented in src/06_stats.py before data)

* **Primary endpoint:** order-averaged PPP accuracy per (judge, foil, domain)
  cell; pooled across domains for the per-judge headline.
* **Uncertainty:** 95% Wilson score intervals on every accuracy.
* **Tests:** exact two-sided binomial vs 0.5 on decisive items;
  Holm–Bonferroni across the judge × foil family; McNemar (exact) for
  original-vs-paraphrased and judge-vs-stylometric on the same items;
  Spearman's ρ for the scale trend (descriptive, n = 5); Cohen's κ with
  1,000-resample bootstrap CI for judge-vs-perplexity-rule agreement;
  IPP: hit rate, false-alarm rate, d′ (log-linear correction), criterion c,
  AUROC with bootstrap CI. AUROC reported alongside accuracy everywhere the
  scale curve is shown (emergence-artifact check).
* **Interpretation rule committed in advance (§9.2):** if the stylometric
  classifier's accuracy ≥ the judge's accuracy on the same items, the
  information needed for "self-recognition" is present in surface statistics,
  and privileged self-access is not required to explain the behaviour.
* **Multiple-comparison family:** all (judge × foil) pooled-domain binomial
  tests (up to 5 × 4 = 20 tests), α = .05.
* **Power (why n≈150/cell):** exact binomial, α = .05, two-sided: n = 150
  gives ~79% power at true accuracy 0.615; pooling 3 domains (n = 450)
  detects ~0.567; effects below ~0.55 need n ≈ 780 (reproduced by
  `power_exact_binomial` in `src/stats_utils.py`; see
  `results/tables/power_analysis.csv`).

## 5. What this study does NOT claim

No result here is evidence for or against consciousness, subjective
experience, or a "self." "Self" denotes only "text generated by the same
weights" (Panickssery et al.'s prosaic usage). The consciousness motivation
appears in at most one sentence of the introduction and one of the
discussion.

## 6. Deviations log

Any change made AFTER the freeze date must be recorded here with a date and
a reason, and disclosed in the paper.

| Date | What changed | Why |
|---|---|---|
| 2026-07-17 | §2 prediction filled (left as a template at the 07-16 freeze). | Recorded before any judging (primary endpoint untouched); disclosed in §2 that the stylometric baseline had already been computed at registration time. |
| 2026-07-17 | Model `revision:` fields in `configs/models.yaml` pinned to the exact HF commits recorded in the generation logs (previously `null` = latest). | Documentation of what actually ran, not a change to it; makes the §3 "pinned revisions" claim true and the appendix reproducible. Paraphraser/base models pin when they first run. |
