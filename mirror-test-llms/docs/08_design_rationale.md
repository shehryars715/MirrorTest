# Design Rationale — why every control exists, and what breaks without it

This is the "advanced" layer: the reasoning that turns a pile of scripts
into a defensible experiment. Reviewers probe exactly these joints. Each
entry: *the threat → the control → what the result would look like if the
control were missing.* Read before writing the Method and Discussion.

---

## 1. Threat: position bias masquerading as recognition

LLM judges systematically prefer one answer position (Zheng et al. 2023).
A judge that "always says A" scores 50% overall but could score far from
50% on any subset where SELF lands in A more often.

**Controls (two, independent):**
* **Counterbalancing (§8.3):** every pair judged twice, orders swapped;
  the item score averages the two runs. Position preference produces ties
  (0.5), not wins.
* **Placebo pairs (§9.1):** SELF-vs-SELF pairs where no right answer exists;
  the chose-A rate directly measures position bias, and ANY pipeline bug
  (swapped fields, template artifacts) surfaces as a deviation from 50%.

**Without them:** the headline accuracy would be uninterpretable, and a
single data-plumbing bug could fabricate the entire "phenomenon."
`06_stats.py` additionally reports *position-consistency* — the fraction of
items answered the same way under both orders. Low consistency is itself a
finding: the "ability" is fragile.

## 2. Threat: verbosity/length confound

Judges prefer longer answers (Zheng et al. 2023); models differ
systematically in verbosity. If SELF texts are longer than FOIL texts,
"self-recognition" could be length-recognition.

**Control:** ≤25% word-count difference filter at pair building (§7.3);
residual length stats reported (`length_stats.csv`).
**Without it:** a "scale curve" could just track how verbosity changes with
model size.

## 3. Threat: boilerplate leaks authorship

Instruction-tuned models decorate output ("Sure! Here is…", "**Summary:**",
`<|im_end|>`). These are near-deterministic authorship tags.

**Control:** one cleaning function applied to EVERY author including humans
(§7.2); the cleaner is unit-tested; per-author cleaning stats reported.
**The alarm for failure:** the stylometric baseline (§9.2). If a bag of
character n-grams hits ~99%, the texts still carry tags and the mirror test
is measuring formatting, not writing. (This is why the baseline is not
optional — it is the *detector* for a broken experiment as well as a
mechanism probe.)

## 4. Threat: near-duplicate candidates

Two models answering "What is the capital of France?" produce near-identical
texts; the judge is at chance for the boring reason that there is nothing to
tell apart. Mixing such pairs in dilutes real signal unpredictably across
cells and domains.

**Control:** drop pairs with ROUGE-L > 0.7 between candidates (§18);
counts reported per cell.

## 5. Threat: refusals and unparseable answers

Aligned models say "As an AI, I have no memory of writing…". Parsing free
text would silently drop exactly the trials where alignment interferes,
biasing the sample.

**Control:** first-token log-probability scoring (§8.4) — the decision is
read from the probability of "A" vs "B" directly; nothing is unparseable.
Free text is still logged on a subsample purely to REPORT refusal rates
(§9.7) — refusal falling with scale is interesting data, not noise.
**Bonus:** the logprob margin is a continuous score → AUROC → the
emergence-artifact check.

## 6. Threat: "emergence" as a metric artifact

Accuracy thresholds a continuous margin at zero. Margins can drift smoothly
toward correct while accuracy sits at chance, then "jumps" — fake emergence
(Schaeffer et al. 2023).

**Control:** AUROC computed from the same runs, plotted beside accuracy
(§9.6). Design detail: PPP AUROC uses score = logp(A)−logp(B) with label
"SELF was in position A" — position bias shifts both classes equally, so
the metric isolates the self-signal.
**Without it:** a jump at 7B would be reported as emergence when it may be
threshold-crossing.

## 7. Threat: prompt-phrasing luck

Self-recognition scores move with instruction wording (Laine et al. 2024).
A result that exists under one phrasing only is a prompt artifact.

**Control:** 3 frozen phrasings (in the config, so they're pre-registered),
all run on the main cell; mean ± range reported (§8.5). Cheap, high value.

## 8. Threat: Yes-bias in the individual protocol

A sycophantic model says "Yes, I wrote this" to everything → 100% on SELF
items, 0% information.

**Control:** signal detection analysis (§9.7): d′ separates discrimination
from bias; criterion c quantifies the Yes-lean. Outcome 4 of the protocol
("high yes-rate, d′≈0") is a publishable methods warning by itself.

## 9. Threat: the deflationary explanations were never tested

The core scientific move of this study: BEFORE interpreting recognition as
anything self-like, give two cheap mechanisms a fair chance to explain it:

* **Stylometric classifier (§9.2):** if TF-IDF character n-grams + logistic
  regression match the judge on the SAME items (McNemar), the information
  is in surface statistics — no privileged self-access required. The
  interpretation rule is committed *in advance* (pre-registration §4).
  The transfer variant (train on news, test on stories) asks whether the
  style signature is model-general or task-bound.
* **Perplexity rule (§9.3):** "pick the likelier-under-me text" is what a
  next-token predictor does natively. High κ between rule and judge choices
  = the judge is plausibly expressing likelihood preference, not identity.
  Note the design subtlety: generation used temperature 0.8 — greedy
  generation would make SELF texts trivially low-perplexity and hand the
  rule a fake win (§7).

* **Paraphrase attack (§9.4):** rewrites both texts, preserving content,
  destroying wording. Collapse ⇒ the signal was lexical/stylistic surface;
  survival ⇒ something deeper (structure, content choices). The quality gate
  (cosine ≥ 0.80, length ±30%) matters: without it, "paraphrase collapse"
  could just mean "the paraphraser mangled the texts."

## 10. Threat: cherry-picked prompts / garden of forking paths

If prompts can be swapped after seeing results, any effect can be
manufactured.

**Controls:** frozen prompt files with SHA-256 checksums, committed before
generation; refusal of the build script to overwrite them; a pre-registered
analysis plan with a deviations log; Holm correction across the test family;
decisive-item testing rules fixed in code before data existed.

## 11. Interpretation discipline (what we may and may not say)

* "SELF" means *text generated by the same weights* — the judge has no
  episodic memory; the only coherent version of the task for stateless
  models is identifying self-TYPICAL text (§14.2).
* Nothing here measures subjective experience, and the paper says so once,
  plainly (§14.1). The consciousness framing gets one hedged sentence in the
  intro and one in the discussion, period (§2).
* Scale conclusions live inside ONE training recipe (Qwen2.5); other
  families may differ (§14.3). That is why judges never come from mixed
  families — size must be the only moving part.
* Contamination caveat (§6): the prompt datasets likely appear in
  pretraining. This does not invalidate a *recognition* comparison (all
  models saw the same data) but is reported in Limitations.

## 12. Implementation interpretations (this repo's judgment calls)

Documented here so the paper can disclose them in one paragraph:

1. **CNN/DailyMail over XSum** — XSum's one-sentence references fail the
   protocol's own 40–160-word reference filter; CNN/DM matches the "3–4
   sentences" instruction. (Config keeps XSum switchable.)
2. **"First 200 valid" sampling** — the protocol's "sample 200 → filter"
   order would leave <200 prompts; we filter-then-take-200 with the same
   seed, keeping cells at full size.
3. **Per-domain prompt-length bounds** — 50–400 words applies to news
   articles; Dolly/WP prompts are naturally short and get their own bounds
   (all in the config, all reported).
4. **Base models judge their instruct sibling's pairs** (§9.5) via a
   completion template — a base model has no generations of its own; "SELF"
   for it means "written by the model sharing my pretraining." The
   perplexity rule is additionally reported for base models as the
   protocol's suggested proxy.
5. **McNemar binary coding** — "correct" = correct under BOTH orders,
   applied identically to both conditions compared. Symmetric, conservative,
   position-proof.
6. **Freetext subsampling** — free-text answers (refusal accounting) are
   logged on a seeded ~300-run subsample per judge rather than every run,
   halving judging cost with no loss for a rate estimate. `--freetext all`
   restores the full logging.
