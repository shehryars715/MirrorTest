# Statistics Primer — every number in the analysis, explained with examples

The protocol says "the difference between a rejected and an accepted first
paper is almost never the idea — it is controls, error bars, and honest
framing." This file is the error-bars part. Every statistic below is
implemented from scratch in `src/stats_utils.py` (read them side by side),
and each section explains: *the problem → the tool → a worked example →
where it appears in the paper.*

Throughout, our basic datum is: a judge answered n forced-choice items and
got k right. Chance = 50%.

---

## 1. Why accuracy alone is meaningless: confidence intervals

**Problem.** A judge scores 60%. Impressive? If n = 10, pure guessing
produces ≥60% about 38% of the time. If n = 500, essentially never. The
number 60% means nothing without n.

**Tool: the 95% Wilson score interval.** A range of "true accuracies"
compatible with what you observed. Roughly: if the true accuracy were
outside this range, data like yours would be surprising.

Why Wilson and not the plus/minus formula from intro stats
(p̂ ± 1.96·√(p̂(1−p̂)/n))? The simple ("Wald") interval misbehaves exactly
where we live — small samples and accuracies near 0.5 or 1.0. It can extend
past 1.0 or collapse to zero width. Wilson never leaves [0, 1] and has
honest coverage at small n. The protocol (§10) mandates it.

**Worked example** (`wilson_ci(0.8, 10)`): 8/10 correct → CI ≈ [0.49, 0.94].
The same 80% with n = 150 → [0.73, 0.86]. Same accuracy, wildly different
knowledge.

**In the paper:** *every* accuracy carries its n and Wilson CI. No
exceptions (§10 "Reporting standards", pre-submission checklist §19).

## 2. "Is it better than guessing?" — the exact binomial test

**Problem.** Even with a CI you want a yes/no answer to "could pure guessing
explain this?"

**Tool.** Under guessing, the number correct follows a Binomial(n, 0.5)
distribution. The exact two-sided p-value sums the probability of every
outcome at least as extreme as yours. "Exact" means no normal approximation —
correct even at n = 30.

**Worked example** (`binom_test_two_sided(95, 150)`): 95/150 = 63.3%
→ p ≈ 0.0013. Guessing produces something this lopsided about once in 800
experiments. Contrast 80/150 = 53.3% → p ≈ 0.46: entirely consistent with
guessing.

**One subtlety in our design:** each pair is judged twice (both A/B orders),
and the two runs of one pair are not independent. So the significance test
uses only *decisive* items — pairs answered the same way (both right or both
wrong) — a standard sign-test treatment. Ties (right once, wrong once) are
position-driven noise and carry no authorship evidence in either direction.
The headline accuracy still counts ties as 0.5 (protocol §8.3).

## 3. Twenty tests, one alpha: Holm–Bonferroni

**Problem.** We test 5 judges × 4 foils = 20 accuracy-vs-chance hypotheses.
At α = .05 each, ~1 false positive is *expected* even if nothing is real.
Reviewers hunt for this mistake.

**Tool: Holm's step-down method.** Sort the 20 p-values ascending. Compare
the smallest to .05/20, the next to .05/19, and so on; stop at the first
failure. It guarantees ≤5% chance of ANY false positive, while rejecting
more than plain Bonferroni.

**Worked example** (`holm_bonferroni`): p-values (.001, .004, .011, .043),
m = 4 → sorted thresholds are .05/4=.0125, .05/3=.0167, .05/2=.025,
.05/1=.05. Check in order: .001<.0125 ✓, .004<.0167 ✓, .011<.025 ✓,
.043<.05 ✓ — all four survive. Now change .011 to .030: .030 > .025 → stop
there, and BOTH .030 and .043 fail (even though .043 < .05 on its own).
That "stop at the first failure" cascade is the whole method.

**In the paper:** "p-values Holm-corrected across the judge × foil family;
corrected α reported" — one sentence, big credibility (§10).

## 4. Same items, two conditions: McNemar's test

**Problem.** Original vs paraphrased is not two independent samples — it is
the SAME pairs measured twice. Comparing "74% vs 61%" with a two-sample test
throws away the pairing and is wrong.

**Tool.** Only the *disagreements* are informative: b = items correct
originally but wrong paraphrased; c = the reverse. If paraphrasing changed
nothing, each disagreement is a fair coin → exact binomial test of b vs c.

**Worked example** (`mcnemar_exact(24, 6)`): of 140 shared pairs, 24 flipped
to wrong after paraphrasing, 6 flipped to right → p ≈ 0.0014. Paraphrasing
reliably damaged recognition. (`mcnemar_exact(9, 6)` → p ≈ 0.61: no
evidence.)

**In the paper:** paraphrase attack (Fig. 2) and judge-vs-stylometric
comparisons, both on identical item sets (§10).

## 5. Do two decision-makers agree beyond luck? Cohen's κ

**Problem.** The perplexity rule picks the same text as the judge 71% of the
time. But if both say "self" ~65% of the time overall, a fair amount of
agreement happens by luck.

**Tool.** κ = (observed agreement − chance agreement) / (1 − chance
agreement), where chance agreement comes from each party's own base rates.
κ = 1 perfect, 0 = luck-level, <0 worse than luck. Rules of thumb: 0.2–0.4
fair, 0.4–0.6 moderate, >0.6 substantial. The pre-registered threshold of
interest is κ > 0.4 (H4).

**Worked example** (`cohen_kappa`): judge says self on 70% of runs, rule on
64%; they agree 71% of the time. Chance agreement = .70×.64 + .30×.36 =
0.556 → κ = (0.71 − 0.556)/(1 − 0.556) ≈ 0.35. Moderate-ish — report with
its bootstrap CI and let the reader see.

## 6. Uncertainty for weird statistics: the bootstrap

**Problem.** κ and AUROC don't have neat CI formulas like proportions do.

**Tool: resampling.** Pretend your sample IS the population: draw n items
*with replacement*, recompute the statistic, repeat 1,000 times; the middle
95% of those values is the CI. One crucial detail in our data: resample
*pairs*, not runs — the two counterbalanced runs of a pair travel together,
otherwise you fake independence you don't have (`bootstrap_ci(...,
items=pair_groups)`).

## 7. The Yes/No trap: signal detection (d′ and criterion c)

**Problem.** In the individual protocol ("Did you write this? Yes/No") a
model that ALWAYS says Yes scores 100% on SELF items. Accuracy alone
rewards sycophancy.

**Tool.** Separate *sensitivity* from *bias*:

* hit rate H = P(Yes | actually SELF)
* false-alarm rate F = P(Yes | actually foil)
* **d′ = z(H) − z(F)** — discrimination. 0 = blind; ~1 = decent; ~2 = strong.
* **criterion c = −(z(H)+z(F))/2** — response style. Negative = trigger-happy
  Yes-sayer; positive = conservative.

(z = inverse of the standard normal CDF; rates get the standard +0.5/+1
correction so 0% and 100% don't produce infinities — `dprime_criterion`.)

**Worked example.** Model A: H=.90, F=.10 → d′ ≈ 2.56, c ≈ 0 — real
discrimination. Model B: H=.98, F=.94 → d′ ≈ 0.50, c ≈ −1.8 — mostly a
Yes-machine. Their raw accuracies can look similar; d′ exposes the
difference. Protocol §11 outcome 4 is exactly this pattern, and it is a
publishable methods finding.

## 8. Threshold-free ability: AUROC (and the "emergence mirage")

**Problem.** Accuracy applies a hard threshold ("was the margin > 0?"). A
model can be *improving* underneath — its logp(A)−logp(B) margins shifting
steadily in the right direction — while accuracy sits flat, then "jumps"
when margins finally cross zero. Schaeffer et al. (2023) showed several
famous "emergent abilities" are exactly this artifact of thresholded
metrics.

**Tool.** AUROC = the probability that a randomly chosen positive item gets
a higher continuous score than a randomly chosen negative item (ties count
half). 0.5 = no signal, 1.0 = perfect. It uses the *whole margin
distribution*, no threshold.

**Our construction for the pairwise test:** every run has a score
s = logp(A) − logp(B) and a label "was SELF in position A?". Position bias
shifts s for both labels equally, so AUROC isolates the genuine self-signal.
For IPP: s = logp(Yes) − logp(No), label = is_self.

**In the paper (§9.6):** plot accuracy AND AUROC vs scale. If accuracy jumps
but AUROC rises smoothly, say exactly that — "the jump is a threshold
artifact, not sudden emergence." Reviewers reward this observation.

## 9. Monotone trends: Spearman's ρ

Pearson correlation asks "is the relationship linear?" — too strong for 5
points. Spearman asks only "does accuracy go UP as size goes up?": correlate
the *ranks*. ρ = +1 means perfectly monotone. With n = 5 sizes it has little
formal power, so the protocol (§10) says report it as descriptive; the
CI-per-point figure carries the real argument.

## 10. "Could we even have detected an effect?" — power analysis

**Problem.** Suppose all results come out ≈ chance (outcome 3). "We found
nothing" is only meaningful if you can add "and we WOULD have found an
effect of size X." Otherwise the null is just underpowered noise.

**Tool.** Power = P(your test rejects | the true accuracy is p_alt). Computed
exactly in `power_exact_binomial`: find every outcome the test would call
significant, sum their probabilities under p_alt.

**Our numbers** (reproduced by `results/tables/power_analysis.csv` and
verified in `tests/`): n = 150/cell → 79% power at true accuracy 0.615;
pooling 3 domains (n = 450) → ~80% at 0.567; detecting 0.55 needs n ≈ 780.
That is the honest sentence for the appendix: effects smaller than ~6 points
above chance may be missed (protocol §14, limitation 8).

## 11. The reporting contract (memorize this)

Every number in the paper carries: its **n**, its **95% CI**, and — if it is
a comparison — the **name of the test and the exact p**. Seeds, model
revisions, decoding parameters, and prompt templates go in the appendix.
All raw judgments ship in the repo. (Protocol §10 + checklist §19.)

## 12. Further reading (free)

* *Seeing Theory* (Brown University) — visual intro to distributions & CIs.
* StatQuest (YouTube): "Confidence intervals", "ROC and AUC", "Bootstrap".
* Stanislas Dehaene's signal-detection lecture notes, or any Psych methods
  text ch. on SDT, for d′/criterion intuition.
* Schaeffer, Miranda & Koyejo (2023), arXiv:2304.15004 — read after §8 here.
