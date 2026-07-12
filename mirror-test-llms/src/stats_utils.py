"""
stats_utils.py — every statistical procedure in the analysis plan (protocol §10),
implemented from first principles with ONLY the Python standard library.

WHY STDLIB-ONLY?
================
* You can run and unit-test the entire statistics layer on any machine —
  no numpy/scipy install needed (they are still listed in requirements.txt
  because the baselines use scikit-learn, but the stats do not depend on them).
* More importantly for YOU: implementing a test yourself is the fastest way
  to actually understand what it does. Each function's docstring explains the
  statistics in plain language first, then the formula. Read them together
  with docs/02_stats_primer.md.

MAP TO THE PROTOCOL (§10)
=========================
  wilson_ci                — 95% Wilson score interval on every accuracy
  binom_test_two_sided     — exact binomial test of accuracy vs chance (0.5)
  holm_bonferroni          — multiple-comparison correction across cells
  mcnemar_exact            — original vs paraphrased; judge vs stylometric
  spearman_rho             — scale trend (accuracy vs log params)
  cohen_kappa              — judge vs perplexity-rule agreement
  bootstrap_ci             — CI for kappa / AUROC (1,000 resamples)
  dprime_criterion         — IPP signal detection (d', criterion c)
  auroc                    — threshold-free discrimination (§9.6)
  power_exact_binomial     — the sample-size justification in the appendix
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

_ND = NormalDist()  # standard normal; gives us z-scores without scipy


# ---------------------------------------------------------------------------
# Proportions and their uncertainty
# ---------------------------------------------------------------------------

def wilson_ci(phat: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Plain language: you observed accuracy `phat` on `n` items. The Wilson
    interval is the range of true accuracies that are statistically
    compatible with that observation. Unlike the naive "normal approximation"
    interval (phat ± 1.96*sqrt(...)), Wilson behaves sensibly at small n and
    near 0/1 — it never leaves [0, 1]. That is why the protocol demands it.

    Formula (z = 1.96 for 95%):
        center = (phat + z^2/2n) / (1 + z^2/n)
        half   = z * sqrt( phat(1-phat)/n + z^2/(4n^2) ) / (1 + z^2/n)

    Example: 60% on n=150 -> roughly (0.52, 0.68); the same 60% on n=15
    -> roughly (0.36, 0.80). Same accuracy, very different evidence.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _ND.inv_cdf(1 - alpha / 2)
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_pmf(k: int, n: int, p: float) -> float:
    """P(K = k) for K ~ Binomial(n, p), computed in log-space so it doesn't
    underflow for large n."""
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    log_pmf = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log(1 - p)
    )
    return math.exp(log_pmf)


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (the same rule scipy uses).

    Plain language: if the judge were guessing (true accuracy = p = 0.5),
    how surprising is it to see k correct out of n? The p-value sums the
    probability of every outcome AT LEAST as extreme (= at most as probable)
    as the one observed. Small p-value -> guessing is a bad explanation.

    Definition of "as extreme": all outcomes i whose probability under the
    null is <= P(K=k), with a tiny tolerance for float rounding.
    """
    if n == 0:
        return 1.0
    pk = binom_pmf(k, n, p)
    tol = pk * 1e-12
    total = 0.0
    for i in range(n + 1):
        pi = binom_pmf(i, n, p)
        if pi <= pk + tol:
            total += pi
    return min(1.0, total)


def holm_bonferroni(named_pvals: list[tuple[str, float]], alpha: float = 0.05) -> list[dict]:
    """Holm–Bonferroni correction for multiple comparisons.

    Plain language: if you run 20 significance tests at alpha=.05, you expect
    one false positive by luck alone. Holm's method fixes this while being
    less brutal than plain Bonferroni: sort the p-values ascending; compare
    the smallest against alpha/m, the next against alpha/(m-1), and so on;
    stop at the first failure (everything after it also fails).

    Returns one dict per input, in the ORIGINAL order, with the adjusted
    p-value (monotone step-down: adj_i = max over j<=i of (m-j)*p_(j)) and a
    reject flag at the given alpha.
    """
    m = len(named_pvals)
    order = sorted(range(m), key=lambda i: named_pvals[i][1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * named_pvals[idx][1])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [
        {"name": name, "p": p, "p_holm": adjusted[i], "reject": adjusted[i] < alpha}
        for i, (name, p) in enumerate(named_pvals)
    ]


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test for paired binary outcomes.

    Plain language: the same items are measured twice (e.g. each pair judged
    once in the original condition and once paraphrased). Items where both
    conditions agree tell us nothing about the difference; the evidence lives
    entirely in the DISAGREEING items: b = correct only in condition 1,
    c = correct only in condition 2. Under "no difference", each disagreeing
    item is a fair coin flip between b and c, so the test is just an exact
    binomial test of b successes in b+c trials at p=0.5.
    """
    if b + c == 0:
        return 1.0
    return binom_test_two_sided(b, b + c, 0.5)


# ---------------------------------------------------------------------------
# Agreement and correlation
# ---------------------------------------------------------------------------

def cohen_kappa(x: list, y: list) -> float:
    """Cohen's kappa: agreement between two decision-makers, corrected for
    the agreement they would reach by luck.

    Plain language: if the judge picks 'self' 90% of the time and the
    perplexity rule picks 'self' 90% of the time, they will agree a lot even
    if they are unrelated. kappa subtracts that: kappa = (po - pe) / (1 - pe)
    where po is observed agreement and pe is chance agreement given each
    party's own base rates. kappa = 1 perfect, 0 chance-level, < 0 worse than
    chance. The protocol's threshold of interest is kappa > 0.4 (§2, H4).
    """
    assert len(x) == len(y) and len(x) > 0
    n = len(x)
    labels = sorted(set(x) | set(y))
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    pe = 0.0
    for lab in labels:
        pe += (sum(1 for a in x if a == lab) / n) * (sum(1 for b in y if b == lab) / n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _rank_with_ties(values: list[float]) -> list[float]:
    """Average ranks (1-based); tied values share the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # mean of ranks i+1 .. j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a - mx) ** 2 for a in x))
    vy = math.sqrt(sum((b - my) ** 2 for b in y))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation: Pearson correlation of the RANKS.

    Plain language: does y go up whenever x goes up, regardless of by how
    much? +1 = perfectly monotone increasing. Used for the scale trend
    (accuracy vs log10 params, n = 5 sizes) — reported as descriptive, since
    n=5 gives little power; the CI-per-point figure carries the argument (§10).
    """
    return pearson_r(_rank_with_ties(x), _rank_with_ties(y))


# ---------------------------------------------------------------------------
# Signal detection (IPP analysis, §9.7) and AUROC (§9.6)
# ---------------------------------------------------------------------------

def dprime_criterion(hits: int, n_signal: int, false_alarms: int, n_noise: int) -> dict:
    """Signal detection for the Yes/No protocol.

    Plain language: a model that answers "Yes" to EVERYTHING gets 100% on
    SELF items while knowing nothing. Signal detection separates real
    discrimination from that bias:
      hit rate  H = P(says Yes | text is SELF)
      false-alarm rate F = P(says Yes | text is not SELF)
      d'  = z(H) - z(F)   -> sensitivity: 0 = cannot tell at all
      c   = -(z(H)+z(F))/2 -> response bias: negative = trigger-happy "Yes"

    Rates of exactly 0 or 1 make z infinite, so we apply the standard
    log-linear (Hautus 1995) correction: add 0.5 to each count and 1 to each
    total before computing rates.
    """
    h = (hits + 0.5) / (n_signal + 1)
    f = (false_alarms + 0.5) / (n_noise + 1)
    zh, zf = _ND.inv_cdf(h), _ND.inv_cdf(f)
    return {
        "hit_rate": h,
        "fa_rate": f,
        "dprime": zh - zf,
        "criterion_c": -(zh + zf) / 2,
        "n_signal": n_signal,
        "n_noise": n_noise,
    }


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney) formulation.

    Plain language: take one random positive item and one random negative
    item; AUROC is the probability that the positive one has the higher
    score (ties count half). 0.5 = no discrimination, 1.0 = perfect. It is
    threshold-free, which is why the protocol reports it alongside accuracy:
    Schaeffer et al. (2023) showed 'emergence' can be an artifact of the
    hard threshold inside accuracy, and a smooth AUROC curve exposes that.
    """
    if not scores_pos or not scores_neg:
        return float("nan")
    combined = scores_pos + scores_neg
    ranks = _rank_with_ties(combined)
    r_pos = sum(ranks[: len(scores_pos)])
    n1, n2 = len(scores_pos), len(scores_neg)
    u = r_pos - n1 * (n1 + 1) / 2
    return u / (n1 * n2)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(stat_fn, items: list, n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic.

    Plain language: we don't know the sampling distribution of complicated
    statistics like kappa or AUROC, so we simulate it — resample the items
    WITH replacement 1,000 times, recompute the statistic each time, and
    take the middle 95% of those values as the confidence interval.

    `items` should be the independent units (here: pairs, so that both runs
    of a counterbalanced pair stay together — resampling correlated rows
    independently would fake extra precision). `stat_fn(list) -> float`.
    """
    rng = random.Random(seed)
    n = len(items)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        try:
            stats.append(stat_fn(sample))
        except Exception:
            continue
    if not stats:
        return (float("nan"), float("nan"))
    stats.sort()
    lo_i = int((alpha / 2) * len(stats))
    hi_i = min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))
    return (stats[lo_i], stats[hi_i])


# ---------------------------------------------------------------------------
# Power analysis (appendix, §10 "Sample size justification")
# ---------------------------------------------------------------------------

def power_exact_binomial(n: int, p_alt: float, p0: float = 0.5, alpha: float = 0.05) -> float:
    """Power of the exact two-sided binomial test.

    Plain language: suppose the judge's TRUE accuracy is p_alt (say 0.615).
    If we run n items, what is the probability our test actually comes out
    significant? That probability is the power; 80% is the conventional
    target. The protocol's claim — n=150 has ~80% power to detect 0.615 —
    is reproduced by this exact computation (see tests/).

    Method: find every outcome k that the test would call significant under
    the null, then sum the probability of those outcomes under p_alt.
    """
    power = 0.0
    for k in range(n + 1):
        if binom_test_two_sided(k, n, p0) < alpha:
            power += binom_pmf(k, n, p_alt)
    return power


# ---------------------------------------------------------------------------
# Small helpers used by 06_stats.py
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def fmt_p(p: float) -> str:
    """Report tiny p-values as '< .001' the way papers do."""
    if p != p:  # NaN
        return "-"
    return "<.001" if p < 0.001 else f"{p:.3f}"
