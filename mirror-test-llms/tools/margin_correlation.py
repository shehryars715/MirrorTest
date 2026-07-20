"""
margin_correlation.py — reviewer-proof version of the dissociation test.

Cohen's kappa saturates toward zero when either decision-maker is
near-constant (which happens twice in our data: the perplexity rule almost
always answers "self" at >=3B, and two judges almost always answer one
letter). This script computes the statistic that does NOT saturate:
Spearman correlation between two CONTINUOUS per-pair preferences —

  judge margin  = position-debiased letter preference
                  = ( (logpA - logpB)@self_A  -  (logpA - logpB)@self_B ) / 2
  rule margin   = nll(foil) - nll(self)   (positive = self more likely)

If verbal choices consult likelihood at all, these correlate; rho ~ 0 makes
the dissociation airtight. Add the result to the paper's §4 dissociation
paragraph ("margin-level Spearman rho = ... across judges").

    python tools/margin_correlation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import read_jsonl, JUDGMENTS_DIR, BASELINES_DIR  # noqa: E402
from stats_utils import spearman_rho, bootstrap_ci  # noqa: E402

JUDGES = ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct",
          "qwen2.5-3b-instruct", "qwen2.5-7b-instruct", "qwen2.5-14b-instruct"]

for j in JUDGES:
    runs = [r for r in read_jsonl(JUDGMENTS_DIR / f"ppp__{j}.jsonl")
            if r["condition"] == "core" and r["phrasing"] == 0]
    by_pair: dict = {}
    for r in runs:
        by_pair.setdefault(r["pair_id"], {})[r["order"]] = r["logp_A"] - r["logp_B"]
    ppl = {r["pair_id"]: r["nll_foil"] - r["nll_self"]
           for r in read_jsonl(BASELINES_DIR / f"ppl__{j}.jsonl")
           if r["condition"] == "core"}
    pts = [((d["self_A"] - d["self_B"]) / 2, ppl[p])
           for p, d in by_pair.items()
           if "self_A" in d and "self_B" in d and p in ppl]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    rho = spearman_rho(xs, ys)
    lo, hi = bootstrap_ci(
        lambda s: spearman_rho([a for a, _ in s], [b for _, b in s]), pts, n_boot=1000)
    print(f"{j}: margin-margin Spearman rho = {rho:+.3f} [{lo:+.3f}, {hi:+.3f}] "
          f"(n={len(pts)} pairs)")
