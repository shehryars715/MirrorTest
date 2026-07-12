"""
06_stats.py — the full analysis: every table, test, and figure in the
statistical analysis plan (protocol §10), computed from the raw judgment
files. Pure CPU; matplotlib is the only non-stdlib dependency (figures are
skippable with --skip-figures).

WHAT COMES OUT (results/tables/ and results/figures/)
=====================================================
  main_ppp.csv/.md      Table 1: per judge x foil (domains pooled) — n, accuracy,
                        95% Wilson CI, exact binomial p vs chance, Holm-corrected
                        significance, AUROC (+bootstrap CI), position-consistency,
                        stylometric pairwise accuracy, perplexity-rule accuracy,
                        Cohen's kappa (judge vs perplexity rule), McNemar p
                        (judge vs stylometric on the same items).
  table1_main.tex       The same table as ready-to-paste LaTeX.
  ppp_by_domain.csv     The un-pooled version.
  placebo.csv           SELF-vs-SELF position-bias check (§9.1).
  ipp.csv               Yes/No protocol: hit rate, false alarms, d', c, AUROC (§9.7).
  paraphrase.csv        Original vs paraphrased, McNemar p (§9.4).
  phrasing_robustness.csv  mean +- range across the 3 instruction phrasings (§8.5).
  refusals.csv          free-text refusal / unparseable rates (§9.7).
  scale_trend.csv       Spearman rho of accuracy vs log10(params) per foil (§9.6).
  base_vs_instruct.csv  §9.5 ablation.
  length_stats.csv      residual SELF-vs-FOIL length differences (§18).
  power_analysis.csv    the appendix sample-size justification (§10).
  all_stats.json        every number above, as one machine-readable blob.
  fig1_scale_curve      accuracy AND AUROC vs parameters (per foil, 95% CIs).
  fig2_paraphrase       original vs paraphrased accuracy bars + McNemar p.
  fig3_placebo          placebo position-bias dot plot (appendix).

KEY ANALYSIS DEFINITIONS (documented once, used everywhere)
===========================================================
* item score: each pair is judged in both orders; score = mean of the two
  runs' correctness in {0, 0.5, 1}. 0.5 = the judge picked the same POSITION
  twice (position-driven, not authorship-driven).
* reported accuracy = mean item score. Wilson CI uses n = number of items.
* significance vs chance: exact binomial test restricted to DECISIVE items
  (score 0 or 1) — ties carry no evidence about authorship either way, and
  dropping them is the standard sign-test treatment.
* "consistently correct" (binary, for McNemar): item score > 0.5, i.e.
  correct under both orders. The same coding is applied to both conditions
  being compared, so the comparison is fair.
* PPP AUROC (threshold-free, §9.6): over runs, score = logp(A) - logp(B),
  positive class = runs where SELF was shown in position A. If the judge has
  any graded self-signal, this score separates the classes; position bias
  alone cannot, because both classes share every position. CI by
  bootstrapping pairs (both orders resampled together).

USAGE
=====
    python src/06_stats.py                 # everything
    python src/06_stats.py --skip-figures  # tables only (no matplotlib needed)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    BASELINES_DIR, FIGURES_DIR, JUDGMENTS_DIR, PAIRS_DIR, TABLES_DIR,
    load_config, now_iso, read_jsonl,
)
from stats_utils import (  # noqa: E402
    auroc, binom_test_two_sided, bootstrap_ci, cohen_kappa, dprime_criterion,
    fmt_p, holm_bonferroni, mcnemar_exact, mean, power_exact_binomial,
    spearman_rho, wilson_ci,
)

# --------------------------------------------------------------------------
# Figure styling — palette + chrome from the validated reference palette
# (dataviz method). Categorical slots in fixed order; slot 4 (green) is
# swapped for slot 5 (violet) because green's grayscale value collides with
# blue's — the paper must stay readable in grayscale, and each series also
# carries a distinct marker + linestyle so identity never rides on color.
# --------------------------------------------------------------------------
SERIES_STYLE = [  # (hex, marker, linestyle) — assigned to foils in config order
    ("#2a78d6", "o", "-"),    # slot 1 blue    (gray 119)
    ("#1baf7a", "s", "--"),   # slot 2 aqua    (gray 152)
    ("#eda100", "^", "-."),   # slot 3 yellow  (gray 174)
    ("#4a3aa7", "D", ":"),    # slot 5 violet  (gray 77)
]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COND_FILL = {"original": "#2a78d6", "paraphrased": "#86b6ef"}  # same hue, 2 steps


# ==========================================================================
# Loading and shaping the raw judgments
# ==========================================================================

def load_ppp_runs() -> list[dict]:
    runs = []
    for path in sorted(JUDGMENTS_DIR.glob("ppp__*.jsonl")):
        runs.extend(read_jsonl(path))
    return runs


def items_from_runs(runs: list[dict]) -> dict:
    """Group counterbalanced runs into items: key = (judge, foil, domain,
    condition, phrasing, pair_id) -> {score, orders seen, per-run rows}."""
    by_item: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        key = (r["judge"], r["foil"], r["domain"], r["condition"],
               r["phrasing"], r["pair_id"])
        by_item[key].append(r)
    items = {}
    for key, rows in by_item.items():
        corrects = [r["correct"] for r in rows if r["correct"] is not None]
        items[key] = {
            "score": mean([1.0 if c else 0.0 for c in corrects]) if corrects else None,
            "n_orders": len(rows),
            "rows": rows,
        }
    return items


def cell_stats(items: dict, judge: str, foil, condition: str = "core",
               phrasing: int = 0, domain: str | None = None,
               alpha: float = 0.05) -> dict | None:
    """All §10 primary-endpoint statistics for one cell (foil=None pools foils)."""
    scores, run_rows, pair_groups = [], [], []
    for (j, f, d, c, ph, pid), it in items.items():
        if j != judge or c != condition or ph != phrasing:
            continue
        if foil is not None and f != foil:
            continue
        if domain is not None and d != domain:
            continue
        if it["score"] is None:
            continue
        scores.append(it["score"])
        run_rows.extend(it["rows"])
        pair_groups.append(it["rows"])
    n = len(scores)
    if n == 0:
        return None
    acc = mean(scores)
    lo, hi = wilson_ci(acc, n, alpha)
    decisive = [s for s in scores if s in (0.0, 1.0)]
    k = sum(1 for s in decisive if s == 1.0)
    p = binom_test_two_sided(k, len(decisive)) if decisive else float("nan")
    consistency = len(decisive) / n

    # AUROC on the continuous logprob margin (see module docstring).
    pos = [r["logp_A"] - r["logp_B"] for r in run_rows if r["self_position"] == "A"]
    neg = [r["logp_A"] - r["logp_B"] for r in run_rows if r["self_position"] == "B"]
    auc = auroc(pos, neg)

    def auc_of(groups):
        rows = [r for g in groups for r in g]
        p_ = [r["logp_A"] - r["logp_B"] for r in rows if r["self_position"] == "A"]
        n_ = [r["logp_A"] - r["logp_B"] for r in rows if r["self_position"] == "B"]
        return auroc(p_, n_)

    auc_lo, auc_hi = bootstrap_ci(auc_of, pair_groups, n_boot=500)
    return {
        "judge": judge, "foil": foil or "ALL", "domain": domain or "pooled",
        "n_items": n, "acc": acc, "ci_lo": lo, "ci_hi": hi,
        "n_decisive": len(decisive), "k_decisive": k, "p_binomial": p,
        "consistency": consistency, "auroc": auc,
        "auroc_ci_lo": auc_lo, "auroc_ci_hi": auc_hi,
    }


# ==========================================================================
# Baselines: joins with the judge's items
# ==========================================================================

def load_stylo() -> tuple[dict, dict]:
    """Returns ({(judge,foil,domain): summary}, {(judge,foil,domain): {pair_id: chose_self}})."""
    summaries, pair_decisions = {}, {}
    for path in sorted(BASELINES_DIR.glob("stylo__*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        summaries[(s["judge"], s["foil"], s["domain"])] = s
    for path in sorted(BASELINES_DIR.glob("stylo_pairs__*.jsonl")):
        _, judge, foil, domain = path.stem.split("__")
        pair_decisions[(judge, foil, domain)] = {
            r["pair_id"]: r["stylo_chose_self"] for r in read_jsonl(path)}
    return summaries, pair_decisions


def load_ppl() -> dict:
    """{(judge, pair_id, condition): rule_chose_self}"""
    out = {}
    for path in sorted(BASELINES_DIR.glob("ppl__*.jsonl")):
        for r in read_jsonl(path):
            out[(r["judge"], r["pair_id"], r["condition"])] = r["rule_chose_self"]
    return out


def judge_vs_stylo_mcnemar(items: dict, judge: str, foil: str,
                           stylo_pairs: dict) -> dict | None:
    """McNemar on the same items: judge 'consistently correct' vs the
    stylometric pair decision (protocol §10)."""
    b = c = n_both = 0
    used = 0
    for (j, f, d, cond, ph, pid), it in items.items():
        if j != judge or f != foil or cond != "core" or ph != 0 or it["score"] is None:
            continue
        dec = stylo_pairs.get((j, f, d), {}).get(pid)
        if dec is None:
            continue
        judge_ok = it["score"] > 0.5
        stylo_ok = bool(dec)
        used += 1
        if judge_ok and not stylo_ok:
            b += 1
        elif stylo_ok and not judge_ok:
            c += 1
        else:
            n_both += 1
    if used == 0:
        return None
    return {"n": used, "judge_only": b, "stylo_only": c, "p_mcnemar": mcnemar_exact(b, c)}


def judge_vs_ppl_kappa(items: dict, judge: str, foil: str, ppl: dict,
                       n_boot: int = 1000) -> dict | None:
    """Cohen's kappa between the judge's per-run choice (self/foil) and the
    perplexity rule's choice on the same pairs, with a bootstrap CI over
    pairs (protocol §9.3, §10)."""
    groups = []  # one entry per pair: list of (judge_chose_self, rule_chose_self)
    acc_rule = []
    for (j, f, d, cond, ph, pid), it in items.items():
        if j != judge or f != foil or cond != "core" or ph != 0:
            continue
        rule = ppl.get((judge, pid, "core"))
        if rule is None:
            continue
        rows = [(r["choice"] == r["self_position"], bool(rule)) for r in it["rows"]]
        groups.append(rows)
        acc_rule.append(1.0 if rule else 0.0)
    if not groups:
        return None
    flat = [t for g in groups for t in g]
    kappa = cohen_kappa([a for a, _ in flat], [b for _, b in flat])

    def kappa_of(gs):
        fl = [t for g in gs for t in g]
        return cohen_kappa([a for a, _ in fl], [b for _, b in fl])

    lo, hi = bootstrap_ci(kappa_of, groups, n_boot=n_boot)
    return {"kappa": kappa, "kappa_ci_lo": lo, "kappa_ci_hi": hi,
            "ppl_rule_acc": mean(acc_rule), "n_pairs": len(groups)}


# ==========================================================================
# Writers
# ==========================================================================

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[table] {path}")


def write_md(path: Path, rows: list[dict], title: str) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table1_tex(path: Path, rows: list[dict]) -> None:
    """Emit Table 1 as paste-ready LaTeX (booktabs)."""
    lines = [
        "% Auto-generated by src/06_stats.py - do not edit by hand.",
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{tabular}{llrccccc}",
        "\\toprule",
        "Judge & Foil & $n$ & Acc. [95\\% CI] & AUROC & Stylo. & PPL rule & $\\kappa$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        if r["foil"] == "ALL":
            continue
        star = "$^{*}$" if r.get("reject_holm") else ""
        acc = f"{r['acc']:.3f}{star} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
        stylo = f"{r['stylo_pair_acc']:.3f}" if isinstance(r.get("stylo_pair_acc"), float) else "--"
        ppl = f"{r['ppl_rule_acc']:.3f}" if isinstance(r.get("ppl_rule_acc"), float) else "--"
        kap = f"{r['kappa']:.2f}" if isinstance(r.get("kappa"), float) else "--"
        lines.append(f"{r['judge']} & {r['foil']} & {r['n_items']} & {acc} & "
                     f"{r['auroc']:.3f} & {stylo} & {ppl} & {kap} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Order-averaged PPP accuracy per judge and foil (domains pooled), "
        "with 95\\% Wilson CIs; $^{*}$ = significant vs.\\ chance after "
        "Holm correction. Stylo.\\ = pairwise accuracy of the character n-gram "
        "classifier on the same items; PPL rule = lower-perplexity rule; "
        "$\\kappa$ = judge--perplexity-rule agreement.}",
        "\\label{tab:main}", "\\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] {path}")


# ==========================================================================
# Figures (matplotlib; every figure also saved as PDF for LaTeX)
# ==========================================================================

def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_scale_curve(main_rows: list[dict], params_of: dict, foil_order: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                         "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), constrained_layout=True)
    for panel, metric, cis in (
        (0, "acc", ("ci_lo", "ci_hi")),
        (1, "auroc", ("auroc_ci_lo", "auroc_ci_hi")),
    ):
        ax = axes[panel]
        for si, foil in enumerate(foil_order):
            color, marker, ls = SERIES_STYLE[si % len(SERIES_STYLE)]
            rows = sorted(
                [r for r in main_rows if r["foil"] == foil and r["judge"] in params_of],
                key=lambda r: params_of[r["judge"]])
            if not rows:
                continue
            xs = [params_of[r["judge"]] for r in rows]
            ys = [r[metric] for r in rows]
            err = [[max(0.0, r[metric] - r[cis[0]]) for r in rows],
                   [max(0.0, r[cis[1]] - r[metric]) for r in rows]]
            ax.errorbar(xs, ys, yerr=err, color=color, marker=marker, linestyle=ls,
                        linewidth=1.4, markersize=4.5, markeredgecolor=INK,
                        markeredgewidth=0.4, capsize=2, elinewidth=0.8,
                        label=foil.replace("-instruct", "").replace("-it", ""))
        ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(ax.get_xlim()[1], 0.505, "chance", ha="right", va="bottom",
                fontsize=6, color=MUTED)
        ax.set_xscale("log")
        # plain ticks at the actual model sizes (0.5, 1.5, 3, 7, 14), not 10^x
        sizes = sorted({params_of[r["judge"]] for r in main_rows if r["judge"] in params_of})
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:g}" for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Judge parameters (B, log scale)")
        ax.set_ylabel("Accuracy" if metric == "acc" else "AUROC")
        ax.set_ylim(0.3, 1.0)
        _style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig1_scale_curve.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {FIGURES_DIR / 'fig1_scale_curve.(png|pdf)'}")


def fig_paraphrase(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.2, 2.5), constrained_layout=True)
    domains = [r["domain"] for r in rows]
    x = range(len(domains))
    w = 0.38
    for off, (cond, acc_k, lo_k, hi_k, hatch) in enumerate((
        ("original", "acc_orig", "orig_ci_lo", "orig_ci_hi", None),
        ("paraphrased", "acc_para", "para_ci_lo", "para_ci_hi", "///"),
    )):
        xs = [i + (off - 0.5) * (w + 0.04) for i in x]
        ys = [r[acc_k] for r in rows]
        err = [[max(0.0, r[acc_k] - r[lo_k]) for r in rows],
               [max(0.0, r[hi_k] - r[acc_k]) for r in rows]]
        ax.bar(xs, ys, width=w, color=COND_FILL[cond], hatch=hatch,
               edgecolor="white", linewidth=0.8, label=cond)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8,
                    capsize=2)
    for i, r in enumerate(rows):
        ax.text(i, 0.97, f"p={fmt_p(r['p_mcnemar'])}", ha="center", fontsize=6,
                color=MUTED)
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_xticks(list(x), domains)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("PPP accuracy")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig2_paraphrase.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {FIGURES_DIR / 'fig2_paraphrase.(png|pdf)'}")


def fig_placebo(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.2, 2.2), constrained_layout=True)
    ys = range(len(rows))
    for i, r in enumerate(rows):
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color="#2a78d6", linewidth=1.2)
        ax.plot(r["chose_A_rate"], i, "o", color="#2a78d6", markersize=4.5,
                markeredgecolor=INK, markeredgewidth=0.4)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_yticks(list(ys), [r["judge"].replace("-instruct", "") for r in rows])
    ax.set_xlabel('P(chose position A) on SELF-vs-SELF pairs')
    ax.set_xlim(0.2, 0.8)
    _style_axes(ax)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig3_placebo.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {FIGURES_DIR / 'fig3_placebo.(png|pdf)'}")


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="All tables, tests, figures (§10).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    alpha = cfg["stats"]["alpha"]
    n_boot = cfg["stats"]["bootstrap_n"]
    judges = [m["key"] for m in cfg["judges"]]
    params_of = {m["key"]: m["params_b"] for m in cfg["judges"]}
    foil_order = [m["key"] for m in cfg["foils"]]
    base_keys = [m["key"] for m in cfg.get("base_judges", [])]

    runs = load_ppp_runs()
    if not runs:
        sys.exit("[error] no judgments in results/judgments/ - run 03_judge_ppp.py first")
    items = items_from_runs(runs)
    stylo_sum, stylo_pairs = load_stylo()
    ppl = load_ppl()
    blob: dict = {"created_at": now_iso(), "n_runs": len(runs)}

    # ---------- Table 1: main PPP (pooled domains) + baselines -------------
    main_rows, pvals = [], []
    for judge in judges:
        for foil in foil_order + [None]:  # None = pooled-over-foils headline
            st = cell_stats(items, judge, foil, alpha=alpha)
            if st is None:
                continue
            if foil is not None:
                pvals.append((f"{judge}|{foil}", st["p_binomial"]))
                # stylometric: pool the per-domain summaries by item count
                accs, ns = [], []
                for (j, f, d), s in stylo_sum.items():
                    if j == judge and f == foil:
                        accs.append(s["acc_pairwise_char_ngram"] * s["n_pairs"])
                        ns.append(s["n_pairs"])
                st["stylo_pair_acc"] = (sum(accs) / sum(ns)) if ns else ""
                mc = judge_vs_stylo_mcnemar(items, judge, foil, stylo_pairs)
                st["p_mcnemar_vs_stylo"] = mc["p_mcnemar"] if mc else ""
                kp = judge_vs_ppl_kappa(items, judge, foil, ppl, n_boot)
                if kp:
                    st.update({"ppl_rule_acc": kp["ppl_rule_acc"], "kappa": kp["kappa"],
                               "kappa_ci_lo": kp["kappa_ci_lo"],
                               "kappa_ci_hi": kp["kappa_ci_hi"]})
                else:
                    st.update({"ppl_rule_acc": "", "kappa": "",
                               "kappa_ci_lo": "", "kappa_ci_hi": ""})
            main_rows.append(st)
    # Holm across the judge x foil family (§10 multiple comparisons)
    holm = {h["name"]: h for h in holm_bonferroni(pvals, alpha)}
    for r in main_rows:
        h = holm.get(f"{r['judge']}|{r['foil']}")
        r["p_holm"] = h["p_holm"] if h else ""
        r["reject_holm"] = h["reject"] if h else ""
    write_csv(TABLES_DIR / "main_ppp.csv", main_rows)
    write_md(TABLES_DIR / "main_ppp.md", main_rows, "Table 1 - Main PPP results")
    write_table1_tex(TABLES_DIR / "table1_main.tex", main_rows)
    blob["main_ppp"] = main_rows

    # ---------- per-domain table -------------------------------------------
    dom_rows = []
    for judge in judges:
        for foil in foil_order:
            for domain in ("news", "dolly", "wp"):
                st = cell_stats(items, judge, foil, domain=domain, alpha=alpha)
                if st:
                    dom_rows.append(st)
    write_csv(TABLES_DIR / "ppp_by_domain.csv", dom_rows)
    blob["ppp_by_domain"] = dom_rows

    # ---------- placebo (§9.1) ---------------------------------------------
    placebo_rows = []
    for judge in judges:
        rows = [r for r in runs if r["judge"] == judge and r["condition"] == "placebo"]
        if not rows:
            continue
        rate = mean([1.0 if r["choice"] == "A" else 0.0 for r in rows])
        lo, hi = wilson_ci(rate, len(rows), alpha)
        placebo_rows.append({"judge": judge, "n_runs": len(rows),
                             "chose_A_rate": rate, "ci_lo": lo, "ci_hi": hi,
                             "flag": "POSITION BIAS" if (lo > 0.5 or hi < 0.5) else "ok"})
    write_csv(TABLES_DIR / "placebo.csv", placebo_rows)
    blob["placebo"] = placebo_rows

    # ---------- IPP (§9.7) --------------------------------------------------
    ipp_rows = []
    for path in sorted(JUDGMENTS_DIR.glob("ipp__*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        judge = rows[0]["judge"]
        hits = sum(1 for r in rows if r["is_self"] and r["choice"] == "Yes")
        n_sig = sum(1 for r in rows if r["is_self"])
        fas = sum(1 for r in rows if not r["is_self"] and r["choice"] == "Yes")
        n_noise = len(rows) - n_sig
        sd = dprime_criterion(hits, n_sig, fas, n_noise)
        pos = [r["logp_yes"] - r["logp_no"] for r in rows if r["is_self"]]
        neg = [r["logp_yes"] - r["logp_no"] for r in rows if not r["is_self"]]
        auc = auroc(pos, neg)
        lo, hi = bootstrap_ci(
            lambda rs: auroc([x["logp_yes"] - x["logp_no"] for x in rs if x["is_self"]],
                             [x["logp_yes"] - x["logp_no"] for x in rs if not x["is_self"]]),
            rows, n_boot=500)
        ipp_rows.append({"judge": judge, "n_items": len(rows),
                         "acc": mean([1.0 if r["correct"] else 0.0 for r in rows]),
                         "hit_rate": sd["hit_rate"], "fa_rate": sd["fa_rate"],
                         "dprime": sd["dprime"], "criterion_c": sd["criterion_c"],
                         "auroc": auc, "auroc_ci_lo": lo, "auroc_ci_hi": hi})
    write_csv(TABLES_DIR / "ipp.csv", ipp_rows)
    blob["ipp"] = ipp_rows

    # ---------- paraphrase attack (§9.4) ------------------------------------
    para_rows = []
    pj, pf = cfg["paraphrase"]["judge"], cfg["paraphrase"]["foil"]
    for domain in ("news", "dolly", "wp"):
        orig, para = {}, {}
        for (j, f, d, cond, ph, pid), it in items.items():
            if j == pj and f == pf and d == domain and ph == 0 and it["score"] is not None:
                if cond == "core":
                    orig[pid] = it["score"]
                elif cond == "paraphrase":
                    para[pid] = it["score"]
        common = sorted(set(orig) & set(para))
        if not common:
            continue
        b = sum(1 for pid in common if orig[pid] > 0.5 and para[pid] <= 0.5)
        c = sum(1 for pid in common if para[pid] > 0.5 and orig[pid] <= 0.5)
        ao, apa = mean([orig[p] for p in common]), mean([para[p] for p in common])
        olo, ohi = wilson_ci(ao, len(common), alpha)
        plo, phi = wilson_ci(apa, len(common), alpha)
        para_rows.append({"judge": pj, "foil": pf, "domain": domain,
                          "n_pairs": len(common), "acc_orig": ao,
                          "orig_ci_lo": olo, "orig_ci_hi": ohi, "acc_para": apa,
                          "para_ci_lo": plo, "para_ci_hi": phi,
                          "flips_to_wrong": b, "flips_to_right": c,
                          "p_mcnemar": mcnemar_exact(b, c)})
    write_csv(TABLES_DIR / "paraphrase.csv", para_rows)
    blob["paraphrase"] = para_rows

    # ---------- phrasing robustness (§8.5) -----------------------------------
    phr_rows = []
    for judge in judges:
        for foil in foil_order:
            accs = {}
            for ph in (0, 1, 2):
                st = cell_stats(items, judge, foil, phrasing=ph, alpha=alpha)
                if st:
                    accs[ph] = st["acc"]
            if len(accs) >= 2:
                phr_rows.append({"judge": judge, "foil": foil,
                                 "n_phrasings": len(accs),
                                 "acc_mean": mean(list(accs.values())),
                                 "acc_min": min(accs.values()),
                                 "acc_max": max(accs.values()),
                                 "range": max(accs.values()) - min(accs.values())})
    write_csv(TABLES_DIR / "phrasing_robustness.csv", phr_rows)
    blob["phrasing_robustness"] = phr_rows

    # ---------- refusal accounting (§9.7) ------------------------------------
    ref_rows = []
    for judge in judges + base_keys:
        logged = [r for r in runs if r["judge"] == judge and r.get("freetext") is not None]
        if not logged:
            continue
        n = len(logged)
        ref_rows.append({
            "judge": judge, "n_freetext_logged": n,
            "refusal_rate": sum(1 for r in logged if r["freetext_status"] == "refusal") / n,
            "unparseable_rate": sum(1 for r in logged if r["freetext_status"] == "unparseable") / n,
            "parse_agrees_with_logprob": mean(
                [1.0 if (r["freetext_status"] == "ok"
                         and (r["freetext"] or "").strip().upper().startswith(r["choice"]))
                 else 0.0 for r in logged if r["freetext_status"] == "ok"] or [float("nan")]),
        })
    write_csv(TABLES_DIR / "refusals.csv", ref_rows)
    blob["refusals"] = ref_rows

    # ---------- scale trend (§9.6) -------------------------------------------
    trend_rows = []
    for foil in foil_order:
        pts = [(params_of[r["judge"]], r["acc"], r["auroc"]) for r in main_rows
               if r["foil"] == foil and r["judge"] in params_of]
        if len(pts) >= 3:
            xs = [math.log10(p) for p, _, _ in pts]
            trend_rows.append({"foil": foil, "n_sizes": len(pts),
                               "spearman_rho_acc": spearman_rho(xs, [a for _, a, _ in pts]),
                               "spearman_rho_auroc": spearman_rho(xs, [u for _, _, u in pts])})
    write_csv(TABLES_DIR / "scale_trend.csv", trend_rows)
    blob["scale_trend"] = trend_rows

    # ---------- base vs instruct (§9.5) --------------------------------------
    bvi_rows = []
    for bkey in base_keys:
        sibling = next((m.get("pairs_of") for m in cfg["base_judges"] if m["key"] == bkey), None)
        for foil in foil_order:
            st_b = cell_stats(items, bkey, foil, alpha=alpha)
            st_i = cell_stats(items, sibling, foil, alpha=alpha) if sibling else None
            if st_b:
                bvi_rows.append({"base_judge": bkey, "foil": foil,
                                 "acc_base": st_b["acc"], "n_base": st_b["n_items"],
                                 "auroc_base": st_b["auroc"],
                                 "acc_instruct": st_i["acc"] if st_i else "",
                                 "auroc_instruct": st_i["auroc"] if st_i else ""})
    write_csv(TABLES_DIR / "base_vs_instruct.csv", bvi_rows)
    blob["base_vs_instruct"] = bvi_rows

    # ---------- residual length stats (§18) -----------------------------------
    len_rows = []
    for path in sorted(PAIRS_DIR.glob("ppp__*.jsonl")):
        pairs = read_jsonl(path)
        if not pairs:
            continue
        _, judge, foil, domain = path.stem.split("__")
        len_rows.append({"judge": judge, "foil": foil, "domain": domain,
                         "n_pairs": len(pairs),
                         "mean_words_self": mean([p["n_words_self"] for p in pairs]),
                         "mean_words_foil": mean([p["n_words_foil"] for p in pairs])})
    write_csv(TABLES_DIR / "length_stats.csv", len_rows)

    # ---------- power analysis (appendix, §10) --------------------------------
    power_rows = [
        {"n": n, "true_acc": p, "power": power_exact_binomial(n, p)}
        for n, p in ((150, 0.615), (450, 0.567), (780, 0.55))
    ]
    write_csv(TABLES_DIR / "power_analysis.csv", power_rows)
    blob["power_analysis"] = power_rows

    with open(TABLES_DIR / "all_stats.json", "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str)
    print(f"[table] {TABLES_DIR / 'all_stats.json'}")

    # ---------- figures --------------------------------------------------------
    if not args.skip_figures:
        try:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            per_foil = [r for r in main_rows if r["foil"] != "ALL"]
            if per_foil:
                fig_scale_curve(per_foil, params_of, foil_order)
            if para_rows:
                fig_paraphrase(para_rows)
            if placebo_rows:
                fig_placebo(placebo_rows)
        except ImportError:
            print("[warn] matplotlib not installed - figures skipped "
                  "(pip install matplotlib, or use --skip-figures to silence this)")
    print("[done] all statistics written. See results/tables/ and results/figures/.")


if __name__ == "__main__":
    main()
