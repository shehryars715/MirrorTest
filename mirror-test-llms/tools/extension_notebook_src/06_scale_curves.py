# ========================= SCALE CURVES FIGURE =============================
# Qwen2.5 vs Gemma-2, three panels vs log judge size:
#   (1) self-recognition PPP accuracy (pooled over LLM foils, human excluded)
#   (2) lower-perplexity-rule accuracy (implicit self-information)
#   (3) human-vs-machine AUROC
# Two family series only, so identity is carried by color AND marker AND
# linestyle (CVD-safe blue-circle-solid vs amber-triangle-dashdot; grayscale
# values 119 vs 174) with a legend always present — never color alone. Chance
# line at 0.5. Reuses the repo's axis chrome for visual identity with the
# paper's Figure 1.
# ===========================================================================

from pathlib import Path

FAMILY_STYLE = {                       # (hex, marker, linestyle, label)
    "qwen":  ("#2a78d6", "o", "-",  "Qwen2.5"),
    "gemma": ("#eda100", "^", "-.", "Gemma-2"),
}
PANELS = [
    ("explicit_ppp_acc", ("explicit_ci_lo", "explicit_ci_hi"),
     "Self-recognition (PPP acc)"),
    ("implicit_ppl_acc", None, "Perplexity-rule acc"),
    ("auroc_human", ("auroc_human_lo", "auroc_human_hi"),
     "Human-vs-machine AUROC"),
]


def _wilson_band(acc, n):
    lo, hi = stats_utils.wilson_ci(acc, n, CFG["stats"]["alpha"])
    return lo, hi


def build_scale_curves(disso_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure] matplotlib unavailable ({e}); skipping scale_curves.png")
        return None

    by_family = {}
    for r in disso_rows:
        by_family.setdefault(r["family"], []).append(r)
    for fam in by_family:
        by_family[fam].sort(key=lambda r: r["params_b"]
                            if isinstance(r["params_b"], (int, float)) else 0.0)
    fams = [f for f in ("qwen", "gemma") if by_family.get(f)]
    if not fams:
        print("[figure] no family rows yet; skipping scale_curves.png")
        return None

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                         "text.color": mod_stats.INK, "axes.labelcolor": mod_stats.INK})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    all_sizes = sorted({r["params_b"] for rows in by_family.values() for r in rows})

    for panel, (metric, ci, title) in enumerate(PANELS):
        ax = axes[panel]
        for fam in fams:
            color, marker, ls, label = FAMILY_STYLE[fam]
            rows = [r for r in by_family[fam]
                    if isinstance(r.get(metric), float)
                    and (r["params_b"] not in (None, ""))]
            if not rows:
                continue
            xs = [r["params_b"] for r in rows]
            ys = [r[metric] for r in rows]
            if ci is None:                       # implicit acc: Wilson from n
                bands = [_wilson_band(r[metric], r["n_llm_pairs"]) for r in rows]
                lo = [max(0.0, y - b[0]) for y, b in zip(ys, bands)]
                hi = [max(0.0, b[1] - y) for y, b in zip(ys, bands)]
            else:
                lo = [max(0.0, r[metric] - r[ci[0]]) for r in rows
                      if isinstance(r.get(ci[0]), float)]
                hi = [max(0.0, r[ci[1]] - r[metric]) for r in rows
                      if isinstance(r.get(ci[1]), float)]
                if len(lo) != len(ys):
                    lo = hi = None
            ax.errorbar(xs, ys, yerr=([lo, hi] if lo is not None else None),
                        color=color, marker=marker, linestyle=(ls if len(xs) > 1 else "None"),
                        linewidth=1.5, markersize=5, markeredgecolor=mod_stats.INK,
                        markeredgewidth=0.4, capsize=2, elinewidth=0.8, label=label)
        ax.axhline(0.5, color=mod_stats.MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(ax.get_xlim()[1], 0.505, "chance", ha="right", va="bottom",
                fontsize=6, color=mod_stats.MUTED)
        ax.set_xscale("log")
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([f"{s:g}" for s in all_sizes])
        ax.minorticks_off()
        ax.set_xlabel("Judge parameters (B, log scale)")
        ax.set_title(title, fontsize=8, color=mod_stats.INK)
        ax.set_ylim(0.3, 1.0)
        mod_stats._style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=8)

    targets = [Path(WORKING_ROOT) / "scale_curves.png",
               Path(OUTPUT_DIR) / "scale_curves.png"]
    fig_dir = utils.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    targets.append(fig_dir / "scale_curves_extended.png")
    for t in targets:
        fig.savefig(t, dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "scale_curves_extended.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] scale_curves.png -> {targets[0]} (families: {', '.join(fams)})")
    return str(targets[0])


SCALE_CURVE_PATH = build_scale_curves(EXT_DISSO_ROWS)
