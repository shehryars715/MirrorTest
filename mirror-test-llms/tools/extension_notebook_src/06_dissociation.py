# ========================= DISSOCIATION FIGURE =============================
# The Tier-1 headline as one picture: for every judge, the IMPLICIT signal
# (lower-perplexity rule, ~1.0 at >=3B) sits far to the right of the EXPLICIT
# verbal choice (PPP, pinned at chance ~0.5). A dumbbell per judge; the length
# of the bar IS the dissociation. Rows are colored by family so the reader sees
# the gap replicate across four training recipes (Qwen2.5, Llama-3.2, Mistral,
# Gemma-2) — not just on Qwen.
#
# Identity is never color-alone: the two measures are distinct marks (filled =
# explicit, open = implicit), a legend is always present, and the chance line is
# drawn. Colors are the validated categorical slots (grayscale-distinct, CVD-safe).
# ===========================================================================

from pathlib import Path

FAM_COLOR = {"qwen": "#2a78d6", "llama": "#4a3aa7",
             "mistral": "#1baf7a", "gemma": "#eda100"}
FAM_LABEL = {"qwen": "Qwen2.5", "llama": "Llama-3.2",
             "mistral": "Mistral", "gemma": "Gemma-2"}
FAM_ORDER = {"qwen": 0, "llama": 1, "mistral": 2, "gemma": 3}


def build_dissociation_figure(disso_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"[figure] matplotlib unavailable ({e}); skipping dissociation.png")
        return None

    rows = [r for r in disso_rows
            if isinstance(r.get("implicit_ppl_acc"), float)
            and isinstance(r.get("explicit_ppp_acc"), float)]
    rows.sort(key=lambda r: (FAM_ORDER.get(r["family"], 9),
                             r["params_b"] if isinstance(r["params_b"], (int, float)) else 0))
    if not rows:
        print("[figure] no dissociation rows yet; skipping dissociation.png")
        return None

    n = len(rows)
    ys = list(range(n))[::-1]                      # first row at top
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                         "text.color": mod_stats.INK, "axes.labelcolor": mod_stats.INK})
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * n + 1.2), constrained_layout=True)

    for y, r in zip(ys, rows):
        c = FAM_COLOR.get(r["family"], "#888888")
        ex, im = r["explicit_ppp_acc"], r["implicit_ppl_acc"]
        ax.plot([ex, im], [y, y], color=c, linewidth=1.8, alpha=0.9, zorder=1)
        if isinstance(r.get("explicit_ci_lo"), float):     # Wilson CI on the explicit dot
            ax.plot([r["explicit_ci_lo"], r["explicit_ci_hi"]], [y, y],
                    color=c, linewidth=0.9, alpha=0.45, zorder=2)
        ax.plot(ex, y, "o", color=c, markersize=6, markeredgecolor=mod_stats.INK,
                markeredgewidth=0.4, zorder=3)                       # explicit = filled
        ax.plot(im, y, "o", color="white", markersize=6.5, markeredgecolor=c,
                markeredgewidth=1.7, zorder=3)                       # implicit = open
        ax.text(1.008, y, f"gap {r['acc_gap_implicit_minus_explicit']:+.2f}",
                va="center", ha="left", fontsize=6, color=mod_stats.MUTED)

    ax.axvline(0.5, color=mod_stats.MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.text(0.5, n - 0.45, "chance", ha="center", va="bottom", fontsize=6,
            color=mod_stats.MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["judge"].replace("-instruct", "").replace("-it", "") for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(FAM_COLOR.get(r["family"], mod_stats.INK))
    ax.set_xlim(0.42, 1.06)
    ax.set_xlabel("Accuracy (pooled over LLM foils)")
    ax.set_title("Implicit (perplexity rule) vs explicit (verbal PPP) self-recognition\n"
                 "the dissociation replicates across training recipes", fontsize=8.5)
    mod_stats._style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=mod_stats.GRID, linewidth=0.5)

    # One legend row BELOW the plot (never overlaps the bottom data rows): the
    # two measures (mark shape) + one swatch per family (color).
    measures = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=mod_stats.INK,
               markeredgecolor=mod_stats.INK, markersize=6, label="explicit (verbal PPP)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=mod_stats.INK, markeredgewidth=1.6, markersize=6.5,
               label="implicit (perplexity rule)"),
    ]
    fams = sorted({r["family"] for r in rows}, key=lambda f: FAM_ORDER.get(f, 9))
    fam_handles = [Line2D([0], [0], marker="s", linestyle="none",
                          markerfacecolor=FAM_COLOR[f], markeredgecolor="none",
                          markersize=7, label=FAM_LABEL.get(f, f)) for f in fams]
    handles = measures + fam_handles
    # "outside lower center" makes constrained_layout RESERVE a strip below the
    # x-label for the legend, so it never overlaps the axis or the data rows
    # (independent of how many judges are plotted).
    try:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   frameon=False, fontsize=7)
    except Exception:                      # older matplotlib without 'outside'
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))

    targets = [Path(WORKING_ROOT) / "dissociation.png",
               Path(OUTPUT_DIR) / "dissociation.png"]
    fig_dir = utils.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    targets.append(fig_dir / "dissociation_extended.png")
    for t in targets:
        fig.savefig(t, dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "dissociation_extended.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] dissociation.png -> {targets[0]} "
          f"({n} judges across {len(fams)} families)")
    return str(targets[0])


DISSOCIATION_PATH = build_dissociation_figure(EXT_DISSO_ROWS)
