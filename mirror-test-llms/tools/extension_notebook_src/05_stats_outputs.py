# ===================== EXTENDED STATS + OUTPUTS =============================
# Recomputes every cell (reused Qwen + new families) from the raw judgment
# files, using the PUBLISHED statistics functions unchanged, so the extended
# table is directly comparable to the paper's Table 1. Holm is recomputed over
# ALL cells present (not hardcoded to the old 20). Writes the task deliverables
# to /kaggle/working/.
#
# Fidelity notes:
#  * cell_stats() is reused verbatim — including its 500-resample AUROC CI and
#    the config's 1000-resample kappa CI — so numbers match the released ones
#    exactly rather than "close".
#  * kappa is reported but is known to saturate toward 0 when either party is
#    near-constant (the paper concedes this). The dissociation is therefore
#    carried by (a) the implicit-vs-explicit ACCURACY GAP and (b) the
#    non-saturating margin-margin Spearman rho (tools/margin_correlation.py).
# ===========================================================================

import csv as _csv
from pathlib import Path
from collections import defaultdict

alpha = CFG["stats"]["alpha"]
KAPPA_BOOT = CFG["stats"]["bootstrap_n"]          # 1000, per frozen config
OUT = Path(OUTPUT_DIR)
(OUT / "raw").mkdir(parents=True, exist_ok=True)

LLM_FOILS_ALL = ["llama-3.2-3b-instruct", "gemma-2-9b-it", "mistral-7b-instruct-v0.3"]
QWEN_FOILS = LLM_FOILS_ALL + ["human"]            # original study foils


def judge_foils(judge_key):
    return QWEN_FOILS if family_of(judge_key) == "qwen" else foils_for(judge_key)


# Params for every judge (Qwen from frozen config; new from the config cell).
PARAMS = {k: v for k, v in QWEN_JUDGES}
PARAMS.update({j["key"]: j["params_b"] for j in NEW_JUDGES})

# Judges of interest = the instruct scale families only. Base checkpoints
# (keys ending in '-base') are a separate mechanism ablation the paper keeps in
# base_vs_instruct.csv, NOT in Table 1 — excluding them keeps the Holm family
# and the scale curve exactly comparable to the published main table.
JUDGES_OF_INTEREST = [k for k, _ in QWEN_JUDGES] + [j["key"] for j in NEW_JUDGES]

# All judges that actually have judgments on disk (partial sessions welcome).
runs = mod_stats.load_ppp_runs()
items = mod_stats.items_from_runs(runs)
runs_judges = {r["judge"] for r in runs}
present_judges = [k for k in JUDGES_OF_INTEREST if k in runs_judges]
stylo_sum, stylo_pairs = mod_stats.load_stylo()
ppl = mod_stats.load_ppl()
print(f"[stats] {len(runs)} runs loaded; judges present: {present_judges}")

# --------------------------- extended_table1 --------------------------------
main_rows, pvals = [], []
cell_n = []
for judge in present_judges:
    for foil in judge_foils(judge):
        st = mod_stats.cell_stats(items, judge, foil, alpha=alpha)
        if st is None:
            continue
        st["family"] = family_of(judge)
        st["params_b"] = PARAMS.get(judge, "")
        pvals.append((f"{judge}|{foil}", st["p_binomial"]))
        # stylometric pairwise acc pooled across domains by item count
        accs, ns = [], []
        for (j, f, d), s in stylo_sum.items():
            if j == judge and f == foil:
                accs.append(s["acc_pairwise_char_ngram"] * s["n_pairs"])
                ns.append(s["n_pairs"])
        st["stylo_pair_acc"] = (sum(accs) / sum(ns)) if ns else ""
        kp = mod_stats.judge_vs_ppl_kappa(items, judge, foil, ppl, KAPPA_BOOT)
        if kp:
            st.update({"ppl_rule_acc": kp["ppl_rule_acc"], "kappa": kp["kappa"],
                       "kappa_ci_lo": kp["kappa_ci_lo"], "kappa_ci_hi": kp["kappa_ci_hi"]})
        else:
            st.update({"ppl_rule_acc": "", "kappa": "", "kappa_ci_lo": "", "kappa_ci_hi": ""})
        main_rows.append(st)
        cell_n.append(st["n_items"])

# Holm across the FULL family of judge x foil cells now present
holm = {h["name"]: h for h in stats_utils.holm_bonferroni(pvals, alpha)}
for r in main_rows:
    h = holm.get(f"{r['judge']}|{r['foil']}")
    r["p_holm"] = h["p_holm"] if h else ""
    r["reject_holm"] = h["reject"] if h else ""

TABLE1_COLS = ["family", "judge", "params_b", "foil", "n_items", "acc", "ci_lo", "ci_hi",
               "auroc", "auroc_ci_lo", "auroc_ci_hi", "stylo_pair_acc", "ppl_rule_acc",
               "kappa", "kappa_ci_lo", "kappa_ci_hi", "consistency", "n_decisive",
               "k_decisive", "p_binomial", "p_holm", "reject_holm"]


def _write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})
    print(f"[out] {path}  ({len(rows)} rows)")


_write_csv(OUT / "extended_table1.csv", TABLE1_COLS, main_rows)
HOLM_FAMILY_SIZE = len(pvals)
print(f"[stats] Holm family recomputed over {HOLM_FAMILY_SIZE} judge x foil cells "
      f"(paper's original was 20).")

# ------------------- pooled-over-LLM-foils helpers --------------------------
def pooled_llm_ppp(judge):
    """Explicit accuracy pooled over the judge's LLM foils (human excluded) —
    the paper's headline definition. Returns (acc, lo, hi, n)."""
    llm = [f for f in judge_foils(judge) if f != "human"]
    scores = [it["score"] for (j, f, d, c, ph, pid), it in items.items()
              if j == judge and f in llm and c == "core" and ph == 0 and it["score"] is not None]
    if not scores:
        return None
    acc = stats_utils.mean(scores)
    lo, hi = stats_utils.wilson_ci(acc, len(scores), alpha)
    return acc, lo, hi, len(scores)


def pooled_llm_ppl(judge, llm_only=True):
    """Implicit (lower-perplexity rule) accuracy. llm_only pools over the same
    LLM foils as the explicit headline (apples-to-apples on one item universe);
    llm_only=False pools over ALL foils incl. human (reproduces the paper's
    prose headline, e.g. 0.795 for the 0.5B)."""
    foils = [f for f in judge_foils(judge)]
    if llm_only:
        foils = [f for f in foils if f != "human"]
    hits = [1.0 if r["rule_chose_self"] else 0.0
            for r in utils.read_jsonl(utils.BASELINES_DIR / f"ppl__{judge}.jsonl")
            if r["condition"] == "core" and r["foil"] in foils]
    return (stats_utils.mean(hits), len(hits)) if hits else None


def margin_rho(judge):
    """Non-saturating companion to kappa (tools/margin_correlation.py): Spearman
    rho between the judge's position-debiased letter-preference margin and the
    perplexity rule's margin, over shared core pairs. Unlike kappa it does not
    collapse when a party is near-constant; larger |rho| means the graded verbal
    margin tracks likelihood more. Reported with a bootstrap CI per judge — the
    DISSOCIATION itself is carried by the implicit-minus-explicit accuracy gap,
    not by assuming rho is ~0 (in Qwen rho is small and rises with scale)."""
    jr = [r for r in utils.read_jsonl(utils.JUDGMENTS_DIR / f"ppp__{judge}.jsonl")
          if r["condition"] == "core" and r["phrasing"] == 0]
    by_pair = defaultdict(dict)
    for r in jr:
        by_pair[r["pair_id"]][r["order"]] = r["logp_A"] - r["logp_B"]
    rule = {r["pair_id"]: r["nll_foil"] - r["nll_self"]
            for r in utils.read_jsonl(utils.BASELINES_DIR / f"ppl__{judge}.jsonl")
            if r["condition"] == "core"}
    pts = [((d["self_A"] - d["self_B"]) / 2, rule[p]) for p, d in by_pair.items()
           if "self_A" in d and "self_B" in d and p in rule]
    if len(pts) < 10:
        return None
    rho = stats_utils.spearman_rho([a for a, _ in pts], [b for _, b in pts])
    lo, hi = stats_utils.bootstrap_ci(
        lambda s: stats_utils.spearman_rho([a for a, _ in s], [b for _, b in s]),
        pts, n_boot=KAPPA_BOOT)
    return rho, lo, hi, len(pts)


# --------------------------- dissociation_summary ---------------------------
disso_rows = []
for judge in present_judges:
    ex = pooled_llm_ppp(judge)
    im = pooled_llm_ppl(judge)
    if ex is None or im is None:
        continue
    kaps = [r["kappa"] for r in main_rows
            if r["judge"] == judge and r["foil"] != "human" and isinstance(r.get("kappa"), float)]
    hum = mod_stats.cell_stats(items, judge, "human", alpha=alpha)
    mr = margin_rho(judge)
    im_all = pooled_llm_ppl(judge, llm_only=False)
    row = {
        "family": family_of(judge), "judge": judge, "params_b": PARAMS.get(judge, ""),
        "n_llm_pairs": ex[3],
        "implicit_ppl_acc": im[0],
        "implicit_ppl_acc_allfoils": (im_all[0] if im_all else ""),
        "explicit_ppp_acc": ex[0],
        "explicit_ci_lo": ex[1], "explicit_ci_hi": ex[2],
        "acc_gap_implicit_minus_explicit": im[0] - ex[0],
        "kappa_median": (sorted(kaps)[len(kaps) // 2] if kaps else ""),
        "kappa_absmax": (max(abs(k) for k in kaps) if kaps else ""),
        "margin_rho": (mr[0] if mr else ""), "margin_rho_lo": (mr[1] if mr else ""),
        "margin_rho_hi": (mr[2] if mr else ""),
        "auroc_human": (hum["auroc"] if hum else ""),
        "auroc_human_lo": (hum["auroc_ci_lo"] if hum else ""),
        "auroc_human_hi": (hum["auroc_ci_hi"] if hum else ""),
    }
    disso_rows.append(row)

DISSO_COLS = ["family", "judge", "params_b", "n_llm_pairs", "implicit_ppl_acc",
              "implicit_ppl_acc_allfoils",
              "explicit_ppp_acc", "explicit_ci_lo", "explicit_ci_hi",
              "acc_gap_implicit_minus_explicit", "kappa_median", "kappa_absmax",
              "margin_rho", "margin_rho_lo", "margin_rho_hi",
              "auroc_human", "auroc_human_lo", "auroc_human_hi"]
disso_rows.sort(key=lambda r: (r["family"], r["params_b"] or 0))
_write_csv(OUT / "dissociation_summary.csv", DISSO_COLS, disso_rows)

# ------------------------------ raw per-cell --------------------------------
# Copy the NEW judges' raw judgments/baselines and write per-(judge,foil)
# JSON + parquet-if-available for reproducibility.
try:
    import pandas as _pd
    _HAVE_PD = True
except Exception:
    _HAVE_PD = False
runs_by_cell = defaultdict(list)
for r in runs:
    runs_by_cell[(r["judge"], r["foil"])].append(r)
new_keys = {j["key"] for j in NEW_JUDGES}
n_raw = 0
for (judge, foil), rows in runs_by_cell.items():
    if judge not in new_keys:
        continue
    stem = OUT / "raw" / f"ppp__{judge}__{foil}"
    utils.write_jsonl(str(stem) + ".jsonl", rows)
    if _HAVE_PD:
        try:
            _pd.DataFrame(rows).to_parquet(str(stem) + ".parquet", index=False)
        except Exception:
            pass
    n_raw += 1
for judge in new_keys:
    for src in (utils.BASELINES_DIR / f"ppl__{judge}.jsonl",
                utils.JUDGMENTS_DIR / f"ipp__{judge}.jsonl"):
        if src.exists():
            (OUT / "raw" / src.name).write_text(src.read_text(encoding="utf-8"),
                                                encoding="utf-8")
print(f"[out] raw per-cell judgments -> {OUT / 'raw'} ({n_raw} PPP cells"
      f"{' + parquet' if _HAVE_PD else ', JSONL only'})")

# Stash for the figure + finalize cells.
CELL_N = cell_n
EXT_MAIN_ROWS = main_rows
EXT_DISSO_ROWS = disso_rows
print("[stats] extended tables written.")
