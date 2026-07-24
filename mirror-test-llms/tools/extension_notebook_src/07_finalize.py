# ============================ FINALIZE ======================================
# Writes run_report.txt (provenance a reviewer needs) and prints the closing
# summary. Safe to run repeatedly; reflects whatever has completed so far.
# ===========================================================================

import datetime, json
from pathlib import Path

WORK = Path(WORKING_ROOT)
lines = []


def out(s=""):
    lines.append(s)
    print(s)


status = globals().get("STATUS", {})
disso = globals().get("EXT_DISSO_ROWS", [])
cell_n = globals().get("CELL_N", [])
holm_n = globals().get("HOLM_FAMILY_SIZE", "n/a")

out("=" * 74)
out("MIRROR-TEST CROSS-FAMILY EXTENSION — run report")
out(f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
out("=" * 74)

# ---- task outcomes + timings ----------------------------------------------
gpu_min = cpu_min = 0.0
outcomes = {}
out("\n[tasks]")
for tid, s in status.items():
    st = s.get("status", "?")
    outcomes[st] = outcomes.get(st, 0) + 1
    mins = s.get("minutes")
    if mins:
        if tid.split(":")[0] in ("gen", "ppp", "ipp", "ppl"):
            gpu_min += mins
        else:
            cpu_min += mins
    out(f"    {tid:<30} {st:<12} {('%.0fm' % mins) if mins else ''}")
out(f"\n    summary: " + " ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
out(f"    GPU time this session ~= {gpu_min/60:.2f} h ; CPU ~= {cpu_min/60:.2f} h")

# ---- reused vs generated ---------------------------------------------------
out("\n[data provenance] (frozen inputs are never regenerated)")
new_keys = [j["key"] for j in NEW_JUDGES]
for key in new_keys:
    s1 = s2 = 0
    rev = None
    for d in utils.DOMAINS:
        for r in utils.read_jsonl(utils.GENERATIONS_DIR / f"{key}__{d}.jsonl"):
            if r.get("sample", "s1") == "s1":
                s1 += 1
            elif r.get("sample") == "s2":
                s2 += 1
            rev = rev or r.get("revision")
    # every Tier-1 judge is a promoted foil: its s1 is the frozen foil text, only s2 is new
    out(f"    {key:<26} s1={s1:<4} s2={s2:<4} [s1 REUSED (frozen), s2 generated]")
    out(f"        resolved revision: {rev or '(not generated yet)'}")
out("    reused unchanged: all Qwen2.5 generations/pairs/judgments; all foil s1 text.")

# ---- model revisions used (all pinned; nothing new to pin) ----------------
# Every Tier-1 judge is a promoted foil, so it reuses the revision already
# pinned in the frozen configs/models.yaml — there is nothing new to pin.
out("\n[model revisions used (all reused from the frozen config)]")
for key in [j["key"] for j in NEW_JUDGES]:
    rev = None
    for d in utils.DOMAINS:
        recs = utils.read_jsonl(utils.GENERATIONS_DIR / f"{key}__{d}.jsonl")
        if recs:
            rev = recs[0].get("revision")
            break
    out(f"    {key}: {rev or '(not generated this session)'}")

# ---- statistics provenance + power ----------------------------------------
out("\n[statistics]")
out(f"    Holm–Bonferroni family recomputed over ALL {holm_n} judge x foil cells "
    f"present (paper's original: 20). Significance flags in extended_table1.csv "
    f"reflect this larger family.")
out(f"    AUROC 95% CI: {500} bootstrap resamples (repo cell_stats default, kept for "
    f"exact comparability with the paper); kappa CI: {CFG['stats']['bootstrap_n']} "
    f"resamples (frozen config).")
if cell_n:
    sn = sorted(cell_n)
    med = sn[len(sn) // 2]
    out(f"    per-cell n: min={min(sn)} median={med} max={max(sn)} across "
        f"{len(sn)} cells.")
    try:
        p80 = None
        for pa in [x / 1000 for x in range(505, 800)]:
            if stats_utils.power_exact_binomial(med, pa) >= 0.80:
                p80 = pa
                break
        out(f"    power note: exact binomial, alpha={CFG['stats']['alpha']}, at the "
            f"median decisive-item count the smallest true accuracy detectable at "
            f"80% power is ~{p80:.3f}; smaller effects may be missed (per-cell CIs "
            f"in extended_table1.csv show the actual precision).")
    except Exception:
        pass

# ---- dissociation headline -------------------------------------------------
if disso:
    out("\n[dissociation replication] implicit (PPL-rule) vs explicit (PPP), per judge")
    out(f"    {'judge':<26}{'impl':>6}{'expl':>7}{'gap':>7}{'margin_rho':>12}")
    for r in disso:
        mr = f"{r['margin_rho']:+.3f}" if isinstance(r.get("margin_rho"), float) else "  n/a"
        out(f"    {r['judge']:<26}{r['implicit_ppl_acc']:>6.3f}"
            f"{r['explicit_ppp_acc']:>7.3f}{r['acc_gap_implicit_minus_explicit']:>7.3f}"
            f"{mr:>12}")
    out("    The large implicit–explicit ACCURACY GAP is the dissociation: it is "
        "robust and does NOT saturate like kappa. margin_rho is a continuous "
        "companion (in Qwen it is small and rises modestly with scale, ~0 to ~0.34) "
        "— report it per judge rather than claiming rho~0.")

# ---- outputs written -------------------------------------------------------
out("\n[outputs in /kaggle/working/]")
for rel in ("extended_table1.csv", "dissociation.png",
            "extended_outputs/dissociation_summary.csv",
            "extended_outputs/extended_table1.csv", "extended_outputs/raw"):
    p = WORK / rel
    if p.exists():
        out(f"    OK  {rel}")
disso_top = WORK / "dissociation_summary.csv"
if not disso_top.exists() and (Path(OUTPUT_DIR) / "dissociation_summary.csv").exists():
    import shutil
    shutil.copy2(Path(OUTPUT_DIR) / "dissociation_summary.csv", disso_top)
    shutil.copy2(Path(OUTPUT_DIR) / "extended_table1.csv", WORK / "extended_table1.csv")
    out("    (copied dissociation_summary.csv + extended_table1.csv to /kaggle/working/ top level)")

# ---- versions --------------------------------------------------------------
out("\n[versions]")
import importlib
for name in ("transformers", "accelerate", "bitsandbytes", "datasets", "sklearn",
             "matplotlib", "torch"):
    try:
        out(f"    {name}: {importlib.import_module(name).__version__}")
    except Exception:
        out(f"    {name}: (n/a)")

(WORK / "run_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[report] wrote {WORK / 'run_report.txt'}")
print("\nNEXT:")
print("  * Resume: if any 'paused/deferred/skipped' tasks remain, Save Version, start a")
print("    fresh session, attach THIS run's saved output as a Dataset, and Run All — the")
print("    repo re-clones automatically and every completed unit is skipped.")
print("  * Persist results: 'Save Version' preserves /kaggle/working (extended tables,")
print("    dissociation.png, raw judgments, checkpoints).")
print("  * If the clone failed: confirm the repo is pushed & public at REPO_GIT_URL with")
print("    Internet ON (or add a GITHUB_TOKEN secret / attach the repo as a Dataset).")
