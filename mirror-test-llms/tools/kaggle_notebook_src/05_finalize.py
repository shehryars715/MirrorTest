# ============================ FINALIZE ======================================
# Runs the (cheap) statistics pass on whatever exists, writes the session
# report, and packages EVERYTHING Claude needs into one zip.

import datetime, json, subprocess, sys, zipfile
from pathlib import Path

# ---- stats snapshot (harmless if too little data exists yet) ---------------
stats = subprocess.run([sys.executable, "src/06_stats.py"], cwd=REPO_DIR,
                       capture_output=True, text=True)
stats_note = ("statistics refreshed - see results/tables/"
              if stats.returncode == 0 else
              "statistics skipped (not enough judgments yet - expected in "
              "early sessions)")
print(f"[finalize] {stats_note}")

# ---- session report ---------------------------------------------------------
state = json.loads((REPO_DIR / "orchestrator_state.json").read_text(encoding="utf-8"))
tasks = state["tasks"]
order = list(tasks)
n_done = sum(1 for t in order if tasks[t]["status"] in ("done", "ran"))
todo = [t for t in order if tasks[t]["status"] in
        ("todo", "blocked", "deferred", "paused", "partial", "would-run")]
skipped = [t for t in order if tasks[t]["status"] == "skipped"]
failed = [t for t in order if tasks[t]["status"] in ("failed", "auth-error")]
EST = {t["id"]: t["est_min"] for t in TASKS}
remaining_h = sum(EST.get(t, 60) for t in todo) / 60

headline = ("ALL GPU WORK COMPLETE - give this bundle to Claude for the "
            "final analysis and paper." if not todo and not failed and not skipped
            else f"IN PROGRESS - {n_done}/{len(order)} tasks done, "
                 f"~{remaining_h:.1f} h of GPU work remaining (run this "
                 "notebook again, attaching this version's output as Input).")

def _tree_counts():
    lines = []
    for sub in ("data/prompts", "data/generations", "data/pairs",
                "results/judgments", "results/baselines", "results/tables"):
        d = REPO_DIR / sub
        n = sum(1 for f in d.rglob("*") if f.is_file()) if d.exists() else 0
        lines.append(f"| {sub} | {n} files |")
    return "\n".join(lines)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
rows = "\n".join(
    f"| {t} | {tasks[t]['status']} | {tasks[t].get('note','')} |" for t in order)
fail_detail = "\n".join(
    f"\n### {t} — last output lines\n```\n" + "\n".join(tasks[t].get("tail", []))
    + "\n```" for t in failed) or "none"

report = f"""# SESSION REPORT — {now}

**{headline}**

- notebook build: `{state['build']}` · budget: {state['budget_hours']} h
- GPU: {HAVE_GPU} · HF token: {HAVE_HF_TOKEN} · {stats_note}

## Task board
| task | status | note |
|---|---|---|
{rows}

`done` = complete before this session · `ran` = completed this session ·
`paused/partial` = mid-way, auto-resumes · `blocked` = waiting on earlier
tasks · `skipped` = needs GPU or HF_TOKEN (fix and re-run) ·
`deferred` = out of time this session.

## Failures needing attention
{fail_detail}

## Data inventory
| location | count |
|---|---|
{_tree_counts()}

## What to do next
1. {"Nothing on Kaggle - download mirror_bundle.zip below and hand it to Claude."
    if not todo and not failed and not skipped else
    "Re-run: open the notebook, + Add Input -> this version's Output, "
    "Save Version -> Save & Run All."}
2. {"Fix first: add the HF_TOKEN secret and accept the Llama/Gemma licenses, "
    "then re-run." if any('HF_TOKEN' in tasks[t].get('note','') for t in skipped)
    else "No settings changes needed."}
3. Download **mirror_bundle.zip** + **SESSION_REPORT.md** from this
   version's Output tab into `Desktop/Mirror/` and tell Claude.
"""

for p in (WORK / "SESSION_REPORT.md", REPO_DIR / "SESSION_REPORT.md"):
    p.write_text(report, encoding="utf-8")
print(report)

# ---- the bundle -------------------------------------------------------------
bundle = WORK / "mirror_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ("data", "results"):
        base = REPO_DIR / sub
        if base.exists():
            for f in base.rglob("*"):
                if f.is_file() and f.suffix != ".zip":
                    zf.write(f, f.relative_to(REPO_DIR))
    for extra in ("orchestrator_state.json", "SESSION_REPORT.md"):
        if (REPO_DIR / extra).exists():
            zf.write(REPO_DIR / extra, extra)
print(f"\n[finalize] bundle ready: {bundle} "
      f"({bundle.stat().st_size/2**20:.1f} MB) - download it from the "
      "Output tab and give it to Claude.")
