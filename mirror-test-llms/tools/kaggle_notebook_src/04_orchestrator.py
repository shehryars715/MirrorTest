# =========================== ORCHESTRATOR ===================================
# Knows the full experiment as an ordered task list with dependencies.
# Each session: evaluate what is already DONE (cheap file-count checks),
# then run the next incomplete tasks until the time budget is spent.
#
# Design note: the done-checks only steer BUDGETING - correctness never
# depends on them, because every pipeline script is internally resumable
# and instantly skips finished items. A wrong "not done" verdict just costs
# one model-load; a wrong "done" verdict is prevented by checking counts.

import json, subprocess, sys, time
from collections import deque
from pathlib import Path

from utils import PAIRS_DIR, GENERATIONS_DIR, JUDGMENTS_DIR, BASELINES_DIR, \
    PROMPTS_DIR, read_jsonl, load_config

CFG = load_config()
T0 = time.monotonic()
STATE_PATH = REPO_DIR / "orchestrator_state.json"
DOMAINS = ["news", "dolly", "wp"]
JUDGES = [m["key"] for m in CFG["judges"]]
FOILS = [m["key"] for m in CFG["foils"]]
GEN_FOILS = [m["key"] for m in CFG["foils"] if m.get("hf_id")]
GATED = {"llama-3.2-3b-instruct", "gemma-2-9b-it"}
BASE_JUDGES = [m["key"] for m in CFG.get("base_judges", [])]
PARA_JUDGE = CFG["paraphrase"]["judge"]
PARA_FOIL = CFG["paraphrase"]["foil"]
N_TARGET = CFG["prompt_filters"]["n_per_domain"]
PLACEBO_N = CFG["generation"]["placebo_n_prompts"]

def hours_left() -> float:
    return BUDGET_HOURS - (time.monotonic() - T0) / 3600

def _n_lines(path: Path) -> int:
    return sum(1 for line in open(path, encoding="utf-8") if line.strip()) \
        if path.exists() else 0

def n_prompts(domain: str) -> int:
    return _n_lines(PROMPTS_DIR / f"{domain}.jsonl")

# ----------------------------- done checks ----------------------------------
def prompts_done() -> bool:
    return all(n_prompts(d) >= N_TARGET for d in DOMAINS)

def gen_done(model: str, placebo: bool) -> bool:
    if not prompts_done():
        return False
    for d in DOMAINS:
        recs = read_jsonl(GENERATIONS_DIR / f"{model}__{d}.jsonl")
        s1 = sum(1 for r in recs if r.get("sample", "s1") == "s1")
        if s1 < n_prompts(d):
            return False
        if placebo:
            s2 = sum(1 for r in recs if r.get("sample") == "s2")
            if s2 < min(PLACEBO_N, n_prompts(d)):
                return False
    return True

def pairs_done() -> bool:
    if not (PAIRS_DIR / "pairs_report.json").exists():
        return False
    for j in JUDGES:
        for f in FOILS:
            for d in DOMAINS:
                if not (PAIRS_DIR / f"ppp__{j}__{f}__{d}.jsonl").exists():
                    return False
        if not (PAIRS_DIR / f"ipp__{j}.jsonl").exists():
            return False
    return True

def paraphrase_done() -> bool:
    for d in DOMAINS:
        src = _n_lines(PAIRS_DIR / f"ppp__{PARA_JUDGE}__{PARA_FOIL}__{d}.jsonl")
        if src == 0 or _n_lines(PAIRS_DIR / f"para__{PARA_JUDGE}__{PARA_FOIL}__{d}.jsonl") \
                < min(src, 200):
            return False
    return True

def _pairs_of(judge: str) -> str:
    for m in CFG.get("base_judges", []):
        if m["key"] == judge:
            return m["pairs_of"]
    return judge

def _judgment_counts(judge: str):
    recs = read_jsonl(JUDGMENTS_DIR / f"ppp__{judge}.jsonl")
    out = {"core0": 0, "placebo": 0, "para": 0, "phr12": 0}
    for r in recs:
        if r["condition"] == "core" and r["phrasing"] == 0:
            out["core0"] += 1
        elif r["condition"] == "placebo":
            out["placebo"] += 1
        elif r["condition"] == "paraphrase":
            out["para"] += 1
        if r["condition"] == "core" and r["phrasing"] in (1, 2):
            out["phr12"] += 1
    return out

def _expected_core_pairs(owner: str) -> int:
    return sum(_n_lines(PAIRS_DIR / f"ppp__{owner}__{f}__{d}.jsonl")
               for f in FOILS for d in DOMAINS)

def ppp_done(judge: str, with_placebo: bool) -> bool:
    if not pairs_done():
        return False
    owner = _pairs_of(judge)
    c = _judgment_counts(judge)
    if c["core0"] < 2 * _expected_core_pairs(owner):
        return False
    if with_placebo:
        exp_pl = sum(_n_lines(PAIRS_DIR / f"placebo__{owner}__{d}.jsonl") for d in DOMAINS)
        if c["placebo"] < 2 * exp_pl:
            return False
    return True

def main_cell_done() -> bool:
    if not paraphrase_done():
        return False
    cell = sum(_n_lines(PAIRS_DIR / f"ppp__{PARA_JUDGE}__{PARA_FOIL}__{d}.jsonl")
               for d in DOMAINS)
    passed = sum(1 for d in DOMAINS for r in
                 read_jsonl(PAIRS_DIR / f"para__{PARA_JUDGE}__{PARA_FOIL}__{d}.jsonl")
                 if r.get("passed_gate"))
    c = _judgment_counts(PARA_JUDGE)
    return c["phr12"] >= 2 * 2 * cell and c["para"] >= 2 * passed

def ipp_done(judge: str) -> bool:
    exp = _n_lines(PAIRS_DIR / f"ipp__{judge}.jsonl")
    return exp > 0 and _n_lines(JUDGMENTS_DIR / f"ipp__{judge}.jsonl") >= exp

def ppl_done(judge: str, with_para: bool) -> bool:
    owner = _pairs_of(judge)
    rows = read_jsonl(BASELINES_DIR / f"ppl__{judge}.jsonl")
    core = sum(1 for r in rows if r["condition"] == "core")
    if core < _expected_core_pairs(owner) or core == 0:
        return False
    if with_para:
        passed = sum(1 for d in DOMAINS for r in
                     read_jsonl(PAIRS_DIR / f"para__{owner}__{PARA_FOIL}__{d}.jsonl")
                     if r.get("passed_gate"))
        if sum(1 for r in rows if r["condition"] == "paraphrase") < passed:
            return False
    return True

def stylo_done() -> bool:
    if not pairs_done():
        return False
    for p in PAIRS_DIR.glob("ppp__*.jsonl"):
        if _n_lines(p) >= 20 and not \
                (BASELINES_DIR / f"stylo__{p.stem[len('ppp__'):]}.json").exists():
            return False
    return True

# ----------------------------- task table -----------------------------------
def T(tid, cmd, est_min, done_fn, after=(), gpu=True, gated=False):
    return dict(id=tid, cmd=cmd, est_min=est_min, done_fn=done_fn,
                after=list(after), gpu=gpu, gated=gated)

PY = [sys.executable]
TASKS = [T("prompts", PY + ["src/00_build_prompts.py"], 12, prompts_done, gpu=False)]
GEN_EST = {"qwen2.5-0.5b-instruct": 35, "qwen2.5-1.5b-instruct": 50,
           "qwen2.5-3b-instruct": 70, "qwen2.5-7b-instruct": 130,
           "qwen2.5-14b-instruct": 170, "llama-3.2-3b-instruct": 70,
           "gemma-2-9b-it": 130, "mistral-7b-instruct-v0.3": 100}
for m in JUDGES:
    TASKS.append(T(f"gen:{m}", PY + ["src/01_generate.py", "--models", m, "--placebo"],
                   GEN_EST.get(m, 90), lambda m=m: gen_done(m, True), after=["prompts"]))
for m in GEN_FOILS:
    TASKS.append(T(f"gen:{m}", PY + ["src/01_generate.py", "--models", m],
                   GEN_EST.get(m, 90), lambda m=m: gen_done(m, False),
                   after=["prompts"], gated=m in GATED))
ALL_GEN = [t["id"] for t in TASKS if t["id"].startswith("gen:")]
TASKS.append(T("pairs", PY + ["src/02_build_pairs.py"], 6, pairs_done,
               after=ALL_GEN, gpu=False))
TASKS.append(T("paraphrase", PY + ["src/02b_paraphrase.py"], 70, paraphrase_done,
               after=["pairs"]))
JUDGE_EST = {"qwen2.5-0.5b-instruct": 35, "qwen2.5-1.5b-instruct": 45,
             "qwen2.5-3b-instruct": 60, "qwen2.5-7b-instruct": 95,
             "qwen2.5-14b-instruct": 130}
for j in JUDGES:
    TASKS.append(T(f"ppp:{j}", PY + ["src/03_judge_ppp.py", "--judge", j,
                                     "--include-placebo"],
                   JUDGE_EST.get(j, 90), lambda j=j: ppp_done(j, True), after=["pairs"]))
TASKS.append(T("main-cell", PY + ["src/03_judge_ppp.py", "--judge", PARA_JUDGE,
                                  "--foils", PARA_FOIL, "--phrasings", "0", "1", "2",
                                  "--include-paraphrase"],
               75, main_cell_done, after=["paraphrase", f"ppp:{PARA_JUDGE}"]))
for j in JUDGES:
    TASKS.append(T(f"ipp:{j}", PY + ["src/04_judge_ipp.py", "--judge", j],
                   15, lambda j=j: ipp_done(j), after=["pairs"]))
for j in BASE_JUDGES:
    TASKS.append(T(f"ppp:{j}", PY + ["src/03_judge_ppp.py", "--judge", j],
                   90, lambda j=j: ppp_done(j, False), after=["pairs"]))
for j in JUDGES:
    para = j == PARA_JUDGE
    cmd = PY + ["src/05_baselines.py", "perplexity", "--judge", j] + \
        (["--include-paraphrase"] if para else [])
    TASKS.append(T(f"ppl:{j}", cmd, 35, lambda j=j, p=para: ppl_done(j, p),
                   after=(["paraphrase"] if para else ["pairs"])))
for j in BASE_JUDGES:
    TASKS.append(T(f"ppl:{j}", PY + ["src/05_baselines.py", "perplexity", "--judge", j],
                   35, lambda j=j: ppl_done(j, False), after=["pairs"]))
TASKS.append(T("stylometric", PY + ["src/05_baselines.py", "stylometric"], 8,
               stylo_done, after=["pairs"], gpu=False))

# ------------------------------ run loop ------------------------------------
STATUS: dict = {}

def save_state():
    STATE_PATH.write_text(json.dumps(
        {"build": NOTEBOOK_BUILD, "budget_hours": BUDGET_HOURS, "tasks": STATUS},
        indent=2), encoding="utf-8")

def run_task(t) -> str:
    tail = deque(maxlen=40)
    print(f"\n===== RUN {t['id']}  (est ~{t['est_min']} min, "
          f"{hours_left():.1f} h left) =====", flush=True)
    proc = subprocess.Popen(t["cmd"], cwd=REPO_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    killed = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        if hours_left() <= 0 and not killed:
            print(f"\n[budget] time is up - pausing {t['id']} "
                  "(it resumes automatically next session)", flush=True)
            proc.terminate()
            killed = True
    rc = proc.wait()
    STATUS[t["id"]]["tail"] = list(tail)[-12:]
    if killed:
        return "paused"
    if rc != 0:
        low = "\n".join(tail).lower()
        if "401" in low or "403" in low or "gated" in low or "unauthorized" in low:
            return "auth-error"
        return "failed"
    return "ran" if t["done_fn"]() else "partial"

print(f"{'TASK':<34}{'STATE':<12}note")
runnable = []
for t in TASKS:
    if t["done_fn"]():
        STATUS[t["id"]] = {"status": "done", "note": "already complete"}
    else:
        STATUS[t["id"]] = {"status": "todo", "note": ""}
        runnable.append(t)
    print(f"{t['id']:<34}{STATUS[t['id']]['status']:<12}"
          f"{STATUS[t['id']]['note']}")
save_state()

for t in runnable:
    st = STATUS[t["id"]]
    deps = [d for d in t["after"] if STATUS.get(d, {}).get("status") not in
            ("done", "ran")]
    if deps:
        st.update(status="blocked", note=f"waiting on: {', '.join(deps)}")
    elif t["gpu"] and not HAVE_GPU:
        st.update(status="skipped", note="no GPU in this session")
    elif t["gated"] and not HAVE_HF_TOKEN:
        st.update(status="skipped",
                  note="needs HF_TOKEN secret + accepted license")
    elif hours_left() <= 0:
        st.update(status="deferred", note="session budget spent")
    elif DRY_RUN:
        st.update(status="would-run", note=f"~{t['est_min']} min")
    else:
        start = time.monotonic()
        outcome = run_task(t)
        mins = (time.monotonic() - start) / 60
        st.update(status=outcome, note=f"{mins:.0f} min")
        print(f"[task] {t['id']} -> {outcome} ({mins:.0f} min)", flush=True)
    save_state()

print("\n[orchestrator] pass complete - finalize cell writes the report.")
