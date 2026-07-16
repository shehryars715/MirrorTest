# =========================== ORCHESTRATOR ===================================
# Knows the full experiment as an ordered task list with dependencies.
# Each session: evaluate what is already DONE (cheap file-count checks),
# then run the next incomplete tasks until the time budget is spent.
#
# Design note: the done-checks only steer BUDGETING - correctness never
# depends on them, because every pipeline script is internally resumable
# and instantly skips finished items. A wrong "not done" verdict just costs
# one model-load; a wrong "done" verdict is prevented by checking counts.

import json, queue, subprocess, sys, threading, time
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
# Time estimates RECALIBRATED from session-1 measurements on Kaggle T4
# (0.5B generation measured 144 min, 1.5B 206 min - about 4x the original
# guesses; larger models extrapolated at the measured tokens/sec trend).
# Foils generate 600 items (no placebo twin), judges 1050.
GEN_EST = {"qwen2.5-0.5b-instruct": 150, "qwen2.5-1.5b-instruct": 210,
           "qwen2.5-3b-instruct": 280, "qwen2.5-7b-instruct": 450,
           "qwen2.5-14b-instruct": 900, "llama-3.2-3b-instruct": 170,
           "gemma-2-9b-it": 380, "mistral-7b-instruct-v0.3": 300}
# Generation ORDER: small judges -> foils -> big (>=10B) judges LAST.
# Rationale: the 14B is by far the most expensive model; running it last
# means the protocol-sanctioned fallback (cap the scale axis at 7B, §18)
# stays available at zero sunk cost if the schedule slips.
_params = {m["key"]: (m.get("params_b") or 0)
           for m in CFG["judges"] + CFG["foils"] if m.get("hf_id")}
GEN_PLAN = ([(m, True) for m in JUDGES if _params.get(m, 0) < 10]
            + [(m, False) for m in GEN_FOILS]
            + [(m, True) for m in JUDGES if _params.get(m, 0) >= 10])
for m, placebo in GEN_PLAN:
    cmd = PY + ["src/01_generate.py", "--models", m] + (["--placebo"] if placebo else [])
    TASKS.append(T(f"gen:{m}", cmd, GEN_EST.get(m, 200),
                   lambda m=m, p=placebo: gen_done(m, p),
                   after=["prompts"], gated=m in GATED))
ALL_GEN = [t["id"] for t in TASKS if t["id"].startswith("gen:")]
TASKS.append(T("pairs", PY + ["src/02_build_pairs.py"], 10, pairs_done,
               after=ALL_GEN, gpu=False))
TASKS.append(T("paraphrase", PY + ["src/02b_paraphrase.py"], 200, paraphrase_done,
               after=["pairs"]))
JUDGE_EST = {"qwen2.5-0.5b-instruct": 150, "qwen2.5-1.5b-instruct": 180,
             "qwen2.5-3b-instruct": 240, "qwen2.5-7b-instruct": 360,
             "qwen2.5-14b-instruct": 540}
for j in JUDGES:
    TASKS.append(T(f"ppp:{j}", PY + ["src/03_judge_ppp.py", "--judge", j,
                                     "--include-placebo"],
                   JUDGE_EST.get(j, 90), lambda j=j: ppp_done(j, True), after=["pairs"]))
TASKS.append(T("main-cell", PY + ["src/03_judge_ppp.py", "--judge", PARA_JUDGE,
                                  "--foils", PARA_FOIL, "--phrasings", "0", "1", "2",
                                  "--include-paraphrase"],
               180, main_cell_done, after=["paraphrase", f"ppp:{PARA_JUDGE}"]))
for j in JUDGES:
    TASKS.append(T(f"ipp:{j}", PY + ["src/04_judge_ipp.py", "--judge", j],
                   30, lambda j=j: ipp_done(j), after=["pairs"]))
for j in BASE_JUDGES:
    TASKS.append(T(f"ppp:{j}", PY + ["src/03_judge_ppp.py", "--judge", j],
                   90, lambda j=j: ppp_done(j, False), after=["pairs"]))
for j in JUDGES:
    para = j == PARA_JUDGE
    cmd = PY + ["src/05_baselines.py", "perplexity", "--judge", j] + \
        (["--include-paraphrase"] if para else [])
    TASKS.append(T(f"ppl:{j}", cmd, 90, lambda j=j, p=para: ppl_done(j, p),
                   after=(["paraphrase"] if para else ["pairs"])))
for j in BASE_JUDGES:
    TASKS.append(T(f"ppl:{j}", PY + ["src/05_baselines.py", "perplexity", "--judge", j],
                   90, lambda j=j: ppl_done(j, False), after=["pairs"]))
TASKS.append(T("stylometric", PY + ["src/05_baselines.py", "stylometric"], 15,
               stylo_done, after=["pairs"], gpu=False))

# ------------------------------ run loop ------------------------------------
STATUS: dict = {}
HEARTBEAT_S = 60      # print a liveness line after this many silent seconds

def clock() -> str:
    """Session clock, e.g. [0:47] = 47 min since the session started."""
    el = int(time.monotonic() - T0)
    return f"[{el // 3600}:{el % 3600 // 60:02d}]"

def save_state():
    STATE_PATH.write_text(json.dumps(
        {"build": NOTEBOOK_BUILD, "budget_hours": BUDGET_HOURS, "tasks": STATUS},
        indent=2), encoding="utf-8")

def session_progress() -> str:
    c = {}
    for v in STATUS.values():
        c[v["status"]] = c.get(v["status"], 0) + 1
    parts = " ".join(f"{k}:{v}" for k, v in sorted(c.items()))
    return f"{parts} | {hours_left():.1f} h budget left"

def run_task(t, seq: int, n_planned: int) -> str:
    """Run one pipeline script as a subprocess, streaming its output live.
    A reader thread feeds a queue so that even when the task is silent
    (model downloads / weight loading), a heartbeat line proves the session
    is alive and shows for how long it has been quiet."""
    tail = deque(maxlen=60)
    print(f"\n{clock()} ===== TASK {seq}/{n_planned}: {t['id']} "
          f"(est ~{t['est_min']} min | {hours_left():.1f} h left) =====",
          flush=True)
    print(f"{clock()} $ {' '.join(t['cmd'])}", flush=True)
    start = time.monotonic()
    proc = subprocess.Popen(t["cmd"], cwd=REPO_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    q: queue.Queue = queue.Queue()

    def _reader():
        for ln in proc.stdout:
            q.put(ln)
        q.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    killed, eof = False, False
    last_output = time.monotonic()
    while not eof:
        try:
            ln = q.get(timeout=HEARTBEAT_S)
        except queue.Empty:
            quiet = time.monotonic() - last_output
            print(f"{clock()} [heartbeat] {t['id']} running "
                  f"{(time.monotonic() - start) / 60:.0f} min, no output for "
                  f"{quiet / 60:.0f} min (downloads/loading can be silent)",
                  flush=True)
        else:
            if ln is None:
                eof = True
            else:
                print(ln, end="", flush=True)
                tail.append(ln.rstrip())
                last_output = time.monotonic()
        if hours_left() <= 0 and not killed:
            print(f"\n{clock()} [budget] time is up - pausing {t['id']} "
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

# ---- status board + session plan --------------------------------------------
print(f"{clock()} evaluating what is already done ...", flush=True)
runnable = []
print(f"\n{'TASK':<34}{'STATE':<10}{'EST':>6}")
for t in TASKS:
    if t["done_fn"]():
        STATUS[t["id"]] = {"status": "done", "note": "already complete"}
    else:
        STATUS[t["id"]] = {"status": "todo", "note": ""}
        runnable.append(t)
    print(f"{t['id']:<34}{STATUS[t['id']]['status']:<10}"
          f"{t['est_min']:>4}m", flush=True)
todo_min = sum(t["est_min"] for t in runnable)
print(f"\n{clock()} plan: {len(TASKS) - len(runnable)}/{len(TASKS)} tasks "
      f"already done; ~{todo_min / 60:.1f} h of work remains; this session's "
      f"budget is {BUDGET_HOURS} h -> expect to run "
      f"~{min(len(runnable), max(1, round(len(runnable) * BUDGET_HOURS * 60 / max(todo_min, 1))))} "
      f"of the {len(runnable)} remaining tasks.", flush=True)
save_state()

seq = 0
for t in runnable:
    st = STATUS[t["id"]]
    deps = [d for d in t["after"] if STATUS.get(d, {}).get("status") not in
            ("done", "ran")]
    if deps:
        st.update(status="blocked", note=f"waiting on: {', '.join(deps)}")
        print(f"{clock()} [skip] {t['id']}: blocked by {', '.join(deps)}", flush=True)
    elif t["gpu"] and not HAVE_GPU:
        st.update(status="skipped", note="no GPU in this session")
        print(f"{clock()} [skip] {t['id']}: no GPU", flush=True)
    elif t["gated"] and not HAVE_HF_TOKEN:
        st.update(status="skipped",
                  note="needs HF_TOKEN secret + accepted license")
        print(f"{clock()} [skip] {t['id']}: needs HF_TOKEN secret", flush=True)
    elif hours_left() <= 0:
        st.update(status="deferred", note="session budget spent")
        print(f"{clock()} [defer] {t['id']}: budget spent", flush=True)
    elif DRY_RUN:
        st.update(status="would-run", note=f"~{t['est_min']} min")
        print(f"{clock()} [dry-run] would run {t['id']} (~{t['est_min']} min)",
              flush=True)
    else:
        seq += 1
        start = time.monotonic()
        outcome = run_task(t, seq, len(runnable))
        mins = (time.monotonic() - start) / 60
        st.update(status=outcome, note=f"{mins:.0f} min")
        print(f"{clock()} [task] {t['id']} -> {outcome.upper()} "
              f"({mins:.0f} min vs est {t['est_min']})", flush=True)
        print(f"{clock()} [session] {session_progress()}", flush=True)
    save_state()

print(f"\n{clock()} [orchestrator] pass complete - {session_progress()}")
print("finalize cell writes the report next.", flush=True)
