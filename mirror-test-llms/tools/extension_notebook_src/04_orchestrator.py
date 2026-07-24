# =========================== ORCHESTRATOR ===================================
# Runs the PUBLISHED src/ scripts (unmodified) as subprocesses, one model in
# memory at a time, with checkpoint/skip, a heartbeat, and a session time
# budget. Using the real scripts is what makes every new measurement
# byte-identical to the released Qwen cells. Correctness never depends on the
# done-checks: each script is internally resumable (skips finished items), so a
# wrong "not-done" verdict costs at most one model load.
#
# Per new judge J (foils P = family-excluded pool + human):
#   gen:J    01_generate.py   --models J --placebo        (only s2 is new for
#                                                           Tier-1 judges)
#   pairs:J  02_build_pairs.py --judges J --foils P...     (PPP + placebo + IPP)
#   ppp:J    03_judge_ppp.py   --judge J --include-placebo
#   ipp:J    04_judge_ipp.py   --judge J
#   ppl:J    05_baselines.py perplexity --judge J
#   stylo:J  05_baselines.py stylometric --judge J         (CPU)
# gemma-2-27b-it is scheduled LAST and sharded across both T4s.
# ===========================================================================

import gc, json, queue, subprocess, threading, time
from collections import deque
from pathlib import Path

T0 = time.monotonic()
STATE_PATH = Path(CHECKPOINT_DIR) / "orchestrator_state.json"
DOMAINS = utils.DOMAINS
N_TARGET = CFG["prompt_filters"]["n_per_domain"]
PLACEBO_N = CFG["generation"]["placebo_n_prompts"]
PY = [sys.executable]
EXT = ["--config", str(EXT_CONFIG)]
GEN, PAIRS, JUDG, BASE = (utils.GENERATIONS_DIR, utils.PAIRS_DIR,
                          utils.JUDGMENTS_DIR, utils.BASELINES_DIR)

ACTIVE_JUDGES = [j for j in NEW_JUDGES
                 if not (j["key"] == "gemma-2-27b-it" and not ENABLE_27B)]


def hours_left():
    return BUDGET_HOURS - (time.monotonic() - T0) / 3600


def _n(path):
    return sum(1 for ln in open(path, encoding="utf-8") if ln.strip()) \
        if Path(path).exists() else 0


# ------------------------------- done checks --------------------------------
def gen_done(key):
    for d in DOMAINS:
        recs = utils.read_jsonl(GEN / f"{key}__{d}.jsonl")
        s1 = sum(1 for r in recs if r.get("sample", "s1") == "s1")
        s2 = sum(1 for r in recs if r.get("sample") == "s2")
        if s1 < N_TARGET or s2 < min(PLACEBO_N, N_TARGET):
            return False
    return True


def pairs_done(key):
    P = foils_for(key)
    if not (PAIRS / f"ipp__{key}.jsonl").exists():
        return False
    for d in DOMAINS:
        if not (PAIRS / f"placebo__{key}__{d}.jsonl").exists():
            return False
        for f in P:
            if not (PAIRS / f"ppp__{key}__{f}__{d}.jsonl").exists():
                return False
    return True


def _expected_core_pairs(key):
    return sum(_n(PAIRS / f"ppp__{key}__{f}__{d}.jsonl")
               for f in foils_for(key) for d in DOMAINS)


def _expected_placebo_pairs(key):
    return sum(_n(PAIRS / f"placebo__{key}__{d}.jsonl") for d in DOMAINS)


def ppp_done(key):
    if not pairs_done(key):
        return False
    recs = utils.read_jsonl(JUDG / f"ppp__{key}.jsonl")
    core0 = sum(1 for r in recs if r["condition"] == "core" and r["phrasing"] == 0)
    plac = sum(1 for r in recs if r["condition"] == "placebo")
    return (core0 >= 2 * _expected_core_pairs(key)
            and plac >= 2 * _expected_placebo_pairs(key) and core0 > 0)


def ipp_done(key):
    exp = _n(PAIRS / f"ipp__{key}.jsonl")
    return exp > 0 and _n(JUDG / f"ipp__{key}.jsonl") >= exp


def ppl_done(key):
    core = sum(1 for r in utils.read_jsonl(BASE / f"ppl__{key}.jsonl")
               if r["condition"] == "core")
    return core > 0 and core >= _expected_core_pairs(key)


def stylo_done(key):
    if not pairs_done(key):
        return False
    for f in foils_for(key):
        for d in DOMAINS:
            pf = PAIRS / f"ppp__{key}__{f}__{d}.jsonl"
            if _n(pf) >= 20 and not (BASE / f"stylo__{key}__{f}__{d}.json").exists():
                return False
    return True


# ------------------------------- task table ---------------------------------
def T(tid, cmd, done_fn, after=(), gpu=True, gated=False, env_extra=None, est=90):
    return dict(id=tid, cmd=cmd, done_fn=done_fn, after=list(after), gpu=gpu,
                gated=gated, env_extra=env_extra or {}, est=est)


GEN_EST = {"llama-3.2-3b-instruct": 60, "gemma-2-9b-it": 90,
           "mistral-7b-instruct-v0.3": 70, "gemma-2-2b-it": 90,
           "gemma-2-27b-it": 200}
JUDGE_EST = {"llama-3.2-3b-instruct": 60, "gemma-2-9b-it": 110,
             "mistral-7b-instruct-v0.3": 90, "gemma-2-2b-it": 60,
             "gemma-2-27b-it": 240}
TASKS, PREFLIGHT = [], []
for j in ACTIVE_JUDGES:
    key = j["key"]
    P = foils_for(key)
    shard = bool(j.get("shard"))
    env_extra = {"MIRROR_MAX_MEMORY": MAX_MEMORY_SHARDED} if shard else {}
    gated = family_of(key) in ("gemma", "llama")
    TASKS.append(T(f"gen:{key}", PY + [str(SRC_DIR / "01_generate.py")] + EXT
                   + ["--models", key, "--placebo"],
                   (lambda k=key: gen_done(k)), gated=gated, env_extra=env_extra,
                   est=GEN_EST.get(key, 120)))
    TASKS.append(T(f"pairs:{key}", PY + [str(SRC_DIR / "02_build_pairs.py")] + EXT
                   + ["--judges", key, "--foils", *P, "--domains", *DOMAINS],
                   (lambda k=key: pairs_done(k)), after=[f"gen:{key}"], gpu=False,
                   est=8))
    TASKS.append(T(f"ppp:{key}", PY + [str(SRC_DIR / "03_judge_ppp.py")] + EXT
                   + ["--judge", key, "--include-placebo"],
                   (lambda k=key: ppp_done(k)), after=[f"pairs:{key}"],
                   gated=gated, env_extra=env_extra, est=JUDGE_EST.get(key, 120)))
    if RUN_IPP:
        TASKS.append(T(f"ipp:{key}", PY + [str(SRC_DIR / "04_judge_ipp.py")] + EXT
                       + ["--judge", key],
                       (lambda k=key: ipp_done(k)), after=[f"pairs:{key}"],
                       gated=gated, env_extra=env_extra, est=25))
    TASKS.append(T(f"ppl:{key}", PY + [str(SRC_DIR / "05_baselines.py"), "perplexity"]
                   + EXT + ["--judge", key],
                   (lambda k=key: ppl_done(k)), after=[f"pairs:{key}"],
                   gated=gated, env_extra=env_extra, est=JUDGE_EST.get(key, 120) // 2))
    if RUN_STYLOMETRY:
        TASKS.append(T(f"stylo:{key}", PY + [str(SRC_DIR / "05_baselines.py"),
                       "stylometric"] + EXT + ["--judge", key],
                       (lambda k=key: stylo_done(k)), after=[f"pairs:{key}"],
                       gpu=False, est=8))

# Order: Tier 1 (cheap, existing text) -> Tier 2 2B -> 27B LAST. Within a judge
# the `after` deps already serialise gen->pairs->{ppp,ipp,ppl,stylo}.
_order = {"llama-3.2-3b-instruct": 0, "mistral-7b-instruct-v0.3": 1,
          "gemma-2-9b-it": 2, "gemma-2-2b-it": 3, "gemma-2-27b-it": 9}
TASKS.sort(key=lambda t: (_order.get(t["id"].split(":", 1)[1], 5),
                          t["id"].startswith(("ppp", "ipp", "ppl", "stylo"))))


# ------------------------- subprocess runner --------------------------------
STATUS, HEARTBEAT_S = {}, 60


def clock():
    el = int(time.monotonic() - T0)
    return f"[{el//3600}:{el%3600//60:02d}]"


def save_state():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(
        {"budget_hours": BUDGET_HOURS, "tasks": STATUS}, indent=2), encoding="utf-8")


def run_task(t, seq, n):
    tail = deque(maxlen=80)
    env = dict(os.environ)
    env.update(t["env_extra"])
    if t["env_extra"]:
        print(f"{clock()} [env] {t['env_extra']}", flush=True)
    print(f"\n{clock()} ===== {seq}/{n}: {t['id']} (est ~{t['est']}m | "
          f"{hours_left():.1f}h left) =====\n{clock()} $ {' '.join(t['cmd'])}", flush=True)
    start = time.monotonic()
    proc = subprocess.Popen(t["cmd"], cwd=str(REPO_DIR), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    q = queue.Queue()
    threading.Thread(target=lambda: ([q.put(ln) for ln in proc.stdout], q.put(None)),
                     daemon=True).start()
    killed = eof = False
    last = time.monotonic()
    while not eof:
        try:
            ln = q.get(timeout=HEARTBEAT_S)
        except queue.Empty:
            print(f"{clock()} [heartbeat] {t['id']} up "
                  f"{(time.monotonic()-start)/60:.0f}m, quiet "
                  f"{(time.monotonic()-last)/60:.0f}m (downloads/loading are silent)",
                  flush=True)
        else:
            if ln is None:
                eof = True
            else:
                print(ln, end="", flush=True)
                tail.append(ln.rstrip())
                last = time.monotonic()
        if hours_left() <= 0 and not killed:
            print(f"\n{clock()} [budget] time up — pausing {t['id']} "
                  "(resumes next session)", flush=True)
            proc.terminate()
            killed = True
    rc = proc.wait()
    STATUS[t["id"]]["tail"] = list(tail)[-15:]
    STATUS[t["id"]]["minutes"] = round((time.monotonic() - start) / 60, 1)
    if killed:
        return "paused"
    if rc != 0:
        low = "\n".join(tail).lower()
        if any(x in low for x in ("401", "403", "gated", "unauthorized", "restricted")):
            return "auth-error"
        if "out of memory" in low or "cuda oom" in low:
            return "oom"
        return "failed"
    return "ran" if t["done_fn"]() else "partial"


# NOTE: there is deliberately NO in-kernel 27B "preflight". An earlier version
# loaded the 27B in this notebook kernel to sanity-check sharded generation+NLL;
# but if that generation hits a CUDA device-side assert (e.g. inf/nan logits
# from a float16 fallback) it poisons the kernel's CUDA context and kills the
# whole run. Instead, ALL model work runs in isolated subprocesses (the src/
# scripts), so a 27B failure is contained: the task is recorded as failed/oom
# and Tier 1 + the 2B/9B curve still ship. Sharded generation AND per-token NLL
# are exercised for real by the subprocess `gen:` and `ppl:` tasks for the 27B.


# ------------------------------- run loop -----------------------------------
print(f"\n{clock()} evaluating what is already done ...")
runnable = []
print(f"\n{'TASK':<32}{'STATE':<10}{'EST':>5}")
for t in TASKS:
    done = False
    try:
        done = t["done_fn"]()
    except Exception:
        done = False
    STATUS[t["id"]] = {"status": "done" if done else "todo"}
    if not done:
        runnable.append(t)
    print(f"{t['id']:<32}{STATUS[t['id']]['status']:<10}{t['est']:>4}m")
print(f"\n{clock()} {len(TASKS)-len(runnable)}/{len(TASKS)} already done; "
      f"~{sum(t['est'] for t in runnable)/60:.1f}h of work remains; budget {BUDGET_HOURS}h")
save_state()

if HAVE_GPU and not FOUR_BIT_OK and not DRY_RUN:
    print(f"{clock()} [4bit] GPU present but 4-bit NF4 unavailable -> GPU tasks will be "
          "SKIPPED (float16 would break comparability and not fit 27B). "
          "Stats/outputs still run on whatever judgments already exist.")

seq = 0
for t in runnable:
    st = STATUS[t["id"]]
    deps = [d for d in t["after"] if STATUS.get(d, {}).get("status") not in ("done", "ran")]
    if deps:
        st.update(status="blocked", note=f"waiting: {','.join(deps)}")
        print(f"{clock()} [skip] {t['id']}: blocked by {','.join(deps)}")
    elif DRY_RUN:
        st.update(status="would-run")
        print(f"{clock()} [dry-run] would run {t['id']} (~{t['est']}m)")
    elif t["gpu"] and not HAVE_GPU:
        st.update(status="skipped", note="no GPU")
        print(f"{clock()} [skip] {t['id']}: no GPU this session")
    elif t["gpu"] and not FOUR_BIT_OK:
        st.update(status="skipped", note="4-bit NF4 unavailable (refusing float16)")
        print(f"{clock()} [skip] {t['id']}: 4-bit NF4 unavailable")
    elif t["gated"] and not HAVE_HF_TOKEN:
        st.update(status="skipped", note="needs HF_TOKEN + accepted license")
        print(f"{clock()} [skip] {t['id']}: needs HF_TOKEN secret + license")
    elif hours_left() <= 0:
        st.update(status="deferred", note="budget spent")
        print(f"{clock()} [defer] {t['id']}: budget spent")
    else:
        seq += 1
        outcome = run_task(t, seq, len(runnable))
        st.update(status=outcome)
        print(f"{clock()} [task] {t['id']} -> {outcome.upper()} "
              f"({st.get('minutes','?')}m vs est {t['est']}m)")
    save_state()

_c = {}
for v in STATUS.values():
    _c[v["status"]] = _c.get(v["status"], 0) + 1
print(f"\n{clock()} [orchestrator] pass complete — "
      + " ".join(f"{k}:{v}" for k, v in sorted(_c.items()))
      + f" | {hours_left():.1f}h budget left")
print("Rerun this notebook (fresh session, re-attach prior output) to continue "
      "any 'paused'/'deferred'/'skipped' tasks — completed units are skipped.")
