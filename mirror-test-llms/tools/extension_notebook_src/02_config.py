# ============================== RUN SETTINGS ================================
# The ONE place to change paths / models / budget / pool. Everything below and
# in later cells reads these globals. Nothing experiment-relevant is hidden in
# the other cells. Printed at the end so a reviewer can audit it at a glance.
#
# Design rule (matches the repo): seeds, decoding params, prompt templates and
# statistical settings are NOT redefined here — they are read from the frozen
# configs/models.yaml in the attached dataset, so the new cells are identical
# to the published ones. This cell only adds the NEW judges + the foil pool +
# the Kaggle plumbing the repo cannot know about.
# ===========================================================================

import os

# ---- Where the repo + data come from ---------------------------------------
# Self-contained, upload-and-run (like kaggle_run_all): the bootstrap cell
# CLONES the pushed repo — code + frozen data + published results — so there is
# NO Kaggle Dataset to attach. The notebook already needs Internet for the model
# weights, so cloning adds no new requirement. If you DO attach the repo as a
# Dataset it is auto-detected and used without cloning; INPUT_ROOT_OVERRIDE
# forces a specific local path (used by LOCAL_TEST).
REPO_GIT_URL        = "https://github.com/shehryars715/MirrorTest.git"
REPO_GIT_REF        = "main"        # branch / tag / commit to clone
INPUT_ROOT_OVERRIDE = None          # force an attached/local repo dir instead of cloning

WORKING_ROOT        = "/kaggle/working"        # writable; MIRROR_ROOT points here
CHECKPOINT_DIR      = "/kaggle/working/checkpoints"
OUTPUT_DIR          = "/kaggle/working/extended_outputs"
# HF weight cache: deliberately NOT under /kaggle/working — that mount has a
# ~20 GB persisted cap and gemma-2-27b-it alone is ~54 GB. /kaggle/temp is the
# large ephemeral scratch. Bootstrap falls back to the default ~/.cache if this
# is not writable.
HF_CACHE_DIR        = "/kaggle/temp/hf_cache"

# ---- Session control -------------------------------------------------------
BUDGET_HOURS = 11.0     # stop launching new work after this; resume next session
DRY_RUN      = False    # True = print the plan and skip all GPU work
ENABLE_27B   = True     # gemma-2-27b-it (sharded 2xT4). Set False to cap Gemma-2
                        # at 2B/9B and guarantee single-session completion (§18
                        # of the protocol sanctions a truncated scale axis).
RUN_STYLOMETRY = True   # CPU char n-gram baseline for the new cells
RUN_IPP        = True   # Yes/No protocol for the new judges

# LOCAL_TEST lets the CPU-only cells (bootstrap file-checks, stats, plotting)
# be exercised off-Kaggle against a normal checkout. When True, INPUT_ROOT and
# WORKING_ROOT are taken from the two env vars below and all GPU tasks are
# skipped. On Kaggle leave this False.
LOCAL_TEST = bool(os.environ.get("MIRROR_LOCAL_TEST"))
if LOCAL_TEST:
    INPUT_ROOT_OVERRIDE = os.environ["MIRROR_LOCAL_INPUT"]      # repo parent dir
    WORKING_ROOT        = os.environ["MIRROR_LOCAL_WORKING"]
    CHECKPOINT_DIR      = WORKING_ROOT + "/checkpoints"
    OUTPUT_DIR          = WORKING_ROOT + "/extended_outputs"
    HF_CACHE_DIR        = WORKING_ROOT + "/hf_cache"
    DRY_RUN = True

# ---- Statistics ------------------------------------------------------------
# The published cell_stats() uses 500 resamples for the Table-1 AUROC CI and
# the config's 1000 for kappa. We recompute EVERY cell (old + new) here with a
# single uniform N so all rows in the extended outputs are mutually comparable;
# point estimates are identical to the paper, only CI width differs by <~0.005.
AUROC_BOOTSTRAP_N = 1000

# ---- The NEW judges --------------------------------------------------------
# family_of(key) classifies by prefix; a judge never foils against its own
# family. Revisions for the three REUSED models are copied verbatim from the
# frozen configs/models.yaml (bootstrap VERIFIES they still match — fail loud
# on drift). The two newly-generated Gemma sizes are not yet pinned: we load
# 'latest', RECORD the resolved commit into run_report.txt, and you pin it for
# camera-ready (exactly how the repo pinned its others; see PREREGISTRATION
# deviations log).
NEW_JUDGES = [
    # Tier 1 — promote existing foils to judges (text exists; only s2 is new)
    {"key": "llama-3.2-3b-instruct",     "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
     "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95", "params_b": 3,
     "tier": 1, "needs_s1": False},
    {"key": "gemma-2-9b-it",             "hf_id": "google/gemma-2-9b-it",
     "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819", "params_b": 9,
     "tier": 1, "needs_s1": False},
    {"key": "mistral-7b-instruct-v0.3",  "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
     "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71", "params_b": 7,
     "tier": 1, "needs_s1": False},
    # Tier 2 — new Gemma-2 sizes for the second scale curve (full generation)
    {"key": "gemma-2-2b-it",             "hf_id": "google/gemma-2-2b-it",
     "revision": None, "params_b": 2, "tier": 2, "needs_s1": True},
    {"key": "gemma-2-27b-it",            "hf_id": "google/gemma-2-27b-it",
     "revision": None, "params_b": 27, "tier": 2, "needs_s1": True, "shard": True},
]

# Per-GPU 4-bit shard budget for models that MUST split across both T4s.
# Consumed by src/utils.load_model_and_tokenizer via env MIRROR_MAX_MEMORY.
MAX_MEMORY_SHARDED = "13GiB,13GiB"

# ---- The common foil pool (family-exclusion) -------------------------------
FOIL_POOL_LLM = ["llama-3.2-3b-instruct", "gemma-2-9b-it", "mistral-7b-instruct-v0.3"]
HUMAN_FOIL    = "human"

# Qwen judges are reused as-is for the family curve + Holm family; params from
# the frozen config (bootstrap cross-checks).
QWEN_JUDGES = [
    ("qwen2.5-0.5b-instruct", 0.5), ("qwen2.5-1.5b-instruct", 1.5),
    ("qwen2.5-3b-instruct", 3), ("qwen2.5-7b-instruct", 7),
    ("qwen2.5-14b-instruct", 14),
]


def family_of(key: str) -> str:
    if key.startswith("qwen2.5"):
        return "qwen"
    if key.startswith("gemma-2"):
        return "gemma"
    if key.startswith("mistral"):
        return "mistral"
    if key.startswith("llama"):
        return "llama"
    if key == "human":
        return "human"
    return key


def foils_for(judge_key: str) -> list:
    """Common pool minus the judge's own family, plus human. All foil text
    pre-exists, so this never triggers new generation."""
    jf = family_of(judge_key)
    foils = [f for f in FOIL_POOL_LLM if family_of(f) != jf]
    return foils + [HUMAN_FOIL]


# ---- Pinned pip environment (T4-compatible) --------------------------------
# bitsandbytes 4-bit NF4 with float16 compute (T4 is Turing: no bf16). These
# pins are known-good on Kaggle's CUDA 12 image; loosen only if Kaggle's base
# image moves under you.
PIP_PACKAGES = [
    "transformers==4.44.2", "accelerate==0.34.2", "bitsandbytes==0.43.3",
    "datasets==2.21.0", "sentence-transformers==3.0.1", "scikit-learn==1.5.1",
    "pyyaml==6.0.2", "matplotlib==3.9.2",
]

if not (0 < len(NEW_JUDGES) == len({j["key"] for j in NEW_JUDGES})):
    raise ValueError("NEW_JUDGES keys must be unique and non-empty")

print("=" * 74)
print("MIRROR-TEST CROSS-FAMILY EXTENSION — run settings")
print("=" * 74)
print(f"  repo source    : clone {REPO_GIT_URL}@{REPO_GIT_REF}"
      f"{'  (OVERRIDE: ' + str(INPUT_ROOT_OVERRIDE) + ')' if INPUT_ROOT_OVERRIDE else ''}")
print(f"  working root   : {WORKING_ROOT}")
print(f"  checkpoints    : {CHECKPOINT_DIR}")
print(f"  outputs        : {OUTPUT_DIR}")
print(f"  HF cache       : {HF_CACHE_DIR}")
print(f"  budget (h)     : {BUDGET_HOURS}   DRY_RUN={DRY_RUN}  LOCAL_TEST={LOCAL_TEST}")
print(f"  enable 27B     : {ENABLE_27B}   (sharded {MAX_MEMORY_SHARDED} if on)")
print(f"  AUROC bootstrap: {AUROC_BOOTSTRAP_N}")
print("  new judges & family-excluded foils:")
for j in NEW_JUDGES:
    if j["key"] == "gemma-2-27b-it" and not ENABLE_27B:
        continue
    rev = (j["revision"] or "latest (will be recorded + flagged to pin)")
    print(f"    - {j['key']:26s} [{j['params_b']:>4}B, tier {j['tier']}, "
          f"fam={family_of(j['key'])}]  rev={rev[:16]}")
    print(f"        foils: {', '.join(foils_for(j['key']))}")
print("  reused Qwen judges: " + ", ".join(k for k, _ in QWEN_JUDGES))
print("=" * 74)
