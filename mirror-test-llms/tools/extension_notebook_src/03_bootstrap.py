# =========================== BOOTSTRAP ======================================
# Fail-loud environment setup. This cell NEVER resamples prompts or regenerates
# reused models: it only locates the frozen inputs, copies them into a writable
# tree, verifies their integrity, and wires up imports. If anything required is
# missing it raises immediately with a report of what was found vs expected —
# silently regenerating would break comparability with the published numbers.
# ===========================================================================

import os, sys, shutil, subprocess, importlib.util
from pathlib import Path

WORK = Path(WORKING_ROOT)
(WORK / "configs").mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---- 1. pip: refresh ONLY the fragile bits, never pin torch ----------------
# An earlier version pinned the whole stack (bitsandbytes==0.43.3, ...). On
# Kaggle that broke the bitsandbytes<->torch CUDA binding, so `import
# bitsandbytes` failed and utils silently fell back to float16 (non-comparable
# to the paper's 4-bit judges, and it triggered an inf/nan generation crash).
# Fix: leave Kaggle's working torch alone and only (re)fresh bitsandbytes /
# transformers / scikit-learn when needed. datasets + sentence-transformers are
# NOT installed (the extension never builds prompts or paraphrases).
def _imports(code: str) -> bool:
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True).returncode == 0


def _pip(*pkgs):
    try:
        _sh([sys.executable, "-m", "pip", "-q", "install", "-U", *pkgs])
    except subprocess.CalledProcessError as e:
        print(f"[pip] WARNING: install {pkgs} returned {e.returncode}; continuing.")


if not LOCAL_TEST:
    if not _imports("import bitsandbytes"):
        print("[pip] bitsandbytes not importable -> installing a CUDA-matching build")
        _pip("bitsandbytes")            # unpinned: pip picks the build for this torch/CUDA
    # gemma-2 needs transformers >= 4.44; only upgrade if Kaggle's is older
    if not _imports("import transformers as t,sys;v=t.__version__.split('.');"
                    "sys.exit(0 if (int(v[0]),int(v[1]))>=(4,44) else 1)"):
        _pip("transformers>=4.44,<5", "accelerate>=0.33")
    if not _imports("import sklearn"):   # stylometry baseline
        _pip("scikit-learn>=1.3")

# ---- 2. obtain the repo: override / attached dataset -> git clone ----------
# Upload-and-run: by default we CLONE the pushed repo (code + frozen data +
# published results), so there is nothing to attach. If the repo is already
# attached as a Dataset or given via INPUT_ROOT_OVERRIDE, that is used instead.
# Frozen data is never regenerated.
def _has_repo(p: Path) -> bool:
    return (p / "mirror-test-llms" / "src" / "utils.py").exists()


def obtain_repo() -> Path:
    # (a) explicit override / LOCAL_TEST
    if INPUT_ROOT_OVERRIDE:
        p = Path(INPUT_ROOT_OVERRIDE)
        if not _has_repo(p) and (p / "src" / "utils.py").exists():
            p = p.parent            # user pointed straight at the repo dir
        if _has_repo(p):
            print(f"[repo] using INPUT_ROOT_OVERRIDE: {p}")
            return p
        raise SystemExit(f"[fatal] INPUT_ROOT_OVERRIDE has no mirror-test-llms/src: {p}")
    # (b) repo attached as a Kaggle Dataset (no clone needed)
    base = Path("/kaggle/input")
    if base.exists():
        cands = sorted({c.parent.parent.parent
                        for c in base.rglob("mirror-test-llms/src/utils.py")},
                       key=lambda x: len(str(x)))
        if cands:
            print(f"[repo] using attached dataset: {cands[0]}")
            return cands[0]
    # (c) git clone the pushed repo (the default upload-and-run path)
    clone_root = WORK / "_repo_src"
    if _has_repo(clone_root):
        print(f"[repo] reusing existing clone: {clone_root}")
        return clone_root
    url = REPO_GIT_URL
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok and url.startswith("https://github.com/"):
        url = url.replace("https://", f"https://{tok}@")   # private-repo auth
    print(f"[repo] cloning {REPO_GIT_URL}@{REPO_GIT_REF} -> {clone_root} "
          "(shallow; needs Internet = ON) ...", flush=True)
    # NB: run directly (not via _sh) so a token in `url` never prints to the log.
    rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_GIT_REF,
                         url, str(clone_root)]).returncode
    if rc != 0 or not _has_repo(clone_root):
        raise SystemExit(
            f"[fatal] could not clone {REPO_GIT_URL}@{REPO_GIT_REF}. Ensure the repo is "
            "pushed and public (or set a GITHUB_TOKEN secret for a private repo), "
            "Internet is ON, or attach the repo as a Dataset / set INPUT_ROOT_OVERRIDE.")
    return clone_root


INPUT_ROOT = obtain_repo()
REPO_DIR = INPUT_ROOT / "mirror-test-llms"
SRC_DIR = REPO_DIR / "src"
ORIG_CONFIG = REPO_DIR / "configs" / "models.yaml"
print(f"[repo] project dir: {REPO_DIR}")
for need in (SRC_DIR / "utils.py", ORIG_CONFIG, REPO_DIR / "data" / "prompts"):
    if not need.exists():
        raise FileNotFoundError(f"[repo] required path missing in repo: {need}")

# ---- 3. environment: writable root, HF cache, token ------------------------
os.environ["MIRROR_ROOT"] = str(WORK)
try:                                     # big weight cache off the 20 GB working cap
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
    print(f"[hf] weight cache: {HF_CACHE_DIR}")
except OSError:
    print(f"[hf] {HF_CACHE_DIR} not writable; using default ~/.cache/huggingface "
          "(fine for the <=9B Tier-1 judges)")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("[auth] HF_TOKEN loaded from Kaggle secret")
    except Exception:
        print("[auth] no HF_TOKEN yet (gated Gemma-2/Llama will 403 until you add "
              "the Kaggle secret and accept the licenses)")

# ---- 4. copy frozen inputs into the writable tree (never clobber) ----------
def seed_tree(src_base: Path, subdirs, label: str) -> None:
    copied = skipped = 0
    for sub in subdirs:
        sdir = src_base / sub
        if not sdir.exists():
            continue
        for f in sdir.rglob("*"):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            dst = WORK / f.relative_to(src_base)
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied += 1
    print(f"[seed:{label}] copied {copied} files, kept {skipped} already-present")


# 4a. frozen data + published results from the repo dataset
seed_tree(REPO_DIR, ["data", "results"], "repo")
# 4b. any RE-ATTACHED prior session output (resume across sessions): other
#     /kaggle/input dirs carrying our working artifacts. Newer local wins.
if Path("/kaggle/input").exists():
    for cand in Path("/kaggle/input").iterdir():
        if cand.resolve() == INPUT_ROOT.resolve():
            continue
        if any((cand / s).exists() for s in
               ("data/generations", "results/judgments", "checkpoints",
                "extended_outputs")):
            seed_tree(cand, ["data", "results", "checkpoints", "extended_outputs"],
                      f"prior:{cand.name}")
            src_ck = cand / "checkpoints"
            if src_ck.exists():
                for f in src_ck.rglob("*"):
                    if f.is_file():
                        dst = Path(CHECKPOINT_DIR) / f.relative_to(src_ck)
                        if not dst.exists():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(f, dst)

# ---- 5. imports (utils + stats + numbered pipeline modules) ----------------
sys.path.insert(0, str(SRC_DIR))
import utils, stats_utils                                  # noqa: E402


def load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(SRC_DIR / filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


mod_build = load_mod("mt_build_pairs", "02_build_pairs.py")
mod_base  = load_mod("mt_baselines", "05_baselines.py")
mod_stats = load_mod("mt_stats", "06_stats.py")
CFG = utils.load_config(str(ORIG_CONFIG))
print(f"[cfg] master seed={CFG['seed']}  domains={utils.DOMAINS}  "
      f"gen(temp={CFG['generation']['temperature']}, "
      f"seed_base={CFG['generation']['seed_base']}, "
      f"placebo_base={CFG['generation']['placebo_seed_base']})")

# ---- 6. verify frozen prompts (checksums) + required generations -----------
report = {"ok": [], "missing": [], "checksum_bad": []}
checks = (REPO_DIR / "data" / "prompts" / "CHECKSUMS.txt")
sums = {}
if checks.exists():
    for line in checks.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            h, name = line.split()[:2]
            sums[name] = h
for d in utils.DOMAINS:
    p = WORK / "data" / "prompts" / f"{d}.jsonl"
    if not p.exists():
        report["missing"].append(f"prompts/{d}.jsonl")
    elif d + ".jsonl" in sums and utils.sha256_file(p) != sums[d + ".jsonl"]:
        report["checksum_bad"].append(f"prompts/{d}.jsonl")
    else:
        report["ok"].append(f"prompts/{d}.jsonl")

N_TARGET = CFG["prompt_filters"]["n_per_domain"]
# Text that must ALREADY exist (never regenerated): every LLM foil in the pool
# AND every new judge (all are promoted foils), all domains, >= N_TARGET s1 recs.
need_s1 = set(FOIL_POOL_LLM) | {j["key"] for j in NEW_JUDGES}
for key in sorted(need_s1):
    for d in utils.DOMAINS:
        f = WORK / "data" / "generations" / f"{key}__{d}.jsonl"
        recs = utils.read_jsonl(f)
        n_s1 = sum(1 for r in recs if r.get("sample", "s1") == "s1")
        if n_s1 >= N_TARGET:
            report["ok"].append(f"gen/{key}__{d} (s1={n_s1})")
        else:
            report["missing"].append(f"gen/{key}__{d} (s1={n_s1}<{N_TARGET})")

qwen_judg = sorted((WORK / "results" / "judgments").glob("ppp__qwen2.5-*.jsonl"))
print("\n[startup] frozen-input verification")
print(f"    OK           : {len(report['ok'])} required files present")
print(f"    reused Qwen  : {len(qwen_judg)} judgment files "
      f"(needed for the Qwen curve + Holm family)")
if report["missing"] or report["checksum_bad"]:
    for m in report["missing"]:
        print(f"    MISSING      : {m}")
    for m in report["checksum_bad"]:
        print(f"    CHECKSUM BAD : {m}")
    raise SystemExit(
        "[fatal] required frozen inputs are missing or altered — refusing to "
        "proceed (regenerating them would break comparability with the paper). "
        "Attach the correct repo Dataset and rerun.")
if not qwen_judg:
    print("    [warn] no Qwen judgments found — the extension will still run, but the "
          "dissociation figure/tables will cover the new judges only, and Holm will "
          "cover the new cells only.")

# ---- 7. write the patched config (adds new judges; foils=[human]) ----------
# WHY foils=[human]: model_index() lets later groups shadow earlier ones, so a
# key present in BOTH judges and foils resolves to group 'foils' and would make
# 01_generate treat it as a non-judge (no placebo s2). Keeping the promoted
# families out of `foils` guarantees is_judge=True so their s2 sample generates.
# Foil TEXT is still supplied explicitly via --foils to 02_build_pairs and read
# straight off disk, so nothing depends on the config foils list.
import copy, yaml                                          # noqa: E402
orig_foils = {m["key"]: m for m in CFG["foils"]}
drift = []
for j in NEW_JUDGES:
    if j["key"] in orig_foils:          # all new judges are promoted foils
        want = orig_foils[j["key"]].get("revision")
        if want and j["revision"] and want != j["revision"]:
            drift.append((j["key"], j["revision"], want))
if drift:
    for k, mine, theirs in drift:
        print(f"    REVISION DRIFT {k}: config-cell={mine[:12]} vs frozen={theirs[:12]}")
    raise SystemExit("[fatal] reused-judge revision mismatch vs frozen config — "
                     "fix NEW_JUDGES revisions in the config cell to match.")

patched = copy.deepcopy(CFG)
patched.pop("_config_path", None)
existing_judge_keys = {m["key"] for m in patched["judges"]}
for j in NEW_JUDGES:
    if j["key"] in existing_judge_keys:
        continue
    patched["judges"].append({"key": j["key"], "hf_id": j["hf_id"],
                              "revision": j["revision"], "params_b": j["params_b"]})
patched["foils"] = [m for m in CFG["foils"] if m["key"] == "human"] or \
    [{"key": "human", "hf_id": None, "params_b": None}]
EXT_CONFIG = WORK / "configs" / "models_ext.yaml"
EXT_CONFIG.write_text(yaml.safe_dump(patched, sort_keys=False, allow_unicode=True),
                      encoding="utf-8")
print(f"\n[cfg] wrote patched config -> {EXT_CONFIG} "
      f"({len(patched['judges'])} judges, foils={[m['key'] for m in patched['foils']]})")

# ---- 8. GPU report + seeds + versions --------------------------------------
HAVE_GPU = False
try:
    import torch                                            # noqa: E402
    HAVE_GPU = torch.cuda.is_available()
    if HAVE_GPU:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"[gpu] cuda:{i} {p.name} ({p.total_memory/2**30:.1f} GB)")
        if torch.cuda.device_count() >= 2:
            print("[gpu] 2 GPUs present (each Tier-1 judge fits on one T4 in 4-bit).")
    else:
        print("[gpu] no CUDA visible (Settings -> Accelerator -> GPU T4 x2).")
    torch.manual_seed(CFG["seed"])
except Exception as e:
    print(f"[gpu] torch unavailable ({e})")
import random as _random                                    # noqa: E402
_random.seed(CFG["seed"])
try:
    import numpy as _np; _np.random.seed(CFG["seed"])       # noqa: E402
except Exception:
    pass

# ---- 8b. verify 4-bit NF4 actually works (in a SUBPROCESS, so the kernel
#      never initialises a CUDA context that a device-side assert could poison).
#      The study REQUIRES 4-bit for comparability with the paper; if it is not
#      available we SKIP GPU work rather than silently run in float16 (which is
#      non-comparable and, on the earlier run, triggered an inf/nan crash).
FOUR_BIT_OK = False
if HAVE_GPU and not LOCAL_TEST:
    chk = subprocess.run(
        [sys.executable, "-c",
         "import torch,bitsandbytes as b;from transformers import BitsAndBytesConfig;"
         "assert torch.cuda.is_available();"
         "BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',"
         "bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16);"
         "print('BNB_OK',b.__version__)"],
        capture_output=True, text=True)
    FOUR_BIT_OK = chk.returncode == 0 and "BNB_OK" in chk.stdout
    if FOUR_BIT_OK:
        print(f"[4bit] bitsandbytes 4-bit NF4 usable ({chk.stdout.strip()})")
    else:
        print("[4bit] *** bitsandbytes 4-bit UNAVAILABLE ***\n"
              f"       {(chk.stderr or chk.stdout).strip()[:300]}\n"
              "       GPU tasks will be SKIPPED (running in float16 would break "
              "comparability with the paper's 4-bit judges). Fix: make "
              "`import bitsandbytes` succeed on this image, then rerun.")

HAVE_HF_TOKEN = bool(os.environ.get("HF_TOKEN"))
print("\n[versions]")
for name in ("transformers", "accelerate", "bitsandbytes", "datasets",
             "sklearn", "matplotlib", "torch"):
    try:
        print(f"    {name}: {importlib.import_module(name).__version__}")
    except Exception:
        print(f"    {name}: (not importable)")
print(f"[env] HAVE_GPU={HAVE_GPU}  HAVE_HF_TOKEN={HAVE_HF_TOKEN}  "
      f"MIRROR_ROOT={os.environ['MIRROR_ROOT']}")
print("[bootstrap] done.")
