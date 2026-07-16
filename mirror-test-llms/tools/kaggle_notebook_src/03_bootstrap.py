# =========================== BOOTSTRAP ======================================
# Unpacks the embedded pipeline code, installs dependencies, authenticates
# to Hugging Face, and pulls in everything already computed by previous
# sessions (attached as Inputs). Idempotent - safe to run every time.

import base64, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else \
    Path(os.environ.get("MIRROR_TEST_DIR", str(Path.home() / "mirror-run")))
REPO_DIR = WORK / "mirror-test-llms"
REPO_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. unpack the embedded repo (code + configs + tests; never data) ------
PAYLOAD_B64 = "__PAYLOAD_B64__"
_zip_bytes = base64.b64decode(PAYLOAD_B64)
with zipfile.ZipFile(io.BytesIO(_zip_bytes)) as zf:
    zf.extractall(REPO_DIR)   # overwrites code; data/ and results/ are not in the zip
print(f"[bootstrap] pipeline code unpacked -> {REPO_DIR} "
      f"({len(_zip_bytes)//1024} KB payload)")

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))

# ---- 2. dependencies (Kaggle image has torch; top up the rest) -------------
if ON_KAGGLE:
    print("[bootstrap] installing/upgrading libraries (2-4 min) ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "transformers", "accelerate", "bitsandbytes", "datasets",
                    "sentence-transformers", "pyyaml", "scikit-learn"],
                   check=False)

import utils  # noqa: E402  (from the unpacked src/)

# ---- 3. Hugging Face auth (Kaggle secret HF_TOKEN or env var) --------------
utils.setup_hf_auth()
HAVE_HF_TOKEN = bool(os.environ.get("HF_TOKEN"))

# ---- 4. GPU inventory -------------------------------------------------------
HAVE_GPU = False
try:
    import torch
    HAVE_GPU = torch.cuda.is_available()
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[gpu] cuda:{i} {p.name} ({p.total_memory/2**30:.1f} GB)")
except Exception as e:  # noqa: BLE001
    print(f"[gpu] torch unavailable: {e}")
if not HAVE_GPU:
    print("[gpu] NO GPU - GPU tasks will be SKIPPED. On Kaggle: Session "
          "options -> Accelerator -> GPU T4 x2, then Save & Run All again.")

# ---- 5. seed data/results from every attached previous output --------------
def _seed_tree(base: Path) -> tuple[int, int]:
    """Copy data/ + results/ files from `base` unless a same-or-bigger local
    copy exists (JSONL outputs only ever grow, so bigger == newer)."""
    copied = kept = 0
    for sub in ("data", "results"):
        src = base / sub
        if not src.is_dir():
            continue
        for f in src.rglob("*"):
            if not f.is_file() or f.suffix == ".zip":
                continue
            dst = REPO_DIR / f.relative_to(base)
            if dst.exists() and dst.stat().st_size >= f.stat().st_size:
                kept += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied += 1
    return copied, kept

def _candidate_bases(root: Path):
    """Places that look like a previous session's tree (possibly nested)."""
    for cand in [root, root / "mirror-test-llms"] + sorted(root.glob("*/")):
        if (cand / "data").is_dir() or (cand / "results").is_dir():
            yield cand

total_seeded = 0
inputs_root = Path("/kaggle/input")
if ON_KAGGLE:
    listed = sorted(inputs_root.iterdir()) if inputs_root.is_dir() else []
    print(f"[seed] attached Inputs: "
          f"{[p.name for p in listed] if listed else 'NONE'}")
    seen = set()
    for inp in listed:
        # a re-uploaded bundle zip? extract it to temp first
        for z in inp.rglob("mirror_bundle*.zip"):
            tmp = WORK / f"_seed_{z.stem}"
            if not tmp.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(tmp)
            inp = tmp  # fall through to directory scan
        for base in _candidate_bases(inp):
            if base in seen:
                continue
            seen.add(base)
            c, k = _seed_tree(base)
            total_seeded += c + k
            print(f"[seed] {base}: copied {c} files, kept {k} local")

# ---- 6. GitHub safety net: recover past results even with no Input ---------
# The laptop-side workflow commits data/ + results/ to a public repo after
# every ingested session, so a forgotten "+ Add Input" no longer restarts
# the project from zero.
_git_url = globals().get("SEED_GIT_URL", "")
if _git_url and total_seeded == 0 and (ON_KAGGLE or os.environ.get("MIRROR_FORCE_GIT_SEED")):
    print(f"[seed] no data found in Inputs - trying git fallback: {_git_url}")
    clone_dir = WORK / "_seed_git"
    try:
        if not clone_dir.exists():
            subprocess.run(["git", "clone", "--depth", "1", _git_url,
                            str(clone_dir)], check=True, capture_output=True,
                           text=True, timeout=300)
        found = False
        for base in _candidate_bases(clone_dir):
            c, k = _seed_tree(base)
            if c or k:
                found = True
                total_seeded += c + k
                print(f"[seed] git: copied {c} files, kept {k} local from {base}")
        if not found:
            print("[seed] git repo cloned but contained no data/ or results/ yet")
    except Exception as e:  # noqa: BLE001
        print(f"[seed] git fallback failed ({e}) - continuing without it")

if ON_KAGGLE and total_seeded == 0:
    print("=" * 72)
    print("[seed] WARNING: no previous results found (no usable Input, no git")
    print("[seed] data). If this is your FIRST session, that is normal.")
    print("[seed] If it is NOT, cancel this run, click '+ Add Input' -> Your")
    print("[seed] Work -> previous version's Output, then Save & Run All.")
    print("=" * 72)

print(f"[bootstrap] ready. repo={REPO_DIR} gpu={HAVE_GPU} hf_token={HAVE_HF_TOKEN} "
      f"seeded_files={total_seeded}")
