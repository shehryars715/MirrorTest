# Free GPUs: Colab & Kaggle survival guide

The whole study fits in free tiers (~15–25 GPU-hours, protocol §15) — IF you
work in a disconnect-proof way. This guide is that way.

---

## 1. The two free options

| | Google Colab (free) | Kaggle Notebooks |
|---|---|---|
| GPU | 1× T4 (16 GB), availability varies | 2× T4 or 1× P100, **30 h/week quota** |
| Session limit | ~up to 12 h, disconnects WILL happen | 12 h, more predictable |
| Persistent disk | no (use Drive or git) | /kaggle/working persists per session output |
| Best for | most runs (0.5B–7B) | the 14B judge (2× T4), long runs |

Strategy: do everything on Colab; switch to Kaggle for the 14B model and
whenever Colab won't give you a GPU. Enable the GPU in Colab via
*Runtime → Change runtime type → T4 GPU* (check with `!nvidia-smi`).

## 2. One-time setup: Hugging Face account + gated models

1. Create an account at huggingface.co.
2. While logged in, open each gated model page and click "Agree/Access":
   * `meta-llama/Llama-3.2-3B-Instruct` (approval is usually quick)
   * `google/gemma-2-9b-it`
3. Settings → Access Tokens → create a token with **read** scope. Treat it
   like a password (never commit it).
4. In every GPU session, authenticate before loading gated models:
   ```python
   from huggingface_hub import login
   login()          # paste the token when prompted
   ```
   On Kaggle you can instead store it as a Secret named `HF_TOKEN` and do
   `login(token=UserSecretsClient().get_secret("HF_TOKEN"))`.

If you skip this you will hit `401 Unauthorized` / `GatedRepoError` — see
the troubleshooting guide.

## 3. The disconnect-proof workflow (this is the important part)

The pipeline is already crash-safe: every script appends each finished item
to its JSONL and re-running skips finished items. Your ONLY job is to make
sure the output files survive the session. Two good patterns:

**Pattern A — git as the checkpoint store (recommended).**
Results are small (MBs). Push them.

```bash
!git clone https://github.com/<you>/mirror-test-llms
%cd mirror-test-llms
!pip -q install -U transformers accelerate bitsandbytes datasets sentence-transformers pyyaml

# ... run a pipeline stage ...

!git config user.email "you@example.com" && git config user.name "Your Name"
!git add -A && git commit -m "generations: qwen 3b (colab)" && git push
# for pushing from Colab, use a GitHub personal access token as the password,
# or `gh auth login` — see GitHub docs on HTTPS tokens.
```
Next session: clone again, and every script resumes exactly where it stopped.

**Pattern B — Google Drive as the working directory.**
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
# first time: !git clone ... ; later sessions: %cd mirror-test-llms
```
Everything the scripts write lands directly on Drive, surviving any crash.
Drive I/O is slower; fine for our file sizes.

**Either way:** never keep results only in the session's RAM/disk, and never
start a >2 h run without knowing where its output lives. (Protocol §15:
"write JSONL incrementally, never keep results only in RAM.")

## 4. Session budget plan (matches the 12-week timeline)

| Session | What | Approx GPU time |
|---|---|---|
| 1 | smoke test: stage-0 commands from the walkthrough | 20 min |
| 2–4 | generation: 0.5B + 1.5B + 3B (+placebo) | ~2.5 h total |
| 5 | generation: 7B (+placebo) | ~2 h |
| 6 (Kaggle) | generation: 14B (+placebo) | ~2.5 h |
| 7–8 | generation: 3 foils | ~3 h |
| 9 | paraphrase set (02b) | ~1 h |
| 10–13 | PPP judging: one judge each (+placebo) | ~0.5–1.5 h each |
| 14 | main-cell phrasings 0 1 2 + paraphrase judging | ~1 h |
| 15 | IPP all judges | ~1 h |
| 16 | perplexity baseline all judges | ~2 h |
| 17 | base 7B/14B judging + their perplexity proxy | ~1.5 h |

Total ≈ 18–20 GPU-hours — inside free tiers with room to redo things.

## 5. Colab-specific tips

* `!nvidia-smi` first — no GPU listed means CPU runtime; fix before running.
* Install cell (every session — Colab resets):
  `!pip -q install -U transformers accelerate bitsandbytes datasets sentence-transformers pyyaml`
* Keep the tab open and occasionally interact; long-idle tabs disconnect
  sooner. Don't fight it — resume is free.
* If you get `CUDA out of memory` on the 14B: use Kaggle 2×T4
  (`device_map="auto"` already spreads layers across both GPUs), or drop the
  14B row from the config and cap the scale axis at 7B (protocol §18
  explicitly blesses this).
* Model downloads count against session time: the 14B download alone is
  ~9 GB. Start big-model sessions with the download.

## 6. Kaggle: the supported flow (use `notebooks/kaggle_pipeline.ipynb`)

The repo has first-class Kaggle support — three pieces work together:

1. **`src/kaggle_setup.py`** — run it first in every session. It reports the
   GPUs, authenticates to Hugging Face from the notebook secret `HF_TOKEN`
   (no interactive login cell needed), verifies the data tree is writable
   (redirecting to `/kaggle/working` via the `MIRROR_ROOT` env var if the
   code sits on a read-only mount), and can seed `data/` + `results/` from a
   previous session's saved output: `--from-input <name>`.
2. **Automatic auth everywhere** — `load_model_and_tokenizer` calls the same
   auth helper, so even a bare `python src/01_generate.py ...` finds the
   token by itself.
3. **`MIRROR_ROOT`** — if set, ALL data/results reads and writes anchor at
   `<MIRROR_ROOT>/data` and `<MIRROR_ROOT>/results` instead of inside the
   repo. You only need it when the repo is attached as a read-only Kaggle
   *dataset*; when you `git clone` into `/kaggle/working` (recommended),
   everything is writable and no override is needed.

Session checklist:

* Settings → Accelerator: **GPU T4 ×2**; Settings → Internet: **ON**.
* Add-ons → Secrets: `HF_TOKEN` (HF read token; licenses accepted on the
  Llama/Gemma pages first). Optional: `GITHUB_PAT` for git-push checkpoints.
* Quota is 30 GPU-hours/week — check the meter in the sidebar.
* **Checkpointing without GitHub:** click *Save Version* at session end —
  the notebook output preserves `/kaggle/working` (your `data/` +
  `results/`). Next session, attach that output as an Input and run
  `python src/kaggle_setup.py --from-input <output-name>`; it copies files
  in without overwriting anything newer.
* The 14B judge and gemma-2-9b generation belong here: `device_map="auto"`
  shards them across both T4s automatically.

## 7. Making runs reproducible ACROSS sessions

* The seeds are in the config and in each record — nothing to do.
* Pin model revisions in `configs/models.yaml` before the real runs: open
  the model page → "Files and versions" → copy the commit hash into
  `revision:`. The generation records also store the resolved hash — put
  those in the paper appendix.
* Record library versions once per real run:
  `!pip freeze | grep -Ei "transformers|torch|bitsandbytes|accelerate" > env_snapshot.txt`
  and commit it.
