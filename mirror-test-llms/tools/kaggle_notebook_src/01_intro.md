# Mirror Test — one-click Kaggle runner

This notebook contains the **entire pipeline inside itself** (no GitHub
needed). Every time you run it, it works out what is already done, does as
much of the remaining GPU work as fits in the session, then packages the
results into **one file** for Claude.

---

## First time only (~15 minutes of clicking)

1. **Kaggle account**: kaggle.com → sign up → *Settings → verify your phone
   number* (required before Kaggle allows GPUs + internet).
2. **Hugging Face access** (for the two gated models):
   - huggingface.co → sign up → open these two pages **while logged in** and
     click *Agree / Access repository*:
     - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
     - https://huggingface.co/google/gemma-2-9b-it
   - then *Settings → Access Tokens → Create new token* (type: **Read**),
     copy it.
3. **This notebook's settings** (right-hand panel ▸ Session options):
   - Accelerator: **GPU T4 x2**
   - Internet: **ON**
4. **The token secret**: top menu *Add-ons → Secrets → Add secret* →
   Label: `HF_TOKEN`, Value: the token you copied → make sure it is
   **attached** to this notebook (checkbox).

## Every run (3 clicks)

1. *(from the 2nd run onward)* right panel → **+ Add Input** → *Your Work* →
   select this notebook's **previous version output** → Add. (This is how
   the new session sees everything already computed.)
2. Click **Save Version → Save & Run All (Commit)** → close the tab.
   Kaggle runs it in the background (up to ~8 h of work per run, then it
   stops itself safely).
3. When Kaggle notifies you it finished: open the version → **Output** tab →
   download **`mirror_bundle.zip`** and **`SESSION_REPORT.md`** → put them
   in `Desktop/Mirror/` on your laptop → tell Claude *“new bundle is in”*.

Repeat until the report's first line says **ALL GPU WORK COMPLETE**
(expect **3–4 runs total**). Claude does all analysis and paper work from
the bundles — you never need to run anything else.

> Safe by design: every result is written incrementally with per-item
> resume, so a crashed or interrupted run costs nothing — the next run
> continues exactly where it stopped. Running this notebook twice never
> repeats finished work.
