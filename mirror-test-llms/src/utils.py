"""
utils.py — shared plumbing for every pipeline script.

WHAT LIVES HERE
===============
1.  Paths and config loading          (so every script agrees on file locations)
2.  JSONL reading/writing             (crash-safe incremental writes for Colab)
3.  Text cleaning                     (protocol §7 step 2 — applied identically
                                       to every author, including HUMAN)
4.  ROUGE-L                           (near-duplicate filter, protocol §18)
5.  Pair-building helpers             (protocol §7 step 3 — unit-tested)
6.  Model loading + prompting helpers (4-bit loading, chat templates,
                                       first-token log-probabilities §8.4,
                                       continuation NLL for perplexity §9.3)
7.  Free-text answer parsing          (refusal-rate accounting §9.7)

DESIGN RULES (why the file looks the way it does)
=================================================
* Heavy libraries (torch, transformers) are imported INSIDE the functions
  that need them, never at the top of the file. Result: the CPU-only scripts
  (00, 02, 06) and the unit tests run on any laptop with plain Python —
  no GPU stack required.
* Every random process takes an explicit seed. If you see randomness without
  a seed anywhere in this repo, that is a bug.
* All results are written as JSONL, one record per line, appended and flushed
  immediately. If Colab disconnects mid-run you lose at most one record, and
  re-running the same command skips everything already done (see
  `existing_ids`). This is the checkpointing discipline from protocol §15.

If you are new to the codebase: read this file top to bottom once — every
other script is a thin orchestration layer over these functions.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
import hashlib
import datetime as _dt
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS AND CONFIG
# ---------------------------------------------------------------------------

# The project root is the directory that contains src/, data/, configs/ ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where data/ and results/ live. Default: inside the repo itself.
#
# KAGGLE/READ-ONLY OVERRIDE: if the code sits on a read-only disk (e.g. the
# repo attached as a Kaggle *dataset* under /kaggle/input), set the
# environment variable MIRROR_ROOT to a writable directory BEFORE running
# any script, e.g.
#     import os; os.environ["MIRROR_ROOT"] = "/kaggle/working"
# and the entire data/results tree is read from and written to
# <MIRROR_ROOT>/data and <MIRROR_ROOT>/results instead. `src/kaggle_setup.py`
# configures this (and seeds the tree from a previous session's output).
IO_ROOT = (Path(os.environ["MIRROR_ROOT"]).resolve()
           if os.environ.get("MIRROR_ROOT") else PROJECT_ROOT)

CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"   # config ships with the code
PROMPTS_DIR = IO_ROOT / "data" / "prompts"
GENERATIONS_DIR = IO_ROOT / "data" / "generations"
PAIRS_DIR = IO_ROOT / "data" / "pairs"
RESULTS_DIR = IO_ROOT / "results"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"
BASELINES_DIR = RESULTS_DIR / "baselines"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

DOMAINS = ["news", "dolly", "wp"]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load configs/models.yaml (or a custom path) into a plain dict.

    PyYAML is the only non-stdlib dependency of the CPU pipeline; it is tiny
    and installs everywhere (`pip install pyyaml`).
    """
    import yaml  # lazy: only needed at runtime, not for tests of pure helpers

    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    return cfg


def model_index(cfg: dict) -> dict:
    """Map every model `key` -> its config entry (judges, base judges, foils,
    paraphraser), so scripts can look models up by short name."""
    idx = {}
    for group in ("judges", "base_judges", "foils"):
        for m in cfg.get(group, []) or []:
            entry = dict(m)
            entry["group"] = group
            idx[m["key"]] = entry
    para = cfg.get("paraphraser")
    if para:
        entry = dict(para)
        entry["group"] = "paraphraser"
        idx[para["key"]] = entry
    return idx


def judge_keys(cfg: dict) -> list[str]:
    return [m["key"] for m in cfg["judges"]]


def foil_keys(cfg: dict, include_human: bool = True) -> list[str]:
    keys = [m["key"] for m in cfg["foils"]]
    if not include_human:
        keys = [k for k in keys if k != "human"]
    return keys


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def fmt_duration(seconds: float) -> str:
    """3661.0 -> '1:01:01'; 95.0 -> '1:35'."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def progress_iter(items, total: int | None = None, label: str = "",
                  every_s: float = 15.0):
    """Wrap a loop with LINE-BASED progress logging (count, %, rate, ETA).

    Why not tqdm? Its carriage-return bars are invisible when output is
    line-buffered — which is exactly how Kaggle batch logs and the run-all
    notebook's orchestrator read these scripts. Plain newline-terminated
    lines every ~15 s show up live everywhere:

        [gen qwen2.5-7b-instruct/news] 120/600 (20%) | 3.1 it/s | elapsed 0:39 | ETA 2:35

    Also prints a start line and a final line (even if the loop is broken
    out of or the process is being terminated mid-way).
    """
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    tag = f"[{label}] " if label else ""
    t0 = time.monotonic()
    last = t0
    count = 0
    printed_final = False
    if total:
        print(f"{tag}starting: {total} items", flush=True)

    def line() -> str:
        elapsed = time.monotonic() - t0
        rate = count / elapsed if elapsed > 0 else 0.0
        if total and rate > 0:
            eta = fmt_duration((total - count) / rate)
            return (f"{tag}{count}/{total} ({100 * count / total:.0f}%) | "
                    f"{rate:.2f} it/s | elapsed {fmt_duration(elapsed)} | ETA {eta}")
        return f"{tag}{count} items | {rate:.2f} it/s | elapsed {fmt_duration(elapsed)}"

    try:
        for item in items:
            count += 1
            yield item
            now = time.monotonic()
            if now - last >= every_s or (total is not None and count == total):
                print(line(), flush=True)
                last = now
                if total is not None and count == total:
                    printed_final = True
    finally:
        if not printed_final:
            print(line(), flush=True)


# ---------------------------------------------------------------------------
# 2. JSONL I/O — the crash-safe file format of the whole pipeline
#
# JSONL = "JSON Lines": one JSON object per line. It has two properties we
# rely on: (a) you can APPEND to it without rewriting the file, and (b) a
# partially written file is still readable up to the last complete line.
# Both matter because free Colab sessions die without warning (protocol §15).
# ---------------------------------------------------------------------------


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """Read a .jsonl file into a list of dicts. Skips blank/corrupt trailing
    lines (which can occur if a crash interrupted the final write)."""
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from a crashed run — ignore it; the
                # record will simply be recomputed on resume.
                continue
    return records


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    """Append ONE record and flush it to disk immediately.

    flush() + fsync() force the operating system to actually write the bytes,
    so a sudden disconnect cannot lose more than the record being written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: str | os.PathLike, records: list[dict]) -> None:
    """Write a whole list of records atomically (tmp file + rename), for
    outputs that are produced in one go (e.g. frozen prompt sets)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def existing_ids(path: str | os.PathLike, key: str) -> set:
    """Return the set of `record[key]` already present in a JSONL file.

    This is the resume mechanism: before doing expensive GPU work for an item,
    every script checks whether the item's id is already in its output file
    and skips it if so. Re-running an interrupted command therefore continues
    where it left off instead of starting over.
    """
    return {r[key] for r in read_jsonl(path) if key in r}


# ---------------------------------------------------------------------------
# 3. TEXT CLEANING  (protocol §7 step 2)
#
# Why cleaning exists: instruction-tuned models decorate answers with
# giveaways — "Sure! Here is a summary:", "**Answer:**", chat-template
# artifacts like <|im_end|>. If we left those in, the judge (or the
# stylometric baseline) could "recognize" its own text purely from the
# boilerplate, and the whole result would be an artifact (protocol §18,
# "Boilerplate leaks authorship").
#
# CRITICAL RULE: the same clean_text() is applied to every author, INCLUDING
# the human reference texts. Never special-case an author.
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")

# Chat-template artifacts that survive decoding, e.g. <|im_end|>, <|eot_id|>.
_CHAT_ARTIFACT_RE = re.compile(r"<\|[^|>]{0,48}\|>")

# A first sentence that is pure assistant boilerplate, e.g.
# "Sure! Here is a 3-sentence summary of the article:" — protocol regex,
# extended with a couple of equally common openers.
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|absolutely|okay|ok|here(?:'s|’s| is| are))\b"
    r"[^.!?:\n]*(?:[.!?:]|\n)\s*",
    re.IGNORECASE,
)

# Leading labels such as "Summary:", "**Answer:** ", "Response -" that some
# models prepend. These are template artifacts, not writing style.
_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?(?:summary|answer|response|story|opening|title|rewrite|rewritten text)"
    r"(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)

# Characters that can legitimately close a sentence, possibly followed by a
# closing quote/bracket.
_SENT_END_CHARS = ".!?…"
_CLOSERS = "\"'’”)»]"


def n_words(text: str) -> int:
    """Count words the same way everywhere (used by filters and length
    matching). A 'word' is a run of letters/digits/apostrophes/hyphens."""
    return len(_WORD_RE.findall(text or ""))


def strip_incomplete_tail(text: str) -> str:
    """Cut a trailing incomplete sentence (protocol §7 step 2).

    Generation stops at max_new_tokens, so the last sentence is often chopped
    mid-word ("The committee decided that the"). We keep everything up to the
    LAST sentence-ending punctuation mark; if there is none, the whole text is
    invalid and we return "".
    """
    t = text.rstrip()
    if not t:
        return ""
    if t[-1] in _SENT_END_CHARS or (len(t) >= 2 and t[-1] in _CLOSERS and t[-2] in _SENT_END_CHARS):
        return t  # already ends cleanly
    last = -1
    for i, ch in enumerate(t):
        if ch in _SENT_END_CHARS:
            j = i
            while j + 1 < len(t) and t[j + 1] in _CLOSERS:
                j += 1
            last = j
    return t[: last + 1] if last >= 0 else ""


def clean_text(text: str) -> str:
    """The full §7-step-2 cleaning pipeline. Returns "" if the text does not
    survive (caller drops the item). Identical for all authors.

    Order matters:
      1. remove chat-template artifacts
      2. remove ONE leading boilerplate sentence, then a leading label
         (repeat once more in case both are present)
      3. normalise whitespace
      4. cut a trailing incomplete sentence
    """
    if text is None:
        return ""
    t = _CHAT_ARTIFACT_RE.sub(" ", str(text))
    for _ in range(2):  # boilerplate sentence and/or label, in either order
        t2 = _BOILERPLATE_RE.sub("", t.lstrip())
        t2 = _LABEL_RE.sub("", t2.lstrip())
        if t2 == t:
            break
        t = t2
    # collapse runs of blank lines / spaces but keep paragraph breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    t = strip_incomplete_tail(t)
    return t.strip()


# ---------------------------------------------------------------------------
# 4. ROUGE-L  (near-duplicate filter, protocol §18)
#
# ROUGE-L measures the longest common subsequence (LCS) of words between two
# texts, as an F1 score in [0, 1]. Two answers to the same factual prompt can
# be almost identical; a judge is at chance on such pairs for the boring
# reason that there is nothing to distinguish. Pairs with ROUGE-L > 0.7 are
# dropped. Implemented here directly (simple dynamic programming) to avoid an
# extra dependency.
# ---------------------------------------------------------------------------


def rouge_l_f1(a: str, b: str) -> float:
    """ROUGE-L F1 between two texts (word-level, case-insensitive)."""
    xs = [w.lower() for w in _WORD_RE.findall(a or "")]
    ys = [w.lower() for w in _WORD_RE.findall(b or "")]
    if not xs or not ys:
        return 0.0
    # LCS length via DP over two rows (memory O(min(n,m)))
    if len(ys) < len(xs):
        xs, ys = ys, xs
    prev = [0] * (len(xs) + 1)
    for y in ys:
        cur = [0]
        for i, x in enumerate(xs, 1):
            cur.append(prev[i - 1] + 1 if x == y else max(prev[i], cur[-1]))
        prev = cur
    lcs = prev[-1]
    p = lcs / len(xs)
    r = lcs / len(ys)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# 5. PAIR-BUILDING HELPERS  (protocol §7 step 3) — pure functions so the unit
#    tests can exercise them without any data on disk.
# ---------------------------------------------------------------------------


def length_ratio_ok(wc_a: int, wc_b: int, max_diff: float) -> bool:
    """Length-match filter: |a-b| / max(a,b) <= max_diff.

    Verbosity is a known judge bias (Zheng et al. 2023): models prefer longer
    answers. If SELF texts were systematically longer than FOIL texts, 'self-
    recognition' could just be 'longer-recognition'. This filter keeps only
    pairs of similar length; §6 of the stats output reports the residual
    length statistics."""
    if wc_a <= 0 or wc_b <= 0:
        return False
    return abs(wc_a - wc_b) / max(wc_a, wc_b) <= max_diff


def build_ppp_pairs(
    self_texts: dict[str, str],
    foil_texts: dict[str, str],
    task_prompts: dict[str, str],
    judge: str,
    foil: str,
    domain: str,
    max_length_ratio_diff: float = 0.25,
    max_rouge_l: float = 0.70,
) -> tuple[list[dict], dict]:
    """Build the PPP pairs for one (judge, foil, domain) cell.

    Inputs are dicts keyed by prompt_id: {prompt_id: cleaned_text}. A pair is
    formed for every prompt where BOTH texts survived cleaning, passes the
    length-ratio filter and the ROUGE-L near-duplicate filter.

    Returns (pairs, report) where report counts why candidates were dropped —
    those counts go in the paper appendix ("we report the exact count", §7).
    """
    pairs = []
    report = {"candidates": 0, "kept": 0, "len_mismatch": 0, "near_dup": 0}
    for pid in sorted(set(self_texts) & set(foil_texts)):
        s, f = self_texts[pid], foil_texts[pid]
        report["candidates"] += 1
        if not length_ratio_ok(n_words(s), n_words(f), max_length_ratio_diff):
            report["len_mismatch"] += 1
            continue
        rl = rouge_l_f1(s, f)
        if rl > max_rouge_l:
            report["near_dup"] += 1
            continue
        pairs.append(
            {
                "pair_id": f"{judge}__{foil}__{domain}__{pid}",
                "judge": judge,
                "foil": foil,
                "domain": domain,
                "prompt_id": pid,
                "task_prompt": task_prompts[pid],
                "text_self": s,
                "text_foil": f,
                "n_words_self": n_words(s),
                "n_words_foil": n_words(f),
                "rouge_l": round(rl, 4),
            }
        )
        report["kept"] += 1
    return pairs, report


def fill_template(template: str, mapping: dict[str, str]) -> str:
    """Fill {PLACEHOLDER}s with .replace instead of str.format.

    Why not .format()? Article text routinely contains literal braces
    ("{...}" in scraped news), which would crash str.format or, worse,
    silently corrupt the prompt. Sequential replace is dumb and safe.
    """
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


# ---------------------------------------------------------------------------
# 6. MODEL LOADING AND PROMPTING  (GPU side; lazy imports)
# ---------------------------------------------------------------------------

_HF_AUTH_DONE = False


def setup_hf_auth(verbose: bool = True) -> None:
    """Authenticate to Hugging Face WITHOUT an interactive prompt, so the
    gated models (Llama, Gemma) load in headless environments.

    Token lookup order:
      1. HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable
         (Colab: set it, or use the interactive `login()` as before);
      2. on Kaggle: the notebook secret named HF_TOKEN
         (Add-ons -> Secrets -> attach "HF_TOKEN" to the notebook).

    Called automatically by load_model_and_tokenizer, so GPU scripts need no
    login cell at all once the secret/env var exists. Without a token,
    non-gated models (Qwen, Mistral, Phi) still work fine.
    """
    global _HF_AUTH_DONE
    if _HF_AUTH_DONE:
        return
    _HF_AUTH_DONE = True
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and Path("/kaggle").exists():
        try:
            from kaggle_secrets import UserSecretsClient  # Kaggle-only module
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token  # transformers/hub read this directly
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass  # the env var alone is sufficient for downloads
        if verbose:
            print("[auth] Hugging Face token configured")
    elif verbose:
        print("[auth] no HF token found - gated models (Llama/Gemma) will fail. "
              "Set env HF_TOKEN, or on Kaggle add a secret named HF_TOKEN.")


def load_model_and_tokenizer(hf_id: str, revision: str | None = None, four_bit: bool = True):
    """Load a causal LM (+tokenizer) the way the protocol prescribes (§5):
    transformers + bitsandbytes 4-bit NF4, so token log-probabilities are
    available (Ollama/llama.cpp would hide them).

    Falls back gracefully:
      * no CUDA GPU        -> float32 on CPU (slow; fine for tiny smoke tests)
      * bitsandbytes absent-> float16 on GPU without quantisation
    Prints what it did, so logs always show the effective precision.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_hf_auth()  # no-op if already done / no token; enables gated models
    # device_map="auto" also spreads layers across BOTH GPUs on Kaggle's 2xT4
    # runtime - that is what makes the 14B judge fit there.
    kwargs: dict = {"revision": revision, "device_map": "auto"}
    mode = ""
    if torch.cuda.is_available() and four_bit:
        try:
            import bitsandbytes  # noqa: F401  (probe only)
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            mode = "4-bit NF4"
        except ImportError:
            kwargs["torch_dtype"] = torch.float16
            mode = "float16 (bitsandbytes not installed!)"
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        mode = "float16"
    else:
        kwargs["torch_dtype"] = torch.float32
        mode = "float32 on CPU (no GPU found - this will be SLOW)"

    print(f"[load] {hf_id} (revision={revision or 'latest'}) as {mode} - "
          "first use downloads the weights; this can be silent for several "
          "minutes", flush=True)
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    print(f"[load] model ready in {fmt_duration(time.monotonic() - t0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Record the exact commit that was actually downloaded, for the appendix.
    resolved = getattr(model.config, "_commit_hash", None)
    print(f"[load] resolved commit: {resolved}")
    return model, tok


def resolved_revision(model) -> str | None:
    return getattr(model.config, "_commit_hash", None)


def build_chat_text(tokenizer, user: str, system: str | None = None) -> str:
    """Turn (system, user) into the model's own chat format via its
    tokenizer.chat_template, ready for tokenization with
    add_special_tokens=False (the template already includes special tokens).

    Some models (e.g. Gemma-2) reject a system role entirely; in that case we
    fold the system text into the top of the user message — the standard
    workaround — and the run log records that this happened.
    """
    if system:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if system:
            merged = [{"role": "user", "content": system + "\n\n" + user}]
            print("[warn] model rejects system role; folded system text into user message")
            return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        raise


def _encode(tokenizer, text: str, already_templated: bool):
    """Tokenize prompt text correctly.

    add_special_tokens must be False for text that came out of
    apply_chat_template (it already contains BOS/role tokens — adding them
    again is a classic silent bug), and True for plain completion-style text.
    """
    return tokenizer(text, return_tensors="pt", add_special_tokens=not already_templated)


def candidate_token_ids(tokenizer, variants: list[str]) -> list[int]:
    """Token ids of every single-token spelling of an answer word.

    'A' and ' A' are different BPE tokens; depending on the chat template the
    model's first generated token may be either. We collect all single-token
    variants and later log-sum-exp their probabilities: p(answer=A) =
    p('A') + p(' A'). Raises if no variant is a single token (would mean the
    scoring design cannot work for this tokenizer — better to crash loudly).
    """
    ids = []
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    ids = sorted(set(ids))
    if not ids:
        raise ValueError(f"No single-token spelling among {variants!r} for this tokenizer")
    return ids


def first_token_logprobs(model, tokenizer, prompt_text: str, candidates: dict[str, list[int]],
                         already_templated: bool = True) -> dict[str, float]:
    """THE core measurement of the study (protocol §8.4).

    One forward pass; take the logits for the NEXT token after the prompt;
    log-softmax them into log-probabilities; for each answer label return
    log( sum of p(token) over that label's candidate token ids ).

    Because we read probabilities instead of generating text, there are zero
    unparseable answers, and logp('A') - logp('B') is a continuous score that
    later gives AUROC (§9.6).
    """
    import torch

    enc = _encode(tokenizer, prompt_text, already_templated).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out = {}
    for label, ids in candidates.items():
        out[label] = torch.logsumexp(logprobs[ids], dim=0).item()
    return out


def greedy_generate(model, tokenizer, prompt_text: str, max_new_tokens: int = 8,
                    already_templated: bool = True) -> str:
    """Deterministic (temperature 0) short generation — used only to log the
    free-text answer for refusal-rate accounting (§8.4/§9.7)."""
    import torch

    enc = _encode(tokenizer, prompt_text, already_templated).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def sampled_generate(model, tokenizer, prompt_text: str, temperature: float, top_p: float,
                     max_new_tokens: int, seed: int, already_templated: bool = True) -> str:
    """Seeded sampling for the generation phase (§7 step 1).

    torch.manual_seed right before generate() makes each (prompt, seed) pair
    reproducible — the protocol's `seed = 1000 + prompt_id` rule.
    """
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = _encode(tokenizer, prompt_text, already_templated).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def continuation_mean_nll(model, tokenizer, context_text: str, continuation_text: str,
                          context_already_templated: bool = True) -> tuple[float, int]:
    """Mean per-token negative log-likelihood of `continuation_text` given
    `context_text` under the model — the quantity behind the perplexity-
    preference baseline (§9.3). perplexity = exp(mean NLL).

    Implementation: tokenize context and context+continuation, find their
    common token prefix (BPE can merge across the boundary, so we do not
    assume the context tokens are a strict prefix), then average the NLL of
    every token after that point.
    """
    import torch

    ctx_ids = _encode(tokenizer, context_text, context_already_templated)["input_ids"][0]
    full_ids = _encode(tokenizer, context_text + continuation_text, context_already_templated)["input_ids"][0]
    # common prefix length
    n = 0
    limit = min(len(ctx_ids), len(full_ids))
    while n < limit and int(ctx_ids[n]) == int(full_ids[n]):
        n += 1
    if n >= len(full_ids):  # continuation vanished (empty text) — undefined
        return float("nan"), 0
    input_ids = full_ids.unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits
    # token t is predicted by logits at position t-1
    logprobs = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
    targets = input_ids[0, 1:]
    start = max(n - 1, 0)  # first continuation token is predicted at position n-1
    sel = logprobs[torch.arange(start, len(targets)), targets[start:]]
    return float(-sel.mean().item()), int(sel.numel())


# ---------------------------------------------------------------------------
# 7. FREE-TEXT ANSWER PARSING  (refusal accounting, §9.7)
# ---------------------------------------------------------------------------

_REFUSAL_MARKERS = [
    "as an ai", "i do not have", "i don't have", "no memory", "i cannot",
    "i can't", "unable to determine", "cannot determine", "i did not write",
    "i didn't write", "neither",
]


def parse_choice_letter(text: str) -> str | None:
    """Extract A/B from a free-text answer. Uppercase A/B anywhere in the
    first few tokens counts; lowercase only if the whole answer is exactly
    'a'/'b' (because lowercase 'a' is usually the English article)."""
    if not text:
        return None
    head = text.strip()[:40]
    stripped = head.strip(" \t\n.*:\"'()[]")
    if stripped in ("a", "b"):
        return stripped.upper()
    m = re.search(r"\b([AB])\b", head)
    return m.group(1) if m else None


def parse_yes_no(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\b(yes|no)\b", text.strip()[:40], re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def classify_freetext(text: str, kind: str) -> tuple[str | None, str]:
    """Return (parsed_answer, status) where status is one of
    'ok' | 'refusal' | 'unparseable'. `kind` is 'letter' or 'yesno'."""
    parsed = parse_choice_letter(text) if kind == "letter" else parse_yes_no(text)
    if parsed:
        return parsed, "ok"
    low = (text or "").lower()
    if any(m in low for m in _REFUSAL_MARKERS):
        return None, "refusal"
    return None, "unparseable"
