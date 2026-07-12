"""
00_build_prompts.py — build and FREEZE the prompt sets (protocol §6).

WHAT THIS SCRIPT DOES
=====================
Downloads three public datasets from Hugging Face, filters them, samples 200
items per domain with a fixed seed, and writes the frozen prompt files:

    data/prompts/news.jsonl    (CNN/DailyMail summarization)
    data/prompts/dolly.jsonl   (Dolly-15k open QA / brainstorming)
    data/prompts/wp.jsonl      (WritingPrompts creative openings)
    data/prompts/CHECKSUMS.txt (SHA-256 of each file — the freeze receipt)
    data/prompts/prompts_report.json (how many items were scanned/rejected and why)

Each record:
    {
      "prompt_id":        "news_0007",     # stable id used everywhere downstream
      "prompt_idx":       7,               # integer used for per-prompt seeds
      "domain":           "news",
      "task_prompt":      "Summarize the following article in 3-4 sentences: ...",
      "human_reference":  "...",           # the HUMAN author class (free!)
      "source_dataset":   "abisee/cnn_dailymail",
      "source_id":        "...",           # id inside the source dataset
      "n_words_prompt":   123,
      "n_words_reference": 58
    }

THE FREEZE RULE (§6 step 3)
===========================
Once written and committed to git, these files must NEVER change. That is
what makes "we did not cherry-pick prompts" credible. The script therefore
REFUSES to overwrite an existing prompt file unless you pass --force.

USAGE
=====
    python src/00_build_prompts.py                    # all three domains
    python src/00_build_prompts.py --domains news wp  # subset
    python src/00_build_prompts.py --force            # explicit re-freeze (danger!)

COST: CPU + internet only (downloads a few hundred MB the first time; the
`datasets` library caches under ~/.cache/huggingface). No GPU needed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    PROMPTS_DIR, load_config, n_words, write_jsonl, sha256_file, now_iso,
)

# --------------------------------------------------------------------------
# Shared filters (protocol §6 step 2)
# --------------------------------------------------------------------------

def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) < 128) / len(text)


def passes_shared_filters(text: str, cfg_filters: dict) -> bool:
    """English-ish, no URLs/handles. Cheap heuristics, applied to both the
    prompt and the human reference."""
    if ascii_ratio(text) < cfg_filters["min_ascii_ratio"]:
        return False
    low = text.lower()
    return not any(s in low for s in cfg_filters["forbid_substrings"])


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def truncate_to_words(text: str, target: int, hard_max: int) -> str | None:
    """Cut a long text at a sentence boundary near `target` words.
    Used to shorten human WritingPrompts stories to ~120 words (§6).
    Returns None if no acceptable cut exists (e.g. one giant sentence)."""
    sentences = _SENT_SPLIT_RE.split(text.strip())
    out, count = [], 0
    for s in sentences:
        w = n_words(s)
        if count + w > hard_max and out:
            break
        out.append(s)
        count += w
        if count >= target:
            break
    if not out or count > hard_max:
        return None
    return " ".join(out).strip()


# --------------------------------------------------------------------------
# WritingPrompts-specific cleanup: the raw dataset carries forum artifacts.
# --------------------------------------------------------------------------

_WP_TAG_RE = re.compile(r"^\s*\[\s*[A-Za-z]{2,4}\s*\]\s*")     # leading [ WP ] tag
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%])")           # "word ." -> "word."


def clean_wp_text(text: str) -> str:
    t = _WP_TAG_RE.sub("", text or "")
    t = t.replace("``", '"').replace("''", '"')
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
    t = re.sub(r"\s+n't\b", "n't", t)  # detokenization artifact: "do n't"
    t = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m)\b", r"\1", t)
    return re.sub(r"[ \t]+", " ", t).strip()


# --------------------------------------------------------------------------
# One candidate-yielding function per domain. Each yields dicts:
#   {task_prompt, human_reference, source_id}
# already domain-filtered; shared filters + sampling happen in main().
# --------------------------------------------------------------------------

def iter_news(cfg: dict):
    from datasets import load_dataset

    d = cfg["datasets"]["news"]
    filters = cfg["prompt_filters"]
    lo, hi = filters["news_article_words"]
    ds = load_dataset(d["hf_id"], d["hf_config"], split=d["split"])
    ds = ds.shuffle(seed=filters["sample_seed"])
    for row in ds:
        article = (row.get("article") or "").strip()
        # CNN/DM 'highlights' are newline-separated bullet sentences ->
        # join into one paragraph so the human text looks like prose.
        reference = " ".join((row.get("highlights") or "").split("\n")).strip()
        if not (lo <= n_words(article) <= hi):
            continue
        yield {
            "task_prompt": d["prompt_template"].replace("{ARTICLE}", article),
            "human_reference": reference,
            "source_id": str(row.get("id", "")),
        }


def iter_dolly(cfg: dict):
    from datasets import load_dataset

    d = cfg["datasets"]["dolly"]
    filters = cfg["prompt_filters"]
    lo, hi = filters["dolly_instruction_words"]
    cats = set(d["categories"])
    ds = load_dataset(d["hf_id"], split=d["split"])
    ds = ds.shuffle(seed=filters["sample_seed"])
    for row in ds:
        if row.get("category") not in cats:
            continue
        if (row.get("context") or "").strip():
            continue  # keep only self-contained questions (no reading passage)
        instruction = (row.get("instruction") or "").strip()
        reference = (row.get("response") or "").strip()
        if not (lo <= n_words(instruction) <= hi):
            continue
        yield {
            "task_prompt": d["prompt_template"].replace("{INSTRUCTION}", instruction),
            "human_reference": reference,
            "source_id": instruction[:80],  # dolly has no row id; use a slug
        }


def iter_wp(cfg: dict):
    from datasets import load_dataset

    d = cfg["datasets"]["wp"]
    filters = cfg["prompt_filters"]
    lo, hi = filters["wp_prompt_words"]
    trunc_to = filters["wp_reference_truncate_to"]
    ds = load_dataset(d["hf_id"], split=d["split"])
    ds = ds.shuffle(seed=filters["sample_seed"])
    for i, row in enumerate(ds):
        prompt = clean_wp_text(row.get("prompt") or "")
        story = clean_wp_text(row.get("story") or "")
        if not (lo <= n_words(prompt) <= hi):
            continue
        reference = truncate_to_words(story, target=trunc_to, hard_max=170)
        if reference is None:
            continue
        yield {
            "task_prompt": d["prompt_template"].replace("{PROMPT}", prompt),
            "human_reference": reference,
            "source_id": f"row{i}",
        }


DOMAIN_ITERATORS = {"news": iter_news, "dolly": iter_dolly, "wp": iter_wp}


# --------------------------------------------------------------------------
# Main: shared filtering + take-first-200-valid + freeze
# --------------------------------------------------------------------------

def build_domain(domain: str, cfg: dict, n_target: int) -> tuple[list[dict], dict]:
    filters = cfg["prompt_filters"]
    ref_lo, ref_hi = filters["reference_words"]
    report = {"scanned": 0, "kept": 0, "ref_length": 0, "shared_filters": 0, "dup_prompt": 0}
    kept, seen_prompts = [], set()

    for cand in DOMAIN_ITERATORS[domain](cfg):
        report["scanned"] += 1
        if report["scanned"] > 200_000:  # safety valve
            break
        tp, ref = cand["task_prompt"], cand["human_reference"]
        if not (ref_lo <= n_words(ref) <= ref_hi):
            report["ref_length"] += 1
            continue
        if not (passes_shared_filters(tp, filters) and passes_shared_filters(ref, filters)):
            report["shared_filters"] += 1
            continue
        if tp in seen_prompts:
            report["dup_prompt"] += 1
            continue
        seen_prompts.add(tp)
        idx = len(kept)
        kept.append({
            "prompt_id": f"{domain}_{idx:04d}",
            "prompt_idx": idx,
            "domain": domain,
            "task_prompt": tp,
            "human_reference": ref,
            "source_dataset": cfg["datasets"][domain]["hf_id"],
            "source_id": cand["source_id"],
            "n_words_prompt": n_words(tp),
            "n_words_reference": n_words(ref),
        })
        report["kept"] += 1
        if len(kept) >= n_target:
            break
    return kept, report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build and freeze the prompt sets (protocol §6). CPU only.",
        epilog="Example: python src/00_build_prompts.py --domains news dolly wp",
    )
    ap.add_argument("--config", default=None, help="path to configs/models.yaml")
    ap.add_argument("--domains", nargs="+", default=["news", "dolly", "wp"],
                    choices=["news", "dolly", "wp"])
    ap.add_argument("--n", type=int, default=None,
                    help="prompts per domain (default: config n_per_domain=200). "
                         "Use a small number like 8 for a quick pipeline test.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing frozen prompt files (breaks the freeze "
                         "guarantee - only do this before your pre-registration commit)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_target = args.n or cfg["prompt_filters"]["n_per_domain"]
    random.seed(cfg["prompt_filters"]["sample_seed"])

    all_reports = {"built_at": now_iso(), "n_per_domain": n_target, "domains": {}}
    for domain in args.domains:
        out_path = PROMPTS_DIR / f"{domain}.jsonl"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} already exists - prompts are FROZEN. "
                  f"Use --force only if you have not pre-registered yet.")
            continue
        print(f"[build] domain={domain} target={n_target} ...")
        records, report = build_domain(domain, cfg, n_target)
        if len(records) < n_target:
            print(f"[warn] only {len(records)} valid items found for {domain} "
                  f"(target {n_target}) - check filters")
        write_jsonl(out_path, records)
        all_reports["domains"][domain] = report
        print(f"[done] {out_path}  kept={report['kept']}  "
              f"rejected: ref_length={report['ref_length']} "
              f"shared={report['shared_filters']} dup={report['dup_prompt']}")

    # Freeze receipt: SHA-256 of every prompt file. Commit this to git.
    checksums = []
    for domain in ["news", "dolly", "wp"]:
        p = PROMPTS_DIR / f"{domain}.jsonl"
        if p.exists():
            checksums.append(f"{sha256_file(p)}  {p.name}")
    (PROMPTS_DIR / "CHECKSUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    with open(PROMPTS_DIR / "prompts_report.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"[done] checksums + report written to {PROMPTS_DIR}")
    print("NEXT: commit data/prompts/ to git (this is the freeze), "
          "then run src/01_generate.py on a GPU.")


if __name__ == "__main__":
    main()
