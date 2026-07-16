"""
02_build_pairs.py — clean all texts and build every evaluation set
(protocol §7 steps 2-4 and §8.2). Pure CPU; runs on any laptop.

WHAT THIS SCRIPT DOES
=====================
1. CLEAN (§7 step 2): applies utils.clean_text IDENTICALLY to every text —
   all model generations AND the human reference answers. Texts that do not
   survive (too short / no complete sentence) are dropped. Per-author
   cleaning statistics are recorded: they go in the paper appendix and are
   themselves interesting (e.g. which models produce the most boilerplate).

2. PPP PAIRS (§7 step 3): for each (judge, foil, domain) cell, pair the
   judge's text with the foil's text FOR THE SAME PROMPT, keeping only pairs
   that pass the length-match filter (<=25% word-count difference) and the
   near-duplicate filter (ROUGE-L <= 0.7).
        -> data/pairs/ppp__{judge}__{foil}__{domain}.jsonl

3. PLACEBO PAIRS (§7 step 4): SELF-vs-SELF pairs from the judge's s1 and s2
   samples. There is no correct answer; the judge's rate of picking
   "position A" measures pure position bias. Any pipeline bug shows up here.
        -> data/pairs/placebo__{judge}__{domain}.jsonl

4. IPP ITEMS (§8.2): a balanced single-text set per judge - 50% SELF texts,
   50% foil texts (all foils incl. HUMAN, all domains, evenly mixed).
        -> data/pairs/ipp__{judge}.jsonl

5. REPORT: data/pairs/pairs_report.json with exact per-cell counts (the
   protocol requires reporting them) and cleaning statistics.

USAGE
=====
    python src/02_build_pairs.py                  # everything
    python src/02_build_pairs.py --judges qwen2.5-0.5b-instruct --domains news

Paraphrased pairs (§7 step 5) need a GPU and live in 02b_paraphrase.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DOMAINS, GENERATIONS_DIR, PAIRS_DIR, PROMPTS_DIR, build_ppp_pairs,
    clean_text, load_config, n_words, now_iso, read_jsonl, write_jsonl,
)


# --------------------------------------------------------------------------
# Loading + cleaning
# --------------------------------------------------------------------------

def load_clean_generations(model_key: str, domain: str, min_words: int,
                           sample: str = "s1") -> tuple[dict, dict]:
    """Return ({prompt_id: cleaned_text}, stats) for one model+domain+sample."""
    path = GENERATIONS_DIR / f"{model_key}__{domain}.jsonl"
    records = [r for r in read_jsonl(path) if r.get("sample", "s1") == sample]
    texts, stats = {}, {"raw": len(records), "kept": 0, "dropped_short": 0, "changed_by_cleaning": 0}
    for r in records:
        cleaned = clean_text(r["text"])
        if cleaned != (r["text"] or "").strip():
            stats["changed_by_cleaning"] += 1
        if n_words(cleaned) < min_words:
            stats["dropped_short"] += 1
            continue
        texts[r["prompt_id"]] = cleaned
        stats["kept"] += 1
    return texts, stats


def load_clean_human(domain: str, min_words: int) -> tuple[dict, dict]:
    """HUMAN pseudo-author: the gold reference answers, cleaned with the SAME
    function as model outputs (protocol §7 step 2 - never special-case an
    author)."""
    prompts = read_jsonl(PROMPTS_DIR / f"{domain}.jsonl")
    texts, stats = {}, {"raw": len(prompts), "kept": 0, "dropped_short": 0, "changed_by_cleaning": 0}
    for p in prompts:
        cleaned = clean_text(p["human_reference"])
        if cleaned != (p["human_reference"] or "").strip():
            stats["changed_by_cleaning"] += 1
        if n_words(cleaned) < min_words:
            stats["dropped_short"] += 1
            continue
        texts[p["prompt_id"]] = cleaned
        stats["kept"] += 1
    return texts, stats


def load_task_prompts(domain: str) -> dict:
    return {p["prompt_id"]: p["task_prompt"] for p in read_jsonl(PROMPTS_DIR / f"{domain}.jsonl")}


# --------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean texts and build PPP / placebo / IPP sets (§7, §8.2). CPU only.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--judges", nargs="+", default=None, help="subset of judge keys")
    ap.add_argument("--foils", nargs="+", default=None, help="subset of foil keys")
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    judges = args.judges or [m["key"] for m in cfg["judges"]]
    foils = args.foils or [m["key"] for m in cfg["foils"]]
    min_words = cfg["cleaning"]["min_words_after_clean"]
    pairing = cfg["pairing"]
    target = pairing["target_pairs_per_cell"]

    report = {"built_at": now_iso(), "cleaning": {}, "cells": {}, "placebo": {}, "ipp": {}}

    # ---- 1. Load + clean everything once --------------------------------
    cleaned: dict[tuple[str, str], dict] = {}          # (author, domain) -> {pid: text}
    for domain in args.domains:
        task_prompts = load_task_prompts(domain)
        if not task_prompts:
            print(f"[warn] no prompts for '{domain}' - run 00_build_prompts.py first")
            continue
        for author in dict.fromkeys(judges + foils):
            if author == "human":
                texts, stats = load_clean_human(domain, min_words)
            else:
                texts, stats = load_clean_generations(author, domain, min_words)
            cleaned[(author, domain)] = texts
            report["cleaning"][f"{author}__{domain}"] = stats
            if stats["raw"] == 0 and author != "human":
                print(f"[warn] no generations found for {author}/{domain} "
                      f"(expected data/generations/{author}__{domain}.jsonl)")

    # ---- 2. PPP pairs per (judge, foil, domain) cell ---------------------
    for judge in judges:
        for foil in foils:
            if foil == judge:
                continue
            for domain in args.domains:
                self_texts = cleaned.get((judge, domain), {})
                foil_texts = cleaned.get((foil, domain), {})
                if not self_texts or not foil_texts:
                    continue
                pairs, cell_report = build_ppp_pairs(
                    self_texts, foil_texts, load_task_prompts(domain),
                    judge, foil, domain,
                    max_length_ratio_diff=pairing["max_length_ratio_diff"],
                    max_rouge_l=pairing["max_rouge_l"],
                )
                out = PAIRS_DIR / f"ppp__{judge}__{foil}__{domain}.jsonl"
                write_jsonl(out, pairs)
                report["cells"][f"{judge}__{foil}__{domain}"] = cell_report
                flag = "" if cell_report["kept"] >= target else f"  <-- BELOW TARGET {target}"
                print(f"[ppp] {judge:26s} vs {foil:26s} {domain:5s} "
                      f"kept={cell_report['kept']:4d} "
                      f"(len={cell_report['len_mismatch']}, dup={cell_report['near_dup']}){flag}")

    # ---- 3. Placebo pairs (SELF s1 vs SELF s2) ---------------------------
    for judge in judges:
        for domain in args.domains:
            s1, _ = load_clean_generations(judge, domain, min_words, sample="s1")
            s2, _ = load_clean_generations(judge, domain, min_words, sample="s2")
            task_prompts = load_task_prompts(domain)
            pairs = []
            for pid in sorted(set(s1) & set(s2)):
                if s1[pid] == s2[pid]:
                    continue  # identical twin samples carry no information
                pairs.append({
                    "pair_id": f"{judge}__placebo__{domain}__{pid}",
                    "judge": judge, "foil": "placebo", "domain": domain,
                    "prompt_id": pid, "task_prompt": task_prompts[pid],
                    # 'self' = s1, 'foil' = s2, but BOTH are the judge's own text;
                    # 'correct' is undefined and 03_judge_ppp records choice only.
                    "text_self": s1[pid], "text_foil": s2[pid],
                    "n_words_self": n_words(s1[pid]), "n_words_foil": n_words(s2[pid]),
                    "is_placebo": True,
                })
            if pairs:
                write_jsonl(PAIRS_DIR / f"placebo__{judge}__{domain}.jsonl", pairs)
                report["placebo"][f"{judge}__{domain}"] = len(pairs)
                print(f"[placebo] {judge:26s} {domain:5s} pairs={len(pairs)}")

    # ---- 4. IPP balanced item sets ---------------------------------------
    rng = random.Random(cfg["seed"])
    n_items = cfg["ipp"]["n_items_per_judge"]
    for judge in judges:
        self_pool, foil_pool = [], []
        for domain in args.domains:
            task_prompts = load_task_prompts(domain)
            for pid, text in cleaned.get((judge, domain), {}).items():
                self_pool.append((domain, pid, task_prompts[pid], "self", text))
            for foil in foils:
                if foil == judge:
                    continue
                for pid, text in cleaned.get((foil, domain), {}).items():
                    foil_pool.append((domain, pid, task_prompts[pid], foil, text))
        if not self_pool or not foil_pool:
            continue
        rng.shuffle(self_pool)
        # round-robin over (foil, domain) buckets so authors/domains are even
        buckets: dict[tuple[str, str], list] = {}
        for item in foil_pool:
            buckets.setdefault((item[3], item[0]), []).append(item)
        for b in buckets.values():
            rng.shuffle(b)
        half = n_items // 2
        chosen_foil, bucket_keys = [], sorted(buckets)
        bi = 0
        while len(chosen_foil) < half and any(buckets[k] for k in bucket_keys):
            k = bucket_keys[bi % len(bucket_keys)]
            if buckets[k]:
                chosen_foil.append(buckets[k].pop())
            bi += 1
        chosen_self = self_pool[:half]
        items = []
        for kind, pool in (("self", chosen_self), ("foil", chosen_foil)):
            for domain, pid, task, author, text in pool:
                items.append({
                    "item_id": f"ipp__{judge}__{author}__{domain}__{pid}",
                    "judge": judge, "domain": domain, "prompt_id": pid,
                    "task_prompt": task, "text": text,
                    "author": author, "is_self": kind == "self",
                })
        rng.shuffle(items)
        write_jsonl(PAIRS_DIR / f"ipp__{judge}.jsonl", items)
        n_self = sum(1 for i in items if i["is_self"])
        report["ipp"][judge] = {"total": len(items), "self": n_self, "foil": len(items) - n_self}
        print(f"[ipp] {judge:26s} items={len(items)} (self={n_self})")

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_DIR / "pairs_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] report -> {PAIRS_DIR / 'pairs_report.json'}")
    print("NEXT: src/02b_paraphrase.py (GPU) for the paraphrase set, "
          "then src/03_judge_ppp.py (GPU).")


if __name__ == "__main__":
    main()
