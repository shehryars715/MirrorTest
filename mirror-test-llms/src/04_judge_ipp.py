"""
04_judge_ipp.py — individual presentation paradigm: "Did you write this?"
(protocol §8.2, analysed with signal detection in §9.7).

WHY IPP EXISTS ALONGSIDE PPP
============================
The pairwise test forces a choice, so response bias mostly cancels. The
individual test shows ONE text and asks Yes/No — here a model that answers
"Yes" to everything scores 100% on SELF items while discriminating nothing.
That is exactly why the analysis uses signal detection theory:

    hit rate          H = P("Yes" | SELF text)
    false-alarm rate  F = P("Yes" | foil text)
    d' = z(H) - z(F)      -> real sensitivity
    criterion c           -> how trigger-happy the "Yes" is

d' ~ 0 with a high yes-rate = sycophancy masquerading as self-recognition
(protocol §11, outcome 4 — a publishable methods finding on its own).

MEASUREMENT: same log-prob trick as PPP — one forward pass, read
logp("Yes") vs logp("No") over the yes/no token spellings; the margin
logp(Yes) - logp(No) is the continuous score used for the IPP AUROC.

OUTPUT
======
results/judgments/ipp__{judge}.jsonl — one record per item:
    {"item_id": ..., "judge": ..., "author": "self|<foil key>",
     "is_self": true, "domain": ..., "logp_yes": ..., "logp_no": ...,
     "choice": "Yes", "correct": true, "freetext": ..., "freetext_status": ...}

USAGE
=====
    python src/04_judge_ipp.py --judge qwen2.5-7b-instruct
    python src/04_judge_ipp.py --judge qwen2.5-7b-instruct --confidence
        (adds the "give a number 0-100" confidence variant, logged from the
         free-text continuation; optional calibration data per §8.2)

COST: one forward pass per item (~300 items/judge) — minutes per judge.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    JUDGMENTS_DIR, PAIRS_DIR, append_jsonl, build_chat_text,
    candidate_token_ids, classify_freetext, existing_ids, fill_template,
    first_token_logprobs, greedy_generate, load_config,
    load_model_and_tokenizer, model_index, now_iso, read_jsonl,
)


def parse_confidence(text: str) -> int | None:
    """Pull a 0-100 number out of the free-text tail, if present."""
    for m in re.finditer(r"\b(\d{1,3})\b", text or ""):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Individual (IPP) Yes/No protocol with log-prob scoring (§8.2).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--confidence", action="store_true",
                    help="append the 0-100 confidence request and log the parsed number")
    ap.add_argument("--limit", type=int, default=None, help="max items (smoke test)")
    ap.add_argument("--freetext", choices=["all", "subsample", "none"], default=None)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    idx = model_index(cfg)
    if args.judge not in idx:
        sys.exit(f"[error] unknown judge '{args.judge}'. Known: {sorted(idx)}")
    jcfg = idx[args.judge]
    if jcfg.get("template") == "completion":
        sys.exit("[error] IPP is defined for instruction-tuned judges only; "
                 "base models are covered by the PPP completion path (§9.5).")

    items = read_jsonl(PAIRS_DIR / f"ipp__{args.judge}.jsonl")
    if not items:
        sys.exit(f"[error] no IPP items for '{args.judge}' - run 02_build_pairs.py first")
    if args.limit:
        items = items[: args.limit]

    jd = cfg["judging"]
    freetext_mode = args.freetext or jd["freetext_mode"]
    tmpl = cfg["ipp_template"]

    model, tok = load_model_and_tokenizer(jcfg["hf_id"], jcfg.get("revision"),
                                          four_bit=not args.no_4bit)
    candidates = {
        "Yes": candidate_token_ids(tok, jd["yes_variants"]),
        "No": candidate_token_ids(tok, jd["no_variants"]),
    }

    out_path = JUDGMENTS_DIR / f"ipp__{args.judge}.jsonl"
    done = existing_ids(out_path, "item_id")
    import random as _random
    ft_rng = _random.Random(cfg["seed"])

    n_new = 0
    for item in items:
        if item["item_id"] in done:
            continue
        user = fill_template(tmpl["user"], {"TASK_PROMPT": item["task_prompt"],
                                            "TEXT": item["text"]})
        if args.confidence:
            user = user + "\n" + tmpl["confidence_suffix"]
        prompt_text = build_chat_text(tok, user=user, system=tmpl["system"])

        logps = first_token_logprobs(model, tok, prompt_text, candidates)
        choice = "Yes" if logps["Yes"] >= logps["No"] else "No"
        correct = (choice == "Yes") == bool(item["is_self"])

        freetext, ft_status, confidence = None, None, None
        want_ft = (freetext_mode == "all"
                   or (freetext_mode == "subsample" and ft_rng.random() < 0.3)
                   or args.confidence)  # confidence needs the generated tail
        if want_ft:
            max_toks = 24 if args.confidence else jd["max_freetext_tokens"]
            freetext = greedy_generate(model, tok, prompt_text, max_new_tokens=max_toks)
            _, ft_status = classify_freetext(freetext, "yesno")
            if args.confidence:
                confidence = parse_confidence(freetext)

        append_jsonl(out_path, {
            "item_id": item["item_id"],
            "judge": args.judge,
            "author": item["author"],
            "is_self": bool(item["is_self"]),
            "domain": item["domain"],
            "prompt_id": item["prompt_id"],
            "logp_yes": round(logps["Yes"], 6),
            "logp_no": round(logps["No"], 6),
            "choice": choice,
            "correct": correct,
            "confidence": confidence,
            "freetext": freetext,
            "freetext_status": ft_status,
            "created_at": now_iso(),
        })
        done.add(item["item_id"])
        n_new += 1
        if n_new % 50 == 0:
            print(f"  {n_new} items done", flush=True)

    print(f"[done] {n_new} new items -> {out_path}")
    print("NEXT: src/05_baselines.py (stylometric on CPU, perplexity on GPU).")


if __name__ == "__main__":
    main()
