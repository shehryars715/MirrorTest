"""
02b_paraphrase.py — build the paraphrased pair set (protocol §7 step 5, §9.4).

WHY THIS EXISTS (the paraphrase attack)
=======================================
If a judge recognizes "its own" text only by surface wording — favourite
phrases, punctuation habits — then rewriting BOTH candidate texts in someone
else's words (content preserved, wording changed) should destroy the signal.
If recognition survives paraphrasing, the signal is deeper than wording.
Either way the result is publishable (§11). This script builds the rewritten
pairs; 03_judge_ppp.py later judges them exactly like the originals.

WHAT IT DOES
============
For the representative cells (judge = largest model, foil = F1, each domain;
set in config `paraphrase:`):
  1. loads the original PPP pairs,
  2. rewrites BOTH texts with the paraphraser model (Phi-3.5-mini,
     temperature 0.3, protocol-verbatim instruction),
  3. cleans the rewrites with the SAME clean_text as everything else,
  4. QUALITY GATE: keeps a pair only if both rewrites have
        cosine similarity >= 0.80 to their originals (MiniLM embeddings —
        meaning preserved) AND length within +-30% (nothing dropped/padded),
  5. writes data/pairs/para__{judge}__{foil}__{domain}.jsonl with the SAME
     pair_id as the original pair (that is what lets 06_stats.py run the
     paired McNemar test), plus the originals and similarity scores.

USAGE
=====
    python src/02b_paraphrase.py                     # cells from config
    python src/02b_paraphrase.py --judge qwen2.5-7b-instruct   # if 14B was dropped

COST: ~1 GPU-hour (§15). Resumable like every GPU script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DOMAINS, PAIRS_DIR, append_jsonl, build_chat_text, clean_text,
    existing_ids, load_config, load_model_and_tokenizer, n_words, now_iso,
    progress_iter, read_jsonl, sampled_generate,
)

PARAPHRASE_INSTRUCTION = (
    "Rewrite the following text preserving its exact meaning but changing the "
    "wording and sentence structure. Output only the rewrite.\n\nText:\n\"{TEXT}\""
)


def length_within(orig: str, new: str, max_ratio: float) -> bool:
    a, b = n_words(orig), n_words(new)
    if a == 0 or b == 0:
        return False
    return abs(a - b) / a <= max_ratio


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paraphrase both texts of the representative PPP cells (§7 step 5).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--judge", default=None, help="override config paraphrase.judge")
    ap.add_argument("--foil", default=None, help="override config paraphrase.foil")
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--max-pairs", type=int, default=200,
                    help="paraphrase at most this many pairs per domain")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pcfg = cfg["paraphrase"]
    judge = args.judge or pcfg["judge"]
    foil = args.foil or pcfg["foil"]
    para = cfg["paraphraser"]

    # -- paraphraser model ---------------------------------------------------
    model, tok = load_model_and_tokenizer(para["hf_id"], para.get("revision"),
                                          four_bit=not args.no_4bit)

    # -- embedder for the quality gate ----------------------------------------
    # (sentence-transformers; ~80 MB model, runs on GPU if available)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(cfg["embedder"])

    def cosine(a: str, b: str) -> float:
        va, vb = embedder.encode([a, b], normalize_embeddings=True)
        return float((va * vb).sum())

    def rewrite(text: str, seed: int) -> str:
        prompt = build_chat_text(
            tok, user=PARAPHRASE_INSTRUCTION.replace("{TEXT}", text), system=None)
        out = sampled_generate(
            model, tok, prompt,
            temperature=para["temperature"], top_p=para.get("top_p", 0.95),
            max_new_tokens=para["max_new_tokens"], seed=seed,
        )
        return clean_text(out.strip().strip('"'))

    for domain in args.domains:
        src_path = PAIRS_DIR / f"ppp__{judge}__{foil}__{domain}.jsonl"
        pairs = read_jsonl(src_path)
        if not pairs:
            print(f"[warn] no pairs at {src_path} - run 02_build_pairs.py first "
                  f"(and check that '{judge}' and '{foil}' have generations)")
            continue
        pairs = pairs[: args.max_pairs]
        out_path = PAIRS_DIR / f"para__{judge}__{foil}__{domain}.jsonl"
        done = existing_ids(out_path, "pair_id")
        todo = [p for p in pairs if p["pair_id"] not in done]
        print(f"[para] {domain}: {len(todo)} pairs to paraphrase "
              f"({len(pairs) - len(todo)} already done)")

        kept = 0
        for p in progress_iter(todo, label=f"para {domain}"):
            seed = para["seed_base"] + int(p["prompt_id"].split("_")[-1])
            new_self = rewrite(p["text_self"], seed)
            new_foil = rewrite(p["text_foil"], seed + 500_000)
            cos_self = cosine(p["text_self"], new_self) if new_self else 0.0
            cos_foil = cosine(p["text_foil"], new_foil) if new_foil else 0.0
            ok = (
                bool(new_self) and bool(new_foil)
                and cos_self >= pcfg["min_cosine"] and cos_foil >= pcfg["min_cosine"]
                and length_within(p["text_self"], new_self, pcfg["max_length_ratio_diff"])
                and length_within(p["text_foil"], new_foil, pcfg["max_length_ratio_diff"])
            )
            append_jsonl(out_path, {
                **{k: p[k] for k in ("pair_id", "judge", "foil", "domain",
                                     "prompt_id", "task_prompt")},
                "condition": "paraphrase",
                "text_self": new_self, "text_foil": new_foil,
                "orig_text_self": p["text_self"], "orig_text_foil": p["text_foil"],
                "n_words_self": n_words(new_self), "n_words_foil": n_words(new_foil),
                "cos_self": round(cos_self, 4), "cos_foil": round(cos_foil, 4),
                "passed_gate": ok, "created_at": now_iso(),
            })
            kept += int(ok)

        total_ok = sum(1 for r in read_jsonl(out_path) if r.get("passed_gate"))
        tgt = pcfg["target_pairs_per_domain"]
        flag = "" if total_ok >= tgt else f"  <-- BELOW TARGET {tgt}: raise --max-pairs"
        print(f"[para] {domain}: {total_ok} pairs passed the quality gate{flag}")

    print("NEXT: src/03_judge_ppp.py --include-paraphrase on the paraphrase judge.")


if __name__ == "__main__":
    main()
