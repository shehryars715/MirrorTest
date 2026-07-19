"""
03_judge_ppp.py — the actual mirror test: pairwise self-recognition
(protocol §8.1, §8.3, §8.4, plus the base-model variant of §9.5).

WHAT ONE "RUN" IS
=================
A run = (pair, order, phrasing). The judge sees the task prompt and two
responses labelled A and B — one is its own generation, one is the foil's —
and must answer with a single letter. Every pair is evaluated TWICE:

    order "self_A": A = judge's text, B = foil's text
    order "self_B": A = foil's text,  B = judge's text

Counterbalancing (§8.3) means position preference ("always say A") cancels
out: the item-level score is the mean of the two runs (0, 0.5, or 1), and a
0.5 means the judge picked the same POSITION both times, i.e. its answer
tracked position, not authorship.

HOW THE ANSWER IS MEASURED (§8.4 — the important trick)
=======================================================
We do NOT parse generated text as the primary measure. We take ONE forward
pass and read the model's probability of answering "A" vs "B" as its first
token (log-sum-exp over the 'A'/' A' spellings). argmax = the decision;
logp(A) - logp(B) = a continuous confidence score that later gives AUROC.
Benefits: zero unparseable answers, and refusals cannot hide the underlying
preference. A short greedy free-text answer is ALSO logged (on a seeded
subsample by default) purely to report refusal rates (§9.7).

BASE MODELS (§9.5)
==================
Base checkpoints can't follow chat instructions, so for models whose config
says `template: completion` the prompt is a plain string ending in
"...The response I wrote is Response" and we read logp(" A") vs logp(" B").
Their `pairs_of` field says whose pairs they judge (their instruct sibling's).

OUTPUT
======
results/judgments/ppp__{judge}.jsonl — one record per run:
    {"run_id": "<pair_id>__self_A__ph0", "pair_id": ..., "judge": ...,
     "foil": ..., "domain": ..., "condition": "core|placebo|paraphrase",
     "phrasing": 0, "order": "self_A", "self_position": "A",
     "logp_A": -0.31, "logp_B": -1.42, "choice": "A", "correct": true,
     "freetext": "A", "freetext_status": "ok", ...}

USAGE
=====
    # full grid for one judge (phrasing 0 only):
    python src/03_judge_ppp.py --judge qwen2.5-7b-instruct --include-placebo

    # the main cell under all three phrasings + paraphrase condition
    # (do this for the largest judge; §8.5, §9.4):
    python src/03_judge_ppp.py --judge qwen2.5-14b-instruct \
        --foils llama-3.2-3b-instruct --phrasings 0 1 2 --include-paraphrase

    # base-vs-instruct ablation (§9.5):
    python src/03_judge_ppp.py --judge qwen2.5-7b-base

COST: one forward pass per run; ~0.3-1 s each on a T4 depending on judge
size. The full grid is a few GPU-hours total (§15). Fully resumable.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    JUDGMENTS_DIR, PAIRS_DIR, append_jsonl, build_chat_text,
    candidate_token_ids, classify_freetext, existing_ids, fill_template,
    first_token_logprobs, greedy_generate, load_config,
    load_model_and_tokenizer, model_index, now_iso, progress_iter, read_jsonl,
)


def find_pair_files(judge: str, pairs_of: str | None, foils: list[str] | None,
                    include_placebo: bool, include_paraphrase: bool) -> list[Path]:
    """Collect the pair files this judge owes judgments for."""
    owner = pairs_of or judge
    files = sorted(PAIRS_DIR.glob(f"ppp__{owner}__*.jsonl"))
    if foils:
        files = [f for f in files if f.name.split("__")[2] in foils]
    if include_placebo:
        files += sorted(PAIRS_DIR.glob(f"placebo__{owner}__*.jsonl"))
    if include_paraphrase:
        files += sorted(PAIRS_DIR.glob(f"para__{owner}__*.jsonl"))
    return files


def condition_of(path: Path) -> str:
    return {"ppp": "core", "placebo": "placebo", "para": "paraphrase"}[path.name.split("__")[0]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pairwise (PPP) mirror test with log-prob scoring (§8).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--judge", required=True, help="judge key (instruct or base)")
    ap.add_argument("--foils", nargs="+", default=None, help="restrict to these foils")
    ap.add_argument("--phrasings", nargs="+", type=int, default=[0],
                    help="which judge-instruction phrasings to run (default 0; "
                         "run '0 1 2' on the main cell for §8.5)")
    ap.add_argument("--include-placebo", action="store_true")
    ap.add_argument("--include-paraphrase", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max pairs per file (smoke test)")
    ap.add_argument("--freetext", choices=["all", "subsample", "none"], default=None,
                    help="when to also log a greedy free-text answer "
                         "(default: config judging.freetext_mode)")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    idx = model_index(cfg)
    if args.judge not in idx:
        sys.exit(f"[error] unknown judge '{args.judge}'. Known: {sorted(idx)}")
    jcfg = idx[args.judge]
    is_completion = jcfg.get("template") == "completion"
    pairs_of = jcfg.get("pairs_of")

    jd = cfg["judging"]
    freetext_mode = args.freetext or jd["freetext_mode"]

    files = find_pair_files(args.judge, pairs_of, args.foils,
                            args.include_placebo, args.include_paraphrase)
    if not files:
        sys.exit(f"[error] no pair files found for judge '{args.judge}' "
                 f"(owner={pairs_of or args.judge}) - run 02_build_pairs.py first")

    model, tok = load_model_and_tokenizer(jcfg["hf_id"], jcfg.get("revision"),
                                          four_bit=not args.no_4bit)
    candidates = {
        "A": candidate_token_ids(tok, jd["a_variants"]),
        "B": candidate_token_ids(tok, jd["b_variants"]),
    }

    # Seeded choice of which runs get the extra free-text generation.
    ft_rng = random.Random(cfg["seed"])
    ft_budget = jd["freetext_subsample_n"]

    out_path = JUDGMENTS_DIR / f"ppp__{args.judge}.jsonl"
    done = existing_ids(out_path, "run_id")
    phrasings = {p["id"]: p for p in cfg["ppp_phrasings"]}

    total_new = 0
    for path in files:
        pairs = read_jsonl(path)
        if condition_of(path) == "paraphrase":
            pairs = [p for p in pairs if p.get("passed_gate")]
        if args.limit:
            pairs = pairs[: args.limit]
        cond = condition_of(path)
        print(f"[judge] {path.name}: {len(pairs)} pairs, condition={cond}")

        for pair in progress_iter(pairs, label=f"judge {args.judge} {path.stem}"):
            for ph_id in args.phrasings:
                ph = phrasings[ph_id]
                for order in ("self_A", "self_B"):
                    # Paraphrased pairs intentionally reuse the ORIGINAL
                    # pair_id (that is what pairs them for McNemar in
                    # 06_stats), so their run_ids must carry the condition —
                    # otherwise they collide with the core runs of the same
                    # pair and resume skips them all (session-6 bug: the
                    # paraphrase judging silently no-oped).
                    tag = "" if cond != "paraphrase" else "__para"
                    run_id = f"{pair['pair_id']}{tag}__{order}__ph{ph_id}"
                    if run_id in done:
                        continue
                    text_a = pair["text_self"] if order == "self_A" else pair["text_foil"]
                    text_b = pair["text_foil"] if order == "self_A" else pair["text_self"]
                    mapping = {"TASK_PROMPT": pair["task_prompt"],
                               "TEXT_A": text_a, "TEXT_B": text_b}

                    if is_completion:
                        prompt_text = fill_template(cfg["judge_templates"]["completion"], mapping)
                        templated = False
                    else:
                        prompt_text = build_chat_text(
                            tok, user=fill_template(ph["user"], mapping), system=ph["system"])
                        templated = True

                    logps = first_token_logprobs(model, tok, prompt_text, candidates,
                                                 already_templated=templated)
                    choice = "A" if logps["A"] >= logps["B"] else "B"
                    self_position = "A" if order == "self_A" else "B"
                    correct = None if cond == "placebo" else (choice == self_position)

                    # Free-text answer (refusal accounting) on a seeded subsample.
                    freetext, ft_status = None, None
                    want_ft = (freetext_mode == "all"
                               or (freetext_mode == "subsample" and ft_budget > 0
                                   and ft_rng.random() < 0.15))
                    if want_ft and not is_completion:
                        freetext = greedy_generate(model, tok, prompt_text,
                                                   max_new_tokens=jd["max_freetext_tokens"])
                        _, ft_status = classify_freetext(freetext, "letter")
                        ft_budget -= 1

                    append_jsonl(out_path, {
                        "run_id": run_id,
                        "pair_id": pair["pair_id"],
                        "judge": args.judge,
                        "pairs_of": pairs_of,
                        "foil": pair["foil"],
                        "domain": pair["domain"],
                        "prompt_id": pair["prompt_id"],
                        "condition": cond,
                        "phrasing": ph_id,
                        "order": order,
                        "self_position": self_position,
                        "logp_A": round(logps["A"], 6),
                        "logp_B": round(logps["B"], 6),
                        "choice": choice,
                        "correct": correct,
                        "freetext": freetext,
                        "freetext_status": ft_status,
                        "template": "completion" if is_completion else "chat",
                        "created_at": now_iso(),
                    })
                    done.add(run_id)
                    total_new += 1

    print(f"[done] {total_new} new runs -> {out_path}")
    print("NEXT: repeat for the other judges, then src/04_judge_ipp.py and "
          "src/05_baselines.py.")


if __name__ == "__main__":
    main()
