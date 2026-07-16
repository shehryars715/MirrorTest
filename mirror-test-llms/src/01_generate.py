"""
01_generate.py — every model answers every frozen prompt (protocol §7 step 1).

WHAT THIS SCRIPT DOES
=====================
For each requested model and domain, loads the model once (4-bit NF4), then
samples ONE response per prompt with the frozen decoding settings
(temperature 0.8, top-p 0.95, max 160 new tokens) and a per-prompt seed:

    seed = 1000 + prompt_idx            (main sample,   "s1")
    seed = 2000 + prompt_idx            (placebo second sample, "s2",
                                         judges only, first 150 prompts)

The second sample exists so 02_build_pairs.py can build SELF-vs-SELF
"placebo" pairs — the critical control that measures pure position bias
(protocol §7 step 4 and §9.1).

OUTPUT
======
    data/generations/{model_key}__{domain}.jsonl, one record per generation:

    {"gen_id": "news_0042__qwen2.5-7b-instruct__s1",
     "prompt_id": "news_0042", "prompt_idx": 42, "domain": "news",
     "model": "qwen2.5-7b-instruct", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
     "revision": "<resolved commit hash>", "sample": "s1", "seed": 1042,
     "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 160,
     "text": "<RAW text - cleaning happens later in 02 so it is auditable>",
     "n_words": 97, "created_at": "..."}

RESUME BEHAVIOUR
================
Records are appended and fsync'd one by one. If Colab dies, re-run the exact
same command: everything already in the output file is skipped. This is why
you can split a big generation run across as many sessions as you need.

USAGE
=====
    # one model, all domains:
    python src/01_generate.py --models qwen2.5-7b-instruct

    # judges also produce the placebo second sample:
    python src/01_generate.py --models qwen2.5-7b-instruct --placebo

    # everything (do this across several Colab sessions, one model at a time):
    python src/01_generate.py --all-judges --placebo
    python src/01_generate.py --all-foils

    # quick smoke test (2 prompts per domain, tiny model, CPU-tolerable):
    python src/01_generate.py --models qwen2.5-0.5b-instruct --max-prompts 2

COST: this is the biggest GPU phase. ~9 models x 600 prompts x <=160 tokens
is roughly 4-8 GPU-hours total on a free T4 (protocol §7/§15).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DOMAINS, GENERATIONS_DIR, PROMPTS_DIR, append_jsonl, build_chat_text,
    existing_ids, load_config, load_model_and_tokenizer, model_index, n_words,
    now_iso, progress_iter, read_jsonl, resolved_revision, sampled_generate,
)


def generate_for_model(model_key: str, domains: list[str], cfg: dict,
                       placebo: bool, max_prompts: int | None, four_bit: bool) -> None:
    idx = model_index(cfg)
    if model_key not in idx:
        sys.exit(f"[error] unknown model key '{model_key}'. Known: {sorted(idx)}")
    mcfg = idx[model_key]
    if mcfg.get("hf_id") is None:
        print(f"[skip] {model_key} is the HUMAN pseudo-author - nothing to generate.")
        return

    gen = cfg["generation"]
    is_judge = mcfg["group"] == "judges"
    model, tok = load_model_and_tokenizer(mcfg["hf_id"], mcfg.get("revision"), four_bit=four_bit)
    revision = resolved_revision(model) or mcfg.get("revision")

    for domain in domains:
        prompts = read_jsonl(PROMPTS_DIR / f"{domain}.jsonl")
        if not prompts:
            print(f"[warn] no prompts for domain '{domain}' - run 00_build_prompts.py first")
            continue
        if max_prompts:
            prompts = prompts[:max_prompts]
        out_path = GENERATIONS_DIR / f"{model_key}__{domain}.jsonl"
        done = existing_ids(out_path, "gen_id")

        # Which (prompt, sample) jobs does this model owe?
        jobs = [(p, "s1", gen["seed_base"]) for p in prompts]
        if placebo and is_judge:
            n_pl = gen["placebo_n_prompts"]
            jobs += [(p, "s2", gen["placebo_seed_base"])
                     for p in prompts if p["prompt_idx"] < n_pl]

        todo = [(p, tag, base) for (p, tag, base) in jobs
                if f"{p['prompt_id']}__{model_key}__{tag}" not in done]
        print(f"[gen] {model_key} / {domain}: {len(todo)} to generate "
              f"({len(jobs) - len(todo)} already done)")

        for p, tag, seed_base in progress_iter(todo, label=f"gen {model_key}/{domain}"):
            seed = seed_base + p["prompt_idx"]
            prompt_text = build_chat_text(tok, user=p["task_prompt"], system=None)
            text = sampled_generate(
                model, tok, prompt_text,
                temperature=gen["temperature"], top_p=gen["top_p"],
                max_new_tokens=gen["max_new_tokens"], seed=seed,
            )
            append_jsonl(out_path, {
                "gen_id": f"{p['prompt_id']}__{model_key}__{tag}",
                "prompt_id": p["prompt_id"],
                "prompt_idx": p["prompt_idx"],
                "domain": domain,
                "model": model_key,
                "hf_id": mcfg["hf_id"],
                "revision": revision,
                "sample": tag,
                "seed": seed,
                "temperature": gen["temperature"],
                "top_p": gen["top_p"],
                "max_new_tokens": gen["max_new_tokens"],
                "text": text,
                "n_words": n_words(text),
                "created_at": now_iso(),
            })

    # Free GPU memory before the caller loads the next model.
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase-1 generation: all models answer the frozen prompts (§7).",
        epilog="Example: python src/01_generate.py --models qwen2.5-0.5b-instruct --placebo",
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=[], help="model keys from configs/models.yaml")
    ap.add_argument("--all-judges", action="store_true")
    ap.add_argument("--all-foils", action="store_true", help="all non-human foils")
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--placebo", action="store_true",
                    help="also produce the second (s2) sample judges need for placebo pairs")
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="cap prompts per domain (smoke tests)")
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit quantisation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    keys = list(args.models)
    if args.all_judges:
        keys += [m["key"] for m in cfg["judges"]]
    if args.all_foils:
        keys += [m["key"] for m in cfg["foils"] if m.get("hf_id")]
    if not keys:
        sys.exit("Nothing to do: pass --models KEY [KEY ...] or --all-judges/--all-foils")

    for key in dict.fromkeys(keys):  # dedupe, keep order
        generate_for_model(key, args.domains, cfg, args.placebo, args.max_prompts,
                           four_bit=not args.no_4bit)
    print("NEXT: run src/02_build_pairs.py (CPU) to clean texts and build pairs.")


if __name__ == "__main__":
    main()
