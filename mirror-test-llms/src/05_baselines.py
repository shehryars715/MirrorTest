"""
05_baselines.py — the two mechanism probes (protocol §9.2, §9.3).

These baselines are the heart of RQ-Mechanism. The judge's accuracy alone
cannot tell us WHAT it used to recognize itself; these two dumb-on-purpose
rules bracket the possibilities:

STYLOMETRIC BASELINE (§9.2)  —  subcommand `stylometric`, CPU only
==================================================================
"The 1990s classifier": TF-IDF over character 3-5-grams -> logistic
regression, predicting the author (judge vs foil) of single texts, with
5-fold GROUPED cross-validation (both texts of a pair stay in the same fold,
so there is no leakage between train and test).

Interpretation rule committed in advance (§9.2): if this classifier's
accuracy >= the judge's accuracy on the same texts, then all the information
needed for "self-recognition" is present in surface statistics — privileged
self-access is not required to explain the behaviour.

Also produced:
  * a "surface features only" variant (sentence length, word length,
    type-token ratio, punctuation frequencies) — even shallower;
  * a cross-domain transfer matrix (train on news, test on dolly, ...) —
    if the style signature transfers, it is model-general, not task-specific;
  * per-pair decisions (pick the candidate with higher out-of-fold
    P(judge-authored)) so 06_stats.py can run McNemar judge-vs-baseline on
    EXACTLY the same items.

PERPLEXITY-PREFERENCE BASELINE (§9.3)  —  subcommand `perplexity`, GPU
======================================================================
For every PPP pair, compute the judge model's mean per-token negative
log-likelihood (NLL) of each candidate text given the ORIGINAL writing task
(the same context it would have generated under). perplexity = exp(NLL).
Rule: "pick the lower-perplexity candidate" — i.e. "pick the text that
sounds most like something I'd say", with no self-concept whatsoever.

06_stats.py then reports (a) this rule's accuracy and (b) Cohen's kappa
between the rule's choices and the judge's actual choices. High kappa =
the judge is plausibly just expressing likelihood preference (§2, H4).

OUTPUT
======
  results/baselines/stylo__{judge}__{foil}__{domain}.json      (cell summary)
  results/baselines/stylo_pairs__{judge}__{foil}__{domain}.jsonl (per-pair)
  results/baselines/transfer__{judge}__{foil}.json             (domain transfer)
  results/baselines/ppl__{judge}.jsonl                          (per-pair NLLs)

USAGE
=====
    python src/05_baselines.py stylometric                    # all cells, CPU
    python src/05_baselines.py perplexity --judge qwen2.5-7b-instruct   # GPU
    python src/05_baselines.py perplexity --judge qwen2.5-7b-base       # §9.5 proxy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    BASELINES_DIR, PAIRS_DIR, append_jsonl, build_chat_text, continuation_mean_nll,
    existing_ids, load_config, load_model_and_tokenizer, model_index, now_iso,
    progress_iter, read_jsonl,
)


# ==========================================================================
# Stylometric baseline (CPU, scikit-learn)
# ==========================================================================

def surface_features(text: str) -> list[float]:
    """Hand-crafted shallow features: if THESE predict authorship, the style
    signature is embarrassingly superficial."""
    words = re.findall(r"[A-Za-z0-9'’-]+", text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    n_chars = max(len(text), 1)
    n_w = max(len(words), 1)
    return [
        len(words) / max(len(sentences), 1),                  # avg sentence length
        sum(len(w) for w in words) / n_w,                     # avg word length
        len({w.lower() for w in words}) / n_w,                # type-token ratio
        *(text.count(ch) / n_chars for ch in ",.;:!?'\"-"),   # punctuation rates
        sum(ch.isdigit() for ch in text) / n_chars,           # digit rate
        sum(ch.isupper() for ch in text) / n_chars,           # uppercase rate
    ]


def run_stylometric_cell(pairs: list[dict], cfg: dict) -> tuple[dict, list[dict]]:
    """5-fold grouped CV on one (judge, foil, domain) cell.

    Texts: each pair contributes one SELF text (label 1) and one FOIL text
    (label 0). Groups = prompt_id, so a pair is never split across folds —
    the classifier is always tested on prompts it has not seen.
    Returns (summary, per_pair_decisions).
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scfg = cfg["stylometric"]
    texts, labels, groups = [], [], []
    for p in pairs:
        texts += [p["text_self"], p["text_foil"]]
        labels += [1, 0]
        groups += [p["prompt_id"], p["prompt_id"]]
    labels = np.array(labels)

    def oof_probs(make_model, X):
        """Out-of-fold P(label=1) for every text, via grouped K-fold."""
        probs = np.zeros(len(labels))
        gkf = GroupKFold(n_splits=scfg["n_folds"])
        for tr, te in gkf.split(X, labels, groups=groups):
            m = make_model()
            m.fit([X[i] for i in tr] if isinstance(X, list) else X[tr], labels[tr])
            Xte = [X[i] for i in te] if isinstance(X, list) else X[te]
            probs[te] = m.predict_proba(Xte)[:, 1]
        return probs

    # --- main variant: char n-gram TF-IDF + LogReg -------------------------
    def make_tfidf():
        return make_pipeline(
            TfidfVectorizer(analyzer="char",
                            ngram_range=tuple(scfg["char_ngram_range"]),
                            max_features=scfg["max_features"], min_df=2),
            LogisticRegression(C=scfg["logreg_C"], max_iter=scfg["logreg_max_iter"]),
        )

    p_char = oof_probs(make_tfidf, texts)
    acc_char = float(((p_char >= 0.5).astype(int) == labels).mean())

    # --- surface-features variant ------------------------------------------
    feats = np.array([surface_features(t) for t in texts])

    def make_surface():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=scfg["logreg_C"],
                                                max_iter=scfg["logreg_max_iter"]))

    p_surf = oof_probs(make_surface, feats)
    acc_surf = float(((p_surf >= 0.5).astype(int) == labels).mean())

    # --- pair-level decisions: pick the candidate with higher OOF P(self) ---
    per_pair, pair_hits = [], 0
    for k, p in enumerate(pairs):
        ps, pf = p_char[2 * k], p_char[2 * k + 1]  # self text, foil text
        chose_self = bool(ps >= pf)
        pair_hits += int(chose_self)
        per_pair.append({"pair_id": p["pair_id"], "prompt_id": p["prompt_id"],
                         "stylo_chose_self": chose_self,
                         "p_self_selftext": round(float(ps), 4),
                         "p_self_foiltext": round(float(pf), 4)})
    summary = {
        "n_pairs": len(pairs), "n_texts": len(texts),
        "acc_singletext_char_ngram": round(acc_char, 4),
        "acc_singletext_surface": round(acc_surf, 4),
        "acc_pairwise_char_ngram": round(pair_hits / len(pairs), 4),
    }
    return summary, per_pair


def run_transfer(judge: str, foil: str, domains: list[str], cfg: dict) -> dict:
    """Train the char-ngram classifier on one domain, test on another (§9.2
    'extra credit'). Transfer = the style signature is model-general."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    scfg = cfg["stylometric"]
    data = {}
    for d in domains:
        pairs = read_jsonl(PAIRS_DIR / f"ppp__{judge}__{foil}__{d}.jsonl")
        if pairs:
            data[d] = ([t for p in pairs for t in (p["text_self"], p["text_foil"])],
                       np.array([1, 0] * len(pairs)))
    out = {}
    for d_tr, (X_tr, y_tr) in data.items():
        model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=tuple(scfg["char_ngram_range"]),
                            max_features=scfg["max_features"], min_df=2),
            LogisticRegression(C=scfg["logreg_C"], max_iter=scfg["logreg_max_iter"]))
        model.fit(X_tr, y_tr)
        for d_te, (X_te, y_te) in data.items():
            if d_te == d_tr:
                continue
            out[f"train_{d_tr}__test_{d_te}"] = round(float(model.score(X_te, y_te)), 4)
    return out


def cmd_stylometric(args, cfg) -> None:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PAIRS_DIR.glob("ppp__*.jsonl"))
    if args.judge:
        files = [f for f in files if f.name.split("__")[1] == args.judge]
    if not files:
        sys.exit("[error] no ppp pair files found - run 02_build_pairs.py first")
    seen_cells = set()
    for path in files:
        _, judge, foil, domain = path.stem.split("__")
        pairs = read_jsonl(path)
        if len(pairs) < 20:
            print(f"[skip] {path.name}: only {len(pairs)} pairs (need >=20 for 5-fold CV)")
            continue
        summary, per_pair = run_stylometric_cell(pairs, cfg)
        summary.update({"judge": judge, "foil": foil, "domain": domain,
                        "created_at": now_iso()})
        with open(BASELINES_DIR / f"stylo__{judge}__{foil}__{domain}.json", "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        from utils import write_jsonl
        write_jsonl(BASELINES_DIR / f"stylo_pairs__{judge}__{foil}__{domain}.jsonl", per_pair)
        print(f"[stylo] {judge} vs {foil} / {domain}: "
              f"char-ngram={summary['acc_singletext_char_ngram']:.3f} "
              f"surface={summary['acc_singletext_surface']:.3f} "
              f"pairwise={summary['acc_pairwise_char_ngram']:.3f}")
        seen_cells.add((judge, foil))
    # transfer matrices once per (judge, foil)
    for judge, foil in sorted(seen_cells):
        transfer = run_transfer(judge, foil, ["news", "dolly", "wp"], cfg)
        if transfer:
            with open(BASELINES_DIR / f"transfer__{judge}__{foil}.json", "w",
                      encoding="utf-8") as f:
                json.dump(transfer, f, indent=2)
            print(f"[transfer] {judge} vs {foil}: {transfer}")


# ==========================================================================
# Perplexity-preference baseline (GPU)
# ==========================================================================

def cmd_perplexity(args, cfg) -> None:
    idx = model_index(cfg)
    if args.judge not in idx:
        sys.exit(f"[error] unknown judge '{args.judge}'. Known: {sorted(idx)}")
    jcfg = idx[args.judge]
    owner = jcfg.get("pairs_of") or args.judge   # base models score their sibling's pairs
    is_chat = jcfg.get("template") != "completion"

    files = sorted(PAIRS_DIR.glob(f"ppp__{owner}__*.jsonl"))
    if args.include_paraphrase:
        files += sorted(PAIRS_DIR.glob(f"para__{owner}__*.jsonl"))
    if not files:
        sys.exit(f"[error] no pair files for owner '{owner}'")

    model, tok = load_model_and_tokenizer(jcfg["hf_id"], jcfg.get("revision"),
                                          four_bit=not args.no_4bit)
    out_path = BASELINES_DIR / f"ppl__{args.judge}.jsonl"
    done = existing_ids(out_path, "row_id")

    for path in files:
        cond = "paraphrase" if path.name.startswith("para__") else "core"
        pairs = read_jsonl(path)
        if cond == "paraphrase":
            pairs = [p for p in pairs if p.get("passed_gate")]
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"[ppl] {path.name}: {len(pairs)} pairs")
        for p in progress_iter(pairs, label=f"ppl {args.judge} {path.stem}"):
            row_id = f"{p['pair_id']}__{cond}"
            if row_id in done:
                continue
            # Context = the judge's own generation context for this task:
            # "how likely would I be to produce this text for this prompt?"
            if is_chat:
                ctx = build_chat_text(tok, user=p["task_prompt"], system=None)
                templated = True
            else:
                ctx = p["task_prompt"] + "\n\n"
                templated = False
            nll_self, nt_s = continuation_mean_nll(model, tok, ctx, p["text_self"],
                                                   context_already_templated=templated)
            nll_foil, nt_f = continuation_mean_nll(model, tok, ctx, p["text_foil"],
                                                   context_already_templated=templated)
            append_jsonl(out_path, {
                "row_id": row_id, "pair_id": p["pair_id"], "judge": args.judge,
                "pairs_of": jcfg.get("pairs_of"), "foil": p["foil"],
                "domain": p["domain"], "condition": cond,
                "nll_self": round(nll_self, 6), "nll_foil": round(nll_foil, 6),
                "n_tok_self": nt_s, "n_tok_foil": nt_f,
                "rule_chose_self": bool(nll_self <= nll_foil),
                "created_at": now_iso(),
            })
            done.add(row_id)
    print(f"[done] -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mechanism baselines: stylometric classifier (§9.2) and "
                    "perplexity-preference rule (§9.3).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("stylometric", help="TF-IDF char-ngram + LogReg, CPU only")
    s1.add_argument("--config", default=None)
    s1.add_argument("--judge", default=None, help="restrict to one judge's cells")

    s2 = sub.add_parser("perplexity", help="lower-perplexity rule under the judge, GPU")
    s2.add_argument("--config", default=None)
    s2.add_argument("--judge", required=True)
    s2.add_argument("--include-paraphrase", action="store_true")
    s2.add_argument("--limit", type=int, default=None)
    s2.add_argument("--no-4bit", action="store_true")

    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "stylometric":
        cmd_stylometric(args, cfg)
    else:
        cmd_perplexity(args, cfg)
    print("NEXT: src/06_stats.py to produce all tables and figures.")


if __name__ == "__main__":
    main()
