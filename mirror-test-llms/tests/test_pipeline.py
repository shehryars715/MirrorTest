"""
Unit tests for the CPU-side logic of the pipeline.

RUN THEM (from the repo root, no GPU or big installs needed):

    pip install pyyaml pytest
    pytest tests/ -v

WHAT IS COVERED (and why it matters)
====================================
* clean_text            — if cleaning is wrong, boilerplate leaks authorship
                          and the whole result is an artifact (§7 step 2, §18).
* rouge_l_f1            — the near-duplicate filter (§18).
* pair building         — length matching + dedup logic (§7 step 3).
* every statistic in stats_utils — wilson, exact binomial, Holm, McNemar,
  kappa, d', AUROC, Spearman, bootstrap, power. Several are checked against
  hand-computable values; the power analysis is checked against the
  protocol's own claim (n=150 detects 0.615 with ~80% power, §10).
* free-text parsing     — refusal accounting (§9.7).

The GPU paths (model loading, logprob scoring) are exercised by running the
scripts with --limit/--max-prompts on the smallest model (see
docs/03_pipeline_walkthrough.md, "smoke test").
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import (  # noqa: E402
    build_ppp_pairs, classify_freetext, clean_text, fill_template,
    length_ratio_ok, n_words, parse_choice_letter, parse_yes_no, rouge_l_f1,
    strip_incomplete_tail,
)
from stats_utils import (  # noqa: E402
    auroc, binom_pmf, binom_test_two_sided, bootstrap_ci, cohen_kappa,
    dprime_criterion, holm_bonferroni, mcnemar_exact, power_exact_binomial,
    spearman_rho, wilson_ci,
)


# ---------------------------------------------------------------------------
# Text cleaning (§7 step 2)
# ---------------------------------------------------------------------------

def test_clean_strips_boilerplate_sentence():
    raw = "Sure! Here is a summary of the article: The economy grew rapidly. It ended well."
    assert clean_text(raw) == "The economy grew rapidly. It ended well."


def test_clean_strips_label_and_artifacts():
    raw = "**Summary:** The cat sat on the mat. It purred loudly.<|im_end|>"
    assert clean_text(raw) == "The cat sat on the mat. It purred loudly."


def test_clean_cuts_incomplete_tail():
    raw = "The plan worked perfectly. Everyone cheered. But then the"
    assert clean_text(raw) == "The plan worked perfectly. Everyone cheered."


def test_clean_no_sentence_end_is_invalid():
    assert clean_text("just a fragment with no ending punctuation") == ""


def test_clean_is_identity_on_clean_human_text():
    human = 'The council approved the budget on Tuesday. "We are pleased," the mayor said.'
    assert clean_text(human) == human


def test_strip_incomplete_tail_keeps_quote_closers():
    assert strip_incomplete_tail('He said "stop." And then he') == 'He said "stop."'


# ---------------------------------------------------------------------------
# ROUGE-L (§18 near-duplicate filter)
# ---------------------------------------------------------------------------

def test_rouge_identical_is_one():
    assert rouge_l_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_rouge_disjoint_is_zero():
    assert rouge_l_f1("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_rouge_known_value():
    # LCS("the cat sat", "the dog sat") = "the sat" -> len 2; P=R=2/3; F1=2/3
    assert abs(rouge_l_f1("the cat sat", "the dog sat") - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Pair building (§7 step 3)
# ---------------------------------------------------------------------------

def _texts(words_a: int, words_b: int):
    return ("word " * words_a).strip() + ".", ("term " * words_b).strip() + "."


def test_length_ratio_filter():
    assert length_ratio_ok(100, 80, 0.25)       # 20% diff -> keep
    assert not length_ratio_ok(100, 70, 0.25)   # 30% diff -> drop


def test_build_ppp_pairs_filters_and_reports():
    a1, b1 = _texts(100, 90)     # keep
    a2, b2 = _texts(100, 50)     # length mismatch -> drop
    dup = "same words here every time."
    self_texts = {"d_0000": a1, "d_0001": a2, "d_0002": dup}
    foil_texts = {"d_0000": b1, "d_0001": b2, "d_0002": dup, "d_0003": b1}
    prompts = {k: f"task {k}" for k in ["d_0000", "d_0001", "d_0002", "d_0003"]}
    pairs, report = build_ppp_pairs(self_texts, foil_texts, prompts,
                                    "judgeX", "foilY", "news")
    assert report == {"candidates": 3, "kept": 1, "len_mismatch": 1, "near_dup": 1}
    assert pairs[0]["pair_id"] == "judgeX__foilY__news__d_0000"
    assert pairs[0]["text_self"] == a1 and pairs[0]["text_foil"] == b1


def test_fill_template_survives_braces_in_text():
    out = fill_template("Task: {TASK_PROMPT}!", {"TASK_PROMPT": "explain {json} dicts"})
    assert out == "Task: explain {json} dicts!"


# ---------------------------------------------------------------------------
# Statistics (§10) — checked against hand-computable values
# ---------------------------------------------------------------------------

def test_wilson_ci_sane_and_reference_value():
    lo, hi = wilson_ci(0.6, 150)
    assert 0.5 < lo < 0.6 < hi < 0.7
    # Known reference: 8/10 -> Wilson 95% CI approx (0.490, 0.943)
    lo2, hi2 = wilson_ci(0.8, 10)
    assert abs(lo2 - 0.490) < 0.01 and abs(hi2 - 0.943) < 0.01
    # never leaves [0, 1] even at the extremes
    assert wilson_ci(1.0, 5)[1] <= 1.0 and wilson_ci(0.0, 5)[0] >= 0.0


def test_binom_pmf_sums_to_one():
    assert abs(sum(binom_pmf(k, 20, 0.3) for k in range(21)) - 1.0) < 1e-9


def test_binom_test_hand_values():
    assert abs(binom_test_two_sided(5, 10, 0.5) - 1.0) < 1e-9  # dead center
    assert abs(binom_test_two_sided(0, 10, 0.5) - 2 / 1024) < 1e-12
    assert binom_test_two_sided(90, 100, 0.5) < 1e-9          # overwhelming
    # symmetry at p0 = 0.5
    assert abs(binom_test_two_sided(60, 100) - binom_test_two_sided(40, 100)) < 1e-12


def test_holm_bonferroni_textbook_example():
    res = holm_bonferroni([("t1", 0.01), ("t2", 0.04), ("t3", 0.03)], alpha=0.05)
    by = {r["name"]: r for r in res}
    assert by["t1"]["p_holm"] == 0.03 and by["t1"]["reject"]           # 0.01*3
    assert abs(by["t3"]["p_holm"] - 0.06) < 1e-12 and not by["t3"]["reject"]
    assert not by["t2"]["reject"]  # blocked by the step-down rule


def test_mcnemar_hand_value():
    # b=1, c=9 -> exact binomial: 2*(C(10,0)+C(10,1))/2^10 = 22/1024
    assert abs(mcnemar_exact(1, 9) - 22 / 1024) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # 50/50 marginals, agreement exactly at chance -> kappa = 0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-12


def test_dprime_directions():
    good = dprime_criterion(hits=90, n_signal=100, false_alarms=10, n_noise=100)
    yes_bot = dprime_criterion(hits=99, n_signal=100, false_alarms=99, n_noise=100)
    assert good["dprime"] > 2.0
    assert abs(yes_bot["dprime"]) < 0.2       # says Yes to everything -> no sensitivity
    assert yes_bot["criterion_c"] < -1.0      # ...and a very liberal criterion


def test_auroc_hand_values():
    assert auroc([2.0, 3.0], [0.0, 1.0]) == 1.0            # perfect separation
    assert auroc([1.0, 1.0], [1.0, 1.0]) == 0.5            # all tied
    assert abs(auroc([1.0, 3.0], [2.0, 4.0]) - 0.25) < 1e-12


def test_spearman_monotone():
    assert abs(spearman_rho([1, 2, 3, 4, 5], [10, 20, 25, 40, 100]) - 1.0) < 1e-12
    assert abs(spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-12


def test_bootstrap_ci_brackets_mean():
    items = [0.0] * 50 + [1.0] * 50
    lo, hi = bootstrap_ci(lambda xs: sum(xs) / len(xs), items, n_boot=500, seed=1)
    assert lo < 0.5 < hi and (hi - lo) < 0.3


def test_power_reproduces_protocol_claims():
    """Protocol §10: n=150 has ~80% power for true acc 0.615; n=450 for 0.567."""
    p150 = power_exact_binomial(150, 0.615)
    p450 = power_exact_binomial(450, 0.567)
    assert 0.72 <= p150 <= 0.88, p150
    assert 0.72 <= p450 <= 0.90, p450


# ---------------------------------------------------------------------------
# Free-text parsing (§9.7)
# ---------------------------------------------------------------------------

def test_parse_choice_letter():
    assert parse_choice_letter("A") == "A"
    assert parse_choice_letter("**B**.") == "B"
    assert parse_choice_letter("I think it was A, honestly") == "A"
    assert parse_choice_letter("a") == "A"                    # bare lowercase ok
    assert parse_choice_letter("I wrote a response") is None  # article 'a' is not an answer
    assert parse_choice_letter("") is None


def test_parse_yes_no_and_refusal():
    assert parse_yes_no("Yes, that one.") == "Yes"
    assert parse_yes_no("no") == "No"
    assert classify_freetext("As an AI, I have no memory of writing this.", "letter") \
        == (None, "refusal")
    assert classify_freetext("banana", "letter") == (None, "unparseable")
    assert classify_freetext("B", "letter") == ("B", "ok")


def test_n_words():
    assert n_words("Hello, world! It's 2 words-ish.") == 5


# ---------------------------------------------------------------------------
# Kaggle compatibility: MIRROR_ROOT redirects the data/results tree
# ---------------------------------------------------------------------------

def test_mirror_root_env_override(tmp_path):
    """With MIRROR_ROOT set, all data/results paths anchor there (read-only
    repo mounts, e.g. Kaggle datasets, write to /kaggle/working instead)."""
    import os
    import subprocess

    src_dir = Path(__file__).resolve().parent.parent / "src"
    code = "import utils; print(utils.PROMPTS_DIR); print(utils.TABLES_DIR)"
    env = dict(os.environ, MIRROR_ROOT=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(src_dir),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == str(tmp_path / "data" / "prompts")
    assert lines[1] == str(tmp_path / "results" / "tables")
