# Glossary — every term in this project, alphabetical

Cross-references: → means "see also". Protocol § numbers refer to
`../mirrortest.md`.

**AUROC** — Area Under the ROC Curve. The probability that a randomly chosen
positive item receives a higher score than a randomly chosen negative item
(ties count ½). 0.5 = no discrimination, 1.0 = perfect. Threshold-free,
which is why it accompanies accuracy in the scale analysis (§9.6).
→ emergence, logprob.

**Base model** — a checkpoint after pretraining only (e.g. `Qwen2.5-7B`),
before instruction tuning. Predicts next tokens; cannot follow "Answer with
one letter." Judged here via a completion template (§9.5).

**Bootstrap** — estimating a statistic's uncertainty by resampling the data
with replacement many times (here 1,000) and taking the middle 95% of the
recomputed values as the CI. Used for κ and AUROC. Resampling unit = the
pair, so counterbalanced runs stay together.

**BPE / tokenizer** — byte-pair encoding; how text becomes tokens. "A" and
" A" are different tokens — the reason answer probabilities are summed over
spelling variants (§8.4).

**Chance level** — the accuracy of blind guessing: 50% in a two-choice test.

**Cleaning** — stripping chat artifacts, boilerplate openers, and incomplete
final sentences from every text, identically for all authors including
humans (§7 step 2). Prevents boilerplate from becoming the recognition
signal.

**Cohen's κ (kappa)** — agreement between two decision-makers corrected for
chance agreement. 0 = luck-level, 1 = perfect. Pre-registered interest
threshold: κ > 0.4 between judge and perplexity rule (H4).

**Counterbalancing** — presenting each pair in both (SELF=A) and (SELF=B)
orders and averaging, so position preference cancels (§8.3). → placebo.

**Criterion (c)** — signal-detection measure of response bias; negative =
liberal ("says Yes a lot"). → d′.

**d′ (d-prime)** — signal-detection sensitivity: z(hit rate) − z(false-alarm
rate). Separates real discrimination from answering "Yes" to everything
(§9.7).

**Decisive item** — a pair answered consistently (correct in both orders or
wrong in both). Significance tests run on these; ties (one right, one wrong
= position-driven) are excluded as uninformative.

**Domain** — a text genre in the study: news summarization (`news`), open
QA (`dolly`), creative writing (`wp`).

**Emergence** — a capability appearing abruptly with scale. Contested:
apparent jumps can be artifacts of thresholded metrics like accuracy
(Schaeffer et al. 2023) — hence AUROC is reported alongside (§9.6).

**Exact binomial test** — significance test for "k correct out of n vs
chance p=0.5" computed from the exact binomial distribution, no
approximations. → Holm.

**Foil** — the "other author" whose text is paired against the judge's own:
another LLM or a human (§4).

**Gated model** — a Hugging Face model requiring license acceptance + token
authentication (Llama, Gemma here).

**Generator / Judge** — roles a model plays: producing texts vs being asked
"which of these did you write?" A judge only ever judges pairs containing
its own generation (§4).

**Holm–Bonferroni** — multiple-comparison correction: sorted p-values are
compared against α/m, α/(m−1), … stopping at the first failure. Controls
the chance of ANY false positive across the 20 judge×foil tests (§10).

**IPP** — individual presentation paradigm: one text, "Did you write this?
Yes/No," analysed with signal detection (§8.2, §9.7). → PPP.

**JSONL** — JSON Lines; one JSON object per line. Appendable and
crash-tolerant — the storage format of every pipeline output.

**Logit / logprob** — a model's raw score per vocabulary token / the log of
its softmax probability. First-token logprobs of "A"/"B" ARE the judge's
decision in this study (§8.4).

**Log-sum-exp** — numerically safe way to add probabilities stored as logs;
used to merge "A" and " A" spellings.

**LoRA** — low-rank adapters; cheap fine-tuning that trains tiny correction
matrices on a frozen model (§9.9, optional extension).

**McNemar's test** — significance test for paired binary outcomes (same
items, two conditions); only disagreements carry evidence. Used for
original-vs-paraphrased and judge-vs-stylometric (§10).

**NF4 / 4-bit quantization** — storing weights in 4 bits (NormalFloat4) via
bitsandbytes so 7B–14B models fit free GPUs while keeping logprobs
available (§5).

**NLL (negative log-likelihood)** — −log p of a token; averaged over a text
= how surprising the text is to a model. → perplexity.

**Pair (PPP pair)** — same prompt, two responses: the judge's (SELF) and a
foil's, length-matched and deduplicated (§7 step 3).

**Paraphrase attack** — rewriting BOTH candidate texts (content preserved,
wording changed) and re-running the test; collapse ⇒ the signal was surface
form (§9.4).

**Perplexity** — exp(mean NLL); "how expected is this text to the model."
The perplexity-preference baseline picks the lower-perplexity candidate
(§9.3).

**Placebo pair** — SELF vs SELF (two seeds). No right answer exists; the
chose-A rate measures pure position bias and catches pipeline bugs (§9.1).

**Position bias / position-consistency** — a judge's preference for a
letter/slot regardless of content; the fraction of items answered the same
under both orders. → counterbalancing.

**Power (statistical)** — the probability of detecting an effect of a given
size if it is real. n=150/cell → ~79% power at true accuracy 0.615 (§10).

**PPP** — pairwise presentation paradigm: two texts, "which is yours?"
Chance = 50%. The primary endpoint (§8.1).

**Pre-registration** — freezing hypotheses + analysis plan in a dated commit
BEFORE running (PREREGISTRATION.md); the antidote to fishing.

**Refusal rate** — fraction of free-text answers where the model declines
("As an AI, I have no memory…"); reported per judge, sidestepped by logprob
scoring (§8.4, §9.7).

**Revision (HF)** — the exact git commit of a model repo; pinned in the
config and reported so "Qwen2.5-7B" is a checkable claim (§5).

**ROUGE-L** — similarity via longest common subsequence of words (F1). Pairs
with ROUGE-L > 0.7 are near-duplicates and dropped (§18).

**Seed** — the number initializing a random generator; fixed seeds make
sampling reproducible (generation seed = 1000 + prompt_idx).

**SELF** — "text generated by the same weights as the judge." Explicitly
NOT a claim about memory or selfhood (§14).

**Stylometry** — authorship identification from surface writing style; here
a TF-IDF character-n-gram logistic regression (§9.2), the deflationary
baseline.

**Temperature / top-p** — sampling controls: temperature rescales the
distribution (0 = greedy), top-p truncates its tail. Generation: 0.8/0.95;
judging: greedy (§7, §8.3).

**TF-IDF** — term frequency × inverse document frequency; weights n-grams by
how distinctive they are. Input features of the stylometric classifier.

**Wilson score interval** — the binomial confidence interval that behaves
well at small n and near the extremes; mandatory on every accuracy (§10).
