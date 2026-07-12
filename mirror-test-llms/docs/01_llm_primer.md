# LLM Primer — every model concept this project uses, from zero

This is not a general LLM course. It is exactly the background needed to
understand what the code in `src/` does and why. Each section ends with
*"where this shows up in the repo."*

---

## 1. Tokens: how a model reads text

A language model never sees words or letters. Text is first chopped into
**tokens** — chunks from a fixed vocabulary (~150,000 entries for Qwen)
learned by a compression algorithm (BPE, byte-pair encoding). Common words
are single tokens; rare words are split:

```
"The economy grew rapidly"  →  ["The", " economy", " grew", " rapidly"]
"Anthropomorphize"          →  ["An", "throp", "omorph", "ize"]
```

Two details matter enormously for us:

* **The leading space is part of the token.** `"A"` and `" A"` are two
  DIFFERENT tokens with different probabilities. When we measure whether the
  model wants to answer "A" or "B", we must account for both spellings —
  which is why `configs/models.yaml` lists `a_variants: ["A", " A"]` and the
  code sums the probability of both (protocol §8.4 warns about exactly this).
* **Different model families use different tokenizers.** That is one reason
  the judges all come from ONE family (Qwen2.5): otherwise "scale" would be
  confounded with "tokenizer".

*Where this shows up:* `utils.candidate_token_ids`, `configs/models.yaml
→ judging`.

## 2. Logits, probabilities, and log-probabilities

At each step the model outputs one number per vocabulary entry — the
**logits**. Softmax turns them into a probability distribution over "what
token comes next":

```
p(token_i) = exp(logit_i) / Σ_j exp(logit_j)
```

A **log-probability** ("logprob") is just log(p). Probabilities multiply
along a sequence; logprobs *add*, and they don't underflow to zero, which is
why all scoring in this repo happens in log space.

**The key trick of the whole study (protocol §8.4):** instead of letting the
judge generate an answer and parsing its text (which fails when the model
rambles or refuses), we take ONE forward pass and read
`logp("A")` vs `logp("B")` for the *next* token directly. The bigger one is
the model's decision; the *difference* is a graded confidence score that
later gives us AUROC. Zero unparseable answers, ever.

To combine the two spellings of an answer we use **log-sum-exp**:
`logp(answer=A) = log( p("A") + p(" A") )` — that's what
`torch.logsumexp` computes without leaving log space.

*Where this shows up:* `utils.first_token_logprobs`, used by
`03_judge_ppp.py` and `04_judge_ipp.py`.

## 3. Perplexity: "how expected is this text to the model?"

For a whole text, average the negative log-probabilities of its tokens
(each conditioned on everything before it) — the **mean NLL**. Perplexity is
`exp(mean NLL)`. Low perplexity = "this is exactly the kind of thing I'd
say"; high = "this text surprises me."

The perplexity-preference baseline (protocol §9.3) uses this as a
no-self-concept-needed decision rule: given two candidate texts, pick the one
with lower perplexity *under the judge model, conditioned on the original
writing task*. If that dumb rule agrees with the judge's actual choices
(high Cohen's κ), the judge's "self-recognition" is plausibly just
likelihood preference.

*Where this shows up:* `utils.continuation_mean_nll`,
`05_baselines.py perplexity`.

## 4. Sampling: temperature, top-p, seeds

Generation = repeatedly sampling the next token from the model's
distribution.

* **Temperature** rescales the logits before softmax. T→0 ("greedy") always
  takes the most likely token — deterministic, flat, unnaturally
  low-perplexity text. T=0.8 (our setting) keeps natural variety.
  **Why generation MUST use T>0 here (protocol §7):** greedy text would be
  trivially low-perplexity, so the perplexity baseline would win for a boring
  mechanical reason and the comparison would be meaningless.
* **Top-p (nucleus) sampling** (ours: 0.95) truncates the distribution to the
  smallest set of tokens covering 95% of the probability, then samples —
  cuts off the crazy tail without flattening everything.
* **Seeds.** Sampling uses a random number generator; fixing its seed makes
  the "random" choices reproducible. Our rule: generation seed
  = `1000 + prompt_idx` (protocol §5), placebo second sample
  = `2000 + prompt_idx`. Anyone re-running gets byte-identical outputs
  (same GPU/library versions assumed — that's why we also pin revisions).
* **Judging is greedy (T=0)** — we want the model's single best answer, and
  the logprob reading is deterministic anyway (protocol §8.3).

*Where this shows up:* `utils.sampled_generate`, `configs/models.yaml
→ generation`.

## 5. Chat templates, and instruct vs base models

Instruction-tuned ("Instruct"/"-it") models are trained on conversations
wrapped in special control tokens. Qwen's format looks like:

```
<|im_start|>system
You are participating in a study...<|im_end|>
<|im_start|>user
The task was: ...<|im_end|>
<|im_start|>assistant
```

You never write these by hand — `tokenizer.apply_chat_template()` produces
the right format for each model family. Two traps the code handles for you:

* **Double-BOS bug:** the templated string already contains the
  begin-of-sequence token, so it must be tokenized with
  `add_special_tokens=False` or the model sees it twice (a classic silent
  quality killer). See `utils._encode`.
* **Gemma has no system role** — sending one raises an error; the code folds
  the system text into the user message when a model demands it.

**Base models** (e.g. `Qwen/Qwen2.5-7B` without "-Instruct") are pure
next-token predictors: they can't follow "Answer with a single letter."
For the base-vs-instruct ablation (protocol §9.5) we instead give them a
plain string ending in "…The response I wrote is Response" and read
logp(" A") vs logp(" B") — pure completion, no instructions needed.

*Where this shows up:* `utils.build_chat_text`,
`configs/models.yaml → judge_templates.completion`, `03_judge_ppp.py`.

## 6. Quantization: how a 14B model fits in a free GPU

Model weights are numbers. Stored at full precision (32-bit floats), a
7B-parameter model needs ~28 GB — more than a free T4's 16 GB. **4-bit NF4
quantization** (via the `bitsandbytes` library) stores each weight in 4 bits
(~5.5 GB for 7B) at a small, well-studied quality cost. Crucially, the model
still runs through the normal `transformers` API, so **log-probabilities
remain available** — the protocol (§5) rules out Ollama/llama.cpp for judging
precisely because they hide logprobs.

VRAM cheat sheet (4-bit): 0.5B→~1 GB, 3B→~2.5 GB, 7B→~5.5 GB,
14B→~9-10 GB (tight on one T4 — use Kaggle's 2×T4 or cap the family at 7B;
protocol §18 says a 0.5→7B curve is still a valid scale axis).

*Where this shows up:* `utils.load_model_and_tokenizer`,
`requirements.txt` notes.

## 7. Hugging Face: models, datasets, gates, revisions

* **The Hub** hosts the models (`Qwen/Qwen2.5-7B-Instruct`) and datasets
  (`abisee/cnn_dailymail`). First use downloads to `~/.cache/huggingface`.
* **Gated models:** Meta (Llama) and Google (Gemma) require you to accept a
  license on the model page while logged in, then authenticate with an
  access token. Do this once — see `04_colab_kaggle_guide.md` §2.
* **Revisions:** a model repo is a git repo; "the model" can silently change.
  Pinning the exact commit hash in `configs/models.yaml` (the `revision:`
  fields) and reporting it in the appendix is what makes "we used Qwen2.5-7B"
  a checkable claim (protocol §5).

## 8. Judge / Foil / SELF / placebo — the experiment's vocabulary

* **Generator:** any model producing texts for prompts (phase 1).
* **Judge:** the model asked "which of these did you write?" — it only ever
  judges pairs containing its OWN generation.
* **Foil:** the other author in the pair (another LLM, or a human).
* **SELF:** the judge's own text. Note the honest caveat (protocol §14): the
  judge has no memory of writing it; stateless models can only recognize
  *self-typical* text. "Self" = "generated by the same weights," nothing more.
* **Placebo pair:** SELF vs SELF (two different-seed samples). There is no
  right answer, so the judge's choice rate for "position A" measures pure
  position bias — and any pipeline bug (mislabeled sides, template leaks)
  shows up as a deviation from 50%. It is the experiment's smoke detector.

## 9. Why cleaning is scientific, not cosmetic

Instruction-tuned models decorate answers: "Sure! Here is a summary…",
"**Answer:**", stray `<|im_end|>` tokens. If SELF texts carry the judge's
signature decoration, a "correct" answer proves only that the judge spotted
its own boilerplate — protocol §18 calls this "boilerplate leaks authorship."
So `utils.clean_text` strips those artifacts from EVERY author identically —
including the human reference texts, because an asymmetric cleaner would
itself become the authorship signal. The stylometric baseline (§9.2) is the
alarm: if the classifier hits ~99%, cleaning failed.

## 10. Further reading (free)

* Hugging Face **LLM course**: https://huggingface.co/learn/llm-course
* Karpathy, "Let's build GPT" (YouTube) — the best mechanical intuition for
  logits/sampling that exists.
* Jurafsky & Martin, *Speech and Language Processing* ch. 10 (free draft
  online) — n-grams to transformers.
* The `transformers` docs on generation strategies and chat templating.
