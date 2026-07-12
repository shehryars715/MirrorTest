# Optional Extensions — Urdu (§9.8) and LoRA fine-tuning (§9.9)

**Rule (protocol §16):** attempt these ONLY after the core (§9.1–§9.7) is
complete. If a week slips, these are what you cut. Either one, done well,
upgrades the paper from "solid" to "novel angle" — the Urdu extension in
particular is a data point nobody has published.

---

## A. Urdu extension (your differentiator)

**Question:** does self-recognition transfer to a lower-resource language
the models saw far less of in training? "Present in English, absent in Urdu
for the same model" (or the reverse) is novel either way.

**Design (protocol §9.8):** 100 Urdu prompts, PPP only, one foil, judges
with credible Urdu ability — e.g. `CohereForAI/aya-expanse-8b` plus your
largest Qwen (verify current best open Urdu-capable models when you start;
c4ai-aya models are gated → accept license).

### A.1 Build the prompt file

The pipeline consumes any `data/prompts/{domain}.jsonl` with the standard
schema, so the Urdu extension is just a new domain file named `urdu.jsonl`.
Write 100 prompts yourself (everyday QA + short creative tasks) or translate
Dolly items, then have **two fellow students sanity-check fluency** —
record their initials in the paper's acknowledgments.

Create `data/prompts/urdu.jsonl` records like:

```json
{"prompt_id": "urdu_0000", "prompt_idx": 0, "domain": "urdu",
 "task_prompt": "<Urdu question or creative task>",
 "human_reference": "<a fluent Urdu answer you or a helper wrote, 40-160 words>",
 "source_dataset": "handwritten", "source_id": "you",
 "n_words_prompt": 12, "n_words_reference": 60}
```

A helper to validate the file (schema + duplicates + non-empty):

```python
# scripts kept inline here to keep the core repo minimal — paste into a cell
import json, sys
seen = set()
for i, line in enumerate(open("data/prompts/urdu.jsonl", encoding="utf-8")):
    r = json.loads(line)
    assert set(r) >= {"prompt_id","prompt_idx","domain","task_prompt",
                      "human_reference"}, f"line {i}: missing fields"
    assert r["prompt_id"] not in seen, f"duplicate {r['prompt_id']}"
    assert r["task_prompt"].strip() and r["human_reference"].strip()
    seen.add(r["prompt_id"])
print(f"OK: {len(seen)} prompts")
```

**Caveats to handle honestly:**
* The English-ascii filter and word-counting in `utils.n_words` are
  Latin-script-oriented; for Urdu, `n_words` still works on
  whitespace-ish tokens but treat length filters as approximate — or relax
  the pair length filter to 35% and say so.
* The boilerplate cleaner's patterns are English; Urdu models may have
  their own openers ("یقیناً!") — inspect 20 generations and extend
  patterns if needed (log it in the deviations table).

### A.2 Run the same pipeline on the new domain

```bash
# add the two Urdu-capable judges + keep one foil in configs/models.yaml
# (e.g. add aya-expanse-8b under judges with its params_b, or run it
#  standalone), then:
python src/01_generate.py --models aya-expanse-8b qwen2.5-14b-instruct \
       llama-3.2-3b-instruct --domains urdu        # NOTE: add 'urdu' to
                                                   # DOMAINS in src/utils.py
python src/02_build_pairs.py --judges aya-expanse-8b qwen2.5-14b-instruct \
       --foils llama-3.2-3b-instruct --domains urdu
python src/03_judge_ppp.py --judge qwen2.5-14b-instruct --foils llama-3.2-3b-instruct
python src/06_stats.py
```

(One code touch: `DOMAINS` in `src/utils.py` and the domain lists in
`02/06` iterate `["news","dolly","wp"]` — add `"urdu"` there. Grep for
`"wp"` to find the three spots.)

**Judge instructions in which language?** Keep the judge prompt in English
(the models' strongest instruction language) with Urdu candidate texts —
and say so in the method. Running a fully-Urdu instruction variant as a
phrasing check is a nice extra robustness line.

**Power honesty:** N = 100 pairs detects only large effects (~0.64 vs
chance at 80% power — compute it: `power_exact_binomial(100, 0.64)`).
Frame the Urdu result as an exploratory data point, not a confirmatory test.

---

## B. LoRA fine-tuning extension (connects to SFT skills)

**Question (protocol §9.9):** is self-recognition *trainable*, and does the
trained ability generalize out-of-domain? This mirrors Panickssery et al.'s
fine-tuning result at hobby scale.

**Design:** fine-tune the 3B judge on 500 labeled PPP items (rank-16 LoRA,
3 epochs, ~30 min on Colab with Unsloth), test on (a) held-out pairs from
the same domains and (b) one entirely held-out domain.

**What LoRA is (one paragraph):** instead of updating all N billion weights,
LoRA freezes the model and learns tiny low-rank correction matrices on the
attention/MLP projections — million-scale trainable parameters, so it fits
in free-tier VRAM and minutes, not days. `unsloth` is a library that makes
this fast on T4s.

### B.1 Build the training data (from your existing pairs)

```python
import json, random, sys
sys.path.insert(0, "src")
from utils import PAIRS_DIR, read_jsonl, fill_template, load_config

cfg = load_config(); ph = cfg["ppp_phrasings"][0]
judge = "qwen2.5-3b-instruct"
rng = random.Random(42)
rows = []
for f in ["llama-3.2-3b-instruct", "gemma-2-9b-it", "human"]:
    for d in ["news", "dolly"]:                      # hold out 'wp' entirely!
        for p in read_jsonl(PAIRS_DIR / f"ppp__{judge}__{f}__{d}.jsonl"):
            order = rng.choice(["self_A", "self_B"])
            a = p["text_self"] if order == "self_A" else p["text_foil"]
            b = p["text_foil"] if order == "self_A" else p["text_self"]
            rows.append({
                "messages": [
                    {"role": "system", "content": ph["system"]},
                    {"role": "user", "content": fill_template(
                        ph["user"], {"TASK_PROMPT": p["task_prompt"],
                                     "TEXT_A": a, "TEXT_B": b})},
                    {"role": "assistant",
                     "content": "A" if order == "self_A" else "B"},
                ]})
rng.shuffle(rows)
train, test = rows[:500], rows[500:700]
for name, split in [("train", train), ("test", test)]:
    with open(f"data/pairs/lora_{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in split:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(train), len(test))
```

### B.2 Train with Unsloth on Colab (sketch — verify against current unsloth docs, APIs drift)

```python
# !pip install unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

ds = load_dataset("json", data_files="data/pairs/lora_train.jsonl", split="train")
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(
    ex["messages"], tokenize=False)})

SFTTrainer(model=model, train_dataset=ds,
           args=SFTConfig(per_device_train_batch_size=2,
                          gradient_accumulation_steps=4, num_train_epochs=3,
                          learning_rate=2e-4, seed=42,
                          output_dir="results/lora_3b")).train()
model.save_pretrained("results/lora_3b/adapter")
```

### B.3 Evaluate: same measurement, tuned model

Load the base 3B + adapter, then score held-out pairs with the SAME
first-token-logprob machinery (import `first_token_logprobs` from
`src/utils.py`) on: (a) held-out news/dolly pairs, (b) the untouched `wp`
domain. Report both accuracies with Wilson CIs next to the untuned 3B row.

**The finding you're after:** tuned in-domain ↑↑ but out-of-domain ≈
untuned ⇒ the model learned dataset-specific style cues, not a general
self-concept. Tuned ↑ everywhere ⇒ a transferable self-typicality signal
exists and is amplifiable. Either is a real result.

**Report honestly:** LoRA results are exploratory (one seed, one rank, no
sweep); put them in the appendix unless they are dramatic.
