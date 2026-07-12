# Reading Guide — the 10 papers, and exactly what to extract from each

Budget (protocol §3): weeks 1–2, one or two papers a day. For each paper
write a **5-sentence summary** the same day you read it — those summaries
become your Related Work section almost verbatim. A template is at the
bottom; keep summaries in `docs/paper_notes/` (create it) or a notebook.

⚠ Verify every arXiv ID/venue yourself before citing (the protocol was
written mid-2026 and papers move). How: search the title on arxiv.org and
Google Scholar; prefer the published venue over the preprint when one exists.

**How to read a paper as a 2nd-year student:** three passes. (1) title,
abstract, figures, conclusion — 10 minutes, decide what it claims. (2)
intro + method skim — 30 minutes, understand HOW. (3) only for the anchor
paper: full read with notes. You do not need to understand every equation;
you need to know what they measured, on what, and what they concluded.

---

## Core — must read and cite

### 1. Gallup (1970), "Chimpanzees: Self-Recognition," *Science* 167.
The original mirror test: chimps with a dye mark on the forehead touch the
mark when given a mirror; monkeys don't.
**Extract for your paper:** the framing sentence for your intro, PLUS the
caveat that even in animals the test is *contested* as a measure of
self-awareness — you inherit a debated instrument, and saying so is
protective. **Feeds:** Introduction ¶1.

### 2. Panickssery, Bowman & Feng (2024), "LLM Evaluators Recognize and Favor Their Own Generations," NeurIPS 2024. arXiv:2404.13076. ⭐ THE ANCHOR
GPT-4/Llama-2 distinguish their own summaries (XSum, CNN/DM) from others'
out of the box; recognition ability correlates with self-*preference* bias
when fine-tuned.
**Extract:** (a) the pairwise + individual protocol designs — your §8 is a
small-model controlled replication of exactly these; (b) their prosaic use
of "self" (no representation claims) — copy that stance verbatim in spirit;
(c) the safety angle (evaluator bias) for your discussion.
**Feeds:** Method (protocols), Related Work ¶1, Discussion.

### 3. Davidson et al. (2024), "Self-Recognition in Language Models." arXiv:2407.06946.
Tests self-recognition with model-generated "security questions"; frames it
as a safety question between interacting agents.
**Extract:** motivation sentence ("mirror risks"), and the existence of an
alternative protocol (yours differs — say why: theirs needs model-authored
questions; yours uses fixed tasks + human class). **Feeds:** Related Work ¶1.

### 4. Laine et al. (2024), "Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs." arXiv:2407.04694.
Broad situational-awareness benchmark including text-continuation
self-recognition; results move with prompt phrasing.
**Extract:** the prompt-sensitivity finding — it is the ENTIRE justification
for your 3-phrasing robustness check (§8.5). One sentence + citation.
**Feeds:** Related Work ¶1, Method (phrasing check).

### 5. Zhou et al. (2025), "From Implicit to Explicit: Enhancing Self-Recognition in LLMs." arXiv:2508.14408.
Recent follow-up; its related-work section maps the subfield — mine its
bibliography. Reports (citing Ackerman et al., 2025) that instruction-tuned
Llama-3-8B succeeds where its base model fails.
**Extract:** (a) the base-vs-instruct claim — your §9.5 ablation confirms or
contradicts it on another family (a real contribution either way);
(b) chase the Ackerman reference through their bibliography and verify it
exists before citing it. **Feeds:** Related Work ¶1, base-vs-instruct
motivation.

## Method-adjacent — cite briefly (one or two sentences each)

### 6. Mitchell et al. (2023), "DetectGPT." arXiv:2301.11305.
Machine-text detection via probability curvature.
**Extract:** your perplexity rule is a cousin of likelihood-based detection;
citing it shows you know that literature. **Feeds:** Related Work ¶3.

### 7. Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685.
LLM judges have position bias and verbosity bias.
**Extract:** the two biases — they justify (a) counterbalancing every pair
(§8.3) and (b) the ≤25% length-match filter (§7). **Feeds:** Related Work
¶2, Method.

### 8. Schaeffer, Miranda & Koyejo (2023), "Are Emergent Abilities of LLMs a Mirage?" arXiv:2304.15004.
Apparent "emergence" often disappears under continuous metrics.
**Extract:** the argument, so you can write: "we therefore report AUROC
alongside accuracy" (§9.6). If your accuracy jumps but AUROC is smooth, this
paper is your interpretation. **Feeds:** Related Work ¶3, Results.

## Framing — one sentence each, no more

### 9. Butlin et al. (2023), "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." arXiv:2308.08708.
### 10. Chalmers (2023), "Could a Large Language Model Be Conscious?" arXiv:2303.07103.
**Extract:** ONE hedged sentence in the intro gesturing at why people care,
e.g. "whether such behaviours bear on machine self-awareness is
philosophically contested [9,10]; we make no such claim." The protocol's
style rule: "conscious(ness)" appears at most twice in your whole paper.
**Feeds:** Introduction ¶1, Discussion (one sentence).

### 11. (optional) MirrorBench (2026, arXiv:2604.14785) — multimodal cousin;
cite in related work only to show the area is active. Verify it exists.

---

## The gap paragraph (assemble after reading; goes in your intro)

> Prior self-recognition work concentrates on large proprietary or 7B+
> models, single scales, and English; none isolates what *signal* the model
> uses. We contribute (i) the first scale curve within one open family,
> (ii) paraphrase and stylometric controls testing whether the signal is
> surface style, and (iii) [optional] the first non-English replication.

## 5-sentence summary template (fill one per paper, same day)

```
PAPER: <authors, year, title, venue, arXiv id — VERIFIED yes/no>
1. Question: what did they ask?
2. Method: on what models/data, with what protocol?
3. Result: the one number or finding that matters.
4. Limitation the authors admit (or you notice).
5. Relevance: which sentence of MY paper does this feed?
```

## Google Scholar alert (do it now)

Create an alert for "self-recognition language models" — protocol §18. If a
new paper does your exact study mid-project, you pivot weight to the
Urdu/mechanism angle instead of being scooped silently.
