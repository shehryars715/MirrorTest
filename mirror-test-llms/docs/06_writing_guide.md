# Writing the Paper — from blank page to submission (weeks 10–12)

The paper is a 4-page ACL-format short paper + unlimited references +
appendix. A complete LaTeX skeleton is in `paper/main.tex` with the
protocol's draft abstract and all three conclusion variants already in
place. This guide covers the process around it.

---

## 1. LaTeX + Overleaf in 15 minutes (if you've never used them)

LaTeX is a typesetting language: you write plain text with commands
(`\section{...}`, `\cite{...}`) and it produces a PDF. Overleaf is a free
web editor for it — no installation.

1. Create an account at overleaf.com.
2. Get the **official ACL style files**: search "acl-style-files" on GitHub
   (repository `acl-org/acl-style-files`). Two ways in:
   * Easiest: in Overleaf, New Project → the ACL template is available in
     the template gallery ("Association for Computational Linguistics
     Rolling Review Submission" or similar official ACL template), OR
   * Download the style files ZIP and upload to a blank Overleaf project.
3. Replace the template's `main.tex` content with `paper/main.tex` from this
   repo, and upload `paper/references.bib`.
4. Compile (Ctrl+S / Recompile). Fix errors from the top down — the first
   error usually causes the rest.

Core LaTeX you need (90% of everything):
```latex
\section{Introduction}            % heading
\textbf{bold} \emph{italic}       % emphasis
\cite{panickssery2024llm}         % [1]-style citation from references.bib
~\ref{tab:main} ~\ref{fig:scale}  % refer to tables/figures by label
$d' = z(H) - z(F)$                % inline math
% anything after a percent sign is a comment
```

## 2. Write in this order (not front to back)

1. **Method** (week 10) — you know it cold; the config file and
   `pairs_report.json` contain every number. Densest section.
2. **Results** (week 10) — paste `results/tables/table1_main.tex`, include
   `results/figures/fig1_scale_curve.pdf` and `fig2_paraphrase.pdf`. One
   short paragraph per finding; numbers do the talking.
3. **Introduction** (week 11) — now that you know what you found, you can
   promise exactly that. Hook → gap → RQ1/RQ2 → contributions bullets.
4. **Related Work** (week 11) — stitch your 5-sentence summaries into the
   three paragraphs (self-recognition; judge biases; detection + emergence
   caveats).
5. **Discussion + Limitations** (week 11) — copy protocol §14's list
   honestly; pick the §13 conclusion matching your outcome row.
6. **Abstract** (last) — fill the bracketed template with real numbers.

## 3. Page budget (4 pages, hard limit at *CL venues)

| Section | Space | Content |
|---|---|---|
| 1 Introduction | 0.75 pp | mirror-test hook (Gallup + one hedged consciousness cite), the 2 RQs, findings preview, 3 contribution bullets |
| 2 Related Work | 0.5 pp | 3 short paragraphs |
| 3 Method | 1 pp | models table, data, pairing + length matching, PPP/IPP (templates → appendix), counterbalancing, logprob scoring, 3 baselines |
| 4 Results | 1.25 pp | Fig 1 (scale curve, acc+AUROC), Table 1 (main + baselines), Fig 2/Table 2 (paraphrase, McNemar p) |
| 5 Discussion | 0.3 pp | what mechanism results imply; 1 consciousness sentence; 1 safety/evaluator-bias sentence |
| 6 Limitations | 0.2 pp | mandatory; from §14 |
| Appendix | free | prompts verbatim, refusal rates, placebo, per-domain tables, phrasing spread, seeds & revisions, power analysis |

## 4. Style rules that get student papers accepted (protocol §12)

* Every claim has a number or a citation. No adjectives like
  "impressive/remarkable".
* Present tense for results ("accuracy rises", not "rose").
* "conscious(ness)" ≤ 2 occurrences, both hedged.
* Figures readable in grayscale (already ensured by `06_stats.py`:
  distinct markers/linestyles/hatches, not color alone).
* The sentence **"We release all code and data."** appears in the abstract
  or introduction.
* Numbers: report as 0.63 [0.55, 0.71], n = 172 — CI everywhere.

## 5. Titles (pick or blend — protocol §12)

1. "Mirror, Mirror: Self-Recognition in Small Open Language Models Is
   [Mostly Stylometry / Scale-Dependent]"
2. "Do Small Language Models Recognize Their Own Reflection? A Controlled
   Study of Scale and Mechanism"
3. "What Does the LLM Mirror Test Measure? Paraphrase and Stylometric
   Controls for Self-Recognition"

## 6. Venue strategy (protocol §17, condensed)

| Target | Fit | Action |
|---|---|---|
| **ACL/EMNLP/NAACL Student Research Workshop (SRW)** | ★★★★★ best first venue | check the next CFP THE WEEK YOU START and plan backwards from its deadline; SRWs often assign a mentor |
| Workshops at NeurIPS/ICLR/ACL (evaluation / model behaviour) | ★★★★★ | workshop lists appear ~4–6 months before each conference |
| Findings of ACL/EMNLP | ★★★☆☆ stretch | only with tight execution + a novel angle (Urdu or decisive mechanism result) |
| Main tracks | ★☆☆☆☆ | wrong scope — the core phenomenon already has a NeurIPS paper |
| Regional IEEE (Pakistan) | ★★★☆☆ | safety net |
| arXiv | always | post simultaneously with submission (CHECK the venue's preprint/anonymity policy first) |

Sequencing: arXiv + SRW first; strong reviews → extend (Urdu + LoRA) into a
Findings submission next cycle. Present at NUST's undergraduate research
office / FYP showcase for free feedback before submitting.

## 7. Anonymization (double-blind venues)

* No names/affiliations in the PDF (the ACL style's `review` option handles
  the header).
* The repo link must be anonymized: use https://anonymous.4open.science to
  mirror your GitHub repo, and cite THAT link in the submission. Swap in the
  real link only in the camera-ready.
* Don't push commits with your name in the paper PDF during review.

## 8. Pre-submission checklist (protocol §19 — verbatim, tick every box)

- [ ] PREREGISTRATION.md committed before main runs (timestamped)
- [ ] Every accuracy has n + 95% Wilson CI; multiple comparisons corrected (Holm)
- [ ] Placebo ≈ 50% reported; refusal rates reported
- [ ] Both A/B orders run; position-consistency reported
- [ ] Stylometric + perplexity baselines in the main table
- [ ] Paraphrase attack with McNemar p-value
- [ ] AUROC alongside accuracy for the scale curve
- [ ] Prompts, seeds, model revisions in appendix; repo public; README
      reproduces Figure 1 in one command
- [ ] "conscious(ness)" ≤ 2 times, both hedged
- [ ] Limitations section present
- [ ] A supervisor/faculty member has read the full draft
- [ ] Venue anonymity respected (anonymized repo link if double-blind)
- [ ] Every reference verified on arXiv/Scholar (week 11 task)
