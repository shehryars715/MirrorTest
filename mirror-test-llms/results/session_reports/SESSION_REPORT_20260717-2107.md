# SESSION REPORT — 2026-07-17 13:44

**IN PROGRESS - 8/32 tasks done, ~62.2 h of GPU work remaining (run this notebook again, attaching this version's output as Input).**

- notebook build: `2026-07-16-a27f93cc` · budget: 8.0 h
- GPU: True · HF token: True · statistics skipped (not enough judgments yet - expected in early sessions)

## Task board
| task | status | note |
|---|---|---|
| prompts | done | already complete |
| gen:qwen2.5-0.5b-instruct | done | already complete |
| gen:qwen2.5-1.5b-instruct | done | already complete |
| gen:qwen2.5-3b-instruct | done | already complete |
| gen:qwen2.5-7b-instruct | done | already complete |
| gen:llama-3.2-3b-instruct | ran | 42 min |
| gen:gemma-2-9b-it | ran | 223 min |
| gen:mistral-7b-instruct-v0.3 | ran | 167 min |
| gen:qwen2.5-14b-instruct | paused | 49 min |
| pairs | blocked | waiting on: gen:qwen2.5-14b-instruct |
| paraphrase | blocked | waiting on: pairs |
| ppp:qwen2.5-0.5b-instruct | blocked | waiting on: pairs |
| ppp:qwen2.5-1.5b-instruct | blocked | waiting on: pairs |
| ppp:qwen2.5-3b-instruct | blocked | waiting on: pairs |
| ppp:qwen2.5-7b-instruct | blocked | waiting on: pairs |
| ppp:qwen2.5-14b-instruct | blocked | waiting on: pairs |
| main-cell | blocked | waiting on: paraphrase, ppp:qwen2.5-14b-instruct |
| ipp:qwen2.5-0.5b-instruct | blocked | waiting on: pairs |
| ipp:qwen2.5-1.5b-instruct | blocked | waiting on: pairs |
| ipp:qwen2.5-3b-instruct | blocked | waiting on: pairs |
| ipp:qwen2.5-7b-instruct | blocked | waiting on: pairs |
| ipp:qwen2.5-14b-instruct | blocked | waiting on: pairs |
| ppp:qwen2.5-7b-base | blocked | waiting on: pairs |
| ppp:qwen2.5-14b-base | blocked | waiting on: pairs |
| ppl:qwen2.5-0.5b-instruct | blocked | waiting on: pairs |
| ppl:qwen2.5-1.5b-instruct | blocked | waiting on: pairs |
| ppl:qwen2.5-3b-instruct | blocked | waiting on: pairs |
| ppl:qwen2.5-7b-instruct | blocked | waiting on: pairs |
| ppl:qwen2.5-14b-instruct | blocked | waiting on: paraphrase |
| ppl:qwen2.5-7b-base | blocked | waiting on: pairs |
| ppl:qwen2.5-14b-base | blocked | waiting on: pairs |
| stylometric | blocked | waiting on: pairs |

`done` = complete before this session · `ran` = completed this session ·
`paused/partial` = mid-way, auto-resumes · `blocked` = waiting on earlier
tasks · `skipped` = needs GPU or HF_TOKEN (fix and re-run) ·
`deferred` = out of time this session.

## Failures needing attention
none

## Data inventory
| location | count |
|---|---|
| data/prompts | 5 files |
| data/generations | 22 files |
| data/pairs | 0 files |
| results/judgments | 0 files |
| results/baselines | 0 files |
| results/tables | 0 files |

## What to do next
1. Re-run: open the notebook, + Add Input -> this version's Output, Save Version -> Save & Run All.
2. No settings changes needed.
3. Download **mirror_bundle.zip** + **SESSION_REPORT.md** from this
   version's Output tab into `Desktop/Mirror/` and tell Claude.
