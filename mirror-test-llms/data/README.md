# data/ — what lives here

| Directory | Contents | Produced by | Committed to git? |
|---|---|---|---|
| `prompts/` | the FROZEN prompt sets (`news/dolly/wp.jsonl`), `CHECKSUMS.txt`, rejection report | `00_build_prompts.py` | **YES — and never edited again** |
| `generations/` | raw model outputs, `{model}__{domain}.jsonl` | `01_generate.py` | yes (small; regenerable from seeds) |
| `pairs/` | cleaned evaluation sets: `ppp__*`, `placebo__*`, `ipp__*`, `para__*`, `pairs_report.json` | `02_build_pairs.py`, `02b_paraphrase.py` | yes |

Full field-by-field schemas: [`../docs/03_pipeline_walkthrough.md`](../docs/03_pipeline_walkthrough.md) → "Data schemas".

**The freeze rule:** once `prompts/` is committed, its files never change
(`00_build_prompts.py` refuses to overwrite without `--force`). If you must
rebuild before pre-registration, delete the files, rebuild, re-commit — and
never after pre-registration without a deviations-log entry.
