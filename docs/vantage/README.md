# docs/vantage/

VANTAGE-specific documentation. All other subdirectories under `docs/` are inherited from the upstream VLMEvalKit fork and pertain to that project, not VANTAGE.

---

## Contents

| File | What it covers |
|------|---------------|
| [`SUBMISSION.md`](SUBMISSION.md) | Full submission guide: portal URL, limits (2/day · 30 lifetime), pillar structure, `.tar.gz` packaging, form fields, JSONL record format, troubleshooting. See also the root-level [`../../SUBMISSION.md`](../../SUBMISSION.md) for a quick reference. |
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) | File-to-file map, all CLI flags, all env vars, three model-registration paths, and exact run commands for every task. |
| [`VANTAGEEvalInputs.md`](VANTAGEEvalInputs.md) | Minimum prediction-file columns required by each evaluator. Read this if you are running `--mode eval` against a pre-existing prediction file or integrating with a custom inference pipeline. |

---

## Other docs locations

| Location | Content |
|----------|---------|
| [`../../README.md`](../../README.md) | Main entry point: installation, dataset setup, run commands, dataset keys, submission workflow, full doc index. |
| [`../../SUBMISSION.md`](../../SUBMISSION.md) | Quick submission reference: 3-step flow, pillar table, packaging command, form overview. |
| [`../../README_VANTAGE.md`](../../README_VANTAGE.md) | Extended reference: environment variables, config-file format, per-model parameter passing, all dataset keys. |
| [`../../scripts/RUN_LMUData.md`](../../scripts/RUN_LMUData.md) | Data download guide: HuggingFace prep script, prerequisites, troubleshooting. |
| [`../../prompt_guide.md`](../../prompt_guide.md) | Exact prompt templates used for each benchmark task. |
