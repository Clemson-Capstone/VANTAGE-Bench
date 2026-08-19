---
name: vantage-validate
description: Use when submission JSONL files exist and need validation for blockers before packaging or submitting to the VANTAGE-Bench portal.
---

Run the validator, interpret its report, and hold the line on blockers
before anything gets packaged.

## Why this skill exists

`emit_submission()` (in `vlmeval/dataset/utils/vantagebench/emit.py`) never
raises on failure — a bad write becomes a `warnings.warn`, so `run.py` can
exit 0 and the prediction `.xlsx` can look fine while the submission JSONL is
missing, truncated, or malformed. This skill is the check that catches that
class of failure before it burns a submission slot.

## Command

```bash
python scripts/validate_submission.py --work-dir <path> --lmu-root <path> --json
```

(or `--archive <path>.tar.gz` to validate an already-packaged archive).
Checks: JSONL parses; required keys (`id`, `task`, `conversations`,
`metadata`) present; id-format shape per task (see
[`tasks-and-pillars.md`](../reference/tasks-and-pillars.md) for the
generators — the validator applies those same rules, not a reimplemented
regex); no duplicate ids; record count vs. source TSV; empty/failure-string
prediction rate; degenerate-answer-distribution check; pillar completeness.

Exit code 0 means zero blockers — but still read the JSON report, because
warnings can exit 0 too.

## Decision logic

| Report says | Action |
|---|---|
| Any blocker (malformed JSONL line, wrong id shape, duplicate id, missing task in an intended pillar) | **STOP.** Do not proceed to `../vantage-package-submit/SKILL.md`. Fix the underlying run (usually via `../vantage-run/SKILL.md` with `--reuse`) and re-validate. |
| Warning only (empty-prediction rate above baseline, degenerate/near-constant answer distribution, record count mismatch with no TSV reference available) | Surface the exact numbers to the user with a recommendation (e.g. "12% empty predictions on `event_verification` — check for API failures before submitting"). Do not silently pass it through. |
| Clean report, zero blockers, zero warnings | Proceed to `../vantage-package-submit/SKILL.md`. |

## Failure handling

| Symptom | Fix |
|---|---|
| `--work-dir` has no `*_submission.jsonl` at all | run finished infer-only; see the `--mode infer` limitation in `../vantage-run/SKILL.md` |
| id-shape check fails for one task | that task's run likely predates a harness fix; re-run inference/eval for just that task |
| `--lmu-root` mismatch errors (can't resolve source TSV for the count check) | pass the same root used for the run (`$LMUData` or `--lmudata-root`) |

## Next

Only after a clean (or explicitly accepted-with-warnings) report, hand off
to `../vantage-package-submit/SKILL.md`.
