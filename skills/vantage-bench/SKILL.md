---
name: vantage-bench
description: Use when the user wants to run VANTAGE-Bench but hasn't said what stage they're at; detects progress on disk and routes to the right sub-skill.
---

Orchestrator for the VANTAGE-Bench submission pipeline: environment check → data prep →
model setup → inference/eval → validate → package → portal submit. It holds no state file;
every routing decision comes from what already exists on disk.

## Read this before doing anything

- Pillars and tasks: `../reference/tasks-and-pillars.md` — submit **all tasks in a pillar
  together** or the portal rejects it.
- Submission budget: **2 submissions/day, 30/lifetime, per email** (`../../SUBMISSION.md`).
  A wasted submission cannot be undone — never invoke `vantage-portal-submit` speculatively;
  only after `vantage-validate` passes clean.

## Stage detection (check in order, act on the first incomplete stage)

| # | Check | Filesystem signal | If incomplete → |
|---|---|---|---|
| 1 | Environment | No signal on disk; cheap to re-run | `../vantage-preflight/SKILL.md` |
| 2 | Data prep | `$LMUData/datasets/<Task>/` missing for a needed task (dir names: `VANTAGE_VQA`, `VANTAGE_EventVerification`, `VANTAGE_DVC`, `VANTAGE_Temporal`, `VANTAGE_2DPointing`, `Astro2D`, `VANTAGE_2DGrounding`, `VANTAGE_SOT`) | `../vantage-data-prep/SKILL.md` |
| 3 | Model target | No model name/config chosen yet | `../vantage-model-config/SKILL.md` (existing model) or `../vantage-add-model/SKILL.md` (new wrapper) |
| 4 | Inference | No `outputs/<model>/<eval_id>/*.xlsx` for a needed dataset key | `../vantage-run/SKILL.md` (single machine/GPU) or `../vantage-cluster-launch/SKILL.md` (multi-node/cluster) |
| 5 | Evaluation | `.xlsx` exists but matching `*_submission.jsonl` doesn't (e.g. a prior `--mode infer`-only run) | `../vantage-run/SKILL.md` with `--mode eval --reuse` |
| 6 | Validation | Submission JSONLs exist but `validate_submission.py` hasn't been run clean | `../vantage-validate/SKILL.md` |
| 7 | Packaging | Validation passed but no `.tar.gz` yet | `../vantage-package-submit/SKILL.md` |
| 8 | Portal | Archive exists and validated | `../vantage-portal-submit/SKILL.md` (fills the form; a human clicks submit) |

If a command in any stage errors out or output looks wrong, route to
`../vantage-troubleshoot/SKILL.md` instead of guessing.

## How to run the checks

For step 2, check one directory per task the user actually wants (map pillar → task via the
reference doc — don't prepare tasks nobody asked for). For steps 4-5, check per dataset key
under the model's `outputs/<model>/` tree; a single run can be mid-pipeline on some tasks and
done on others — route to whichever stage the *earliest* incomplete task needs.

Do not skip stage 6. `emit_submission()` never raises on failure (see reference doc) — a
prediction file can look complete while its submission JSONL is empty or malformed, and
`run.py` exiting 0 proves nothing about JSONL validity.

## All sub-skills

- `../vantage-preflight/SKILL.md` — environment report, dependency install, backend recommendation.
- `../vantage-data-prep/SKILL.md` — download/stage LMUData via `run_lmudata.py`.
- `../vantage-model-config/SKILL.md` — point a run at a registered model or author a config JSON.
- `../vantage-add-model/SKILL.md` — write and register a new model wrapper class.
- `../vantage-run/SKILL.md` — execute `run.py` for chosen tasks/model, infer/eval modes.
- `../vantage-cluster-launch/SKILL.md` — multi-GPU/multi-node launch.
- `../vantage-validate/SKILL.md` — wraps `scripts/validate_submission.py`.
- `../vantage-package-submit/SKILL.md` — wraps `scripts/package_submission.py` and
  `scripts/run_manifest.py`.
- `../vantage-portal-submit/SKILL.md` — fills the upload form; never clicks submit.
- `../vantage-troubleshoot/SKILL.md` — failure triage across all stages.

## Gates

This skill only routes — it runs no mutating commands itself. Each sub-skill states its own
gates (installs, large downloads, destructive flags, portal submit) at the point they apply;
honor those, don't pre-empt them here.
