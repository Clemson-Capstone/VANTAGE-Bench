---
name: vantage-package-submit
description: Use when validated submission JSONLs need bundling into submission.tar.gz and a draft form_metadata.md for the portal.
---

Wrap `scripts/package_submission.py` (read it, don't modify it) with a
pre-check that stops on a missing pillar task, and a post-step that drafts
the portal form.

## Precondition

Only run this after `../vantage-validate/SKILL.md` reports zero blockers. This
skill does not re-validate JSONL contents.

## 1. Pre-check: pillar completeness against intent

Before packaging, cross-check which `*_submission.jsonl` files are present in
`<work-dir>` against the pillars the user actually intends to submit, using
the pillar → task map in
[`tasks-and-pillars.md`](../reference/tasks-and-pillars.md). If any task in
an intended pillar is missing its submission file:

**STOP — do not package.** A missing task in a submitted pillar is rejected
server-side, and that burns one of the 2/day · 30/lifetime submission slots.
This is a hard stop, not a warning: send the user back to
`../vantage-run/SKILL.md` (or `--mode eval --reuse` if only evaluation is
missing) for the missing task before packaging anything.

Note: `package_submission.py` itself only warns on a missing task
(`INCOMPLETE — missing: ...` in its pillar-coverage printout) and still
builds the archive — this skill's job is to make that condition fatal when
the user meant to submit that pillar.

## 2. Package

```bash
python scripts/package_submission.py --work-dir <path> --out submission.tar.gz
```

Read its printed pillar-coverage report even after the pre-check — it also
flags duplicate submission files per task (e.g. both an 8frame and a 16frame
run present) and keeps only the most-recent by sort order, which may not be
the one the user intended. Confirm with the user which run should win if
duplicates are reported.

## 3. Gate — confirm before overwriting an existing archive

**STOP and confirm before overwriting an existing `.tar.gz`** at the target
`--out` path. Name exactly what would be lost (the existing archive's path,
size, and mtime) before proceeding.

## 4. Post-package checks

- Confirm the archive is **under 500 MB** (the script prints size; the
  portal's hard limit is 500 MB). If it's large, check for stray non-JSONL
  files rather than assuming it's the JSONL content.
- Draft the portal form:

```bash
python scripts/run_manifest.py --work-dir <path>
```

This writes `run_manifest.json` (git commit, package versions, GPU, dataset
keys, per-task counts, boolean-only API-key presence) and `form_metadata.md`
(every live portal field, `<FILL IN>` for anything needing human judgment)
into `<work-dir>`.

## Failure handling

| Symptom | Fix |
|---|---|
| `No submission files found` | nothing matched `*_submission.jsonl`; re-check `--work-dir` points at the eval output dir, not a parent |
| `unrecognized submission file, skipping` | filename doesn't match any `TASK_PATTERNS` entry; verify the dataset key used is a registered VANTAGE key |
| Archive > 500 MB | re-run `package_submission.py` from a clean `--work-dir` (stray files, not JSONL size, are the usual cause) |

## Next

Hand off `submission.tar.gz` and `form_metadata.md` to
`../vantage-portal-submit/SKILL.md`.
