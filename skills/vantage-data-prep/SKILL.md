---
name: vantage-data-prep
description: Use when LMUData is missing tasks needed for a run — stages the public VANTAGE-Bench dataset locally via run_lmudata.py.
---

Wraps `scripts/run_lmudata.py` (do not modify it — treat `scripts/RUN_LMUData.md` and `--help`
as the source of truth for exact flags if anything here looks stale).

## State detection

Check `$LMUData/datasets/<Task>/` for each task the user needs (map pillar → task via
`../reference/tasks-and-pillars.md`). Expected dir names and contents are in
`scripts/RUN_LMUData.md` §K. A task already passing its integrity check is skipped
automatically by the script even without this check — but do the check first anyway so you
can size the plan (dry-run) before touching the network.

## Decision logic

**Which tasks**: only prepare tasks for the pillar(s) the user actually wants
(`../reference/tasks-and-pillars.md`). Don't default to `--all` if they only asked for one
pillar — SOT alone adds ~16GB.

**`--symlink` vs `--copy`**: the script defaults to `--symlink`. On **Windows, default to
`--copy` instead** — symlinks require Developer Mode or admin privileges and are unreliable
without them, so a symlink-mode LMUData can silently end up with broken media links on a
typical Windows setup. On Linux/macOS, keep the `--symlink` default unless disk-constrained
relative to the HF cache location, or the user wants a portable/self-contained folder.

## Commands, in order

1. **Dry run first** — no downloads, no writes, shows the plan and disk estimate:
   ```bash
   python scripts/run_lmudata.py --tasks <task1,task2,...> --lmu-root <path> --dry-run
   ```
   (or `--all` in place of `--tasks ...` for every task). Read the printed summary before
   proceeding — it names the HF files each task would fetch.

2. **Gate: confirm with the user before the real (non-dry-run) run.** State explicitly:
   SOT alone pulls ~16GB into the HF cache; `--copy` mode additionally duplicates that media
   into LMUData (tens of GB total across all tasks). Get explicit go-ahead before spending
   that bandwidth/disk, especially if SOT is in the task list.

3. **Real run**:
   ```bash
   python scripts/run_lmudata.py --tasks <task1,task2,...> --lmu-root <path> --copy
   # or, on Linux/macOS with the default media mode:
   python scripts/run_lmudata.py --tasks <task1,task2,...> --lmu-root <path>
   ```

4. **Post-check**: confirm the resulting layout matches `scripts/RUN_LMUData.md` §K
   (per-task TSV + media dir, or `<seq>/gt.json` + `<seq>/frames/` for SOT).

## Retry and destructive flags

- **Failed task retry**: each task fails independently; re-run only the failed ones with
  `--force` to rebuild just that task's index (it does not re-download files already cached):
  ```bash
  python scripts/run_lmudata.py --tasks <failed_task> --lmu-root <path> --force
  ```
- **`--force-clean` is destructive — gate: confirm before using it.** It wipes the named
  task's media directory under `$LMUData/datasets/<Task>/` before re-staging. State exactly
  which directory will be deleted and confirm the user wants a full rebuild, not a retry.

## Failure handling

| Symptom | Fix |
|---|---|
| Broken/dangling symlinks | HF cache moved or was cleaned. Re-run prep, or rebuild with `--copy`. See `RUN_LMUData.md` §J. |
| `ffmpeg` missing (SOT only) | `conda install -c conda-forge ffmpeg`; auto-detected from common conda envs too. §H. |
| HF 401/403 | `hf auth login`, `export HF_TOKEN=...`, or `--hf-token`; check dataset license acceptance. §J. |
| VLMEvalKit can't find data after prep | `$LMUData` doesn't match `--lmu-root`; `export LMUData=<same path>` and verify with `python -c "from vlmeval.smp import LMUDataRoot; print(LMUDataRoot())"`. |
| Grounding image download fails | Falls back to `gdown` only if the primary GitHub mirror fails; `pip install gdown` only then. §I. |

## Further reading

`scripts/RUN_LMUData.md` in full for edge cases (local dataset-repo mode, RefDrone
prerequisites, idempotency guarantees). `../reference/tasks-and-pillars.md` for the pillar →
task map and dataset-key defaults used once data prep is done.
