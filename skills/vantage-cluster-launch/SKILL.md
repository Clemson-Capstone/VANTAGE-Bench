---
name: vantage-cluster-launch
description: Use when the user wants to launch a VANTAGE-Bench run on a SLURM cluster or a rented/cloud GPU VM instead of a local machine.
---

Launch the same `run.py` commands from `../vantage-run/SKILL.md` on shared or
remote compute, with the extra care that entails: shared-resource etiquette,
scratch-storage risk, and disconnect-safe monitoring.

Still do the mandatory smoke test from `../vantage-run/SKILL.md` (step 3) before
any cluster/VM launch — it does not need its own gate. Launching the **full**
run on a cluster does need the gate below.

## Path A — SLURM

Wrap `scripts/srun.sh`, which hardcodes `-n1 --ntasks-per-node=1
--gres=gpu:8 --quotatype=reserved --cpus-per-task=64` and launches
`torchrun --nproc-per-node=8 run.py ${@:2}`:

```bash
bash scripts/srun.sh <partition> --data <task1> <task2> ... --model <ModelName> \
  --work-dir <work-dir> --verbose
```

Everything after `<partition>` is forwarded verbatim to `run.py` via
torchrun — use the exact `run.py` flags from `../vantage-run/SKILL.md`. This
script requests a fixed 8-GPU node; if the target partition doesn't have
8-GPU nodes, don't use it — call `srun`/`sbatch` directly with matching
`--gres` instead.

## Path B — cloud VM / rented GPU

No scheduler wrapper exists for this path; run `run.py` directly, but:

- **Scratch storage may be wiped** on instance stop/reclaim. Point
  `--work-dir` (and `--lmudata-root` if data lives there too) at persistent
  storage, or copy outputs off the instance before it can be reclaimed.
- **Watch quota** on scratch/persistent volumes — video datasets and model
  checkpoints are large; check free space before a multi-task run.
- The session may disconnect. Launch detached and tail the log rather than
  relying on a foreground shell:

```bash
nohup python run.py --data <task1> <task2> ... --model <ModelName> \
  --work-dir <work-dir> --verbose > run.log 2>&1 &
tail -f run.log
```

## Preemption / interruption resume

`--reuse` is the key resilience mechanism for both SLURM preemption and
cloud spot-instance interruption: it picks inference back up from existing
prediction files instead of repeating it. If a job was killed mid-evaluation
(predictions exist, no `*_submission.jsonl` yet), `--mode eval --reuse`
finishes evaluation without touching inference at all.

## Gate — confirm before submitting a cluster job

**STOP and confirm with the user before submitting to SLURM or launching on
a rented/cloud GPU.** This is a shared resource (SLURM) or a metered/billed
resource (cloud), and other users or the user's budget are affected — treat
it the same as the full-run gate in `../vantage-run/SKILL.md`.

## Failure handling

| Symptom | Fix |
|---|---|
| Job disappears from `squeue`, no output | check `sacct -j <jobid>` for `OUT_OF_MEMORY`/`PREEMPTED`/`TIMEOUT`; resume with `--reuse` |
| `srun.sh` hangs at partition allocation | partition likely lacks free 8-GPU nodes; check `sinfo -p <partition>` |
| Disk quota exceeded mid-run | move `--work-dir`/`--lmudata-root` off the full volume; resume with `--reuse` |
| SSH session drops, run stops | relaunch with `nohup`/`tmux`/`screen`, not a bare foreground shell |

## Next

Once tasks finish, hand off to `../vantage-validate/SKILL.md`.
