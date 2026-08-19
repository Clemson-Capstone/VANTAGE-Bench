---
name: vantage-run
description: Use when the user wants to run VANTAGE-Bench inference and evaluation for chosen pillars, from smoke test through a monitored full run.
---

Run one or more VANTAGE-Bench tasks end to end: smoke test first, then (after
confirmation) the full run, with monitoring and resume support.

## 1. Build the task list

Map the pillars/tasks the user wants to `--data` dataset keys via
[`skills/reference/tasks-and-pillars.md`](../reference/tasks-and-pillars.md) —
do not hand-write dataset key names. Default to the `8frame`/default variant
per task unless the user asks for a different sampling rate.

## 2. State detection

Check `<work-dir>/<model>/<eval_id>/` for existing `*.xlsx` / `*_submission.jsonl`.
If predictions already exist for a task, prefer `--reuse` over re-running
inference from scratch.

## 3. Mandatory smoke test (no gate needed for this step)

Before touching the full task list, run the smallest task available —
`VANTAGE_VQA_8frame_200` (200-sample subset) is the standard smoke target:

```bash
python run.py --data VANTAGE_VQA_8frame_200 --model <ModelName> \
  --work-dir <work-dir> --verbose
```

Inspect the resulting `*_VANTAGE_VQA_8frame_200.xlsx` for empty predictions or
the API failure string (`FAIL_MSG = 'Failed to obtain answer via API.'`, per
`scripts/apires_scan.py`):

```bash
python -c "from vlmeval.smp import load; d=load('<path-to-smoke-xlsx>'); \
p=d['prediction'].astype(str); \
print((p.str.strip().eq('') | p.str.contains('Failed to obtain answer via API.')).sum(), 'of', len(d))"
```

Any nonzero count here, or predictions that look like garbage on manual read,
means fix the model/config before scaling up — do not extrapolate from a
broken smoke sample.

From the smoke run's wall-clock and (for API models) token/cost usage,
extrapolate to the full task list: `full_task_row_count / 200 * smoke_wall_clock`
per task, summed across tasks. Note this is rough — parallelism
(`--api-nproc`) and per-task video length both skew it.

## 4. Gate — confirm before the full run

**STOP and confirm with the user before launching the full multi-task run.**
State the task list, the extrapolated wall-clock, and (for API models) the
extrapolated cost. This gate applies only to the full run — the smoke test in
step 3 does not need it.

## 5. Launch the full run

```bash
python run.py --data <task1> <task2> ... --model <ModelName> \
  --work-dir <work-dir> --verbose
```

Run in the background (`nohup ... &`, `screen`, or equivalent) and tail the
log rather than blocking the foreground session, since multi-task/multi-GPU
runs can take hours.

## 6. The `--mode infer` trap

`--mode infer` never writes `*_submission.jsonl` — that file is written by
`dataset.evaluate()` during the eval phase. If someone ran infer-only,
recover with `--mode eval --reuse` (does not repeat inference) rather than
re-running everything. Default `--mode all` avoids the trap entirely.

## 7. Recognizing failure signatures while tailing logs

| Signature | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | vLLM/HF OOM | lower `gpu_memory_utilization`, lower `max_model_len`, lower `tensor_parallel_size` (config file) |
| No new log lines for a long stretch, GPU util ~0 | stalled worker / hung download | check `nvidia-smi`; kill and resume with `--reuse` |
| Repeated `401`/`403` in API model logs | bad/expired API key | fix env var (`OPENAI_API_KEY` etc.), resume with `--reuse` |
| `ValueError` about `fps`/`nframe` | both set in config | see `tasks-and-pillars.md` — they're mutually exclusive |

Resume any interrupted run with `--reuse` (add `--ignore` only if you
intend to skip already-failed indices rather than retry them).

## Next

Once every intended task has a `*_submission.jsonl`, hand off to
`../vantage-validate/SKILL.md`.
