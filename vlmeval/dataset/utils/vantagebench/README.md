# vlmeval/dataset/utils/vantagebench/

Submission JSONL emission system. Every VANTAGE task calls into this module to produce the `*_submission.jsonl` file that participants upload to the leaderboard.

---

## How it works

When `dataset.evaluate(result_file)` is called, it calls `emit_submission(data, stem, out_path, task=...)` from `emit.py`. The emitter:

1. Validates that the required columns are present (e.g. `index`, `prediction`).
2. Resolves a canonical submission ID for each row using the task-specific `id_resolver`.
3. Writes one JSON record per row to the output JSONL:
   ```json
   {"id": "<canonical-id>", "task": "<task-key>", "conversations": [{"role": "assistant", "content": "<prediction>"}], "metadata": {}}
   ```
4. Never raises — any error is converted to a `warnings.warn` so the prediction xlsx is always written even if emission fails.

---

## Files

| File | Purpose |
|------|---------|
| `emit.py` | Canonical emitter. Shared policy for all tasks. |
| `adapter_vqa.py` | Thin wrapper: calls `emit.py` with task=`vqa`. |
| `adapter_temporal.py` | Thin wrapper: task=`temporal`. |
| `adapter_dvc.py` | Thin wrapper: task=`dvc`. |
| `adapter_event_verification.py` | Thin wrapper: task=`event_verification`. |
| `adapter_sot.py` | Thin wrapper: task=`sot`. Requires `dataset` kwarg. |
| `adapter_grounding.py` | Thin wrapper: task=`grounding`. |
| `adapter_pointing.py` | Thin wrapper: task=`pointing`. |
| `adapter_astro.py` | Thin wrapper: task=`astro`. |
| `id_rules.py` | Per-task canonical ID generators (`make_vqa_id`, `make_temporal_id`, …). |
| `evaluator.py` | Local evaluation helpers (used when GT is present). |
| `io.py` | JSONL read/write helpers. |
| `validate.py` | Column presence validators. |
| `__init__.py` | Package init. |

---

## Calling from a dataset class

```python
from vlmeval.dataset.utils.vantagebench.emit import emit_submission

def evaluate(self, eval_file, **judge_kwargs):
    data = load(eval_file)
    stem = osp.splitext(osp.basename(eval_file))[0]
    out_path = eval_file.replace('.xlsx', '_submission.jsonl')
    emit_submission(data, stem, out_path, task='vqa')
    # ... local scoring if GT present
```

---

## Output format

Each line in `*_submission.jsonl` is one JSON object:

```json
{
  "id": "vqa_<index>",
  "task": "vqa",
  "conversations": [
    {"role": "assistant", "content": "B"}
  ],
  "metadata": {}
}
```

The `id` format is task-specific and defined in `id_rules.py`. The leaderboard server uses `id` to match predictions against withheld ground truth.
