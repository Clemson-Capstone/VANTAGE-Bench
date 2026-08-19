# VANTAGE-Bench tasks, pillars, and IDs — single source of truth

Every skill and script links here instead of restating this table. If you are editing this
file, you are the only place it needs to change.

## Pillars (submit all tasks in a pillar together, or the submission is rejected)

| Pillar | Task (short key) | Default dataset key | Primary metric | Submission filename |
|---|---|---|---|---|
| I — Semantic | `vqa` | `VANTAGE_VQA_8frame` | Top-1 Accuracy | `vqa.jsonl` |
| I — Semantic | `event_verification` | `VANTAGE_EventVerification_8frame` | Macro F1 | `event_verification.jsonl` |
| II — Spatial | `grounding` | `VANTAGE_2DGrounding` | mIoU | `grounding.jsonl` |
| II — Spatial | `pointing` | `VANTAGE_2DPointing` | Top-1 Accuracy | `pointing.jsonl` |
| II — Spatial | `astro` | `Astro2D` | F1@0.5 | `astro.jsonl` |
| III — Temporal | `temporal` | `VANTAGE_Temporal_8frame` | mIoU | `temporal.jsonl` |
| III — Temporal | `dvc` | `VANTAGE_DVC_8frame` | SODAc | `dvc.jsonl` |
| IV — Spatio-Temporal | `sot` | `VANTAGE_SOT` | Success AUC | `sot.jsonl` |

This is the same map `scripts/package_submission.py`'s `PILLAR_TASKS`/`TASK_PATTERNS` encode
in code — `validate_submission.py` and `package_submission.py` should import that dict rather
than re-deriving it here.

## All registered dataset keys (`--data` values)

Full list and per-key sampling notes: `README_VANTAGE.md` §4. Skills default to the
**8frame / default** variant per task unless the user asks for a different sampling rate:

- VQA: `VANTAGE_VQA_8frame` (default), `_16frame`, `_64frame`, `_4fps`, `_1fps`, `_0.5fps`, `_8frame_200` (200-sample smoke-test subset)
- Temporal: `VANTAGE_Temporal_8frame` (default), `_16frame`, `_64frame`, `_1fps`, `_0.5fps`, `_10fps`
- DVC: `VANTAGE_DVC_8frame` (default), `_64frame`, `_1fps`, `_2fps`, `_4fps`
- EventVerification: `VANTAGE_EventVerification_8frame` (default), `_16frame`, `_1fps`, `_4fps`
- SOT: `VANTAGE_SOT` (default, 8f/stride15), `_16f`, `_32f`
- Image (no frame variants): `VANTAGE_2DGrounding`, `VANTAGE_2DPointing`, `Astro2D`

**Config-file requirement:** video-task keys above are in `supported_video_datasets` and can be
instantiated with `{}` in a config JSON. The three image datasets are **not** in that
registry — a config entry for them must name `"class"` explicitly
(`VANTAGE_2DGroundingDataset`, `VANTAGE_2DPointing`, `Astro2DDetectionDataset`).

**`fps` vs `nframe`:** mutually exclusive, setting both raises `ValueError`. `VANTAGE_EventVerification`
defaults to `fps=4`; pass `fps=0` to use `nframe` instead (all registered nframe variants of
this task already do this).

## Canonical submission ID formats and `task` field values

Defined once in `vlmeval/dataset/utils/vantagebench/id_rules.py` (id generators) and
`emit.py`'s `_TASK_SPECS` registry (the `task` string written into each record) — **always
import these, never reimplement the regex or hand-map the task string**. Note the short key
(used for filenames, `--work-dir` matching, and the `task=` kwarg you pass *into*
`emit_submission()`) is **not** what ends up in the JSON record's `task` field — that's a
longer canonical string looked up from the registry:

| Short key (filenames, pillar map) | `task` field written to JSONL | Generator | ID format | Example |
|---|---|---|---|---|
| `vqa` | `video_qa` | `make_vqa_id(video, index)` | `{video_stem}__q_{index:06d}` | `C0065_clip01__q_000042` |
| `event_verification` | `event_verification` | `make_event_verification_id` | `{video_stem}__ev_{index:06d}` | `C0065_clip01__ev_000007` |
| `temporal` | `temporal_grounding` | `make_temporal_id` | `{video_stem}__tg_{index:06d}` | `C0065_clip01__tg_000013` |
| `dvc` | `dense_video_captioning` | `make_dvc_id` | `{video_stem}__dvc_{index:06d}` | `C0065_clip01__dvc_000001` |
| `grounding` | `referring_expressions` | `make_grounding_id(image, index)` | `{image_stem}__rx_{index:06d}` | `frame_0042__rx_000003` |
| `pointing` | `spatial_pointing` | `make_pointing_id(image_path, index)` | `{image_stem}__sp_{index:06d}` | `000000_000000__sp_000099` |
| `astro` | `object_localization` | `make_astro_id(image_filename, index)` | `{image_stem}__ol_{index:06d}` | `IVA_frame_0001__ol_000000` |
| `sot` | `single_object_tracking` | `make_sot_id(seq_dir_name)` | normalizes to `{scene}__{camera}_{frame:07d}__obj{id}`; no index suffix | `Warehouse_000__Camera_0003_0005648__obj37` |

SOT differs from the other tasks: its id takes only the sequence dir name (no enumeration index), and
`make_sot_id` normalizes an alternate doubled-separator/unpadded-frame spelling that
the public dataset sometimes uses — passing a name through the wrong path can join 0/N
against ground truth without erroring. Always call the generator; never string-format by hand.

## Submission JSONL record shape

One JSON object per line, written by `emit_submission()` in
`vlmeval/dataset/utils/vantagebench/emit.py` during `dataset.evaluate()` (the eval phase,
not inference). **The repo's own docs (`SUBMISSION.md`, `docs/vantage/SUBMISSION.md`,
`vlmeval/dataset/utils/vantagebench/README.md`) previously showed `{"role": "assistant",
"content": ...}` and `"metadata": {}` — that was wrong; verified against `emit.py` source
and corrected here and in those docs:

```json
{"id": "<canonical-id>", "task": "<canonical-task-string, e.g. video_qa>", "conversations": [{"from": "assistant", "value": "<raw-model-output>"}], "metadata": {"model": "<model-name>", "box_coord_order": "xyxy", "extra": {}}}
```

The conversation turn uses `from`/`value`, not `role`/`content`. `metadata` is never empty —
it always carries `model` and `box_coord_order` (read by the grounding/astro/SOT evaluators
to score models like Gemini that emit boxes in `yxyx` order instead of `xyxy`).

**Critical property:** `emit_submission()` never raises — a failure becomes a
`warnings.warn`, so the prediction `.xlsx` can look fine while the submission JSONL is
missing, truncated, or malformed. This is why `scripts/validate_submission.py` exists —
never assume a JSONL is correct just because `run.py` exited 0.

## Output layout

```
outputs/<model>/<eval_id>/
├── <model>_<dataset_key>.xlsx              # raw predictions
├── <model>_<dataset_key>_submission.jsonl  # written by evaluate(), not infer
└── <model>_<dataset_key>_<metrics-suffix>  # acc.csv / metrics.json / acc.json, task-specific
```

`--mode infer` alone never writes `_submission.jsonl`. Recover with `--mode eval --reuse`
(does not repeat inference) or just use the default `--mode all`.
