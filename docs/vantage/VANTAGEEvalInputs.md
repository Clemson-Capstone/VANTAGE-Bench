# VANTAGE Eval Input Schemas

This note documents the minimum prediction-file columns needed to run evaluation without embedding ground-truth fields inside the model output file.

## General rule

- preferred formats: `.xlsx`, `.json`, `.tsv`
- every prediction file should include a `prediction` column
- use dataset row identifiers from the benchmark TSV whenever possible
- avoid copying GT fields like `answer`, `gt_bboxes`, or labels into exported predictions

## VANTAGE VQA

Minimum columns:

- `index`
- `prediction`

Notes:

- GT answer, task type, and category are resolved from the dataset TSV by `index`
- prediction text can be free-form as long as the final answer letter is recoverable

## VANTAGE Temporal

Minimum columns:

- `index`
- `prediction`

Notes:

- predictions may be JSON with `start` / `end` or timestamp text that the parser can recover
- GT spans and video duration are resolved from the dataset TSV by `index`

## VANTAGE DVC

Minimum columns:

- `index`
- `prediction`

Notes:

- prediction can be JSON event lists or timestamped text lines
- GT event spans and captions are resolved from the dataset TSV by `index`

## VANTAGE Event Verification

Minimum columns:

- `prediction`

Recommended additional identifier columns:

- `index`, or
- `id`, or
- `video`

Notes:

- GT is resolved from dataset metadata in that priority order
- prediction should contain a recoverable `yes` / `no`

## VANTAGE SOT

Minimum columns:

- `index`
- `prediction`

Notes:

- prediction should be a JSON object keyed by frame number, for example `frame_1`, `frame_2`, ...
- GT track metadata is resolved from the prepared SOT dataset cache

## Astro2D

Minimum columns:

- `image_path`
- `prediction`

Recommended:

- `image_filename`

Notes:

- GT boxes are loaded from label files on disk
- predictions should be parseable as bbox JSON or list-style box outputs

## 2D Grounding

Minimum columns:

- `index`
- `prediction`

Notes:

- evaluator can fall back to the dataset TSV for GT boxes if they are not embedded in the prediction file
- if `image_width` / `image_height` are absent, the evaluator can recover them from the image file when needed

## 2D Pointing

Minimum columns:

- `index`
- `prediction`

Notes:

- this is a multiple-choice task; the prediction should contain a recoverable answer letter (A/B/C/D)
- GT answer and spatial reference are resolved from the dataset TSV by `index`
