# Submitting to VANTAGE-Bench

Submit at: **https://vantage-bench.org/submit**

---

## Limits

- 2 submissions per day per email
- 30 submissions lifetime per email

Scores are emailed to you after evaluation runs server-side against withheld ground truth.

---

## Pillars

VANTAGE-Bench is organized into four pillars. **You must submit all tasks within a pillar** — partial-pillar submissions are rejected. The overall leaderboard score is computed only over the pillars you submit.

| Pillar | Name | Tasks (dataset keys) | Primary metric |
|--------|------|----------------------|---------------|
| **I** | Semantic | Event Verification (`VANTAGE_EventVerification_*`) | Macro F1 |
| | | Video QA (`VANTAGE_VQA_*`) | Top-1 Accuracy |
| **II** | Spatial | Referring Expressions (`VANTAGE_2DGrounding`) | mIoU |
| | | Spatial Pointing (`VANTAGE_2DPointing`) | Top-1 Accuracy |
| | | Object Localization (`Astro2D`) | F1@0.5 |
| **III** | Temporal | Temporal Localization (`VANTAGE_Temporal_*`) | mIoU |
| | | Dense Video Captioning (`VANTAGE_DVC_*`) | SODAc |
| **IV** | Spatio-Temporal | Single Object Tracking (`VANTAGE_SOT`) | Success AUC |

You may submit any combination of pillars (e.g., I + III only). Pillars not submitted are excluded from scoring.

---

## Step 1 — Run inference and evaluation for all tasks in your chosen pillars

Submission JSONL files are written during the **evaluation phase** (`dataset.evaluate()`). Use the default `--mode all` so that both inference and evaluation run in one step. If you need to re-run without repeating inference, use `--mode eval --reuse`.

```bash
export LMUData=~/LMUData

# Example: all four pillars (inference + evaluation in one step)
python run.py \
  --data VANTAGE_VQA_8frame \
        VANTAGE_EventVerification_8frame \
        VANTAGE_Temporal_8frame \
        VANTAGE_DVC_8frame \
        VANTAGE_SOT \
        VANTAGE_2DGrounding \
        VANTAGE_2DPointing \
        Astro2D \
  --model <YourModel> \
  --work-dir ./outputs
```

Each task writes a `*_submission.jsonl` file next to its prediction xlsx:

```
outputs/<model>/<eval_id>/
├── <model>_VANTAGE_VQA_8frame.xlsx
├── <model>_VANTAGE_VQA_8frame_submission.jsonl
├── <model>_VANTAGE_EventVerification_8frame.xlsx
├── <model>_VANTAGE_EventVerification_8frame_submission.jsonl
├── ...
└── <model>_Astro2D_submission.jsonl
```

---

## Step 2 — Package into a `.tar.gz` archive

The submission form requires **one `.tar.gz` archive containing one `.jsonl` file per task**. Use the provided helper script:

```bash
python scripts/package_submission.py \
  --work-dir ./outputs/<model>/<eval_id> \
  --out submission.tar.gz
```

This collects all `*_submission.jsonl` files, renames them to their canonical task names, and bundles them:

```
submission.tar.gz
├── vqa.jsonl
├── event_verification.jsonl
├── temporal.jsonl
├── dvc.jsonl
├── sot.jsonl
├── grounding.jsonl
├── pointing.jsonl
└── astro.jsonl
```

Only files for tasks you actually ran are included — the server validates that all tasks in each submitted pillar are present.

---

## Step 3 — Fill out the submission form

Go to **https://vantage-bench.org/submit** and complete the seven sections:

| Section | Fields |
|---------|--------|
| **Identity** | Candidate name (leaderboard display name), Organization, Model card / paper URL, Contact email |
| **Submission type** | Pipeline type (Single model / System pipeline), System access type (Fully open-weight / Mixed / Proprietary) |
| **Model configuration** | Primary model / checkpoint, Parameter count, Inference precision, Training type (Zero-shot / Fine-tuned) |
| **Inference setup** | Primary inference infrastructure, Official harness used (Yes / No), Additional hyperparameters |
| **Pillars submitted** | Select the pillars you are submitting (at least one) |
| **Predictions file** | Upload `submission.tar.gz` (max 500 MB) |
| **Acknowledgements** | Confirm four statements before submitting |

---

## JSONL record format

Each line in a task's `.jsonl` is:

```json
{
  "id": "<canonical-id>",
  "task": "<task-key>",
  "conversations": [{"role": "assistant", "content": "<raw-model-output>"}],
  "metadata": {}
}
```

The canonical ID format is task-specific and generated automatically by the harness:

| Task | ID format | Example |
|------|-----------|---------|
| VQA | `{video_stem}__q_{index:06d}` | `C0065_clip01__q_000042` |
| EventVerification | `{video_stem}__ev_{index:06d}` | `C0065_clip01__ev_000007` |
| Temporal | `{video_stem}__tg_{index:06d}` | `C0065_clip01__tg_000013` |
| DVC | `{video_stem}__dvc_{index:06d}` | `C0065_clip01__dvc_000001` |
| SOT | `{seq_dir_name}` | `Warehouse_000__Camera_0003__obj37` |
| 2DGrounding | `{image_stem}__rx_{index:06d}` | `frame_0042__rx_000003` |
| 2DPointing | `{image_stem}__sp_{index:06d}` | `000000_000000__sp_000099` |
| Astro2D | `{image_stem}__ol_{index:06d}` | `IVA_frame_0001__ol_000000` |

The `task` field in each record is the canonical short key: `vqa`, `event_verification`, `temporal`, `dvc`, `sot`, `grounding`, `pointing`, `astro`.

---

## Troubleshooting

**"Partial pillar submission rejected"** — Make sure you ran inference for every task in each pillar you selected. See the pillar table above.

**Missing `*_submission.jsonl` file** — Submission files are written during `dataset.evaluate()`, not during inference. Re-run with `--mode eval --reuse` (keeps existing predictions, runs evaluation only) or use the default `--mode all`.

**Archive too large (>500 MB)** — Each JSONL record is small. If the archive is large, check for accidentally bundled non-JSONL files. Run `package_submission.py` again from a clean output directory.

**Did not receive score email** — Check your spam folder. Scores are sent to the contact email provided in the form. Evaluation may take up to 24 hours.
