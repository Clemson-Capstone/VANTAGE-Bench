# Submitting to VANTAGE-Bench

**Portal:** [https://vantage-bench.org/submit](https://vantage-bench.org/submit)  
**Limits:** 2 submissions / day · 30 submissions / lifetime per email  
**Scores** are returned by email; ground truth is withheld from the public dataset.

---

## Pillars

VANTAGE-Bench is organized into four pillars. Submit **all tasks within a pillar** together — partial-pillar submissions are rejected. You may submit any combination of complete pillars.

| Pillar | Name | Tasks | Primary metric |
|--------|------|-------|----------------|
| **I** | Semantic | Event Verification, Video QA | Macro F1, Accuracy |
| **II** | Spatial | Referring Expressions, Spatial Pointing, Object Localization (Astro2D) | mIoU, Accuracy, F1@0.5 |
| **III** | Temporal | Temporal Localization, Dense Video Captioning | mIoU, SODAc |
| **IV** | Spatio-Temporal | Single Object Tracking | Success AUC |

---

## Step 1 — Run inference and evaluation

The submission JSONL files are written during the **evaluation** phase (not inference-only). Use the default `--mode all` or explicitly pass `--mode eval` after inference.

```bash
export LMUData=~/LMUData

# Run all four pillars in one command (inference + evaluation)
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

Each task writes a `*_submission.jsonl` alongside its prediction xlsx:

```
outputs/<model>/<eval_id>/
├── <model>_VANTAGE_VQA_8frame.xlsx
├── <model>_VANTAGE_VQA_8frame_submission.jsonl
├── <model>_VANTAGE_EventVerification_8frame.xlsx
├── <model>_VANTAGE_EventVerification_8frame_submission.jsonl
├── ...
└── <model>_Astro2D_submission.jsonl
```

> **Note:** Submission JSONLs are produced by the evaluation phase. If you ran with `--mode infer`, re-run with `--mode eval --reuse` to generate them without repeating inference.

---

## Step 2 — Package into a `.tar.gz` archive

The portal requires **one `.tar.gz` containing one `.jsonl` per task**:

```bash
python scripts/package_submission.py \
  --work-dir ./outputs/<model>/<eval_id> \
  --out submission.tar.gz
```

This renames files to canonical task names and bundles them:

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

---

## Step 3 — Fill out the form and upload

Go to [https://vantage-bench.org/submit](https://vantage-bench.org/submit) and complete:

| Section | Fields |
|---------|--------|
| **Identity** | Leaderboard name, Organization, Model card / paper URL, Contact email |
| **Submission type** | Single model vs. system pipeline; open-weight / mixed / proprietary |
| **Model configuration** | Checkpoint, parameter count, precision, zero-shot vs. fine-tuned |
| **Inference setup** | Infrastructure, official harness used, additional hyperparameters |
| **Pillars submitted** | Select each pillar you are submitting |
| **Predictions file** | Upload `submission.tar.gz` (max 500 MB) |
| **Acknowledgements** | Confirm four statements |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Partial pillar rejected | Run all tasks in the pillar; see pillar table above |
| Missing `*_submission.jsonl` | Re-run with `--mode eval --reuse`; submission files require the eval phase |
| Archive > 500 MB | Run `package_submission.py` from a clean output dir; check for extra files |
| No score email | Check spam; evaluation can take up to 24 hours |

---

Full details including JSONL record format and canonical ID schemas: [docs/vantage/SUBMISSION.md](docs/vantage/SUBMISSION.md)
