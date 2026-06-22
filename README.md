# VANTAGE-Bench

**VANTAGE-Bench** is a multi-task benchmark for evaluating Vision-Language Models on fixed-camera footage captured in operational environments — spanning warehouse, transportation, and smart-spaces deployments.

This repository is a fork of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), an open-source VLM evaluation toolkit. VANTAGE-Bench adds new benchmark tasks, dataset loaders, and evaluation metrics on top of that foundation.

**[[Website]](https://vantage-bench.org)  [[Leaderboard]](https://huggingface.co/spaces/clemson-computing/VANTAGE-Bench-Leaderboard)  [[Dataset]](https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench)  [[Submit]](https://vantage-bench.org/submit)**

---

## Contents

- [About VANTAGE-Bench](#about-vantage-bench)
- [Quick Start](#quick-start)
- [End-to-End Flow](#end-to-end-flow)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running Evaluations](#running-evaluations)
- [Submission Workflow](#submission-workflow)
- [Model Backends](#model-backends)
- [Output Structure](#output-structure)
- [Prediction File Schemas](#prediction-file-schemas)
- [Hardware Requirements](#hardware-requirements)
- [Repository Layout](#repository-layout)
- [Documentation Index](#documentation-index)
- [Built on VLMEvalKit](#built-on-vlmevalkit)
- [Citation](#citation)

---

## About VANTAGE-Bench

VANTAGE-Bench is a multi-task benchmark for Real-World Video Understanding, designed to evaluate Vision-Language Models on fixed-camera footage captured in operational environments.

The benchmark spans three deployment domains: **Warehouse**, **Transportation**, and **Smart Spaces**, and evaluates model capability across four complementary pillars of video intelligence.

Unlike benchmarks built around curated internet media or short trimmed clips, VANTAGE-Bench emphasizes the perceptual and reasoning capabilities required for real-world Infrastructure AI systems — including localization, tracking, temporal reasoning, and grounded understanding of events.

### Benchmark Structure

| Pillar | Description | Tasks |
|--------|-------------|-------|
| **Spatial** | 2D scene understanding | 2D Object Localization, 2D Referring Expressions, 2D Pointing |
| **Spatio-Temporal** | Tracking and spatial reasoning over time | Single Object Tracking |
| **Temporal** | Event timing and temporal reasoning | Temporal Localization, Dense Video Captioning |
| **Semantic** | High-level video understanding | Event Verification, Video Question Answering |

### Tasks and Primary Metrics

| Pillar | Task | Primary Metric |
|--------|------|----------------|
| Spatial | 2D Object Localization | F1@0.5 |
| Spatial | 2D Referring Expressions | mIoU |
| Spatial | 2D Pointing | Accuracy |
| Spatio-Temporal | Single Object Tracking | AUC |
| Temporal | Temporal Localization | mIoU |
| Temporal | Dense Video Captioning | SODA_c |
| Semantic | Event Verification | Macro F1 |
| Semantic | Video Question Answering | Accuracy |

---

## Quick Start

```bash
# 1. Clone and install (Python 3.10+ required)
git clone https://github.com/Clemson-Capstone/VANTAGE-Bench.git
cd VANTAGE-Bench
conda create -n vantage python=3.10 -y && conda activate vantage
conda install -c conda-forge ffmpeg -y          # required for VANTAGE-SOT
pip install -r requirements.txt && pip install -e .

# 2. Download benchmark data from HuggingFace
hf auth login                                   # once — needed for SOT data
python scripts/run_lmudata.py --all --lmu-root ~/LMUData

# 3. Run inference + evaluation (produces predictions and submission files)
export LMUData=~/LMUData
export OPENAI_API_KEY=<your-key>                # if using an API model
python run.py \
  --data VANTAGE_VQA_8frame \
  --model GPT4o \
  --work-dir ./outputs

# 4. Package and submit
python scripts/package_submission.py --work-dir ./outputs/<model>/<eval_id> --out submission.tar.gz
# Upload submission.tar.gz at https://vantage-bench.org/submit
```

> **Data prep guide:** [`scripts/RUN_LMUData.md`](scripts/RUN_LMUData.md) covers all tasks, options, troubleshooting, and the SOT/grounding prerequisites.
>
> **Prompt formats:** [`prompt_guide.md`](prompt_guide.md) documents the exact prompt templates used for each benchmark.

---

## End-to-End Flow

This section traces exactly what happens from a fresh clone to a submitted result.

### Participant flow

```
git clone / pip install
        │
        ▼
scripts/run_lmudata.py                 downloads nvidia/PhysicalAI-VANTAGE-Bench from HF
        │                              reshapes into LMUData/datasets/<Task>/ layout
        │                              (symlinks media into ~/.cache/huggingface/)
        ▼
export LMUData=~/LMUData
        │
        ▼
python run.py                          builds dataset object → calls dataset.prepare_dataset()
        │                              loads TSV from LMUData/datasets/<Task>/<Task>.tsv
        │                              for each row: calls dataset.build_prompt()
        │                                 → packages video/image paths + question text
        │                              feeds prompt to model → gets raw prediction string
        │                              writes predictions to outputs/<model>/<eval_id>/<model>_<task>.xlsx
        │
        ▼
dataset.evaluate(result_file)          called at end of run.py (skipped in --mode infer)
        │                              calls emit_submission() in vlmeval/dataset/utils/vantagebench/
        │                              writes outputs/<model>/<eval_id>/<model>_<task>_submission.jsonl
        │                              (no local leaderboard metrics — GT is withheld from public dataset)
        ▼
scripts/package_submission.py          collects all *_submission.jsonl files from the output dir
        │                              renames them to canonical task names (vqa.jsonl, temporal.jsonl, …)
        │                              bundles into submission.tar.gz
        ▼
upload submission.tar.gz               to https://vantage-bench.org/submit
                                       scores emailed back; 2 submissions/day · 30 lifetime
```

### Organizer / leaderboard pipeline

```
outputs/<model>/<eval_id>/
        │
        ▼  parser/get_all_outputs.py
vlmevalkit_outputs.json                one JSON keyed by model, task → metric values
        │
        ▼  parser/prepare_outputs_hf.py
hf/leaderboard.json                    leaderboard schema with overall scores and task breakdowns
        │
        ▼  hf/up.py
HF Space (hf/app.py)                   Gradio leaderboard UI at
                                        https://huggingface.co/spaces/clemson-computing/VANTAGE-Bench-Leaderboard
```

### Key code paths per task

| Task | Dataset class | Prompt built in | Evaluator / emitter |
|------|--------------|-----------------|---------------------|
| VQA | `vantage_vqa.py` → `VANTAGE_VQA` | `build_prompt()` | `evaluate()` → `adapter_vqa.py` |
| Temporal | `vantage_temporal.py` → `VANTAGE_Temporal` | `build_prompt()` | `evaluate()` → `adapter_temporal.py` |
| DVC | `vantage_dvc.py` → `VANTAGE_DVC` | `build_prompt()` | `evaluate()` → `adapter_dvc.py` |
| EventVerification | `vantage_event_verification.py` → `VANTAGE_EventVerification` | `build_prompt()` | `evaluate()` → `adapter_event_verification.py` |
| SOT | `vantage_sot.py` → `VANTAGE_SOT` | `build_prompt()` | `evaluate()` → `adapter_sot.py` |
| 2DGrounding | `vantage2d/grounding_2d_dataset.py` | `build_prompt()` | `evaluate()` → `adapter_grounding.py` |
| 2DPointing | `vantage2d/pointing_dataset.py` | inherited MCQ | `evaluate()` → `adapter_pointing.py` |
| Astro2D | `vantage2d/astro_2d_dataset.py` | `build_prompt()` | `evaluate()` → `adapter_astro.py` |

---

## Benchmarks

VANTAGE covers eight tasks across video and image modalities. Each benchmark is independently runnable.

### Video Benchmarks

| Benchmark | Task | Primary Metrics | Dataset key (example) |
|-----------|------|-----------------|----------------------|
| **VANTAGE-VQA** | Multiple-choice video question answering | Accuracy | `VANTAGE_VQA_8frame` |
| **VANTAGE-Temporal** | Temporal event localization | mIoU, Precision@0.5 | `VANTAGE_Temporal_8frame` |
| **VANTAGE-DVC** | Dense video captioning | SODA-c, mIoU, IoU-F1, BERTScore-F1 | `VANTAGE_DVC_8frame` |
| **VANTAGE-EventVerification** | Binary event physics verification (Yes/No) | Macro F1, Accuracy, Balanced Accuracy | `VANTAGE_EventVerification_8frame` |
| **VANTAGE-SOT** | Single-object tracking across frames | Success AUC, Mean IoU, Precision@0.5 | `VANTAGE_SOT` |

### Image Benchmarks

| Benchmark | Task | Primary Metrics | Dataset key |
|-----------|------|-----------------|-------------|
| **VANTAGE-2DGrounding** | Referring expression grounding | Acc@0.5, Acc@0.25, Mean IoU | `VANTAGE_2DGrounding` |
| **VANTAGE-2DPointing** | Spatial pointing (multiple-choice) | Accuracy | `VANTAGE_2DPointing` |
| **Astro2D** | Person detection on aerial imagery | mAP, AP50 | `Astro2D` |

All dataset keys and their frame/fps variants are listed in [All Registered Dataset Names](#all-registered-dataset-names).

---

## Installation

```bash
# Python 3.10 or later required
conda create -n vantage python=3.10 -y
conda activate vantage

# ffmpeg is required for VANTAGE-SOT frame extraction
conda install -c conda-forge ffmpeg -y

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: vLLM backend for local model inference
pip install vllm
```

> **Fresh-clone note.** `run.py` imports `MMMU_result_transfer` / `MMTBench_result_transfer`
> from `vlmeval/utils/result_transfer.py` at startup, so that file must be present or the
> script fails to import before any inference runs (it is only exercised by the non-VANTAGE
> `MMMU_TEST` / `MMT-Bench_ALL` datasets). The `.gitignore` `result*` rule is anchored
> (`/result*`, `*.result`) specifically so it does **not** exclude that source file; do not
> revert it to a bare `result*`.

---

## Dataset Setup

### Download from HuggingFace (recommended)

The benchmark data is hosted at [`nvidia/PhysicalAI-VANTAGE-Bench`](https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench). Use the provided prep script to download and reshape it into the layout VLMEvalKit expects:

```bash
# Prepare all eight tasks (symlink mode — disk-efficient)
hf auth login                    # one-time setup; required for SOT data
python scripts/run_lmudata.py --all --lmu-root ~/LMUData
```

To skip the large SOT download (~16 GB), prepare individual tasks:

```bash
python scripts/run_lmudata.py \
  --tasks vqa,event_verification,dvc,temporal,pointing,astro2d,grounding \
  --lmu-root ~/LMUData
```

Full documentation, prerequisites (ffmpeg, gdown), troubleshooting, and advanced options are in **[`scripts/RUN_LMUData.md`](scripts/RUN_LMUData.md)**.

#### Source layout for EventVerification and 2DPointing

The prep script reads these two tasks directly from the public release layout:

- **EventVerification** — annotations and videos are downloaded from
  `data/event_verification/filtered/**`. Annotation files are named
  `test_annotation*.json` and live in **per-group subdirectories**; the item list inside
  each is wrapped under a single (dataset-named) top-level key. Each item's `video` path is
  resolved **relative to its own annotation file's directory** — videos are in nested
  subtrees, not a single flat `videos/` folder. Output video basenames are de-duplicated.
- **2DPointing** — the source is `data/pointing/Vantage2DPointing.tsv`, a TSV already in
  the benchmark schema (read directly with `csv.DictReader`). There is no
  `VANTAGE_2DPointing.jsonl`.

### Local layout

After running the prep script, or if you place data manually, VLMEvalKit looks for data under `$LMUData/datasets/<DatasetName>/`. Override the root with:

```bash
export LMUData=/path/to/your/data
# or pass it directly:
python run.py --lmudata-root /path/to/your/data ...
```

Expected layout:

```
$LMUData/                                      # default: ~/LMUData
└── datasets/
    ├── VANTAGE_VQA/
    │   ├── VANTAGE_VQA.tsv
    │   └── videos/
    ├── VANTAGE_Temporal/
    │   ├── VANTAGE_Temporal.tsv
    │   └── videos/
    ├── VANTAGE_DVC/
    │   ├── VANTAGE_DVC.tsv
    │   └── videos/
    ├── VANTAGE_EventVerification/
    │   ├── VANTAGE_EventVerification.tsv
    │   └── videos/
    ├── VANTAGE_SOT/
    │   └── <seq_name>/                # one directory per sequence: gt.json + frames/
    ├── VANTAGE_2DGrounding/
    │   ├── images/
    │   └── annotations.json
    ├── VANTAGE_2DPointing/
    │   ├── VANTAGE_2DPointing.tsv
    │   └── images_annotated/
    └── Astro2D/
        ├── images/
        └── labels/
```

### S3 fallback (internal use only)

For environments with access to private S3-compatible storage, the dataset classes can fall back to downloading from S3 if the local directory is absent. Set these variables before running:

| Variable | Default | Description |
|----------|---------|-------------|
| `VANTAGE_S3_PROFILE` | `default` | AWS credentials profile in `~/.aws/credentials` |
| `VANTAGE_S3_REGION` | — | AWS region override |
| `VANTAGE_S3_ENDPOINT_URL` | — | S3-compatible endpoint |
| `VANTAGE_S3_DOWNLOAD_WORKERS` | `8` | Parallel download threads |

> **Note:** The S3 bucket is not publicly accessible. External users should use the HuggingFace download path above.

---

## Running Evaluations

All commands run from the repository root. Replace `<ModelName>` with any key from `supported_VLM` in `vlmeval/config.py`.

### Run inference + evaluation together

```bash
python run.py --data VANTAGE_VQA_8frame --model <ModelName> --verbose
```

### Run inference only (no evaluation)

```bash
python run.py --data VANTAGE_VQA_8frame --model <ModelName> --mode infer --work-dir ./outputs
```

### Run evaluation only (from existing prediction file)

Prediction files do **not** need to contain ground-truth columns — the evaluator resolves GT from the dataset TSV at evaluation time.

```bash
python run.py --data VANTAGE_VQA_8frame --model <ModelName> --mode eval --reuse --work-dir ./outputs
```

### Run multiple benchmarks at once

```bash
python run.py \
  --data VANTAGE_VQA_8frame VANTAGE_Temporal_8frame VANTAGE_DVC_8frame VANTAGE_EventVerification_8frame \
  --model <ModelName> \
  --verbose
```

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--work-dir <path>` | `./outputs` | Directory for all output files |
| `--lmudata-root <path>` | `$LMUData` or `~/LMUData` | Override the dataset root directory |
| `--reuse` | off | Reuse an existing prediction file; skip inference |
| `--mode infer` | `all` | Inference only |
| `--mode eval` | `all` | Evaluation only (requires existing prediction file) |
| `--api-nproc 8` | `4` | Parallel threads for API model calls |
| `--retry 5` | model default | Retry count for failed API calls |
| `--verbose` | off | Verbose logging |

---

## All Registered Dataset Names

Pass any of these strings as the `--data` argument.

### VANTAGE-VQA

| Key | Sampling |
|-----|----------|
| `VANTAGE_VQA_8frame` | 8 frames uniformly sampled |
| `VANTAGE_VQA_16frame` | 16 frames |
| `VANTAGE_VQA_64frame` | 64 frames |
| `VANTAGE_VQA_4fps` | 4 frames per second |
| `VANTAGE_VQA_1fps` | 1 frame per second |
| `VANTAGE_VQA_0.5fps` | 0.5 fps |
| `VANTAGE_VQA_8frame_200` | 8 frames, 200-sample subset (seed 42) |

### VANTAGE-Temporal

| Key | Sampling |
|-----|----------|
| `VANTAGE_Temporal_8frame` | 8 frames |
| `VANTAGE_Temporal_16frame` | 16 frames |
| `VANTAGE_Temporal_64frame` | 64 frames |
| `VANTAGE_Temporal_1fps` | 1 fps |
| `VANTAGE_Temporal_0.5fps` | 0.5 fps |
| `VANTAGE_Temporal_10fps` | 10 fps |

### VANTAGE-DVC

| Key | Sampling |
|-----|----------|
| `VANTAGE_DVC_8frame` | 8 frames |
| `VANTAGE_DVC_64frame` | 64 frames |
| `VANTAGE_DVC_1fps` | 1 fps |
| `VANTAGE_DVC_2fps` | 2 fps |
| `VANTAGE_DVC_4fps` | 4 fps |

### VANTAGE-EventVerification

| Key | Sampling |
|-----|----------|
| `VANTAGE_EventVerification_8frame` | 8 frames |
| `VANTAGE_EventVerification_16frame` | 16 frames |
| `VANTAGE_EventVerification_1fps` | 1 fps |
| `VANTAGE_EventVerification_4fps` | 4 fps |

> **Note:** The EventVerification class defaults to `fps=4`. All registered variants override this with `fps=0` when using frame-count-based sampling. If you instantiate the class directly, pass `fps=0` alongside `nframe` to avoid unexpected behavior.

### VANTAGE-SOT

| Key | Notes |
|-----|-------|
| `VANTAGE_SOT` | Default: 8 frames, stride 15 |
| `VANTAGE_SOT_16f` | 16 frames |
| `VANTAGE_SOT_32f` | 32 frames |

### Image benchmarks

| Key | Class | Task | Submit? |
|-----|-------|------|---------|
| `VANTAGE_2DGrounding` | `VANTAGE_2DGroundingDataset` | Referring expression grounding | ✓ |
| `VANTAGE_2DPointing` | `VANTAGE_2DPointing` | Spatial pointing MCQ | ✓ |
| `Astro2D` | `Astro2DDetectionDataset` | Person detection, aerial imagery | ✓ |
| `VANTAGE_2DGrounding_val` | `VANTAGE_2DGroundingDataset` | Grounding — validation split | dev only |
| `VANTAGE_2DGrounding_small` | `VANTAGE_2DGroundingDataset` | Grounding — small debug subset | dev only |

---

## Submission Workflow

Submit at: **https://vantage-bench.org/submit** — Limits: 2 per day · 30 lifetime per email.

> **Ground truth is withheld from the public dataset.** Scoring is server-side; you cannot compute leaderboard metrics locally.

### Pillars

VANTAGE-Bench is organized into four pillars. **You must submit all tasks within a pillar** — partial-pillar submissions are rejected. Submit any combination of complete pillars.

| Pillar | Name | Tasks | Primary metric |
|--------|------|-------|----------------|
| **I** | Semantic | Event Verification, Video QA | Macro F1, Accuracy |
| **II** | Spatial | Referring Expressions, Spatial Pointing, Object Localization (Astro2D) | mIoU, Accuracy, F1@0.5 |
| **III** | Temporal | Temporal Localization, Dense Video Captioning | mIoU, SODAc |
| **IV** | Spatio-Temporal | Single Object Tracking | Success AUC |

### Step 1 — Run inference + evaluation

Submission JSONL files are written during the **evaluation phase**, not inference-only. Use the default mode (`--mode all`) to run both in one step:

```bash
python run.py \
  --data VANTAGE_VQA_8frame VANTAGE_EventVerification_8frame \
         VANTAGE_Temporal_8frame VANTAGE_DVC_8frame VANTAGE_SOT \
         VANTAGE_2DGrounding VANTAGE_2DPointing Astro2D \
  --model <YourModel> --work-dir ./outputs
```

Each task produces a `*_submission.jsonl` alongside its prediction xlsx. If you already ran inference with `--mode infer`, add `--mode eval --reuse` instead of re-running inference.

### Step 2 — Package into a `.tar.gz`

The portal requires **one `.tar.gz` containing one `.jsonl` per task**:

```bash
python scripts/package_submission.py \
  --work-dir ./outputs/<model>/<eval_id> \
  --out submission.tar.gz
```

The script collects submission files, renames them to canonical task names (`vqa.jsonl`, `temporal.jsonl`, …), prints pillar coverage, and writes the archive.

### Step 3 — Upload

Go to **https://vantage-bench.org/submit**, complete the form (identity, model config, inference setup, pillars), and upload `submission.tar.gz` (max 500 MB). Scores arrive by email.

Quick reference: [`SUBMISSION.md`](SUBMISSION.md) · Full details and JSONL format: [`docs/vantage/SUBMISSION.md`](docs/vantage/SUBMISSION.md).

---

## Model Backends

VANTAGE benchmarks work with any model supported by VLMEvalKit. Three backends are available:

### 1. API model (OpenAI-compatible endpoint)

Set the endpoint and key via environment variables:

```bash
export OPENAI_API_BASE=https://your-endpoint/v1/chat/completions
export OPENAI_API_KEY=your-key
```

Then run with any API-backed model name from `vlmeval/config.py`:

```bash
python run.py --data VANTAGE_VQA_8frame --model <ApiModelName>
```

### 2. Local HuggingFace model

```bash
python run.py --data VANTAGE_VQA_8frame --model <HFModelName>
```

Model weights are loaded from HuggingFace Hub by default. Set `HF_HUB_CACHE` to control the local cache directory.

### 3. Local vLLM model (multi-GPU)

Use a config file to pass `use_vllm` and `tensor_parallel_size`:

```json
{
    "model": {
        "MyModel-4gpu": {
            "class": "<VLMClassName>",
            "model_path": "<hf-model-id>",
            "use_vllm": true,
            "tensor_parallel_size": 4
        }
    },
    "data": {
        "VANTAGE_VQA_8frame": {}
    }
}
```

```bash
python run.py --config my_config.json
```

To list all registered model names:

```bash
python -c "from vlmeval.config import supported_VLM; print(list(supported_VLM.keys()))"
```

---

## Output Structure

`<eval_id>` is a run stamp in the format `T<YYYYMMDD>_G<8-char-git-hash>` (e.g. `T20250614_Gabc12345`). Symlinks to the latest run's files appear directly under `<model_name>/`.

```
./outputs/
└── <model_name>/
    ├── <model>_VANTAGE_VQA_8frame.xlsx              ← symlink to latest run
    ├── <model>_VANTAGE_VQA_8frame_submission.jsonl  ← symlink to latest run
    └── T<YYYYMMDD>_G<hash>/                         ← timestamped run folder
        ├── <model>_VANTAGE_VQA_8frame.xlsx              # raw predictions
        ├── <model>_VANTAGE_VQA_8frame_submission.jsonl  # bundle this for upload
        ├── <model>_VANTAGE_Temporal_8frame.xlsx
        ├── <model>_VANTAGE_Temporal_8frame_submission.jsonl
        ├── <model>_VANTAGE_DVC_8frame.xlsx
        ├── <model>_VANTAGE_DVC_8frame_submission.jsonl
        ├── model_config.txt                             # model __dict__ dump
        └── VANTAGE_VQA_8frame_config.json               # dataset config dump
```

> **Note:** VANTAGE public tasks do **not** produce local metric files (`_acc.csv`, `_metrics.json`) because ground truth is withheld from the public dataset. The `*_submission.jsonl` files are what you package and upload for server-side scoring.

Override the output root with `--work-dir` or the `MMEVAL_ROOT` environment variable.

---

## Prediction File Schemas

Ground truth is always resolved from the dataset TSV at evaluation time. Prediction files only need to contain the model's raw outputs alongside an identifier column.

| Benchmark | Required columns | GT resolution |
|-----------|-----------------|---------------|
| VANTAGE-VQA | `index`, `prediction` | GT resolved from dataset TSV by `index` |
| VANTAGE-Temporal | `index`, `prediction` | GT spans resolved by `index` |
| VANTAGE-DVC | `index`, `prediction` | GT events resolved by `index` |
| VANTAGE-EventVerification | `prediction` + one of: `index`, `id`, or `video` | GT resolved in that priority order |
| VANTAGE-SOT | `index`, `prediction` | GT track metadata from SOT cache |
| VANTAGE-2DGrounding | `index`, `prediction` | GT boxes resolved by `index` |
| VANTAGE-2DPointing | `index`, `prediction` | GT answer resolved from dataset TSV by `index` |
| Astro2D | `image_path`, `prediction` | GT loaded from KITTI label files on disk |

The `prediction` column should contain the raw model output string. Evaluators apply task-specific parsers (answer letter extraction, JSON span parsing, bbox parsing) internally.

Full schema details: [docs/vantage/VANTAGEEvalInputs.md](docs/vantage/VANTAGEEvalInputs.md).

---

## Hardware Requirements

Requirements vary by model size and backend.

| Scenario | Minimum GPU memory |
|----------|--------------------|
| API model inference (any size) | None (API calls only) |
| Small VLM local inference (≤7B, HuggingFace) | 16 GB VRAM (1× GPU) |
| Medium VLM local inference (7B–13B, vLLM) | 24 GB VRAM (1× GPU) |
| Large VLM local inference (30B+, vLLM) | 2–4× 40 GB VRAM |

Video benchmarks (VANTAGE-Temporal, VANTAGE-DVC, VANTAGE-SOT) load up to 256 frames per video when using fps-based sampling. Memory usage scales with the number of frames and frame resolution. Use `max_frames` and `total_pixels` parameters to limit memory consumption — pass them via a config file with explicit `nframe`, `max_frames`, and `total_pixels` values.

---

## Repository Layout

```
run.py                                  # main entry point
SUBMISSION.md                           # quick submission reference
README_VANTAGE.md                       # extended reference (config files, edge cases)
prompt_guide.md                         # prompt templates for each task

scripts/
├── run_lmudata.py                      # data download + prep
├── package_submission.py               # bundles *_submission.jsonl → .tar.gz
└── RUN_LMUData.md                      # data prep guide

docs/vantage/
├── SUBMISSION.md                       # full submission guide (JSONL format, IDs)
├── DEVELOPER_GUIDE.md                  # file map, all flags, model registration
└── VANTAGEEvalInputs.md                # prediction file schema reference

vlmeval/
├── config.py                           # supported_VLM dict (model name → class)
├── dataset/
│   ├── vantage_vqa.py                  # VANTAGE-VQA
│   ├── vantage_temporal.py             # VANTAGE-Temporal
│   ├── vantage_dvc.py                  # VANTAGE-DVC
│   ├── vantage_event_verification.py   # VANTAGE-EventVerification
│   ├── vantage_sot.py                  # VANTAGE-SOT
│   ├── vantage2d/
│   │   ├── grounding_2d_dataset.py     # VANTAGE-2DGrounding
│   │   ├── astro_2d_dataset.py         # Astro2D
│   │   ├── pointing_dataset.py         # VANTAGE-2DPointing
│   │   ├── datasets.yaml               # per-dataset path config (image tasks)
│   │   └── utils.py                    # shared bbox / AP helpers
│   ├── utils/vantagebench/             # submission emitter, adapters, ID rules
│   ├── __init__.py                     # dataset registration
│   └── video_dataset_config.py         # video variant registrations
├── vlm/
│   └── <model>.py                      # local model wrappers (HuggingFace / vLLM)
└── api/
    └── <model>.py                      # API wrappers (OpenAI-compatible)
```

---

## Documentation Index

| Document | What it covers |
|----------|---------------|
| [`SUBMISSION.md`](SUBMISSION.md) | Quick submission reference: 3-step flow, pillar table, packaging, form fields |
| [`docs/vantage/SUBMISSION.md`](docs/vantage/SUBMISSION.md) | Full submission guide: JSONL record format, canonical IDs, troubleshooting |
| [`docs/vantage/DEVELOPER_GUIDE.md`](docs/vantage/DEVELOPER_GUIDE.md) | File-to-file map, all CLI flags, all env vars, model registration paths |
| [`configs/README.md`](configs/README.md) | Sample config files for every supported model; GPU/package requirements table |
| [`docs/vantage/VANTAGEEvalInputs.md`](docs/vantage/VANTAGEEvalInputs.md) | Minimum prediction-file columns required by each evaluator |
| [`README_VANTAGE.md`](README_VANTAGE.md) | Extended reference: config files, per-model parameter passing, all dataset keys |
| [`scripts/RUN_LMUData.md`](scripts/RUN_LMUData.md) | Data download guide: prerequisites, per-task flags, troubleshooting |
| [`prompt_guide.md`](prompt_guide.md) | Exact prompt templates used for each benchmark task |

---

## Built on VLMEvalKit

This repository is a fork of **VLMEvalKit** ([open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)), an open-source toolkit for evaluating large vision-language models. VLMEvalKit provides the core infrastructure: dataset base classes, model wrappers, the `run.py` entry point, and evaluation utilities used throughout VANTAGE.

All VLMEvalKit benchmarks and models remain available in this fork. To evaluate any of the 70+ VLMEvalKit benchmarks alongside VANTAGE tasks, refer to the [VLMEvalKit documentation](https://github.com/open-compass/VLMEvalKit).

To add a new model or benchmark to this repository, follow the VLMEvalKit contribution guide: [docs/en/Development.md](docs/en/Development.md).

---

## Citation

If you use VANTAGE-Bench in your research, please cite:

```bibtex
@misc{vantagebench2026,
  title        = {VANTAGE-Bench: A Benchmark for Vision-Language Models on Fixed-Camera Infrastructure AI},
  author       = {{VANTAGE-Bench Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/Clemson-Capstone/VANTAGE-Bench}},
  note         = {Benchmark, dataset, evaluation framework, and public leaderboard. Leaderboard: https://huggingface.co/spaces/clemson-computing/VANTAGE-Bench-Leaderboard}
}
```

If you use the VLMEvalKit infrastructure, please also cite:

```bibtex
@inproceedings{duan2024vlmevalkit,
  title     = {VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models},
  author    = {Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin
               and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi
               and others},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  pages     = {11198--11201},
  year      = {2024}
}
```
