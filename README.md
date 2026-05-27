# VANTAGE Benchmark Suite

**VANTAGE** is a benchmark suite for evaluating large vision-language models (VLMs) on real-world video and image understanding tasks — spanning multiple-choice QA, temporal localization, dense captioning, event verification, single-object tracking, 2D detection, and referring expression grounding.

This repository is a fork of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), an open-source VLM evaluation toolkit. VANTAGE adds new benchmark tasks, dataset loaders, and evaluation metrics on top of that foundation.

---

## Contents

- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running Evaluations](#running-evaluations)
- [Model Backends](#model-backends)
- [Output Structure](#output-structure)
- [Prediction File Schemas](#prediction-file-schemas)
- [Hardware Requirements](#hardware-requirements)
- [Repository Layout](#repository-layout)
- [Built on VLMEvalKit](#built-on-vlmevalkit)
- [Citation](#citation)

---

## Benchmarks

VANTAGE covers nine tasks across video and image modalities. Each benchmark is independently runnable.

### Video Benchmarks

| Benchmark | Task | Primary Metrics | Dataset key (example) |
|-----------|------|-----------------|----------------------|
| **VANTAGE-VQA** | Multiple-choice video question answering | Accuracy | `VANTAGE_VQA_8frame` |
| **VANTAGE-Temporal** | Temporal event localization | mIoU, Precision@0.5 | `VANTAGE_Temporal_8frame` |
| **VANTAGE-DVC** | Dense video captioning | SODA-c, CIDEr, METEOR, IoU | `VANTAGE_DVC_8frame` |
| **VANTAGE-EventVerification** | Binary event physics verification (Yes/No) | Macro F1, Accuracy, Balanced Accuracy | `VANTAGE_EventVerification_8frame` |
| **VANTAGE-SOT** | Single-object tracking across frames | Success AUC, Mean IoU, Precision@0.5 | `VANTAGE_SOT` |

### Image Benchmarks

| Benchmark | Task | Primary Metrics | Dataset key |
|-----------|------|-----------------|-------------|
| **VANTAGE-2DDetection** | Object detection (KITTI format) | mAP, AP50 | `VANTAGE_2DDetection` |
| **VANTAGE-2DGrounding** | Referring expression grounding | Acc@0.5, Acc@0.25, Mean IoU | `VANTAGE_2DGrounding` |
| **VANTAGE-2DPointing** | Spatial pointing (multiple-choice) | Accuracy | `VANTAGE_2DPointing` |
| **Astro2D** | Person detection on aerial imagery | mAP, AP50 | `Astro2D` |

All dataset keys and their frame/fps variants are listed in [All Registered Dataset Names](#all-registered-dataset-names).

---

## Installation

```bash
# Python 3.10+ recommended
conda create -n vantage python=3.10 -y
conda activate vantage

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: vLLM backend for local model inference
pip install vllm
```

---

## Dataset Setup

### Local layout

Each dataset is resolved from a local directory first. Place data under `$LMUDataRoot/datasets/<DatasetName>/`:

```
$LMUDataRoot/                          # default: ~/LMUData
└── datasets/
    ├── VANTAGE_VQA/
    │   ├── VANTAGE_VQA.tsv            # annotation file (index, video, question, answer, ...)
    │   └── videos/
    │       ├── <video_id>.mp4
    │       └── ...
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
    ├── VANTAGE_2DDetection/
    │   ├── images/
    │   └── labels/                    # KITTI-format label files
    ├── VANTAGE_2DGrounding/
    │   ├── images/
    │   └── annotations/
    ├── VANTAGE_2DPointing/
    │   ├── VANTAGE_2DPointing.tsv
    │   └── images/
    └── Astro2D/
        ├── images/
        └── labels/
```

Override the root directory with the environment variable:

```bash
export LMUDataRoot=/path/to/your/data
```

### S3 download (optional)

For datasets hosted on S3-compatible storage, set:

| Variable | Default | Description |
|----------|---------|-------------|
| `VANTAGE_S3_PROFILE` | `default` | AWS credentials profile in `~/.aws/credentials` |
| `VANTAGE_S3_REGION` | — | AWS region override |
| `VANTAGE_S3_ENDPOINT_URL` | — | S3-compatible endpoint (omit for standard AWS S3) |
| `VANTAGE_S3_DOWNLOAD_WORKERS` | `8` | Parallel download threads |

When S3 is configured, the dataset classes fall back to downloading from S3 automatically if the local directory is absent or incomplete.

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

> **Note:** The EventVerification class defaults to `fps=4`. All registered variants override this with `fps=0` when using frame-count-based sampling. If you instantiate the class directly, pass `fps=0` alongside `nframe` to avoid unexpected behavior.

### VANTAGE-SOT

| Key | Notes |
|-----|-------|
| `VANTAGE_SOT` | Default: 8 frames, stride 15 |
| `VANTAGE_SOT_16f` | 16 frames |
| `VANTAGE_SOT_32f` | 32 frames |
| `VANTAGE_SOT_tiny` | Small validation subset |

### Image benchmarks

Image benchmarks are registered via their `supported_datasets()` class methods and the `IMAGE_DATASET` list in `vlmeval/dataset/__init__.py`. The following names are available:

| Key | Class | Task |
|-----|-------|------|
| `VANTAGE_2DDetection` | `VANTAGE_2DDetectionDataset` | Object detection (KITTI format) |
| `VANTAGE_2DGrounding` | `VANTAGE_2DGroundingDataset` | Referring expression grounding |
| `VANTAGE_2DGrounding_val` | `VANTAGE_2DGroundingDataset` | Grounding (validation split) |
| `VANTAGE_2DPointing` | `VANTAGE_2DPointing` | Spatial pointing MCQ |
| `Astro2D` | `Astro2DDetectionDataset` | Person detection, aerial imagery |

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

```
./outputs/
└── <model_name>/
    └── <eval_id>/
        ├── <model>_VANTAGE_VQA_8frame.xlsx             # raw predictions
        ├── <model>_VANTAGE_VQA_8frame_acc.csv          # per-sample accuracy
        ├── <model>_VANTAGE_Temporal_8frame.xlsx
        ├── <model>_VANTAGE_Temporal_8frame_metrics.json
        ├── <model>_VANTAGE_DVC_8frame.xlsx
        ├── <model>_VANTAGE_DVC_8frame_metrics.json
        ├── <model>_VANTAGE_EventVerification_8frame.xlsx
        ├── <model>_VANTAGE_EventVerification_8frame_acc.json
        └── <model>_VANTAGE_SOT.xlsx
```

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
| Astro2D | `image_path`, `prediction` | GT loaded from KITTI label files on disk |
| VANTAGE-2DGrounding | `index`, `prediction` | GT boxes resolved by `index` |

The `prediction` column should contain the raw model output string. Evaluators apply task-specific parsers (answer letter extraction, JSON span parsing, bbox parsing) internally.

Full schema details: [docs/en/VANTAGEEvalInputs.md](docs/en/VANTAGEEvalInputs.md).

---

## Hardware Requirements

Requirements vary by model size and backend.

| Scenario | Minimum GPU memory |
|----------|--------------------|
| API model inference (any size) | None (API calls only) |
| Small VLM local inference (≤7B, HuggingFace) | 16 GB VRAM (1× GPU) |
| Medium VLM local inference (7B–13B, vLLM) | 24 GB VRAM (1× GPU) |
| Large VLM local inference (30B+, vLLM) | 2–4× 40 GB VRAM |

Video benchmarks (VANTAGE-Temporal, VANTAGE-DVC, VANTAGE-SOT) load up to 256 frames per video when using fps-based sampling. Memory usage scales with the number of frames and frame resolution. Use `max_frames` and `total_pixels` parameters to limit memory consumption.

```bash
# Example: limit frame count and pixel budget for a memory-constrained setup
python run.py \
  --data VANTAGE_Temporal_8frame \
  --model <ModelName> \
  --verbose
```

For custom limits, use a config file with explicit `nframe`, `max_frames`, and `total_pixels` parameters.

---

## Repository Layout

```
vlmeval/
├── dataset/
│   ├── vantage_vqa.py                  # VANTAGE-VQA
│   ├── vantage_temporal.py             # VANTAGE-Temporal
│   ├── vantage_dvc.py                  # VANTAGE-DVC
│   ├── vantage_event_verification.py   # VANTAGE-EventVerification
│   ├── vantage_sot.py                  # VANTAGE-SOT
│   ├── vantage2d/
│   │   ├── detection_2d_dataset.py     # VANTAGE-2DDetection
│   │   ├── grounding_2d_dataset.py     # VANTAGE-2DGrounding
│   │   ├── astro_2d_dataset.py         # Astro2D
│   │   ├── datasets.yaml               # per-dataset path config
│   │   └── utils.py                    # shared bbox / AP helpers
│   ├── __init__.py                     # dataset registration
│   └── video_dataset_config.py         # video variant registrations
├── vlm/
│   └── <model>.py                       # local vLLM wrapper for models
├── api/
│   └── <model>.py                # API wrapper for OpenAI-compatible endpoints
└── config.py                           # supported_VLM dict (model name → class)

run.py                                  # main entry point
README_VANTAGE.md                       # extended reference (config files, edge cases)
docs/en/VANTAGEEvalInputs.md           # prediction file schema reference
```

---

## Built on VLMEvalKit

This repository is a fork of **VLMEvalKit** ([open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)), an open-source toolkit for evaluating large vision-language models. VLMEvalKit provides the core infrastructure: dataset base classes, model wrappers, the `run.py` entry point, and evaluation utilities used throughout VANTAGE.

All VLMEvalKit benchmarks and models remain available in this fork. To evaluate any of the 70+ VLMEvalKit benchmarks alongside VANTAGE tasks, refer to the [VLMEvalKit documentation](https://github.com/open-compass/VLMEvalKit).

To add a new model or benchmark to this repository, follow the VLMEvalKit contribution guide: [docs/en/Development.md](docs/en/Development.md).

---

## Citation

If you use VANTAGE in your research, please cite this work. If you use the VLMEvalKit infrastructure, please also cite the VLMEvalKit paper.

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
