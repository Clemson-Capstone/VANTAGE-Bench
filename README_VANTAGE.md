# VANTAGE Benchmark Suite — Complete Reference

This document is the single source of truth for running, configuring, and extending the VANTAGE benchmarks inside this VLMEvalKit fork.

---

## Table of Contents

1. [Where VANTAGE code lives](#1-where-vantage-code-lives)
2. [Setup](#2-setup)
3. [Environment variables](#3-environment-variables)
4. [All registered dataset names](#4-all-registered-dataset-names)
5. [Exact run commands — CLI mode](#5-exact-run-commands--cli-mode)
6. [Config-file mode](#6-config-file-mode)
7. [Passing parameters to models](#7-passing-parameters-to-models)
8. [Inference-only and eval-only modes](#8-inference-only-and-eval-only-modes)
9. [Output structure](#9-output-structure)
10. [Data directory layout](#10-data-directory-layout)
11. [Prediction file schema (eval inputs)](#11-prediction-file-schema-eval-inputs)
12. [Per-model requirements](#12-per-model-requirements)

---

## 1. Where VANTAGE code lives

```
vlmeval/
├── dataset/
│   ├── vantage_vqa.py                  # VANTAGE_VQA — multi-choice video QA
│   ├── vantage_temporal.py             # VANTAGE_Temporal — temporal event localization
│   ├── vantage_dvc.py                  # VANTAGE_DVC — dense video captioning
│   ├── vantage_event_verification.py   # VANTAGE_EventVerification — event physics QA
│   ├── vantage_sot.py                  # VANTAGE_SOT — single-object tracking
│   ├── vantage2d/
│   │   ├── __init__.py
│   │   ├── grounding_2d_dataset.py     # VANTAGE_2DGroundingDataset (image)
│   │   ├── astro_2d_dataset.py         # Astro2DDetectionDataset (image)
│   │   ├── pointing_dataset.py         # VANTAGE_2DPointing (image)
│   │   ├── datasets.yaml               # per-dataset config (classes, S3 paths)
│   │   └── utils.py                    # shared bbox / AP helpers
│   ├── __init__.py                     # imports + IMAGE_DATASET / VIDEO_DATASET lists
│   └── video_dataset_config.py         # all video variant registrations
├── vlm/                                # local model wrappers (HuggingFace / vLLM)
├── api/                                # API model wrappers (OpenAI-compatible)
└── config.py                           # supported_VLM dict — model name → class
```

Key lookup path for dataset names:

```
video_dataset_config.py  →  supported_video_datasets dict  →  build_dataset()
__init__.py              →  IMAGE_DATASET list              →  build_dataset()
```

Supported model names are defined in `vlmeval/config.py`. To list all registered model names:

```bash
python -c "from vlmeval.config import supported_VLM; print(list(supported_VLM.keys()))"
# or, if vlmeval is installed:
vlmutil mlist all
```

---

## 2. Setup

```bash
# 1. Create environment (Python 3.10+)
conda create -n vantage python=3.10 -y
conda activate vantage

# 2. Install ffmpeg (required for VANTAGE-SOT frame extraction)
conda install -c conda-forge ffmpeg -y

# 3. Install dependencies
cd /path/to/VANTAGE-Bench
pip install -r requirements.txt
pip install -e .

# 4. (Optional) vLLM backend for local inference
pip install vllm
```

---

## 3. Environment variables

### S3 dataset download (internal use only)

The S3 bucket is not publicly accessible. External users should download data via `scripts/run_lmudata.py` (see [`scripts/RUN_LMUData.md`](scripts/RUN_LMUData.md)). These variables are only relevant for internal NVIDIA infrastructure where data is not already present under `$LMUData/datasets/<DatasetName>/`.

| Variable | Default | Description |
|----------|---------|-------------|
| `VANTAGE_S3_PROFILE` | `default` | AWS credentials profile in `~/.aws/credentials` |
| `VANTAGE_S3_REGION` | — | AWS region override |
| `VANTAGE_S3_ENDPOINT_URL` | — | S3-compatible endpoint (omit for standard AWS) |
| `VANTAGE_S3_DOWNLOAD_WORKERS` | `8` | Parallel download threads |

### API model inference

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Bearer token for OpenAI or OpenAI-compatible API endpoints |
| `OPENAI_API_BASE` | Endpoint URL override (default: OpenAI). Set for self-hosted or third-party servers. |

Most API wrappers in `vlmeval/api/` read these standard variables. Per-model env var names may differ; check the relevant wrapper class for details.

### Work directory

| Variable | Description |
|----------|-------------|
| `MMEVAL_ROOT` | Override `--work-dir`; outputs are written here when set |
| `LMUData` | Root for cached dataset TSVs and media files (default: `~/LMUData`) |

You can also override the data root at the CLI level without setting an env var:

```bash
python run.py --lmudata-root /path/to/data --data VANTAGE_VQA_8frame --model <ModelName>
```

---

## 4. All registered dataset names

Pass any of these strings as the `--data` argument or as a key in the config JSON `data` block.

### Video datasets (registered in `video_dataset_config.py`)

#### VANTAGE_VQA — multi-choice video question answering

| Dataset key | Sampling |
|-------------|----------|
| `VANTAGE_VQA_8frame` | 8 frames uniformly sampled |
| `VANTAGE_VQA_16frame` | 16 frames |
| `VANTAGE_VQA_64frame` | 64 frames |
| `VANTAGE_VQA_4fps` | 4 fps (up to 256 frames, pixel-capped) |
| `VANTAGE_VQA_1fps` | 1 frame per second |
| `VANTAGE_VQA_0.5fps` | 0.5 fps |
| `VANTAGE_VQA_8frame_200` | 8 frames, 200-sample subset (seed 42) |

#### VANTAGE_Temporal — temporal event localization

| Dataset key | Sampling |
|-------------|----------|
| `VANTAGE_Temporal_8frame` | 8 frames |
| `VANTAGE_Temporal_16frame` | 16 frames |
| `VANTAGE_Temporal_64frame` | 64 frames |
| `VANTAGE_Temporal_1fps` | 1 fps |
| `VANTAGE_Temporal_0.5fps` | 0.5 fps |
| `VANTAGE_Temporal_10fps` | 10 fps |

#### VANTAGE_DVC — dense video captioning

| Dataset key | Sampling |
|-------------|----------|
| `VANTAGE_DVC_8frame` | 8 frames |
| `VANTAGE_DVC_64frame` | 64 frames |
| `VANTAGE_DVC_1fps` | 1 fps |
| `VANTAGE_DVC_2fps` | 2 fps |
| `VANTAGE_DVC_4fps` | 4 fps |

#### VANTAGE_EventVerification — event physics yes/no

| Dataset key | Sampling |
|-------------|----------|
| `VANTAGE_EventVerification_8frame` | 8 frames (fps override=0 to force nframe mode) |
| `VANTAGE_EventVerification_16frame` | 16 frames |
| `VANTAGE_EventVerification_1fps` | 1 fps |
| `VANTAGE_EventVerification_4fps` | 4 fps |

#### VANTAGE_SOT — single-object tracking

| Dataset key | Notes |
|-------------|-------|
| `VANTAGE_SOT` | default 8 frames, stride 15 |
| `VANTAGE_SOT_16f` | 16 frames |
| `VANTAGE_SOT_32f` | 32 frames |

### Image datasets (registered in `__init__.py → IMAGE_DATASET`)

| Dataset class | Registered name(s) | Task | Submit? |
|---------------|--------------------|------|---------|
| `VANTAGE_2DGroundingDataset` | `VANTAGE_2DGrounding` | Referring expression grounding | ✓ |
| `VANTAGE_2DGroundingDataset` | `VANTAGE_2DGrounding_val` | Grounding — validation split | dev only |
| `VANTAGE_2DGroundingDataset` | `VANTAGE_2DGrounding_small` | Grounding — small debug subset | dev only |
| `Astro2DDetectionDataset` | `Astro2D` | Astro 2D person detection | ✓ |
| `VANTAGE_2DPointing` | `VANTAGE_2DPointing` | Spatial pointing MCQ | ✓ |

---

## 5. Exact run commands — CLI mode

Run all commands from the repo root with your conda environment active. Replace `<ModelName>` with any key from `supported_VLM` in `vlmeval/config.py`.

### VANTAGE VQA

```bash
python run.py \
  --data VANTAGE_VQA_8frame \
  --model <ModelName> \
  --verbose
```

### VANTAGE Temporal

```bash
python run.py \
  --data VANTAGE_Temporal_8frame \
  --model <ModelName> \
  --verbose
```

### VANTAGE DVC

```bash
python run.py \
  --data VANTAGE_DVC_8frame \
  --model <ModelName> \
  --verbose
```

### VANTAGE EventVerification

```bash
python run.py \
  --data VANTAGE_EventVerification_8frame \
  --model <ModelName> \
  --verbose
```

### VANTAGE SOT

```bash
python run.py \
  --data VANTAGE_SOT \
  --model <ModelName> \
  --verbose
```

### Run multiple benchmarks at once

```bash
python run.py \
  --data VANTAGE_VQA_8frame VANTAGE_Temporal_8frame VANTAGE_DVC_8frame \
  --model <ModelName> \
  --verbose
```

### Common optional flags

| Flag | Default | Effect |
|------|---------|--------|
| `--work-dir ./my_results` | `./outputs` | Write all outputs here |
| `--reuse` | off | Reuse an existing prediction file; skip inference |
| `--mode infer` | `all` | Run inference only, skip evaluation |
| `--mode eval` | `all` | Run evaluation only (requires `--reuse` or existing file) |
| `--api-nproc 8` | `4` | Parallel threads for API calls |
| `--retry 5` | model default | Retry count for API calls |
| `--verbose` | off | Verbose logging |

---

## 6. Config-file mode

Config files give per-model and per-dataset control beyond what CLI flags expose (e.g. `tensor_parallel_size`, custom temperatures, custom `nframe` values).

### Invoking with a config file

```bash
# Named flag
python run.py --config path/to/my_config.json

# Positional shortcut (identical effect)
python run.py path/to/my_config.json
```

### Config file format

```json
{
    "model": {
        "<run-label>": {
            "class": "<ClassName in vlmeval.api or vlmeval.vlm>",
            "<param>": "<value>"
        }
    },
    "data": {
        "<run-label>": {
            "class": "<ClassName in vlmeval.dataset>",
            "dataset": "<dataset string accepted by the class>",
            "<param>": "<value>"
        }
    }
}
```

- The **run-label** (the dict key) becomes the filename stem for output files — choose something descriptive.
- For **model**, if the run-label matches a key in `supported_VLM` (`vlmeval/config.py`), you can pass `{}` to use the default configuration.
- For **data**, if the run-label matches a key in `supported_video_datasets` (`video_dataset_config.py`), you can pass `{}` to use the default configuration.
- `--data` / `--model` CLI flags must **not** be used alongside `--config`.

### VANTAGE config example

```json
{
    "model": {
        "MyModel_default": {},

        "MyModel_custom": {
            "class": "GPT4V",
            "model": "gpt-4o",
            "temperature": 0,
            "img_detail": "high"
        },

        "MyLocalModel_vllm": {
            "class": "LLaVANextVideo",
            "model_path": "lmms-lab/LLaVA-NeXT-Video-7B-DPO",
            "use_vllm": true,
            "tensor_parallel_size": 2
        }
    },
    "data": {
        "VANTAGE_VQA_8frame": {},

        "VANTAGE_VQA_16frame": {},

        "VANTAGE_Temporal_custom_32f": {
            "class": "VANTAGE_Temporal",
            "dataset": "VANTAGE_Temporal",
            "nframe": 32,
            "total_pixels": 8192,
            "max_frames": 256
        },

        "VANTAGE_DVC_2fps": {},

        "VANTAGE_EventVerification_custom": {
            "class": "VANTAGE_EventVerification",
            "dataset": "VANTAGE_EventVerification",
            "nframe": 32,
            "fps": 0,
            "total_pixels": 8192,
            "max_frames": 256
        }
    }
}
```

### Notes on fps vs nframe

- `fps` and `nframe` are mutually exclusive. Setting both raises `ValueError`.
- For `VANTAGE_EventVerification`, the class defaults to `fps=4`; pass `fps=0` when using `nframe` to override that default (as done in all registered variants).

---

## 7. Passing parameters to models

### Option A: CLI flags (apply globally to all models in the run)

```bash
python run.py \
  --data VANTAGE_VQA_8frame \
  --model <ModelName> \
  --retry 5 \
  --api-nproc 8 \
  --verbose
```

`--retry` and `--verbose` propagate to all API model wrappers whose constructors accept those kwargs.

### Option B: Config file (per-model, full control)

```json
{
    "model": {
        "MyModel": {
            "class": "GPT4V",
            "model": "gpt-4o",
            "temperature": 0,
            "retry": 15,
            "timeout": 300
        }
    },
    "data": { "VANTAGE_VQA_8frame": {} }
}
```

### Option C: Use a pre-registered name as a shortcut

Any name defined in `supported_VLM` inside `vlmeval/config.py` can be used directly without a config file:

```bash
python run.py --data VANTAGE_VQA_8frame --model <RegisteredModelName>
```

### Three model backends

VLMEvalKit supports three ways to run a model:

| Backend | When to use | Typical kwargs |
|---------|-------------|----------------|
| **Local HuggingFace** | Small models, no vLLM | `model_path`, `torch_dtype` |
| **Local vLLM** | Large/multi-GPU models | `model_path`, `use_vllm=true`, `tensor_parallel_size` |
| **API (OpenAI-compat)** | Hosted endpoints | `model`, `api_base`, `api_key`, `temperature`, `timeout` |

For **multi-GPU vLLM**, pass `tensor_parallel_size` via config file:

```json
{
    "model": {
        "MyModel-vllm-4gpu": {
            "class": "<VLMClass>",
            "model_path": "<hf-model-id>",
            "use_vllm": true,
            "tensor_parallel_size": 4
        }
    },
    "data": { "VANTAGE_Temporal_8frame": {} }
}
```

---

## 8. Inference-only and eval-only modes

### Inference only (no evaluation)

```bash
python run.py \
  --data VANTAGE_VQA_8frame \
  --model <ModelName> \
  --mode infer \
  --work-dir ./outputs
```

Writes `./outputs/<model>/<eval_id>/<model>_VANTAGE_VQA_8frame.xlsx` (or `.tsv`).

### Evaluation only (reuse existing prediction file)

```bash
python run.py \
  --data VANTAGE_VQA_8frame \
  --model <ModelName> \
  --mode eval \
  --reuse \
  --work-dir ./outputs
```

Reads the existing prediction file and writes evaluation results alongside it. GT is resolved from the dataset TSV at eval time — the prediction file does **not** need GT columns.

See [docs/vantage/VANTAGEEvalInputs.md](docs/vantage/VANTAGEEvalInputs.md) for the minimum columns required by each evaluator.

---

## 9. Output structure

```
./outputs/
└── <model_name>/
    └── <eval_id>/
        ├── <model>_VANTAGE_VQA_8frame.xlsx          # raw predictions
        ├── <model>_VANTAGE_VQA_8frame_acc.csv        # accuracy results
        ├── <model>_VANTAGE_Temporal_8frame.xlsx
        ├── <model>_VANTAGE_Temporal_8frame_metrics.json
        ├── <model>_VANTAGE_DVC_8frame.xlsx
        ├── <model>_VANTAGE_DVC_8frame_metrics.json
        ├── <model>_VANTAGE_EventVerification_8frame.xlsx
        ├── <model>_VANTAGE_EventVerification_8frame_acc.json
        └── <model>_VANTAGE_SOT.xlsx
```

---

## 10. Data directory layout

Dataset files are loaded from `$LMUData/datasets/<DatasetName>/`. Populate this using `scripts/run_lmudata.py`. Internal NVIDIA infrastructure may additionally fall back to S3 via `VANTAGE_S3_*` env vars, but S3 is not accessible to external users.

Expected local layout for video datasets:

```
$LMUData/datasets/VANTAGE_VQA/
├── VANTAGE_VQA.tsv          # main annotation file
└── videos/
    ├── <video_id_1>.mp4
    └── <video_id_2>.mp4
```

The TSV must contain at minimum: `index`, `video`, `question`, `answer` columns. Additional columns are dataset-specific (e.g. `choices` for VQA, `start_time`/`end_time` for Temporal).

For image datasets, layout is configured in `vlmeval/dataset/vantage2d/datasets.yaml`.

---

## 11. Prediction file schema (eval inputs)

GT is always resolved from the dataset TSV at eval time. Prediction files only need to contain:

| Column | Required | Description |
|--------|----------|-------------|
| `index` | yes | Row identifier matching the dataset TSV |
| `prediction` | yes | Raw model output string |
| `video` | fallback | Used to resolve GT when `index` lookup fails |

Each evaluator's minimum required columns are documented in [docs/vantage/VANTAGEEvalInputs.md](docs/vantage/VANTAGEEvalInputs.md).

---

## 12. Per-model requirements

Each model backend has different install and hardware requirements. Use the table below to pick the right setup before running.

| Model class | Config file | Min GPU VRAM | Required packages | Required env vars |
|---|---|---|---|---|
| `Cosmos` (vLLM) | `configs/cosmos_reason2_8b.json` | 1 × 24 GB | `vllm`, `qwen-vl-utils` | — |
| `Cosmos` (vLLM) | `configs/cosmos_reason2_32b.json` | 2 × 80 GB | `vllm`, `qwen-vl-utils` | — |
| `Cosmos` (vLLM) | `configs/cosmos_reason1_7b.json` | 1 × 16 GB | `vllm`, `qwen-vl-utils` | — |
| `CosmosHF` | `configs/cosmos_reason2_8b_hf.json` | 1 × 16 GB | `qwen-vl-utils` | — |
| `Cosmos3` | `configs/cosmos3_nano.json` | 1 × 8 GB | `transformers_cosmos3`\* | — |
| `Cosmos3vLLM` | (config or CLI key) | 1+ GPU | `vllm`, `qwen-vl-utils` | — |
| `CosmosReason2` (API) | `configs/cosmos_reason2_api.json` | None | `qwen-vl-utils`, `ffmpeg` | `COSMOS_REASON2_API_BASE`, `COSMOS_API_KEY` or `NVIDIA_API_KEY` |
| `GPT4V` | `configs/gpt4o.json` | None | — | `OPENAI_API_KEY` |

\* Install `transformers_cosmos3` from source:
```bash
pip install "transformers_cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3"
```

**Cosmos3-Nano-vLLM note**: `Cosmos3vLLM` requires a pre-converted Qwen3VL checkpoint — the raw HuggingFace model will not work directly. Run the conversion script first:
```bash
python scripts/convert_cosmos3_for_vllm.py --model nvidia/Cosmos3-Nano --output /path/to/converted
```
Then pass the output directory as `model_path` in your config.

**CosmosReason2 API note**: `COSMOS_REASON2_API_BASE` must be a **full endpoint URL** ending in `/v1/chat/completions` (e.g. `https://your-deployment.lepton.run/v1/chat/completions`). `ffmpeg` must be on `PATH` — it is used to encode video frames before sending them to the API.

See [`configs/README.md`](configs/README.md) for ready-to-run config files and hardware notes.
