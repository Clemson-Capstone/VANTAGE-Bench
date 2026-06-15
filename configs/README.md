# Sample Configs

Each JSON file here is a ready-to-use config for `run.py --config`. Every config runs **all eight VANTAGE tasks** at once.

## Usage

```bash
export LMUData=/path/to/data      # where datasets are cached
python run.py --config configs/<model>.json --work-dir ./outputs
```

To run a subset of tasks, remove entries from the `"data"` block.

---

## Available Configs

| File | Model | Backend | GPU | Extra packages | Env vars required |
|---|---|---|---|---|---|
| `cosmos_reason2_8b.json` | Cosmos-Reason2-8B | vLLM | 1 × 24 GB | `vllm`, `qwen-vl-utils` | — |
| `cosmos_reason2_32b.json` | Cosmos-Reason2-32B | vLLM | 2 × 80 GB (or 4 × 40 GB) | `vllm`, `qwen-vl-utils` | — |
| `cosmos_reason1_7b.json` | Cosmos-Reason1-7B | vLLM | 1 × 16 GB | `vllm`, `qwen-vl-utils` | — |
| `cosmos_reason2_8b_hf.json` | Cosmos-Reason2-8B | HuggingFace | 1 × 16 GB | `qwen-vl-utils` | — |
| `cosmos3_nano.json` | Cosmos3-Nano | HuggingFace | 1 × 8 GB | `transformers_cosmos3`\* | — |
| `cosmos_reason2_api.json` | Cosmos-Reason2-8B | API (Lepton/NIM) | None | `qwen-vl-utils`, `ffmpeg` | `COSMOS_REASON2_API_BASE`, `COSMOS_API_KEY` or `NVIDIA_API_KEY` |
| `gpt4o.json` | GPT-4o | OpenAI API | None | — | `OPENAI_API_KEY` |

\* `transformers_cosmos3` is installed from source:
```bash
pip install "transformers_cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3"
```

---

## Config File Format

```jsonc
{
    "_comment": "ignored by the parser — human-readable note",
    "model": {
        "<display-name>": {
            "class": "<VLM class from vlmeval/vlm/ or vlmeval/api/>",
            // ... constructor kwargs passed directly to the class
        }
    },
    "data": {
        // Video tasks: empty dict — loaded from supported_video_datasets registry
        "VANTAGE_VQA_8frame": {},
        // Image tasks: must specify class (not in video registry)
        "VANTAGE_2DGrounding": {
            "class": "VANTAGE_2DGroundingDataset",
            "dataset": "VANTAGE_2DGrounding"
        }
    }
}
```

**Why image tasks need `"class"`**: video-task names are in the `supported_video_datasets` registry and can be instantiated with `{}`. Image datasets (`VANTAGE_2DGrounding`, `VANTAGE_2DPointing`, `Astro2D`) are not in that registry, so the config must name the Python class explicitly.

---

## Adjusting for Your Hardware

**Fewer GPUs**: lower `tensor_parallel_size` (must divide evenly into the model's attention heads).

**Less VRAM**: lower `gpu_memory_utilization` (e.g. 0.75) or reduce `max_model_len`.

**Subset of tasks**: remove entries from `"data"`.

**Custom model checkpoint**: override `model_path` with a local directory path.

---

## See Also

- [SUBMISSION.md](../SUBMISSION.md) — packaging and submitting results
- [docs/vantage/DEVELOPER_GUIDE.md](../docs/vantage/DEVELOPER_GUIDE.md) — full flag reference and model registration details
- [README_VANTAGE.md](../README_VANTAGE.md) — per-model install requirements
