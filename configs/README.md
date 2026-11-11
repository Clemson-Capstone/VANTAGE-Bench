# Leaderboard configs

Each config here holds the exact settings behind our leaderboard scores for one model: the model settings, the per-task overrides, the dataset sampling and the prompts.

| Config | Model | Backend |
|---|---|---|
| `qwen3_vl_8b.json` | Qwen3-VL-8B-Instruct | vLLM, 1 × 80 GB |
| `qwen3_vl_32b.json` | Qwen3-VL-32B-Instruct | vLLM, 2 × 80 GB |
| `qwen3_5_9b.json` | Qwen3.5-9B | vLLM ≥ 0.23, 1 × 80 GB |
| `qwen3_5_27b.json` | Qwen3.5-27B | vLLM ≥ 0.23, 2 × 80 GB |
| `cosmos_reason2_8b.json` | Cosmos-Reason2-8B | vLLM, 1 × 80 GB |
| `cosmos3_nano.json` | Cosmos3-Nano | vLLM, 2 × 80 GB, converted weights |
| `cosmos3_super.json` | Cosmos3-Super | vLLM, 2 × 80 GB, converted weights |

All configs use the built-in prompts, which are the prompts behind our scores (see `prompt_guide.md`).

## Run

```bash
export LMUData=/path/to/data
python scripts/reproduce.py configs/qwen3_vl_8b.json
python scripts/reproduce.py configs/qwen3_vl_8b.json --tasks VANTAGE_VQA_4fps   # one task
python scripts/reproduce.py configs/qwen3_vl_8b.json --dry-run                  # check only
```

Some tasks need different model settings. For example, dense captioning and tracking need a larger output limit, and the Qwen models track at a low temperature. `run.py --config` applies one set of model settings to every dataset, so `reproduce.py` groups the tasks that share settings and runs one `run.py` call per group. The model therefore loads once per group: six times for the Qwen models and four times for the Cosmos models. All groups write to `outputs/<name>/`, so `scripts/package_submission.py` packages them as one submission. Arguments that `reproduce.py` does not know are passed to `run.py`.

### Cosmos3 weights

vLLM cannot load the nvidia/Cosmos3-Nano and nvidia/Cosmos3-Super checkpoints directly. Our runs loaded a conversion of each one to the Qwen3-VL format (`Qwen3VLForConditionalGeneration`). Convert the checkpoint yourself, then set `model_path` in `cosmos3_nano.json` or `cosmos3_super.json` to the converted folder. The configs default to `converted/Cosmos3-<Nano|Super>-qwen3vl`, relative to the repository root.

## How close a rerun gets

The scores themselves are on the leaderboard at [vantage-bench.org](https://vantage-bench.org/). Expect most tasks to land within about one to three points of them, not exactly on them:

- **Sampling.** The Qwen models sample at temperature 0.7 (0.3 for Qwen3.5 captioning, 0.1 for Qwen3.5 tracking), so every run differs a little.
- **Data.** Most of our v1.0 runs read an internal copy of the data. The public release has anonymized video names, and 46 Event Verification rows carry a differently worded per-clip system prompt.
- **Cosmos Event Verification.** Our runs also sent the system message "You are a helpful assistant."; `Cosmos3vLLM` cannot send a system message.
- **Qwen3-VL-32B tracking** uses a maximum context of 98,304 tokens instead of the 131,072 our runs used, so the KV cache fits on two 80 GB GPUs. This only matters for a request longer than 98,304 tokens.
- **Score review.** A few leaderboard cells were adjusted during the v1.0 score review (for example Astro2D moved to pooled F1), so a raw rerun can differ from them for that reason too.
- **Dense video captioning scored locally.** The local evaluator skips videos with no usable prediction, while the leaderboard scores them as 0, so a local DVC score can come out higher for a model that fails on some videos.
