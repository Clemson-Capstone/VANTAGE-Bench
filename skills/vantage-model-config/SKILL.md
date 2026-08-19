---
name: vantage-model-config
description: Use when pointing a run at a model — choosing a registered model name, or authoring a config JSON with custom params and env vars.
---

Two ways to target a model with `run.py`; pick the simpler one unless the user needs
per-parameter control.

## State detection

No filesystem state — this is a per-run decision. Check whether the desired model name
already appears in `supported_VLM`:

```bash
python -c "from vlmeval.config import supported_VLM; print(list(supported_VLM.keys()))"
```

If it's there, use path (a). If not (custom checkpoint, custom sampling params, multi-GPU
`tensor_parallel_size`, or a brand-new model class), use path (b). A model that doesn't exist
at all yet is out of scope here — see `../vantage-add-model/SKILL.md`.

## (a) Registered name, no config needed

```bash
python run.py --data <dataset_key> --model <NameFromSupportedVLM> --verbose
```

## (b) Config JSON

Start from an existing file under `configs/*.json` rather than writing from scratch — they're
already correct, runnable templates (`configs/README.md` lists all of them with their
GPU/package/env-var requirements). Copy the closest match and edit `model.<label>` in place;
leave the `data` block as-is unless trimming to a subset of tasks.

```bash
python run.py --config configs/<closest-match>.json --work-dir ./outputs
# or: python run.py configs/<closest-match>.json   (positional shortcut, same effect)
```

Do not combine `--config` with `--data`/`--model` — `run.py` treats them as mutually exclusive
input modes.

## Gotchas (full detail in `../reference/tasks-and-pillars.md`, don't re-derive from memory)

- **Image tasks need an explicit `"class"`** in the config `data` block
  (`VANTAGE_2DGroundingDataset`, `VANTAGE_2DPointing`, `Astro2DDetectionDataset`) — they are
  not in `supported_video_datasets`, so `{}` alone fails for them. Video-task keys can use
  `{}`.
- **`fps` and `nframe` are mutually exclusive** — setting both raises `ValueError`.
  `VANTAGE_EventVerification` defaults to `fps=4`; pass `fps=0` explicitly when supplying
  `nframe` instead (every registered nframe variant already does this).

## Backends and env vars

| Backend | Typical kwargs | Env vars |
|---|---|---|
| Local HuggingFace | `model_path`, `torch_dtype` | — |
| Local vLLM | `model_path`, `use_vllm=true`, `tensor_parallel_size` | — |
| API (OpenAI-compatible) | `model`, `api_base`, `api_key`, `temperature`, `timeout` | `OPENAI_API_KEY`, `OPENAI_API_BASE` |
| `CosmosReason2` (API) | `model`, `temperature`, `top_p`, `max_tokens` | `COSMOS_REASON2_API_BASE` (full URL, must end `/v1/chat/completions`), `COSMOS_API_KEY` or `NVIDIA_API_KEY` |

For multi-GPU vLLM, `tensor_parallel_size` must divide evenly into the model's attention
heads (`configs/README.md`) — see `../vantage-preflight/SKILL.md` for a sizing recommendation
based on detected GPU count.

## Setting env vars

Session-only export needs no gate:

```bash
export OPENAI_API_KEY=sk-...              # bash
$env:OPENAI_API_KEY = "sk-..."            # PowerShell, current session only
```

**Gate: confirm before writing env vars into a shell profile** (`~/.bashrc`, `$PROFILE`,
etc.) — that persists the secret beyond this session and this repo.

## Failure handling

| Symptom | Fix |
|---|---|
| `ValueError` on dataset construction, both `fps`/`nframe` set | Drop one; see the config-file requirement above. |
| Config error naming an image dataset | Add explicit `"class"` for that entry. |
| API 401/auth error | Confirm the exact env var name the wrapper expects (`README_VANTAGE.md` §3, §12) — names differ per model, don't assume `OPENAI_API_KEY` covers all APIs. |
| `COSMOS_REASON2_API_BASE` requests fail with 404 | URL must be the full chat-completions endpoint, not just the host. |

## Further reading

`README_VANTAGE.md` §6 (config format), §7 (param passing), §12 (per-model requirements);
`configs/README.md` (all templates + hardware notes).
