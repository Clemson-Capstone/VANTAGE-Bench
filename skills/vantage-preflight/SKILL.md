---
name: vantage-preflight
description: Use when starting a fresh environment or before a run to check GPU/VRAM, disk, ffmpeg, HF token, and network, then recommend a model backend.
---

Runs the environment report script, interprets its findings, and — only for items a human
should decide on — proposes a backend and installs missing dependencies.

## State detection

No state file; the check is cheap and safe to re-run any time. Always run it fresh rather than
assuming a prior pass still holds (drivers, disk space, and tokens all drift).

## Command

```bash
python scripts/preflight_check.py --lmu-root <LMUData path> --work-dir <outputs path> --json
```

Findings are classified `ok` / `warning` / `blocker`. Exit code 0 means no blocker (warnings
may still remain); exit 1 means at least one blocker — resolve it before continuing to any
other `vantage-*` skill.

## Decision logic — backend recommendation

Only propose this for a `warning`/`blocker` on GPU/VRAM; don't second-guess an `ok` finding.

| Detected GPU/VRAM | Recommended backend | Notes |
|---|---|---|
| No GPU | Hosted API (`GPT4V`, `CosmosReason2` API) | See `../vantage-model-config/SKILL.md` for env vars. |
| 1 GPU, < 16 GB | Local HF, small model (e.g. `Cosmos3-Nano`, 1×8GB) | Check `README_VANTAGE.md` §12 for the model's minimum VRAM. |
| 1 GPU, 16-24 GB | Local HF (`CosmosHF`) or local vLLM `tensor_parallel_size=1` | |
| 1 GPU, ≥ 24 GB | Local vLLM, `tensor_parallel_size=1` | |
| N GPUs (N > 1) | Local vLLM, `tensor_parallel_size=N` | Must divide evenly into the model's attention heads (`configs/README.md`). If N isn't a clean divisor for the target model, use the largest power-of-2 divisor of N (e.g. 6 GPUs → 4, leaving 2 idle) rather than guessing an odd size. |

## Installs — gate: confirm with the user before running any pip/conda command

```bash
pip install -r requirements.txt
pip install -e .
```

Conditional, only if the chosen model needs them (`README_VANTAGE.md` §12):

```bash
pip install vllm                       # any local-vLLM backend (Cosmos, Cosmos-Reason1/2)
pip install qwen-vl-utils               # Cosmos family (vLLM, HF, or API)
pip install "transformers_cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3"   # Cosmos3-Nano only
```

`ffmpeg` (only needed for SOT frame extraction) is not a pip package — install via
`conda install -c conda-forge ffmpeg` or the OS package manager; see
`scripts/RUN_LMUData.md` §H if `preflight_check.py` reports it missing.

## Verify

```bash
python -c "from vlmeval.config import supported_VLM; print(len(supported_VLM))"
```

A number > 0 confirms `vlmeval` imports cleanly and the registry is populated. If this errors,
the install did not complete — re-check the traceback before moving on.

## Failure handling

| Symptom | Fix |
|---|---|
| `ffmpeg` blocker, preparing SOT | `conda install -c conda-forge ffmpeg`; see `RUN_LMUData.md` §H. |
| No HF token / 401 on download | `hf auth login`, or `export HF_TOKEN=...`; see `RUN_LMUData.md` §C.2. |
| GPU not detected but hardware exists | Check driver/CUDA install outside this skill's scope; not a preflight-script bug. |
| Disk warning near a task's size estimate | Prefer `--symlink` in `vantage-data-prep` (Linux/macOS), or free space before `--copy`. |
| Verify step raises `ImportError` | Re-run the two base `pip install` commands; check for a stale/partial env. |

For anything not listed here, hand off to `../vantage-troubleshoot/SKILL.md`.

## Further reading

`README_VANTAGE.md` §2 (setup), §3 (env vars), §12 (per-model requirements);
`scripts/RUN_LMUData.md` §C (checklist).
