---
name: vantage-troubleshoot
description: Use as a symptom-to-fix lookup when a VANTAGE-Bench run, validation, packaging, or submission step fails or looks wrong.
---

Dense symptom → fix table, not a tutorial. The last column names the skill
that owns the full flow around that fix.

| Symptom | Fix | Skill |
|---|---|---|
| Dangling `LMUData` symlink (video/image files 0 bytes or broken link) | re-run `scripts/run_lmudata.py --all --lmu-root <root>`; the default layout symlinks into the HF cache instead of copying | data-prep |
| `$LMUData` root doesn't match what a run used | pass `--lmudata-root <path>` explicitly on every `run.py` call rather than relying on the env var being set consistently | run |
| Missing `ffmpeg` (SOT frame extraction / CosmosReason2 API frame encoding fails) | `conda install -c conda-forge ffmpeg -y`; confirm with `ffmpeg -version` | preflight |
| HF `401`/`403` on dataset or model download | `hf auth login`; confirm the account has access to the gated repo if applicable | preflight |
| Portal rejects a pillar as partial | every task in that pillar must have a submission file — see the pillar table in `tasks-and-pillars.md`; re-run the missing task | package-submit |
| Missing `*_submission.jsonl` after a run | that file is written during the **eval** phase, not infer; `--mode infer` never writes it — recover with `--mode eval --reuse` | run |
| Archive `> 500 MB` | re-run `scripts/package_submission.py` from a clean `--work-dir`; stray non-JSONL files are the usual cause, not JSONL size | package-submit |
| vLLM `CUDA out of memory` | lower `gpu_memory_utilization`, lower `max_model_len`, or lower `tensor_parallel_size` in the model config | run |
| `tensor_parallel_size` doesn't divide the model's attention heads evenly | pick a divisor of the head count (commonly a power of 2 ≤ GPU count); check the model config, not `run.py` | run / model-config |
| `ValueError` — `fps` and `nframe` both set | they're mutually exclusive; for `VANTAGE_EventVerification` (which defaults `fps=4`), pass `fps=0` to use `nframe` instead | run / model-config |
| No score email after 24h | check spam first; scores can take up to 24h; if still nothing, re-verify the contact email entered in Identity (section 01) | portal-submit |
| `--lmudata-root` path doesn't exist | `run.py` raises `FileNotFoundError` immediately — the path must exist before the flag is set, this flag does not create it | run |
| `--data`/`--model` used together with `--config` | not supported; config-file mode replaces the CLI `--data`/`--model` flags entirely | model-config |
| Image dataset (`VANTAGE_2DGrounding`/`Astro2D`/`VANTAGE_2DPointing`) fails to build from a bare `{}` config entry | these three are **not** in `supported_video_datasets`; the config entry must name `"class"` explicitly | model-config |
| Duplicate submission files for one task (e.g. 8frame and 16frame both ran) | `package_submission.py` keeps only the most-recent by sort order — confirm that's the intended run before packaging | package-submit |
| `COSMOS_REASON2_API_BASE` requests fail | must be a full URL ending in `/v1/chat/completions`, not just a host | preflight |

See `skills/README.md` for the full skill index these fixes point into.
