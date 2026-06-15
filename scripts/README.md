# scripts/

Scripts in this folder fall into two groups: **participant tools** (used when running the benchmark) and **internal/dev tools** (used to manage infra or debug).

---

## Participant tools

### `package_submission.py`

Bundles per-task `*_submission.jsonl` files from an inference run into the `.tar.gz` archive required by the submission portal at https://vantage-bench.org/submit.

```bash
python scripts/package_submission.py \
  --work-dir ./outputs/<model>/<eval_id> \
  --out submission.tar.gz
```

Automatically maps each file to its canonical task name, prints pillar coverage, and warns if a pillar is incomplete. See [`docs/vantage/SUBMISSION.md`](../docs/vantage/SUBMISSION.md) for the full submission guide.

---

### `run_lmudata.py` + `RUN_LMUData.md`

The primary data-prep script. Downloads `nvidia/PhysicalAI-VANTAGE-Bench` from HuggingFace and reshapes it into the LMUData folder layout that VLMEvalKit expects.

```bash
# One-time setup: prepare all eight tasks
hf auth login
python scripts/run_lmudata.py --all --lmu-root ~/LMUData
```

See [`RUN_LMUData.md`](RUN_LMUData.md) for the full guide: prerequisites (ffmpeg, gdown), per-task flags, troubleshooting, and copy vs. symlink modes.

---

## Internal / dev tools

These are not needed to run the benchmark. They're kept here for maintainer use.

| Script | Purpose |
|--------|---------|
| `auto_run.py` | Iterates all registered non-API local models and emits run commands. Upstream VLMEvalKit utility. |
| `apires_scan.py` | Scans a model output directory for prediction files and checks for API failure strings. Debugging aid. |
| `data_browser.py` | Gradio app for visually browsing dataset samples. Run with `python scripts/data_browser.py`. |
| `summarize.py` | Aggregates benchmark scores from output files. Upstream VLMEvalKit utility (references non-VANTAGE datasets). |
| `convert_macbench.py` | Format converter for MacBench data. Upstream utility. |
| `setup_gemma4_instance.sh` | One-time setup script for an internal Gemma4 eval instance. Internal infra. |
| `srun.sh` | SLURM job submission wrapper. Internal HPC use. |
| `cover.sh` | Internal coverage helper. |
| `AI2D_preproc.ipynb` | Notebook for preprocessing AI2D dataset. Upstream utility. |
| `visualize.ipynb` | Notebook for visualizing benchmark outputs. |
