# Preparing VANTAGE-Bench data for inference (`run_lmudata.py`)

A beginner-friendly guide to turning the public **VANTAGE-Bench** Hugging Face
dataset into a local **LMUData** folder you can run VLMEvalKit inference against.

> **TL;DR** — most participants just run:
> ```bash
> hf auth login                                   # once
> python scripts/run_lmudata.py --all --lmu-root ~/LMUData
> ```
> Then run your model with `python run.py --data <task> --model <model>` (default mode runs inference + evaluation and produces submission files). See [`../README.md`](../README.md) for full run commands.
>
> Running this from a clone of the **PhysicalAI-VANTAGE-Bench dataset repo**? It
> auto-uses the local `data/` folder — see *"Where are you running this from?"*.

---

## A. What this script does

`run_lmudata.py` downloads the **public, no-ground-truth** VANTAGE-Bench dataset
from Hugging Face (`nvidia/PhysicalAI-VANTAGE-Bench`) and reshapes it into the
exact folder layout that VLMEvalKit's dataset loaders expect (called
**LMUData**).

- ✅ It prepares data **for inference and submission generation**.
- ❌ It is **not** for local scoring. Ground-truth answers are withheld from the
  public dataset; scoring happens **server-side** on the leaderboard.
- It never fabricates answers. Evaluation-only columns are simply not written.

You run this once per machine. After it finishes, you run your model with
`python run.py` (default mode runs inference + evaluation, producing a **submission JSONL** you upload).

---

## B. The mental model (how the pieces fit)

```
Hugging Face dataset repo              (nvidia/PhysicalAI-VANTAGE-Bench)
        │  download
        ▼
Local HF cache                         (~/.cache/huggingface/ — files reused)
        │  run_lmudata.py reshapes + links/copies
        ▼
LMUData/                               (datasets/<TASK>/… — what VLMEvalKit reads)
        │  python run.py  (default --mode all: inference + evaluation)
        ▼
Predictions  ─►  submission JSONL      (upload to the leaderboard for scoring)
```

Key idea: the HF cache is the real local copy of the media. By **default**, your
LMUData folder *symlinks* into that cache instead of duplicating tens of GB.

---

## Where are you running this from?

This same script ships in **two** repos and picks its data source automatically.
The source is shown in the run summary as `Source: …`.

**Source resolution priority:**
1. `--local-source PATH` (explicit) — use that local checkout's `data/` folder.
2. **Auto-local** — if the script *file itself* lives inside a valid
   PhysicalAI-VANTAGE-Bench checkout (detected by walking the script's own
   parent folders — **no filesystem search**).
3. **HF remote** — download from `--hf-repo` (default
   `nvidia/PhysicalAI-VANTAGE-Bench`) via the HF cache.

### A. Running from the VLMEvalKit repo (most users)

- Default: **HF remote**. Pulls data through the HF cache.
- Nothing special to do — this is the normal path.
  ```bash
  python scripts/run_lmudata.py --all --lmu-root ~/LMUData
  ```

### B. Running from the PhysicalAI-VANTAGE-Bench dataset repo

- If you cloned the dataset repo and run its bundled copy of this script, it
  **auto-detects** the repo and reads `data/` **locally** — no re-download of the
  primary dataset.
  ```bash
  python scripts/run_lmudata.py --all --lmu-root ~/LMUData
  # Source: local-auto:/path/to/PhysicalAI-VANTAGE-Bench
  ```
- ⚠️ **SOT and Grounding still need network.** The dataset repo ships the SOT
  benchmark + prep script and the RefDrone prep script, but **not** the SOT
  source camera videos (from `nvidia/PhysicalAI-SmartSpaces`) or the VisDrone
  images. Those still download. Local mode only avoids re-fetching the primary
  VANTAGE data.

### C. Explicit local source (advanced / developer)

```bash
python scripts/run_lmudata.py --all --lmu-root ~/LMUData \
  --local-source /path/to/PhysicalAI-VANTAGE-Bench
```
- Takes precedence over auto-detect and `--hf-repo`.
- The checkout must match the **public release layout** (validated per task).

### D. Warnings for local mode

- **Wrong / incomplete clone fails validation.** Each task checks for its expected
  public-release paths (e.g. `data/pointing/Vantage2DPointing.tsv`,
  `data/event_verification/filtered/`). A missing marker fails
  *that task* with a clear message — it never silently serves the wrong layout.
- **Missing Git LFS files** (videos/images not pulled) will fail media checks.
  Run `git lfs pull` in the clone first.
- **Symlink mode points into the clone.** In local mode the default symlinks
  reference files inside your dataset clone; moving or `git clean`-ing the clone
  breaks them. Use `--copy` for a portable LMUData.
- **`--hf-repo` is ignored** when a local source is active (a warning is logged).

---

## C. Before you start (checklist)

1. **Python environment** with the project installed and `huggingface_hub`
   available (the repo's normal VLMEvalKit env works). If a snapshot fails with
   `huggingface_hub is required`, run `pip install huggingface_hub`.
2. **Hugging Face login** (recommended): `hf auth login`. The script
   auto-detects this token — you won't need to pass `--hf-token`.
3. **ffmpeg** — only needed for the **SOT** task (frame extraction). Skip if you
   aren't preparing SOT. Easiest install: `conda install -c conda-forge ffmpeg`.
4. **Disk space** — symlink mode (default) needs little extra space (media stays
   in the HF cache, ~40 GB there). `--copy` mode duplicates that media into
   LMUData. SOT adds ~16 GB of source videos to the HF cache.
5. **Choose an LMUData location** — an absolute path you control, e.g.
   `~/LMUData` or `/data/LMUData`. Pass it with `--lmu-root`. The script never
   writes to the current working directory by default.

---

## D. What is the HF cache?

When `huggingface_hub` downloads files, it stores them in a local **cache**,
usually:

```
~/.cache/huggingface/
```

(overridable with the `HF_HOME` env var or this script's `--hf-cache`).

- Downloads are **reused**: re-running the script, or preparing another task
  that shares files, won't re-download what's already cached.
- **Symlink mode (default) depends on this cache.** Your LMUData media entries
  are symlinks pointing into the cache. If you delete or move the HF cache,
  those symlinks break. (Fix: re-run the prep, or use `--copy`.)

---

## E. Recommended participant command (default: symlink)

```bash
python scripts/run_lmudata.py \
  --all \
  --lmu-root /path/to/LMUData
```

- `--all` prepares all eight tasks.
- Media is **symlinked** from the HF cache (disk-efficient).
- Already-prepared tasks are skipped automatically (safe to re-run).

If you only want some tasks (e.g. skip the large SOT download):

```bash
python scripts/run_lmudata.py \
  --tasks vqa,event_verification,dvc,temporal,pointing,astro2d,grounding \
  --lmu-root /path/to/LMUData
```

---

## F. Portable / self-contained command (copy mode)

Use `--copy` when you want a LMUData folder with **real media files** inside it —
portable across machines, and unaffected by HF-cache cleanup:

```bash
python scripts/run_lmudata.py \
  --all \
  --lmu-root /path/to/LMUData \
  --copy
```

Trade-off: this duplicates tens of GB of media into LMUData.

---

## G. Dry-run (simulation, no writes)

Preview exactly what would happen — **no downloads, no files written**, the
LMUData folder isn't even created:

```bash
python scripts/run_lmudata.py \
  --all \
  --lmu-root /path/to/LMUData \
  --dry-run
```

The summary header shows the media mode (`media=symlink` by default), and each
task prints the HF files it would fetch and the paths it would write.

---

## H. SOT-specific prerequisites

SOT (single-object tracking) is the heaviest task. It downloads source camera
videos from
[`nvidia/PhysicalAI-SmartSpaces`](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
and extracts frames.

- **ffmpeg is required** for frame extraction. The shipped prep script has no
  ffmpeg-free path. The wrapper auto-discovers ffmpeg on your `PATH` **and** in
  common conda envs (`~/miniconda3/envs/*/bin`, `/opt/conda/envs/*/bin`, …) and
  bridges it onto the subprocess automatically.
- **HF token is auto-detected** in this order: `--hf-token` → `HF_TOKEN` env →
  the token from `hf auth login`. If you've logged in, nothing to pass.
- **Source videos:** ~16 GB pulled into the HF cache; expect a multi-minute run.
- **`gt.json` contains only the public `init_bbox`** — no hidden per-frame
  trajectories.

**If ffmpeg is missing**, you'll get a clear message with install options:
```
conda install -c conda-forge ffmpeg          # recommended; auto-detected
sudo apt-get install -y ffmpeg               # Debian/Ubuntu
# or a static build from https://johnvansickle.com/ffmpeg/
```

**If no token is found**, the message lists the three ways to provide one
(`hf auth login`, `HF_TOKEN`, `--hf-token`).

---

## I. RefDrone / Grounding prerequisites

The grounding task materializes 1503 VisDrone images via the shipped
`prep_refdrone_data.py`.

| Requirement | Needed? | Notes |
|---|---|---|
| Internet access | yes | Downloads from `github.com` + `huggingface.co`. |
| GitHub HTTPS mirror | primary | Ultralytics release (~311 MB), size + SHA-256 verified. **Sufficient on its own.** |
| Google Drive / `gdown` | **optional** | Fallback only, used if the HTTPS mirror fails. Install with `pip install gdown` only then. |
| Disk | ~600 MB transient | Zip downloaded, images extracted, **zip then deleted** (~290 MB remains). |
| System packages | none | Pure-Python extraction. |

If you already have the images staged, pass `--skip-grounding-images` to write
`annotations.json` without re-downloading.

---

## J. Common troubleshooting

- **Wrong LMUData path / VLMEvalKit can't find data.** VLMEvalKit resolves
  `LMUDataRoot()` from `$LMUData` (if it points to an existing dir), else
  `~/LMUData`. Make them match:
  ```bash
  export LMUData=/path/to/LMUData
  python -c "from vlmeval.smp import LMUDataRoot; print(LMUDataRoot())"
  ```
- **Broken / dangling symlinks.** In symlink mode, deleting or moving the HF
  cache breaks LMUData media links. Fix by re-running the prep, or rebuild with
  `--copy` for a self-contained folder.
- **Missing ffmpeg (SOT).** Install via conda/apt (see section H). A conda-env
  ffmpeg is auto-detected.
- **HF auth / token issues.** Run `hf auth login`, or `export HF_TOKEN=hf_xxx`,
  or pass `--hf-token`. If a download 401/403s, confirm any required dataset
  license acceptance and that your token has read access.
- **Use the default mode (inference + evaluation).** These TSVs omit GT columns, so `evaluate()` produces no local accuracy metrics — but it does emit the submission JSONL needed for leaderboard scoring. Scoring itself is server-side. Do not use `--mode infer` alone if you need the submission JSONL; use `--mode all` (the default) or `--mode eval --reuse` after an existing inference run.

---

## K. What gets created under LMUData

```
LMUData/
└── datasets/
    ├── VANTAGE_VQA/                 VANTAGE_VQA.tsv                + videos/
    ├── VANTAGE_EventVerification/   VANTAGE_EventVerification.tsv  + videos/
    ├── VANTAGE_DVC/                 VANTAGE_DVC.tsv                + videos/
    ├── VANTAGE_Temporal/            VANTAGE_Temporal.tsv           + videos/
    ├── VANTAGE_2DPointing/          VANTAGE_2DPointing.tsv         + images_annotated/
    ├── Astro2D/                     images/ + labels/ (empty placeholders)
    ├── VANTAGE_2DGrounding/         annotations.json               + images/
    └── VANTAGE_SOT/                 <seq>/gt.json + <seq>/frames/f0X.png
```

Inference-only TSV schemas (no GT columns):

| Task | Columns |
|---|---|
| VANTAGE_VQA | `index, video, question, options` |
| VANTAGE_EventVerification | `index, video, system_prompt, question` |
| VANTAGE_DVC | `index, video, question` |
| VANTAGE_Temporal | `index, video, question` (video = bare stem, no `.mp4`) |
| VANTAGE_2DPointing | `index, question_id, image_path, question, A, B, C, D` |

- **Astro2D `labels/*.txt` are intentionally empty** — they exist only so the
  loader doesn't drop images; they contain no ground truth.
- **VANTAGE_2DGrounding `annotations.json` omits `bboxes`** — the loader takes
  its no-GT branch.

> **Note:** a `LMUData/images/` folder may appear **later** — that is a
> VLMEvalKit *runtime* artifact (frame caching / image dumping during
> inference), **not** produced by this prep script. It's safe to ignore.

---

## L. Advanced options

| Flag | Purpose |
|---|---|
| `--tasks a,b,c` | Prepare a subset. Choices: `vqa, event_verification, dvc, temporal, pointing, astro2d, grounding, sot`. |
| `--all` | Prepare all eight tasks (default if neither `--tasks` nor `--all` given). |
| `--lmu-root PATH` | Output root (absolute). Default `~/LMUData`. Never CWD. |
| `--local-source PATH` | Use a local PhysicalAI-VANTAGE-Bench checkout's `data/` instead of HF. Wins over `--hf-repo`. Auto-enabled when the script lives inside such a repo. |
| `--symlink` | **Default.** Symlink media (from the HF cache, or the local checkout in local mode). |
| `--copy` | Copy real media into LMUData (portable, self-contained; uses more disk). |
| `--force` | Rebuild the index file even if the task already looks complete; re-places missing media. |
| `--force-clean` | **Destructive.** Wipe a task's media dir before re-staging. |
| `--dry-run` | Print the plan; no HF calls, no writes. |
| `--hf-token TOKEN` | HF token. Optional — auto-detected from `HF_TOKEN` / `hf auth login`. Needed for SOT. |
| `--hf-repo REPO` | Source repo override (testing/simulation only). Production default: `nvidia/PhysicalAI-VANTAGE-Bench`. |
| `--skip-grounding-images` | For grounding: write `annotations.json` but don't download VisDrone images. |
| `--write-manifest` | Write `.vantage_prep_manifest.json` telemetry at the LMU root (off by default). |
| `--verbose` / `-v` | Debug logging (snapshot paths, per-file decisions). |

`--symlink` and `--copy` are mutually exclusive.

### Idempotency & safety

- **Re-running is safe.** A task that already passes its integrity check is
  **skipped** (no download, no writes) unless you pass `--force`.
- **Interrupted runs recover.** A partial task fails its integrity check, so the
  next normal run rebuilds just that task. SOT resumes cheaply — already
  downloaded videos and extracted frames are reused from the HF cache.
- **Non-destructive by default.** Without `--force-clean`, the script only adds
  missing files and (under `--force`) overwrites the index file. It never
  deletes media on its own.

### Per-task failure isolation

Each task runs independently. If one fails (missing source, no token, no
ffmpeg, mirror down), it's marked `failed` in the summary **and the other tasks
continue**. The process exits non-zero if any task failed, zero otherwise.

---

## After preparing: run inference

```bash
export LMUData=/path/to/LMUData
python run.py --data VANTAGE_VQA_8frame --model <YourModel>
```

Repeat per task (or pass multiple to `--data`). The default mode (inference + evaluation) emits a `*_submission.jsonl` next to the prediction file — that's what you upload. Scoring is done server-side against the withheld ground truth. See [`../SUBMISSION.md`](../SUBMISSION.md) for packaging and upload steps.
