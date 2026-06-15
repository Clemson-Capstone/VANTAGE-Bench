#!/usr/bin/env python3
"""
run_lmudata.py — Prepare a VLMEvalKit-compatible LMUData layout from
``nvidia/PhysicalAI-VANTAGE-Bench`` (HF dataset).

PHASE 2A implemented:
  - VANTAGE_VQA, VANTAGE_EventVerification, VANTAGE_DVC,
    VANTAGE_Temporal, VANTAGE_2DPointing
PHASE 2B implemented:
  - Astro2D, VANTAGE_2DGrounding, VANTAGE_SOT

The target LMUData layout per task (matched against loader expectations
in vlmeval/dataset/vantage_*.py and vlmeval/dataset/image_mcq.py):

  $LMU_ROOT/datasets/
    VANTAGE_VQA/                 VANTAGE_VQA.tsv                videos/*.mp4
    VANTAGE_EventVerification/   VANTAGE_EventVerification.tsv  videos/*.mp4
    VANTAGE_DVC/                 VANTAGE_DVC.tsv                videos/*.mp4
    VANTAGE_Temporal/            VANTAGE_Temporal.tsv           videos/*.mp4
    VANTAGE_2DPointing/          VANTAGE_2DPointing.tsv         images_annotated/*.jpg
    Astro2D/                     images/<flat>.jpg              labels/<flat>.txt  (empty)
    VANTAGE_2DGrounding/         annotations.json               images/*.jpg
    VANTAGE_SOT/                 <seq>/gt.json                  <seq>/frames/f0X.png

No GT fields are fabricated. ``answer`` is an empty string where the
public source omits it; ``duration`` falls back to 30.0; ``category``
falls back to ``"Unknown"``.

Runtime source: HF repo ``nvidia/PhysicalAI-VANTAGE-Bench``
(repo_type=dataset). No local repo is read at runtime.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Production runtime source — DO NOT change this default.
# Override only for testing/simulation via the --hf-repo CLI flag.
HF_REPO_ID = "nvidia/PhysicalAI-VANTAGE-Bench"
HF_REPO_TYPE = "dataset"

DEFAULT_LMU_ROOT = Path("~/LMUData").expanduser()
MANIFEST_FILENAME = ".vantage_prep_manifest.json"

# Top-level files/dirs that mark a PhysicalAI-VANTAGE-Bench checkout.
REPO_TOP_MARKERS = ["data", "README.md", "LICENSE.md"]

# Per-task required paths (relative to repo root) for a LOCAL source to be
# usable. These encode the post-PR layout; a stale/pre-PR clone that lacks the
# marker fails validation for that task with a clear message rather than
# silently producing wrong data.
LOCAL_TASK_MARKERS: Dict[str, List[str]] = {
    "vqa": ["data/vqa/data_jsons/annotations"],
    "event_verification": ["data/event_verification/data_jsons/annotations"],
    "dvc": ["data/dense_captioning/metadata.jsonl"],
    "temporal": ["data/temporal_localization/data_jsons/annotations"],
    "pointing": ["data/pointing/VANTAGE_2DPointing.jsonl"],
    "astro2d": ["data/2dbbox/metadata.jsonl"],
    "grounding": [
        "data/referring/refdrone_test_llava.json",
        "data/referring/prep_refdrone_data.py",
    ],
    "sot": [
        "data/tracking/sot_benchmark.jsonl",
        "data/tracking/prepare_sot_dataset.py",
    ],
}

IMPLEMENTED_TASKS = [
    "vqa", "event_verification", "dvc", "temporal", "pointing",
    "astro2d", "grounding", "sot",
]
DEFERRED_TASKS: List[str] = []
ALL_TASKS = IMPLEMENTED_TASKS + DEFERRED_TASKS

# nvidia/PhysicalAI-SmartSpaces is the source for SOT camera videos.
# Access requires HF token + license acceptance at:
#   https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
SOT_SOURCE_REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
SOT_SOURCE_REPO_SUBDIR = "MTMC_Tracking_2025"

log = logging.getLogger("vantage-prep")


# ---------------------------------------------------------------------------
# Options + result containers
# ---------------------------------------------------------------------------

@dataclass
class Options:
    lmu_root: Path
    hf_cache: Optional[Path]
    hf_token: Optional[str]
    symlink: bool
    force: bool
    force_clean: bool
    dry_run: bool
    verbose: bool
    hf_repo: str = HF_REPO_ID
    skip_grounding_images: bool = False
    write_manifest: bool = False
    local_source: Optional[Path] = None


@dataclass
class TaskResult:
    task: str
    lmu_name: str
    target_dir: Path
    status: str  # "skipped" | "built" | "rebuilt" | "deferred" | "failed" | "dry-run"
    rows: int = 0
    media_count: int = 0
    source_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    source_mode: str = ""  # "local-explicit" | "local-auto" | "hf" | ""


# ---------------------------------------------------------------------------
# Per-task config
# ---------------------------------------------------------------------------

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "vqa": {
        "lmu_name": "VANTAGE_VQA",
        "index_file": "VANTAGE_VQA.tsv",
        "media_dir": "videos",
        "media_glob": "*.mp4",
        "hf_patterns": [
            "data/vqa/data_jsons/annotations/*.json",
            "data/vqa/videos/*",
        ],
    },
    "event_verification": {
        "lmu_name": "VANTAGE_EventVerification",
        "index_file": "VANTAGE_EventVerification.tsv",
        "media_dir": "videos",
        "media_glob": "*.mp4",
        "hf_patterns": [
            "data/event_verification/data_jsons/annotations/*.json",
            "data/event_verification/videos/*",
        ],
    },
    "dvc": {
        "lmu_name": "VANTAGE_DVC",
        "index_file": "VANTAGE_DVC.tsv",
        "media_dir": "videos",
        "media_glob": "*.mp4",
        "hf_patterns": [
            "data/dense_captioning/metadata.jsonl",
            "data/dense_captioning/prompt.json",
            "data/dense_captioning/videos/*",
        ],
    },
    "temporal": {
        "lmu_name": "VANTAGE_Temporal",
        "index_file": "VANTAGE_Temporal.tsv",
        "media_dir": "videos",
        "media_glob": "*.mp4",
        "hf_patterns": [
            "data/temporal_localization/data_jsons/annotations/*.json",
            "data/temporal_localization/videos/*",
        ],
    },
    "pointing": {
        "lmu_name": "VANTAGE_2DPointing",
        "index_file": "VANTAGE_2DPointing.tsv",
        "media_dir": "images_annotated",
        "media_glob": "*",
        "hf_patterns": [
            "data/pointing/VANTAGE_2DPointing.jsonl",
            "data/pointing/images_annotated/*",
        ],
    },
    # ---- PHASE 2B ----
    "astro2d": {
        "lmu_name": "Astro2D",
        # Astro2D has no top-level index file; loader reads images/ and labels/ directly.
        "index_file": None,
        "media_dir": "images",
        "media_glob": "*",
        "hf_patterns": [
            "data/2dbbox/metadata.jsonl",
            "data/2dbbox/prompt.json",
            "data/2dbbox/sequence_a/images/*",
            "data/2dbbox/sequence_b/images/*",
            "data/2dbbox/sequence_c/images/*",
        ],
    },
    "grounding": {
        "lmu_name": "VANTAGE_2DGrounding",
        "index_file": "annotations.json",
        "media_dir": "images",
        "media_glob": "*",
        "hf_patterns": [
            "data/referring/refdrone_test_llava.json",
            "data/referring/prep_refdrone_data.py",
            "data/referring/RUN.md",
        ],
    },
    "sot": {
        "lmu_name": "VANTAGE_SOT",
        # SOT has no top-level index file; integrity check is per-sequence.
        "index_file": None,
        "media_dir": ".",  # per-sequence dirs live at the task root
        "media_glob": "*",
        "hf_patterns": [
            "data/tracking/sot_benchmark.jsonl",
            "data/tracking/prepare_sot_dataset.py",
            "data/tracking/README.md",
        ],
    },
}


# ---------------------------------------------------------------------------
# Logging + path helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_lmu_root(arg_value: Optional[str]) -> Path:
    if arg_value:
        root = Path(arg_value).expanduser().resolve()
    else:
        root = DEFAULT_LMU_ROOT
    if not root.is_absolute():
        raise SystemExit(f"--lmu-root must be absolute (got: {root})")
    return root


def _target_dir(lmu_root: Path, task: str) -> Path:
    return lmu_root / "datasets" / TASK_CONFIG[task]["lmu_name"]


# ---------------------------------------------------------------------------
# HF snapshot
# ---------------------------------------------------------------------------

def _snapshot(task: str, opts: Options) -> Path:
    """Download (or reuse cached) HF snapshot restricted to this task's patterns."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    cfg = TASK_CONFIG[task]
    patterns = list(cfg["hf_patterns"])
    log.info("[%s] snapshot from %s (patterns=%d)", task, opts.hf_repo, len(patterns))
    kwargs: Dict[str, Any] = dict(
        repo_id=opts.hf_repo,
        repo_type=HF_REPO_TYPE,
        allow_patterns=patterns,
    )
    if opts.hf_cache is not None:
        kwargs["cache_dir"] = str(opts.hf_cache)
    if opts.hf_token:
        kwargs["token"] = opts.hf_token
    snap_dir = snapshot_download(**kwargs)
    log.debug("[%s] snapshot dir: %s", task, snap_dir)
    return Path(snap_dir)


# ---------------------------------------------------------------------------
# Source resolution: local dataset repo vs HF remote
# ---------------------------------------------------------------------------

def _task_data_subdir(task: str) -> str:
    """Return the `data/<subdir>` name for a task (e.g. dvc -> dense_captioning),
    derived from the task's first HF pattern so there is one source of truth."""
    first = TASK_CONFIG[task]["hf_patterns"][0]  # e.g. "data/dense_captioning/metadata.jsonl"
    parts = first.split("/")
    return parts[1] if len(parts) > 1 else task


def _is_valid_local_repo(path: Path, task: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that `path` is a PhysicalAI-VANTAGE-Bench checkout.

    Top-level: must contain data/, README.md, LICENSE.md.
    Per-task (when `task` given): the post-PR marker paths in LOCAL_TASK_MARKERS
    must all exist. A stale/pre-PR clone missing a marker fails here with a
    clear reason instead of silently serving the wrong layout.
    """
    if not path.is_dir():
        return False, f"not a directory: {path}"
    for m in REPO_TOP_MARKERS:
        if not (path / m).exists():
            return False, f"missing top-level marker: {m}"
    if task is not None:
        for rel in LOCAL_TASK_MARKERS.get(task, []):
            if not (path / rel).exists():
                return False, f"missing {task} marker: {rel}"
    return True, "ok"


# Cache the (single) auto-detected root so we don't walk parents per task.
_AUTODETECT_CACHE: Dict[str, Optional[Path]] = {}


def _autodetect_local_root() -> Optional[Path]:
    """Return the dataset-repo root iff the script itself lives inside one.

    Walks only Path(__file__).resolve().parents — never searches arbitrary
    filesystem locations. Top-level markers only (per-task checks happen in
    _resolve_source). Returns None when the script is not inside a valid repo
    (e.g. when shipped in the VLMEvalKit repo).
    """
    if "root" in _AUTODETECT_CACHE:
        return _AUTODETECT_CACHE["root"]
    here = Path(__file__).resolve()
    found: Optional[Path] = None
    for parent in here.parents:
        ok, _why = _is_valid_local_repo(parent, task=None)
        if ok:
            found = parent
            break
    _AUTODETECT_CACHE["root"] = found
    return found


def _resolve_source(task: str, opts: Options) -> Tuple[Optional[Path], str]:
    """Decide where this task's source data comes from.

    Priority:
      1. --local-source PATH        -> ("local-explicit")
      2. script inside a repo       -> ("local-auto")
      3. HF remote (--hf-repo)      -> ("hf")

    Returns (source_root, source_mode). For "hf", source_root is the snapshot
    dir — downloaded here for a real run, but left None during --dry-run so no
    network/cache work happens. For local modes, source_root is the repo root
    and is validated (per-task) before returning.
    """
    # 1. Explicit local source.
    if opts.local_source is not None:
        root = opts.local_source
        ok, why = _is_valid_local_repo(root, task)
        if not ok:
            raise SystemExit(
                f"[{task}] --local-source {root} is not a usable dataset repo: {why}. "
                f"Ensure it is a complete, post-PR PhysicalAI-VANTAGE-Bench checkout "
                f"(and that Git LFS media is pulled)."
            )
        return root, "local-explicit"

    # 2. Auto-local: only when the script itself sits inside a valid repo.
    auto = _autodetect_local_root()
    if auto is not None:
        ok, why = _is_valid_local_repo(auto, task)
        if not ok:
            raise SystemExit(
                f"[{task}] local dataset repo detected at {auto} but it is missing "
                f"required layout: {why}. The clone looks stale or pre-PR. Update it, "
                f"or run from the VLMEvalKit repo to use the HF remote."
            )
        return auto, "local-auto"

    # 3. HF remote. Defer the actual download during dry-run.
    if opts.dry_run:
        return None, "hf"
    return _snapshot(task, opts), "hf"


# ---------------------------------------------------------------------------
# File operations (idempotent + non-destructive)
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path, dry_run: bool) -> None:
    if p.exists():
        return
    log.debug("mkdir -p %s", p)
    if not dry_run:
        p.mkdir(parents=True, exist_ok=True)


def _link_or_copy_file(src: Path, dst: Path, opts: Options) -> bool:
    """Place src at dst as symlink (default) or copy. Returns True if action taken."""
    if dst.exists() or dst.is_symlink():
        return False
    if opts.dry_run:
        log.debug("would %s %s -> %s", "symlink" if opts.symlink else "copy", src, dst)
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if opts.symlink:
        os.symlink(os.fspath(src.resolve()), os.fspath(dst))
    else:
        shutil.copy2(src, dst)
    return True


def _link_or_copy_dir(src_dir: Path, dst_dir: Path, opts: Options) -> int:
    """Mirror src_dir into dst_dir. Returns count of new entries placed."""
    if not src_dir.exists():
        log.warning("source media dir missing: %s", src_dir)
        return 0
    placed = 0
    _ensure_dir(dst_dir, opts.dry_run)
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        dst = dst_dir / src.name
        if _link_or_copy_file(src, dst, opts):
            placed += 1
    return placed


def _wipe_dir(p: Path, dry_run: bool) -> None:
    if not p.exists():
        return
    log.warning("--force-clean: removing %s", p)
    if dry_run:
        return
    shutil.rmtree(p)


def _write_tsv(path: Path, rows: List[Dict[str, Any]], columns: List[str], dry_run: bool) -> None:
    log.info("write TSV %s (%d rows, %d cols)", path, len(rows), len(columns))
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, delimiter="\t",
            quoting=csv.QUOTE_MINIMAL, lineterminator="\n",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


# ---------------------------------------------------------------------------
# Integrity check (mirrors loader check_integrity predicates)
# ---------------------------------------------------------------------------

def _check_integrity(task: str, target_dir: Path) -> Tuple[bool, str]:
    # Per-task overrides for layouts that don't fit "index file + media dir".
    if task == "astro2d":
        images = target_dir / "images"
        labels = target_dir / "labels"
        if not images.exists() or not any(images.iterdir()):
            return False, "images/ empty/missing"
        if not labels.exists() or not any(labels.iterdir()):
            return False, "labels/ empty/missing (placeholders required)"
        return True, "ok"
    if task == "sot":
        if not target_dir.exists():
            return False, "target dir missing"
        seq_dirs = [d for d in target_dir.iterdir()
                    if d.is_dir() and (d / "gt.json").exists() and (d / "frames").is_dir()]
        if not seq_dirs:
            return False, "no valid sequence dirs (need <seq>/gt.json + frames/)"
        return True, f"{len(seq_dirs)} sequence dirs"

    cfg = TASK_CONFIG[task]
    idx_name = cfg["index_file"]
    if idx_name is None:
        # Shouldn't happen — handled above per-task.
        return False, "no integrity check defined"
    idx = target_dir / idx_name
    media = target_dir / cfg["media_dir"]
    if not idx.exists():
        return False, f"index file missing: {idx.name}"
    if idx.stat().st_size == 0:
        return False, f"index file empty: {idx.name}"
    if not media.exists() or not any(media.iterdir()):
        return False, f"media dir empty/missing: {media.name}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Common helpers for prep functions
# ---------------------------------------------------------------------------

def _normalize_category(cat: Optional[str]) -> str:
    if not cat:
        return "Unknown"
    s = str(cat).strip()
    if not s:
        return "Unknown"
    if s == "Smart Spaces":
        return "Smart_Spaces"
    return s


def _strip_mp4(name: str) -> str:
    return name[:-4] if name.endswith(".mp4") else name


def _ensure_mp4(name: str) -> str:
    return name if name.endswith(".mp4") else name + ".mp4"


def _bare_stem(name: str) -> str:
    """Strip a leading dir and trailing .mp4 from a HF file_name field."""
    base = os.path.basename(name)
    return _strip_mp4(base)


def _strip_video_ext(name: str) -> str:
    """Strip one trailing .json OR .mp4 extension (matches VANTAGE_VQA loader's logic).

    q_uid values in VQA annotations come in two flavors:
      "concat_wh_52_2925_4.mp4"     -> "concat_wh_52_2925_4"
      "temporal_cb00ec82cd.json"    -> "temporal_cb00ec82cd"
    """
    base = os.path.basename(name)
    for ext in (".json", ".mp4"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("jsonl parse error %s:%d %s", path.name, ln, e)
    return out


# ---------------------------------------------------------------------------
# VQA
# ---------------------------------------------------------------------------

VQA_QUESTION_PREFIX = "You are analyzing a surveillance or traffic monitoring video. Watch the video carefully before answering. Answer based only on what you observe in the video."

# Mirror of VANTAGE_VQA.generate_question (vlmeval/dataset/vantage_vqa.py)
def _vqa_format_question(base_question: str, options: List[str]) -> str:
    labels = ["A", "B", "C", "D"]
    out = "Question: " + base_question + "\n"
    out += "Select your answer from the choices below:\n"
    for i, lab in enumerate(labels[: len(options)]):
        out += f"{lab}. {options[i]}\n"
    out += (
        "Respond with ONLY the letter corresponding to your answer (A, B, C, or D). "
        "Do not provide any explanation or other text.\n"
    )
    return out


def _vqa_process_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    video = item.get("q_uid") or item.get("vid") or ""
    if not video:
        return None
    # q_uid may end in .json (template file ref) or .mp4 (video ref) or nothing.
    # Match the loader's _process_annotation_item logic: strip one of those.
    video = _ensure_mp4(_strip_video_ext(video))
    question = item.get("question", "")
    if not question:
        return None
    raw_opts = item.get("options", []) or []
    options: List[str] = []
    for opt in raw_opts:
        if isinstance(opt, str):
            # Strip leading "A: " / "B: " labels if present (matches loader)
            parts = opt.split(": ", 1)
            options.append(parts[1] if len(parts) == 2 else opt)
        else:
            options.append(str(opt))
    if not options:
        return None
    formatted = _vqa_format_question(question, options)
    category = _normalize_category(item.get("industry") or item.get("category"))
    return {
        "video": video,
        "question": formatted,
        "answer": "",
        "options": json.dumps(options, ensure_ascii=False),
        "category": category,
        "_qid": item.get("question_id", ""),
    }


def _prep_vqa(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="vqa", lmu_name="VANTAGE_VQA", target_dir=target_dir, status="dry-run")
    ann_dir = snap_dir / "data" / "vqa" / "data_jsons" / "annotations"
    src_videos = snap_dir / "data" / "vqa" / "videos"
    if not ann_dir.exists():
        raise SystemExit(f"VQA annotations dir missing in snapshot: {ann_dir}")

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for jf in sorted(ann_dir.glob("*.json")):
        data = _load_json(jf)
        if not isinstance(data, list):
            log.debug("VQA: skipping non-list JSON %s", jf.name)
            continue
        res.source_files.append(jf.name)
        for item in data:
            proc = _vqa_process_item(item)
            if proc is None:
                continue
            dedup_key = (proc["video"], proc["_qid"] or proc["question"][:80])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            rows.append(proc)

    rows.sort(key=lambda r: (r["video"], r["_qid"]))
    for i, r in enumerate(rows):
        r["index"] = i
        r.pop("_qid", None)

    # Inference-only schema: GT columns (answer, category) are dropped.
    # build_prompt needs question + options; submission emit needs index + video.
    columns = ["index", "video", "question", "options"]
    tsv = target_dir / "VANTAGE_VQA.tsv"
    media_dst = target_dir / "videos"

    if opts.dry_run:
        res.rows = len(rows)
        res.media_count = sum(1 for _ in src_videos.glob("*.mp4")) if src_videos.exists() else 0
        res.notes.append(f"plan: write {tsv} and link {res.media_count} videos")
        return res

    if opts.force_clean:
        _wipe_dir(media_dst, opts.dry_run)
    _write_tsv(tsv, rows, columns, opts.dry_run)
    placed = _link_or_copy_dir(src_videos, media_dst, opts)
    res.rows = len(rows)
    res.media_count = placed
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# Event Verification
# ---------------------------------------------------------------------------

EV_DEFAULT_SYSTEM_PROMPT = (
    "You are a warehouse safety monitoring system analyzing surveillance video. "
    "Determine if a near-miss incident has occurred between a person and a forklift. "
    "A near-miss is defined as a situation where a person and an operating forklift "
    "come into dangerously close proximity without a collision occurring — for example, "
    "a person crossing the path of a moving forklift, a forklift passing close behind "
    "or in front of a person, or a person narrowly avoiding being struck. "
    "Answer \"Yes\" if a near-miss is clearly visible. Otherwise, answer \"No\"."
)


def _ev_load_items(path: Path) -> List[Dict[str, Any]]:
    raw = _load_json(path)
    if isinstance(raw, dict) and "bcq" in raw:
        return list(raw["bcq"])
    if isinstance(raw, list):
        return raw
    log.warning("EV: unrecognized JSON shape in %s", path.name)
    return []


def _prep_event_verification(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="event_verification", lmu_name="VANTAGE_EventVerification",
                     target_dir=target_dir, status="dry-run")
    ann_dir = snap_dir / "data" / "event_verification" / "data_jsons" / "annotations"
    src_videos = snap_dir / "data" / "event_verification" / "videos"
    if not ann_dir.exists():
        raise SystemExit(f"EV annotations dir missing in snapshot: {ann_dir}")

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for jf in sorted(ann_dir.glob("*.json")):
        items = _ev_load_items(jf)
        if not items:
            continue
        res.source_files.append(jf.name)
        for item in items:
            video = item.get("video") or item.get("video_id") or ""
            if not video:
                continue
            video = _ensure_mp4(os.path.basename(video))
            question = item.get("question", "")
            if not question:
                continue
            iid = item.get("id", "")
            key = (video, iid or question[:80])
            if key in seen:
                continue
            seen.add(key)
            sys_prompt = item.get("system_prompt") or EV_DEFAULT_SYSTEM_PROMPT
            category = _normalize_category(item.get("category"))
            rows.append({
                "video": video,
                "system_prompt": sys_prompt,
                "question": question,
                "answer": "",
                "category": category,
                "_id": iid,
            })

    rows.sort(key=lambda r: (r["video"], r["_id"]))
    for i, r in enumerate(rows):
        r["index"] = i
        r.pop("_id", None)

    # Inference-only schema: GT columns (answer, category) dropped.
    # system_prompt is REQUIRED — build_prompt reads line['system_prompt']
    # unconditionally and it is prompt framing, not ground truth.
    columns = ["index", "video", "system_prompt", "question"]
    tsv = target_dir / "VANTAGE_EventVerification.tsv"
    media_dst = target_dir / "videos"

    if opts.dry_run:
        res.rows = len(rows)
        res.media_count = sum(1 for _ in src_videos.glob("*.mp4")) if src_videos.exists() else 0
        res.notes.append(f"plan: write {tsv} and link {res.media_count} videos")
        return res

    if opts.force_clean:
        _wipe_dir(media_dst, opts.dry_run)
    _write_tsv(tsv, rows, columns, opts.dry_run)
    placed = _link_or_copy_dir(src_videos, media_dst, opts)
    res.rows = len(rows)
    res.media_count = placed
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# DVC
# ---------------------------------------------------------------------------

def _prep_dvc(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="dvc", lmu_name="VANTAGE_DVC", target_dir=target_dir, status="dry-run")
    meta_path = snap_dir / "data" / "dense_captioning" / "metadata.jsonl"
    src_videos = snap_dir / "data" / "dense_captioning" / "videos"
    if not meta_path.exists():
        raise SystemExit(f"DVC metadata.jsonl missing in snapshot: {meta_path}")

    items = _load_jsonl(meta_path)
    res.source_files.append(meta_path.name)

    rows: List[Dict[str, Any]] = []
    for item in items:
        file_name = item.get("file_name") or item.get("video") or ""
        if not file_name:
            continue
        video = _ensure_mp4(os.path.basename(file_name))
        prompt = item.get("prompt") or item.get("question") or ""
        if not prompt:
            continue
        rows.append({
            "video": video,
            "question": prompt,
            "answer": "",
            "category": "Unknown",
        })

    rows.sort(key=lambda r: r["video"])
    for i, r in enumerate(rows):
        r["index"] = i

    # Inference-only schema: GT columns (answer, category) dropped.
    # DVC build_prompt uses a constant query; question is kept for readability.
    columns = ["index", "video", "question"]
    tsv = target_dir / "VANTAGE_DVC.tsv"
    media_dst = target_dir / "videos"

    if opts.dry_run:
        res.rows = len(rows)
        res.media_count = sum(1 for _ in src_videos.glob("*.mp4")) if src_videos.exists() else 0
        res.notes.append(f"plan: write {tsv} and link {res.media_count} videos")
        return res

    if opts.force_clean:
        _wipe_dir(media_dst, opts.dry_run)
    _write_tsv(tsv, rows, columns, opts.dry_run)
    placed = _link_or_copy_dir(src_videos, media_dst, opts)
    res.rows = len(rows)
    res.media_count = placed
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

TEMPORAL_QUESTION_PREFIX = (
    "Localize a series of activity events in the video, output the start and end "
    "timestamp for each event. Provide the result in json format with 'mm:ss.ff' "
    "format for time depiction for this event. Use keywords 'start' and 'end' in "
    "the json output."
)


def _prep_temporal(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="temporal", lmu_name="VANTAGE_Temporal", target_dir=target_dir, status="dry-run")
    ann_dir = snap_dir / "data" / "temporal_localization" / "data_jsons" / "annotations"
    src_videos = snap_dir / "data" / "temporal_localization" / "videos"
    if not ann_dir.exists():
        raise SystemExit(f"Temporal annotations dir missing in snapshot: {ann_dir}")

    rows: List[Dict[str, Any]] = []
    seen_qids: set = set()
    for jf in sorted(ann_dir.glob("*.json")):
        data = _load_json(jf)
        if not isinstance(data, list):
            log.debug("Temporal: skipping non-list JSON %s", jf.name)
            continue
        res.source_files.append(jf.name)
        for item in data:
            vid = item.get("vid") or item.get("video") or ""
            if not vid:
                continue
            # Loader appends ".mp4" itself, so store bare stem.
            vid = _strip_mp4(os.path.basename(vid))
            base_q = item.get("question", "")
            if not base_q:
                continue
            qid = item.get("question_id", "")
            if qid and qid in seen_qids:
                continue
            if qid:
                seen_qids.add(qid)
            duration = item.get("duration", 30.0)
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = 30.0
            question = TEMPORAL_QUESTION_PREFIX + "\n" + base_q
            category = _normalize_category(item.get("category"))
            rows.append({
                "video": vid,
                "question": question,
                "answer": "",
                "duration": duration,
                "category": category,
                "_qid": qid or f"{vid}_0",
            })

    rows.sort(key=lambda r: (r["video"], r["_qid"]))
    for i, r in enumerate(rows):
        r["index"] = i
        r.pop("_qid", None)

    # Inference-only schema: GT columns (answer, category) and the
    # evaluation-only 'duration' field are dropped. build_prompt reads only
    # question + video; duration is consumed solely by evaluate().
    columns = ["index", "video", "question"]
    tsv = target_dir / "VANTAGE_Temporal.tsv"
    media_dst = target_dir / "videos"

    if opts.dry_run:
        res.rows = len(rows)
        res.media_count = sum(1 for _ in src_videos.glob("*.mp4")) if src_videos.exists() else 0
        res.notes.append(f"plan: write {tsv} and link {res.media_count} videos")
        return res

    if opts.force_clean:
        _wipe_dir(media_dst, opts.dry_run)
    _write_tsv(tsv, rows, columns, opts.dry_run)
    placed = _link_or_copy_dir(src_videos, media_dst, opts)
    res.rows = len(rows)
    res.media_count = placed
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# 2DPointing  (JSONL -> TSV)
# ---------------------------------------------------------------------------

POINTING_COLUMNS = ["index", "question_id", "image_path", "question", "A", "B", "C", "D"]


def _prep_pointing(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="pointing", lmu_name="VANTAGE_2DPointing",
                     target_dir=target_dir, status="dry-run")
    jsonl_path = snap_dir / "data" / "pointing" / "VANTAGE_2DPointing.jsonl"
    src_images = snap_dir / "data" / "pointing" / "images_annotated"
    if not jsonl_path.exists():
        raise SystemExit(
            f"Pointing JSONL missing in snapshot: {jsonl_path}\n"
            "The script targets the post-PR layout where pointing is JSONL. If the "
            "live nvidia/PhysicalAI-VANTAGE-Bench still ships a TSV, this task will "
            "fail until the PR merges."
        )
    res.source_files.append(jsonl_path.name)

    items = _load_jsonl(jsonl_path)
    rows: List[Dict[str, Any]] = []
    missing_images: List[str] = []
    for i, item in enumerate(items):
        row = {col: item.get(col, "") for col in POINTING_COLUMNS}
        # Re-index for safety in case source skips indices.
        row["index"] = item.get("index", i)
        # Sanity: image_path should resolve under images_annotated/
        ip = str(row.get("image_path", ""))
        if not ip.startswith("images_annotated/"):
            res.notes.append(f"unexpected image_path prefix: {ip}")
        else:
            rel = ip[len("images_annotated/"):]
            if src_images.exists() and not (src_images / rel).exists():
                missing_images.append(rel)
        rows.append(row)

    if missing_images:
        res.notes.append(f"{len(missing_images)} image paths not resolvable in snapshot")
        if opts.verbose:
            for m in missing_images[:5]:
                log.warning("pointing: missing image %s", m)

    tsv = target_dir / "VANTAGE_2DPointing.tsv"
    media_dst = target_dir / "images_annotated"

    if opts.dry_run:
        res.rows = len(rows)
        res.media_count = sum(1 for p in src_images.iterdir() if p.is_file()) if src_images.exists() else 0
        res.notes.append(f"plan: write {tsv} and link {res.media_count} images")
        return res

    if opts.force_clean:
        _wipe_dir(media_dst, opts.dry_run)
    _write_tsv(tsv, rows, POINTING_COLUMNS, opts.dry_run)
    placed = _link_or_copy_dir(src_images, media_dst, opts)
    res.rows = len(rows)
    res.media_count = placed
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# Astro2D
# ---------------------------------------------------------------------------

# The KITTI labels are NOT publicly released. The VLMEvalKit loader
# (vlmeval/dataset/vantage2d/astro_2d_dataset.py:_build_data_structure) silently
# DROPS any image that lacks a matching labels/<base>.txt file. To keep all
# images visible to the loader for inference, we emit zero-byte placeholder
# label files. They are NOT inferred or fabricated GT — parse_kitti_label on
# an empty file returns an empty object list, which is the no-GT semantic.
ASTRO2D_SEQUENCES = ("sequence_a", "sequence_b", "sequence_c")


def _prep_astro2d(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="astro2d", lmu_name="Astro2D", target_dir=target_dir, status="dry-run")
    src_root = snap_dir / "data" / "2dbbox"
    if not src_root.exists():
        raise SystemExit(f"Astro2D source missing in snapshot: {src_root}")

    images_dst = target_dir / "images"
    labels_dst = target_dir / "labels"

    if opts.force_clean:
        _wipe_dir(images_dst, opts.dry_run)
        _wipe_dir(labels_dst, opts.dry_run)
    _ensure_dir(images_dst, opts.dry_run)
    _ensure_dir(labels_dst, opts.dry_run)

    image_count = 0
    label_count = 0
    res.source_files.append("sequence_a/b/c/images/*")
    for seq in ASTRO2D_SEQUENCES:
        seq_dir = src_root / seq / "images"
        if not seq_dir.exists():
            res.notes.append(f"missing source: {seq}/images")
            continue
        for img in sorted(seq_dir.iterdir()):
            if not img.is_file():
                continue
            ext = img.suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                continue
            # Prefix with sequence name to avoid cross-sequence basename collisions.
            flat_name = f"{seq}_{img.name}"
            img_dst = images_dst / flat_name
            _link_or_copy_file(img, img_dst, opts)
            image_count += 1
            # Empty placeholder label (NOT inferred GT — see header comment).
            stem = Path(flat_name).stem
            lbl_dst = labels_dst / f"{stem}.txt"
            if not lbl_dst.exists() and not opts.dry_run:
                lbl_dst.touch()
                label_count += 1
            elif not lbl_dst.exists() and opts.dry_run:
                label_count += 1

    res.media_count = image_count
    res.notes.append(f"emitted {label_count} empty label placeholders (no-GT)")
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# Grounding (VANTAGE_2DGrounding)
# ---------------------------------------------------------------------------

def _grounding_build_annotations(ann_path: Path) -> List[Dict[str, Any]]:
    """Convert HF refdrone_test_llava.json -> loader-compatible no-GT entries.

    Output schema (per loader vlmeval/dataset/vantage2d/grounding_2d_dataset.py
    parse_refcoco_annotations no-GT branch):
        {image, sentence, width, height, category}
    No `bboxes` key is emitted (loader's no-GT branch is triggered when the
    first item lacks `bboxes`).
    """
    raw = _load_json(ann_path)
    if not isinstance(raw, list):
        raise SystemExit(f"unexpected grounding annotation shape in {ann_path}")
    out: List[Dict[str, Any]] = []
    for item in raw:
        meta = item.get("_meta", {}) or {}
        sentence = meta.get("sentence") or ""
        if not sentence:
            # Try to recover sentence from human conversation if _meta is absent.
            for conv in item.get("conversations", []):
                if conv.get("from") == "human":
                    txt = conv.get("value", "")
                    m = re.search(r'"(.+?)"', txt)
                    if m:
                        sentence = m.group(1)
                    break
        if not sentence:
            continue
        media = item.get("media") or ""
        image = os.path.basename(media) if media else ""
        if not image:
            continue
        out.append({
            "image": image,
            "sentence": sentence,
            "width": meta.get("image_width"),
            "height": meta.get("image_height"),
            "category": meta.get("object_category", ""),
        })
    return out


def _run_refdrone_prep_script(script_src: Path, work_root: Path, force: bool,
                              opts: Options) -> Path:
    """Stage prep_refdrone_data.py in a writable workdir at the depth it
    expects (<work_root>/scripts/refdrone/...), run it, and return the
    resulting images dir.

    The script hard-codes REPO_ROOT = __file__.parent.parent.parent, so the
    output ends up at <work_root>/LMUData/Spatial/2d_referring_expressions/refdrone/.
    """
    script_dir = work_root / "scripts" / "refdrone"
    script_dir.mkdir(parents=True, exist_ok=True)
    staged = script_dir / "prep_refdrone_data.py"
    if not staged.exists() or force:
        shutil.copy2(script_src, staged)
    cmd = [sys.executable, str(staged)]
    if force:
        cmd.append("--force")
    log.info("[grounding] running %s", " ".join(cmd))
    try:
        # Stream output so the user sees download progress.
        subprocess.run(cmd, check=True, cwd=str(work_root))
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"[grounding] prep_refdrone_data.py failed (exit {e.returncode}). "
            f"VisDrone mirrors may be down or unreachable. Re-run with --skip-grounding-images "
            f"if you already have images/, or retry later."
        ) from e
    except FileNotFoundError as e:
        raise SystemExit(f"[grounding] failed to launch prep script: {e}") from e

    images_dir = work_root / "LMUData" / "Spatial" / "2d_referring_expressions" / "refdrone" / "images"
    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise SystemExit(f"[grounding] prep script produced no images at {images_dir}")
    return images_dir


def _prep_grounding(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="grounding", lmu_name="VANTAGE_2DGrounding",
                     target_dir=target_dir, status="dry-run")
    src_root = snap_dir / "data" / "referring"
    # HF dataset uses this filename for grounding annotations (named after annotation format origin)
    ann_path = src_root / "refdrone_test_llava.json"
    prep_script = src_root / "prep_refdrone_data.py"
    if not ann_path.exists():
        raise SystemExit(f"Grounding annotation missing in snapshot: {ann_path}")
    if not prep_script.exists() and not opts.skip_grounding_images:
        raise SystemExit(
            f"Grounding prep script missing in snapshot: {prep_script}\n"
            "Pass --skip-grounding-images if you have pre-staged images."
        )
    res.source_files.append(ann_path.name)

    # 1) Convert annotations (no-GT entries).
    entries = _grounding_build_annotations(ann_path)
    ann_dst = target_dir / "annotations.json"
    log.info("write %s (%d entries, no-GT)", ann_dst, len(entries))
    if not opts.dry_run:
        ann_dst.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_dst, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    # 2) Materialize images.
    images_dst = target_dir / "images"
    if opts.force_clean:
        _wipe_dir(images_dst, opts.dry_run)
    _ensure_dir(images_dst, opts.dry_run)

    if opts.skip_grounding_images:
        res.notes.append("--skip-grounding-images: images/ left as-is")
        # Count whatever is already there for reporting.
        if images_dst.exists():
            res.media_count = sum(1 for p in images_dst.iterdir() if p.is_file())
    else:
        work_root = target_dir.parent.parent / ".work" / "grounding"
        work_root.mkdir(parents=True, exist_ok=True)
        try:
            src_images = _run_refdrone_prep_script(prep_script, work_root, opts.force, opts)
        except SystemExit:
            raise
        placed = _link_or_copy_dir(src_images, images_dst, opts)
        res.media_count = placed
        res.notes.append(f"linked {placed} VisDrone images from prep workdir")

    res.rows = len(entries)
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# SOT
# ---------------------------------------------------------------------------

def _sot_write_gt_jsons(benchmark_path: Path, target_dir: Path, opts: Options) -> Tuple[int, int]:
    """For each entry in sot_benchmark.jsonl, write <target>/<seq_id>/gt.json
    with public init_bbox only (no hidden trajectories).

    Returns (gt_written, frames_present).
    """
    items = _load_jsonl(benchmark_path)
    gt_written = 0
    frames_present = 0
    for item in items:
        seq_id = item.get("seq_id")
        if not seq_id:
            continue
        seq_dir = target_dir / seq_id
        if not seq_dir.exists():
            # prepare_sot_dataset.py may have skipped this seq (download failure, etc.)
            continue
        canonical = item.get("canonical_frame_ids") or []
        n_frames = len(canonical) if canonical else 8
        init_bbox = item.get("init_bbox")
        if init_bbox is None:
            log.warning("SOT: %s has no init_bbox in benchmark — skipping gt.json", seq_id)
            continue
        # NOTE: only frame 0 (init) gets a public bbox. Other frames are
        # absent on purpose — server-side scoring uses hidden trajectories.
        gt = {
            "label": f"{item.get('scene','')}/{item.get('camera','')}/"
                     f"{item.get('init_frame_id','')}/obj{item.get('object_id','')}",
            "scene": item.get("scene", ""),
            "camera": item.get("camera", ""),
            "object_id": str(item.get("object_id", "")),
            "object_type": item.get("object_type", "object"),
            "frame_ids": list(range(n_frames)),
            "source_frame_ids": canonical,
            "init_bbox": init_bbox,
            "gt_bboxes": {"0": init_bbox},
        }
        gt_path = seq_dir / "gt.json"
        if not opts.dry_run:
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt, f, indent=2)
        gt_written += 1
        frames_dir = seq_dir / "frames"
        if frames_dir.is_dir():
            frames_present += sum(1 for p in frames_dir.iterdir() if p.suffix == ".png")
    return gt_written, frames_present


def _discover_ffmpeg_dir() -> Optional[str]:
    """Return a directory containing an ffmpeg binary, or None.

    Searches PATH first, then common conda-env bin dirs (the prep script's own
    find_ffmpeg() does NOT look inside conda envs, so we bridge that gap by
    prepending the discovered dir to the subprocess PATH).
    """
    found = shutil.which("ffmpeg")
    if found:
        return os.path.dirname(found)
    import glob
    patterns = [
        os.path.expanduser("~/miniconda3/envs/*/bin/ffmpeg"),
        os.path.expanduser("~/anaconda3/envs/*/bin/ffmpeg"),
        "/opt/conda/envs/*/bin/ffmpeg",
        os.path.expanduser("~/miniconda3/bin/ffmpeg"),
        os.path.expanduser("~/anaconda3/bin/ffmpeg"),
        "/opt/conda/bin/ffmpeg",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return os.path.dirname(hits[0])
    return None


def _resolve_hf_token(cli_token: Optional[str]) -> Optional[str]:
    """Resolve an HF token: --hf-token, then HF_TOKEN env, then the token
    saved by `hf auth login` (huggingface_hub.get_token())."""
    if cli_token:
        return cli_token
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            return tok
    except Exception:
        pass
    return None


def _run_sot_prep_script(script_path: Path, benchmark_path: Path, output_dir: Path,
                         opts: Options) -> None:
    """Invoke prepare_sot_dataset.py to download videos + extract frames."""
    token = opts.hf_token  # already resolved (cli/env/stored) in main()
    if not token:
        raise SystemExit(
            "[sot] No HF token found. Provide one of:\n"
            "  - run `hf auth login` (token is then auto-detected), or\n"
            "  - export HF_TOKEN=hf_xxx, or\n"
            "  - pass --hf-token hf_xxx\n"
            f"Source videos come from {SOT_SOURCE_REPO_ID}. If it is gated, accept\n"
            f"the license at https://huggingface.co/datasets/{SOT_SOURCE_REPO_ID}"
        )
    # ffmpeg check — the prep script requires it for frame extraction.
    ffmpeg_dir = _discover_ffmpeg_dir()
    if ffmpeg_dir is None:
        raise SystemExit(
            "[sot] ffmpeg not found. Frame extraction in prepare_sot_dataset.py "
            "requires it. Easiest installs:\n"
            "  - conda install -c conda-forge ffmpeg\n"
            "  - (Debian/Ubuntu) sudo apt-get install -y ffmpeg\n"
            "  - or a static build from https://johnvansickle.com/ffmpeg/\n"
            "Then re-run. (Tip: an ffmpeg inside a conda env is auto-detected.)"
        )
    # Bridge conda-env ffmpeg onto the subprocess PATH (the prep script's
    # find_ffmpeg does not search conda envs).
    child_env = os.environ.copy()
    if shutil.which("ffmpeg") is None:
        child_env["PATH"] = ffmpeg_dir + os.pathsep + child_env.get("PATH", "")
        log.info("[sot] using ffmpeg from %s", ffmpeg_dir)
    cmd = [
        sys.executable, str(script_path),
        "--benchmark", str(benchmark_path),
        "--output-dir", str(output_dir),
        "--hf-token", token,
        "--repo-id", SOT_SOURCE_REPO_ID,
        "--repo-subdir", SOT_SOURCE_REPO_SUBDIR,
    ]
    if opts.hf_cache is not None:
        cmd += ["--hf-cache-dir", str(opts.hf_cache)]
    log.info("[sot] running %s", " ".join(cmd[:6] + ["..."]))
    try:
        subprocess.run(cmd, check=True, env=child_env)
    except subprocess.CalledProcessError as e:
        # Most common failure: 401/403 from HF (gated dataset).
        raise SystemExit(
            f"[sot] prepare_sot_dataset.py failed (exit {e.returncode}). "
            f"If HTTP 401/403: confirm license acceptance at "
            f"https://huggingface.co/datasets/{SOT_SOURCE_REPO_ID} and that your "
            f"HF token has read access."
        ) from e


def _prep_sot(snap_dir: Path, target_dir: Path, opts: Options) -> TaskResult:
    res = TaskResult(task="sot", lmu_name="VANTAGE_SOT",
                     target_dir=target_dir, status="dry-run")
    src_root = snap_dir / "data" / "tracking"
    benchmark = src_root / "sot_benchmark.jsonl"
    prep_script = src_root / "prepare_sot_dataset.py"
    if not benchmark.exists():
        raise SystemExit(f"SOT benchmark missing in snapshot: {benchmark}")
    if not prep_script.exists():
        raise SystemExit(f"SOT prep script missing in snapshot: {prep_script}")
    res.source_files.append(benchmark.name)

    if opts.force_clean:
        # Wipe per-seq dirs (but keep target_dir itself).
        if target_dir.exists():
            for child in list(target_dir.iterdir()):
                if child.is_dir():
                    _wipe_dir(child, opts.dry_run)
    _ensure_dir(target_dir, opts.dry_run)

    # 1) Run the prep script (downloads videos + extracts frames).
    _run_sot_prep_script(prep_script, benchmark, target_dir, opts)

    # 2) Write per-sequence gt.json from public init_bbox.
    gt_written, frames_present = _sot_write_gt_jsons(benchmark, target_dir, opts)
    res.rows = gt_written
    res.media_count = frames_present
    res.notes.append(f"wrote {gt_written} gt.json files (init_bbox only, no hidden trajectories)")
    res.notes.append(f"{frames_present} frame .png files present")
    res.status = "built"
    return res


# ---------------------------------------------------------------------------
# Deferred-task stubs
# ---------------------------------------------------------------------------

# (No deferred stubs in PHASE 2B — all eight tasks are implemented.)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PREP_FNS: Dict[str, Callable[..., TaskResult]] = {
    "vqa": _prep_vqa,
    "event_verification": _prep_event_verification,
    "dvc": _prep_dvc,
    "temporal": _prep_temporal,
    "pointing": _prep_pointing,
    "astro2d": _prep_astro2d,
    "grounding": _prep_grounding,
    "sot": _prep_sot,
}


def _dry_run_plan(task: str, target_dir: Path, opts: Options,
                  source_mode: str, source_root: Optional[Path]) -> TaskResult:
    """Pure dry-run summary. No disk writes. For hf mode no download happens."""
    cfg = TASK_CONFIG[task]
    res = TaskResult(
        task=task,
        lmu_name=cfg["lmu_name"],
        target_dir=target_dir,
        status="dry-run",
    )
    if source_mode in ("local-explicit", "local-auto"):
        res.notes.append(f"source: {source_mode}:{source_root}")
        # Per-task markers already validated by _resolve_source.
        res.notes.append(f"would read from local data/{_task_data_subdir(task)}/ "
                         f"(no HF download)")
    else:
        res.notes.append(f"source: hf:{opts.hf_repo}")
        res.notes.append(f"would download HF patterns: {cfg['hf_patterns']}")
    if task == "astro2d":
        res.notes.append(f"would flatten sequence_{{a,b,c}}/images -> {target_dir}/images/")
        res.notes.append(f"would emit empty placeholder labels at {target_dir}/labels/ (no-GT)")
    elif task == "grounding":
        res.notes.append(f"would write {target_dir}/annotations.json (no-GT)")
        if opts.skip_grounding_images:
            res.notes.append("would skip VisDrone image download (--skip-grounding-images)")
        else:
            res.notes.append("would invoke prep_refdrone_data.py to download VisDrone (~297 MB)")
        res.notes.append(f"would populate: {target_dir}/images/")
    elif task == "sot":
        res.notes.append(f"would invoke prepare_sot_dataset.py against {SOT_SOURCE_REPO_ID}")
        res.notes.append(f"would write per-seq <seq>/gt.json under {target_dir}/")
        token = opts.hf_token  # resolved (cli/env/stored) in main()
        if token:
            res.notes.append("preflight: HF token detected")
        else:
            res.notes.append("PREFLIGHT FAIL: no HF token (run `hf auth login`, "
                             "export HF_TOKEN, or pass --hf-token)")
        ffdir = _discover_ffmpeg_dir()
        if ffdir:
            res.notes.append(f"preflight: ffmpeg found ({ffdir})")
        else:
            res.notes.append("PREFLIGHT FAIL: ffmpeg not found "
                             "(conda install -c conda-forge ffmpeg)")
    else:
        idx_name = cfg.get("index_file")
        if idx_name:
            res.notes.append(f"would write: {target_dir / idx_name}")
        res.notes.append(f"would populate: {target_dir / cfg['media_dir']}/")
    return res


def _run_task(task: str, opts: Options) -> TaskResult:
    target_dir = _target_dir(opts.lmu_root, task)

    # Idempotency: skip if integrity passes and not forcing
    if not opts.force and target_dir.exists():
        ok, why = _check_integrity(task, target_dir)
        if ok:
            log.info("[%s] skip — already populated at %s", task, target_dir)
            return TaskResult(
                task=task,
                lmu_name=TASK_CONFIG[task]["lmu_name"],
                target_dir=target_dir,
                status="skipped",
                notes=[why],
            )
        else:
            log.info("[%s] partial state (%s) — will rebuild", task, why)

    # Resolve where the source data comes from (validates local sources;
    # for hf mode this downloads, unless dry-run, in which case it returns None).
    source_root, source_mode = _resolve_source(task, opts)

    # Dry-run short-circuits before any disk write.
    if opts.dry_run:
        log.info("[%s] dry-run (source=%s) — no writes", task, source_mode)
        res = _dry_run_plan(task, target_dir, opts, source_mode, source_root)
        res.source_mode = source_mode
        return res

    _ensure_dir(target_dir, opts.dry_run)
    fn = PREP_FNS[task]
    res = fn(source_root, target_dir, opts)
    res.source_mode = source_mode
    if opts.force and res.status == "built":
        res.status = "rebuilt"

    if res.status in ("built", "rebuilt"):
        ok, why = _check_integrity(task, target_dir)
        if not ok:
            res.notes.append(f"post-build integrity check FAILED: {why}")
            res.status = "failed"
        else:
            res.notes.append(f"integrity ok: {why}")
    return res


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(lmu_root: Path, results: List[TaskResult], opts: Options) -> None:
    manifest_path = lmu_root / MANIFEST_FILENAME
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hf_repo": opts.hf_repo,
        "hf_repo_production_default": HF_REPO_ID,
        "hf_repo_type": HF_REPO_TYPE,
        "lmu_root": str(lmu_root),
        "local_source": str(opts.local_source) if opts.local_source else None,
        "options": {
            "media_mode": "symlink" if opts.symlink else "copy",
            "force": opts.force,
            "force_clean": opts.force_clean,
            "dry_run": opts.dry_run,
            "skip_grounding_images": opts.skip_grounding_images,
        },
        "tasks": [
            {
                "task": r.task,
                "lmu_name": r.lmu_name,
                "target_dir": str(r.target_dir),
                "status": r.status,
                "source_mode": r.source_mode,
                "rows": r.rows,
                "media_count": r.media_count,
                "source_files": r.source_files,
                "notes": r.notes,
            }
            for r in results
        ],
    }
    if opts.dry_run:
        log.info("would write manifest %s (--write-manifest)", manifest_path)
        return
    log.info("write manifest %s", manifest_path)
    lmu_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tasks(raw: Optional[str], all_flag: bool) -> List[str]:
    if all_flag:
        return list(IMPLEMENTED_TASKS)  # deferred tasks intentionally excluded
    if not raw:
        return list(IMPLEMENTED_TASKS)
    out: List[str] = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t == "all":
            return list(IMPLEMENTED_TASKS)
        if t not in ALL_TASKS:
            raise SystemExit(f"unknown task '{t}' (choose from: {','.join(ALL_TASKS)})")
        out.append(t)
    if not out:
        raise SystemExit("no tasks selected")
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_lmudata.py",
        description=(
            "Prepare an LMUData layout for VLMEvalKit from "
            "nvidia/PhysicalAI-VANTAGE-Bench (no-GT inference layout)."
        ),
    )
    p.add_argument("--tasks", type=str, default=None,
                   help=f"comma-separated tasks (choices: {','.join(ALL_TASKS)}). Default = all implemented.")
    p.add_argument("--all", action="store_true",
                   help="alias for --tasks=all (implemented tasks only)")
    p.add_argument("--lmu-root", type=str, default=None,
                   help=f"LMUData output root (default: {DEFAULT_LMU_ROOT})")
    p.add_argument("--local-source", type=str, default=None,
                   help="Path to a local PhysicalAI-VANTAGE-Bench checkout. Use its data/ "
                        "folder directly instead of downloading from HF. Takes precedence "
                        "over --hf-repo. (Auto-enabled when this script lives inside such a repo.)")
    p.add_argument("--hf-repo", type=str, default=HF_REPO_ID,
                   help=(f"HF dataset repo id (default: {HF_REPO_ID}). "
                         "Override is for testing/simulation only — production runs MUST use the default. "
                         "Ignored when a local source is active."))
    p.add_argument("--hf-cache", type=str, default=None,
                   help="Override HF hub cache_dir")
    p.add_argument("--hf-token", type=str, default=None,
                   help=("HF token (or set HF_TOKEN env). Required for the SOT task, "
                         "which downloads source camera videos from the gated "
                         "nvidia/PhysicalAI-SmartSpaces dataset."))
    media = p.add_mutually_exclusive_group()
    media.add_argument("--symlink", dest="symlink", action="store_true",
                       help="symlink media into the HF cache (default) — saves disk, "
                            "but LMUData depends on the HF cache staying in place")
    media.add_argument("--copy", dest="symlink", action="store_false",
                       help="copy media files into LMUData instead of symlinking "
                            "(self-contained / portable; duplicates tens of GB of media)")
    p.set_defaults(symlink=True)
    p.add_argument("--force", action="store_true",
                   help="rebuild index files even if integrity check passes")
    p.add_argument("--force-clean", action="store_true",
                   help="wipe existing media dirs before relinking (destructive)")
    p.add_argument("--dry-run", action="store_true",
                   help="print plan, do not write files or call HF")
    p.add_argument("--skip-grounding-images", action="store_true",
                   help="skip VisDrone image download for grounding (use pre-staged images/)")
    p.add_argument("--write-manifest", action="store_true",
                   help="write a .vantage_prep_manifest.json telemetry file at the LMU root "
                        "(off by default; participant LMUData stays clean)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    local_source = None
    if args.local_source:
        local_source = Path(args.local_source).expanduser().resolve()

    opts = Options(
        lmu_root=_resolve_lmu_root(args.lmu_root),
        hf_cache=Path(args.hf_cache).expanduser().resolve() if args.hf_cache else None,
        hf_token=_resolve_hf_token(args.hf_token),
        symlink=args.symlink,
        force=args.force,
        force_clean=args.force_clean,
        dry_run=args.dry_run,
        verbose=args.verbose,
        hf_repo=args.hf_repo,
        skip_grounding_images=args.skip_grounding_images,
        write_manifest=args.write_manifest,
        local_source=local_source,
    )

    # Determine the effective source for logging/summary (per-task validation
    # still happens in _resolve_source).
    auto_root = None if opts.local_source else _autodetect_local_root()
    if opts.local_source:
        source_label = f"local-explicit:{opts.local_source}"
        local_active = True
    elif auto_root is not None:
        source_label = f"local-auto:{auto_root}"
        local_active = True
    else:
        source_label = f"hf:{opts.hf_repo}"
        local_active = False

    if local_active and opts.hf_repo != HF_REPO_ID:
        log.warning("--hf-repo %s is IGNORED because a local source is active (%s).",
                    opts.hf_repo, source_label)
    elif opts.hf_repo != HF_REPO_ID:
        log.warning("--hf-repo override active: %s (production default: %s)",
                    opts.hf_repo, HF_REPO_ID)

    tasks = _parse_tasks(args.tasks, args.all)

    log.info("VANTAGE prep — source=%s lmu_root=%s tasks=%s dry_run=%s media=%s force=%s",
             source_label, opts.lmu_root, ",".join(tasks), opts.dry_run,
             "symlink" if opts.symlink else "copy", opts.force)
    if not opts.dry_run:
        opts.lmu_root.mkdir(parents=True, exist_ok=True)
        (opts.lmu_root / "datasets").mkdir(parents=True, exist_ok=True)

    results: List[TaskResult] = []
    for task in tasks:
        try:
            res = _run_task(task, opts)
        except SystemExit as e:
            # Per-task SystemExits (missing source dir, missing HF token, etc.)
            # become per-task failures so other tasks can still proceed.
            msg = str(e) if str(e) else f"SystemExit code {e.code!r}"
            log.error("[%s] %s", task, msg)
            res = TaskResult(
                task=task,
                lmu_name=TASK_CONFIG.get(task, {}).get("lmu_name", task),
                target_dir=_target_dir(opts.lmu_root, task) if task in TASK_CONFIG else opts.lmu_root,
                status="failed",
                notes=[msg[:500]],
            )
        except Exception as e:
            log.exception("[%s] failed: %s", task, e)
            res = TaskResult(
                task=task,
                lmu_name=TASK_CONFIG.get(task, {}).get("lmu_name", task),
                target_dir=_target_dir(opts.lmu_root, task) if task in TASK_CONFIG else opts.lmu_root,
                status="failed",
                notes=[f"exception: {e!r}"],
            )
        results.append(res)

    if opts.write_manifest:
        _write_manifest(opts.lmu_root, results, opts)

    # Summary
    print()
    print("=" * 78)
    print("VANTAGE prep summary"
          + ("  [TEST OVERRIDE]" if (not local_active and opts.hf_repo != HF_REPO_ID) else ""))
    print(f"Source:   {source_label}")
    print(f"LMU root: {opts.lmu_root}")
    print(f"Mode:     {'DRY-RUN' if opts.dry_run else 'WRITE'}  "
          f"media={'symlink' if opts.symlink else 'copy'}  "
          f"force={opts.force}  force_clean={opts.force_clean}")
    print("-" * 78)
    print(f"{'task':<28}{'status':<12}{'rows':>8}{'media':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r.lmu_name:<28}{r.status:<12}{r.rows:>8}{r.media_count:>10}")
        for note in r.notes:
            print(f"    - {note}")
    print("=" * 78)
    if opts.write_manifest:
        print(f"Manifest: {opts.lmu_root / MANIFEST_FILENAME}")
    return 0 if all(r.status in ("built", "rebuilt", "skipped", "deferred", "dry-run") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
