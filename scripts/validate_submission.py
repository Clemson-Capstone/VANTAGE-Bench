#!/usr/bin/env python3
"""
validate_submission.py — Safety net for VANTAGE-Bench *_submission.jsonl
files before upload.

``emit_submission()`` (vlmeval/dataset/utils/vantagebench/emit.py) never
raises: any failure during submission-JSONL writing becomes a
``warnings.warn`` and the run exits 0 anyway, because the legacy prediction
xlsx is still written. That means a submission JSONL can be missing,
truncated, or malformed while everything else about the run looked fine.
This script is the check that catches that before upload.

Validates, per task file:
  - every line is valid JSON
  - every record has {id, task, conversations, metadata} with a resolvable
    assistant turn
  - `task` matches the canonical task string this task actually emits
    (NOTE: this is a long-form string like "video_qa", not the short key
    "vqa" — see the drift note in the module docstring below)
  - `id` matches the canonical shape for its task (vlmeval...id_rules.py)
  - no duplicate ids within a file
  - (with --lmu-root) record count / id-set cross-check against the source
    TSV or dataset dir
  - empty/whitespace predictions and API-failure strings
  - degenerate single-letter answer distribution for MCQ-style tasks
  - pillar completeness (reusing package_submission.py's PILLAR_TASKS)

Usage:
    python scripts/validate_submission.py --work-dir ./outputs/<model>/<eval_id>
    python scripts/validate_submission.py --archive submission.tar.gz
    python scripts/validate_submission.py --work-dir ./outputs/<model>/<eval_id> \\
        --lmu-root ~/LMUData --json

Exit code: 0 only if zero blockers (malformed JSON, wrong id shape,
duplicate ids, missing required keys) across all files found. Warnings
(empty predictions, degenerate distribution, count mismatch) are reported
but do not affect the exit code.

--- Drift note (read this before trusting "task" field checks elsewhere) ---
docs/vantage/SUBMISSION.md, vlmeval/dataset/utils/vantagebench/README.md and
skills/reference/tasks-and-pillars.md all document the record shape as
``{"role": "assistant", "content": "..."}`` and the `task` field as the
short key (e.g. "vqa"). The actual emitter (emit.py) writes
``{"from": "assistant", "value": "..."}`` and a long-form canonical task
string (e.g. "video_qa" for vqa; see validate.py's _TASK_REGISTRY). This
script validates against what the code actually produces, not the docs, and
accepts either turn shape defensively. See the final task report for the
full list of short-key -> canonical-task-string mappings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FAIL_MSG = 'Failed to obtain answer via API.'  # mirrors scripts/apires_scan.py's
# FAIL_MSG constant. Not imported from there: that script runs sys.argv-driven
# code at module scope and isn't safe to import.

STATUS_ORDER = {"ok": 0, "warning": 1, "blocker": 2}

# Regex-only shape check (no source data required). Derived from the format
# strings documented in id_rules.py's docstrings; the authoritative check is
# calling the real make_*_id() generators when --lmu-root is given.
TASK_ID_REGEX = {
    "vqa": re.compile(r"^.+__q_\d{6}$"),
    "event_verification": re.compile(r"^.+__ev_\d{6}$"),
    "temporal": re.compile(r"^.+__tg_\d{6}$"),
    "dvc": re.compile(r"^.+__dvc_\d{6}$"),
    "grounding": re.compile(r"^.+__rx_\d{6}$"),
    "pointing": re.compile(r"^.+__sp_\d{6}$"),
    "astro": re.compile(r"^.+__ol_\d{6}$"),
    "sot": re.compile(r"^.+_\d{7}__obj.+$"),
}

MCQ_TASKS = {"vqa", "pointing"}
MCQ_LETTER_RE = re.compile(r"^[A-D]$")

# Where each task's source lives under $LMUData/datasets/, per
# scripts/RUN_LMUData.md section K / run_lmudata.py's TASK_CONFIG.
LMU_TASK_LAYOUT = {
    "vqa": {"lmu_name": "VANTAGE_VQA", "tsv": "VANTAGE_VQA.tsv"},
    "event_verification": {"lmu_name": "VANTAGE_EventVerification", "tsv": "VANTAGE_EventVerification.tsv"},
    "temporal": {"lmu_name": "VANTAGE_Temporal", "tsv": "VANTAGE_Temporal.tsv"},
    "dvc": {"lmu_name": "VANTAGE_DVC", "tsv": "VANTAGE_DVC.tsv"},
    "pointing": {"lmu_name": "VANTAGE_2DPointing", "tsv": "VANTAGE_2DPointing.tsv"},
    "grounding": {"lmu_name": "VANTAGE_2DGrounding"},
    "astro": {"lmu_name": "Astro2D"},
    "sot": {"lmu_name": "VANTAGE_SOT"},
}

def _load_vantagebench_helpers():
    """Import the id_rules generators + validate.py's task registry —
    deliberately deferred to a function called from main() *after*
    argparse has run, not at module import time.

    Importing anything under vlmeval.* — even a leaf module like id_rules.py
    that itself only uses `re` — forces Python to first execute
    vlmeval/__init__.py, which unconditionally does `from .dataset import *`,
    `from .vlm import *`, `from .config import *` (the entire model zoo).
    That makes even `--help` pay for a fully working torch/numpy/vlm-zoo
    install. run_lmudata.py and package_submission.py sidestep this by not
    importing vlmeval at all; this script can't avoid it (the id generators
    must come from id_rules.py per the reference doc), so the best available
    mitigation is deferring the cost past argument parsing.
    """
    from types import SimpleNamespace
    from vlmeval.dataset.utils.vantagebench.id_rules import (
        make_astro_id,
        make_dvc_id,
        make_event_verification_id,
        make_grounding_id,
        make_pointing_id,
        make_sot_id,
        make_temporal_id,
        make_vqa_id,
    )
    # _TASK_REGISTRY: short-key -> canonical task string actually written
    # into each record's "task" field. This is the real source of truth
    # (see the module drift note above); tasks-and-pillars.md's short keys
    # are not what is on disk.
    from vlmeval.dataset.utils.vantagebench.validate import _TASK_REGISTRY

    return SimpleNamespace(
        make_vqa_id=make_vqa_id,
        make_event_verification_id=make_event_verification_id,
        make_temporal_id=make_temporal_id,
        make_dvc_id=make_dvc_id,
        make_grounding_id=make_grounding_id,
        make_pointing_id=make_pointing_id,
        make_astro_id=make_astro_id,
        make_sot_id=make_sot_id,
        task_registry=_TASK_REGISTRY,
        id_generators={
            "vqa": make_vqa_id,
            "event_verification": make_event_verification_id,
            "temporal": make_temporal_id,
            "dvc": make_dvc_id,
            "pointing": make_pointing_id,
        },
    )


def _load_package_submission():
    """Import scripts/package_submission.py by path (scripts/ has no
    __init__.py) so its TASK_PATTERNS/PILLAR_TASKS/find_submission_files
    stay the single source of truth instead of being re-derived here."""
    path = Path(__file__).resolve().parent / "package_submission.py"
    spec = importlib.util.spec_from_file_location("_vantage_package_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pkg = _load_package_submission()
TASK_PATTERNS = _pkg.TASK_PATTERNS
PILLAR_TASKS = _pkg.PILLAR_TASKS


class Report:
    def __init__(self):
        self.findings: List[dict] = []

    def add(self, check: str, status: str, message: str) -> None:
        assert status in STATUS_ORDER
        self.findings.append({"check": check, "status": status, "message": message})

    def worst_status(self) -> str:
        if not self.findings:
            return "ok"
        return max((f["status"] for f in self.findings), key=lambda s: STATUS_ORDER[s])


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_files_work_dir(work_dir: Path) -> Dict[str, Path]:
    return _pkg.find_submission_files(work_dir)


def find_files_archive(archive_path: Path) -> Tuple[Dict[str, Path], Path]:
    """Extract task jsonl members from a package_submission.py-built
    .tar.gz (members are named exactly '<task_key>.jsonl'). Returns
    (found, tmp_dir) — caller is not required to clean up tmp_dir."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="vantage_validate_"))
    # tarfile's `filter=` kwarg (safe-extraction, PEP 706) only exists on
    # Python 3.12+; setup.py declares python_requires>=3.10, so guard it.
    extract_kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
    found: Dict[str, Path] = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        for task_key, _pattern in TASK_PATTERNS:
            name = f"{task_key}.jsonl"
            if name in members:
                tar.extract(members[name], path=tmp_dir, **extract_kwargs)
                found[task_key] = tmp_dir / name
    return found, tmp_dir


# ---------------------------------------------------------------------------
# Record-level helpers
# ---------------------------------------------------------------------------

def _load_jsonl_tolerant(path: Path) -> Tuple[List[dict], List[Tuple[int, str]]]:
    """Parse JSONL, collecting (line_no, error) for lines that don't parse
    instead of raising on the first bad line."""
    records: List[dict] = []
    errors: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append((i, str(e)))
    return records, errors


def _extract_assistant_content(record: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (content, shape, error). shape is "canonical" for the
    {"from": "assistant", "value": ...} turns emit.py actually writes, or
    "legacy" for the {"role": "assistant", "content": ...} shape that used to
    be (wrongly) documented across this repo — accepted for backward
    compatibility, but callers should warn on it: a submission built from the
    stale docs will silently produce turns the leaderboard doesn't expect."""
    convs = record.get("conversations")
    if not isinstance(convs, list) or not convs:
        return None, None, "conversations missing or not a non-empty list"
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") == "assistant" and isinstance(turn.get("value"), str):
            return turn["value"], "canonical", None
        if turn.get("role") == "assistant" and isinstance(turn.get("content"), str):
            return turn["content"], "legacy", None
    return None, None, "no assistant turn found (checked from/value and role/content shapes)"


# ---------------------------------------------------------------------------
# --lmu-root cross-check
# ---------------------------------------------------------------------------

def _source_ids_and_count(task_key: str, lmu_root: Path, vb) -> Tuple[Optional[set], Optional[int], str]:
    """Best-effort recomputation of expected ids + row count from the
    source LMUData layout. Returns (expected_ids_or_None, count_or_None,
    note-describing-confidence)."""
    layout = LMU_TASK_LAYOUT[task_key]
    target_dir = lmu_root / "datasets" / layout["lmu_name"]

    if task_key in vb.id_generators:
        tsv_path = target_dir / layout["tsv"]
        if not tsv_path.exists():
            return None, None, f"source TSV not found at {tsv_path}"
        from vlmeval.smp import load
        df = load(str(tsv_path))
        gen = vb.id_generators[task_key]
        col = "image_path" if task_key == "pointing" else "video"
        ids = {gen(row[col], row["index"]) for _, row in df.iterrows()}
        return ids, len(df), "exact (recomputed from source TSV video/index columns)"

    if task_key == "grounding":
        ann_path = target_dir / "annotations.json"
        if not ann_path.exists():
            return None, None, f"annotations.json not found at {ann_path}"
        with open(ann_path, encoding="utf-8") as f:
            entries = json.load(f)
        ids = {vb.make_grounding_id(e.get("image", ""), i) for i, e in enumerate(entries)}
        return ids, len(entries), "best-effort (assumes annotation order == inference row order)"

    if task_key == "astro":
        images_dir = target_dir / "images"
        if not images_dir.exists():
            return None, None, f"images/ not found at {images_dir}"
        names = sorted(p.name for p in images_dir.iterdir() if p.is_file())
        ids = {vb.make_astro_id(n, i) for i, n in enumerate(names)}
        return ids, len(names), "best-effort (assumes sorted-filename order == inference row order)"

    if task_key == "sot":
        if not target_dir.exists():
            return None, None, f"target dir not found at {target_dir}"
        seq_names = [d.name for d in target_dir.iterdir()
                     if d.is_dir() and (d / "gt.json").exists() and (d / "frames").is_dir()]
        ids = {vb.make_sot_id(n) for n in seq_names}
        return ids, len(seq_names), "exact (make_sot_id needs only the sequence dir name)"

    return None, None, "no source-count method implemented for this task"


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------

def validate_file(task_key: str, path: Path, report: Report, lmu_root: Optional[Path], vb) -> dict:
    prefix = f"{task_key}"
    detail: dict = {"file": str(path), "task_key": task_key}

    records, parse_errors = _load_jsonl_tolerant(path)
    detail["records"] = len(records)
    detail["parse_errors"] = len(parse_errors)
    if parse_errors:
        sample = "; ".join(f"line {ln}: {msg}" for ln, msg in parse_errors[:3])
        report.add(f"{prefix}:malformed_json", "blocker",
                    f"{len(parse_errors)} line(s) failed to parse as JSON in {path.name}. e.g. {sample}")

    expected_task = vb.task_registry.get(task_key)
    id_regex = TASK_ID_REGEX.get(task_key)

    malformed = 0
    task_mismatches = 0
    bad_id_shape = 0
    seen_ids: Dict[str, int] = {}
    duplicates = 0
    empty_content = 0
    fail_msg_count = 0
    legacy_shape = 0
    non_empty_contents: List[str] = []

    for pos, r in enumerate(records):
        if not isinstance(r, dict):
            malformed += 1
            continue
        rid = r.get("id")
        task_val = r.get("task")
        has_conv = "conversations" in r
        has_meta = "metadata" in r
        if not isinstance(rid, str) or not rid or task_val is None or not has_conv or not has_meta:
            malformed += 1
            continue
        content, shape, err = _extract_assistant_content(r)
        if err is not None:
            malformed += 1
            continue
        if shape == "legacy":
            legacy_shape += 1

        if expected_task is not None and task_val != expected_task:
            task_mismatches += 1

        if id_regex is not None and not id_regex.match(rid):
            bad_id_shape += 1

        if rid in seen_ids:
            duplicates += 1
        else:
            seen_ids[rid] = pos

        if not content.strip():
            empty_content += 1
        else:
            non_empty_contents.append(content)
            if FAIL_MSG in content:
                fail_msg_count += 1

    detail.update({
        "malformed_records": malformed,
        "task_mismatches": task_mismatches,
        "bad_id_shape": bad_id_shape,
        "duplicate_ids": duplicates,
        "empty_content": empty_content,
        "fail_msg_count": fail_msg_count,
        "legacy_shape": legacy_shape,
        "unique_ids": len(seen_ids),
    })

    if malformed:
        report.add(f"{prefix}:required_keys", "blocker",
                    f"{malformed} record(s) missing id/task/conversations/metadata or an assistant "
                    f"turn value, out of {len(records)} in {path.name}.")
    if task_mismatches:
        report.add(f"{prefix}:task_field", "blocker",
                    f"{task_mismatches} record(s) have task != '{expected_task}' (the canonical "
                    f"string this task actually emits — see id_rules/validate.py) in {path.name}.")
    if bad_id_shape:
        report.add(f"{prefix}:id_shape", "blocker",
                    f"{bad_id_shape} record(s) have an id not matching the canonical shape "
                    f"{id_regex.pattern if id_regex else '(n/a)'} in {path.name}.")
    if duplicates:
        report.add(f"{prefix}:duplicate_ids", "blocker",
                    f"{duplicates} duplicate id(s) in {path.name}.")
    if not malformed and not task_mismatches and not bad_id_shape and not duplicates:
        report.add(f"{prefix}:structure", "ok", f"{len(records)} record(s), all structurally valid.")
    if legacy_shape:
        report.add(f"{prefix}:legacy_turn_shape", "warning",
                    f"{legacy_shape} record(s) in {path.name} use the legacy "
                    f"{{'role': 'assistant', 'content': ...}} turn shape instead of the canonical "
                    f"{{'from': 'assistant', 'value': ...}} shape emit.py actually writes. Accepted "
                    f"here for compatibility, but this may not be what the leaderboard scorer "
                    f"expects — regenerate via run.py's evaluate() rather than a custom writer.")

    total = len(records)
    if total:
        empty_rate = empty_content / total
        if empty_content:
            report.add(f"{prefix}:empty_content", "warning",
                        f"{empty_content}/{total} ({empty_rate * 100:.1f}%) record(s) have empty/"
                        f"whitespace-only prediction content in {path.name}.")
        if fail_msg_count:
            fail_rate = fail_msg_count / total
            report.add(f"{prefix}:api_failures", "warning",
                        f"{fail_msg_count}/{total} ({fail_rate * 100:.1f}%) record(s) contain the "
                        f"API-failure string {FAIL_MSG!r} in {path.name}.")

    # Degenerate MCQ answer distribution.
    if task_key in MCQ_TASKS and non_empty_contents:
        letters = [c.strip().upper() for c in non_empty_contents if MCQ_LETTER_RE.match(c.strip().upper())]
        if letters:
            from collections import Counter
            counts = Counter(letters)
            letter, n = counts.most_common(1)[0]
            frac = n / len(non_empty_contents)
            detail["mcq_top_letter"] = letter
            detail["mcq_top_fraction"] = round(frac, 3)
            if frac > 0.8:
                report.add(f"{prefix}:mcq_degenerate", "warning",
                            f"answer '{letter}' makes up {frac * 100:.1f}% of {len(non_empty_contents)} "
                            f"non-empty predictions in {path.name} — looks like a parsing bug, not "
                            f"genuine model behavior.")

    # --lmu-root cross-check.
    if lmu_root is not None:
        expected_ids, source_count, note = _source_ids_and_count(task_key, lmu_root, vb)
        detail["source_cross_check"] = {"source_count": source_count, "note": note}
        if source_count is None:
            report.add(f"{prefix}:count_cross_check", "warning",
                        f"could not cross-check against source data: {note}")
        else:
            if total != source_count:
                report.add(f"{prefix}:count_cross_check", "warning",
                            f"{total} record(s) in {path.name} vs {source_count} in the source "
                            f"({note}) — {'under' if total < source_count else 'over'}-count by "
                            f"{abs(total - source_count)}.")
            else:
                report.add(f"{prefix}:count_cross_check", "ok",
                            f"record count matches source ({total}); {note}.")
            if expected_ids is not None:
                sub_ids = set(seen_ids)
                missing = expected_ids - sub_ids
                extra = sub_ids - expected_ids
                if missing or extra:
                    report.add(f"{prefix}:id_cross_check", "warning",
                                f"id set differs from recomputed source ids ({note}): "
                                f"{len(missing)} missing, {len(extra)} unexpected.")
                else:
                    report.add(f"{prefix}:id_cross_check", "ok",
                                f"id set matches recomputed source ids exactly ({note}).")

    return detail


# ---------------------------------------------------------------------------
# Pillar coverage
# ---------------------------------------------------------------------------

def pillar_coverage(found_tasks: set) -> dict:
    out = {}
    for pillar, tasks in PILLAR_TASKS.items():
        covered = [t for t in tasks if t in found_tasks]
        missing = [t for t in tasks if t not in found_tasks]
        out[pillar] = {"complete": not missing, "covered": covered, "missing": missing}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_submission.py",
        description="Validate VANTAGE-Bench *_submission.jsonl files before upload.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--work-dir", type=Path, default=None,
                      help="Directory containing *_submission.jsonl files (same convention as "
                           "package_submission.py).")
    src.add_argument("--archive", type=Path, default=None,
                      help="A built submission .tar.gz (as produced by package_submission.py).")
    p.add_argument("--lmu-root", type=str, default=None,
                    help="$LMUData root, used to cross-check record counts/ids against the source "
                         "TSVs/dataset dirs under <lmu-root>/datasets/<Task>/. Optional.")
    p.add_argument("--json", action="store_true", help="print a single JSON report object instead of text")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = Report()

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser().resolve()
        if not work_dir.is_dir():
            print(f"Error: --work-dir does not exist: {work_dir}")
            return 1
        found = find_files_work_dir(work_dir)
        source_desc = str(work_dir)
    else:
        archive_path = args.archive.expanduser().resolve()
        if not archive_path.is_file():
            print(f"Error: --archive does not exist: {archive_path}")
            return 1
        found, _tmp_dir = find_files_archive(archive_path)
        source_desc = str(archive_path)

    if not found:
        report.add("discovery", "blocker", f"no submission files found under {source_desc}.")
        found = {}
    else:
        report.add("discovery", "ok", f"found {len(found)} task file(s) under {source_desc}: "
                                       f"{', '.join(sorted(found))}.")

    lmu_root = Path(args.lmu_root).expanduser().resolve() if args.lmu_root else None
    if lmu_root is not None and not lmu_root.exists():
        report.add("lmu_root", "warning", f"--lmu-root {lmu_root} does not exist; skipping source cross-checks.")
        lmu_root = None

    # Deferred until here (see _load_vantagebench_helpers docstring): --help
    # and a bad --work-dir/--archive already exited above without needing a
    # working vlmeval install.
    vb = _load_vantagebench_helpers()

    files_detail: Dict[str, dict] = {}
    for task_key, path in sorted(found.items()):
        files_detail[task_key] = validate_file(task_key, path, report, lmu_root, vb)

    coverage = pillar_coverage(set(found))
    for pillar, info in coverage.items():
        if info["complete"]:
            report.add(f"pillar:{pillar}", "ok", f"complete ({', '.join(info['covered'])}).")
        elif info["covered"]:
            report.add(f"pillar:{pillar}", "warning",
                        f"incomplete — have {info['covered']}, missing {info['missing']}. A partially "
                        f"submitted pillar is not scored; submit all its tasks or none.")
        # A pillar with zero files present is not a finding — it simply wasn't attempted.

    status = report.worst_status()

    if args.json:
        print(json.dumps({
            "status": status,
            "findings": report.findings,
            "files": files_detail,
            "pillar_coverage": coverage,
        }, indent=2))
    else:
        print("=" * 78)
        print("VANTAGE-Bench submission validation")
        print(f"Source: {source_desc}")
        print("=" * 78)
        width = max((len(f["check"]) for f in report.findings), default=10) + 2
        for f in report.findings:
            tag = {"ok": "OK", "warning": "WARN", "blocker": "BLOCK"}[f["status"]]
            print(f"[{tag:<5}] {f['check']:<{width}} {f['message']}")
        print("-" * 78)
        print(f"Overall status: {status.upper()}")
        if status == "blocker":
            print("At least one blocker was found — fix it before packaging/uploading.")
        print("=" * 78)

    return 0 if status != "blocker" else 1


if __name__ == "__main__":
    sys.exit(main())
