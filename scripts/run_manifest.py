#!/usr/bin/env python3
"""
run_manifest.py — Produce a structured manifest and a draft submission-portal
form for a completed VANTAGE-Bench run.

Given ``--work-dir outputs/<model>/<eval_id>/`` (the layout run.py produces —
see README_VANTAGE.md section 9), writes two files into that directory:

  - run_manifest.json  — git commit/branch/dirty flag, python + key package
    versions, GPU inventory, model name + dataset keys inferred from
    filenames, per-task record counts, and file mtimes as a rough timing
    signal. Secret-looking env vars (*API_KEY*, *TOKEN*, *SECRET*) are
    recorded as booleans only — their values are never written.

  - form_metadata.md  — a human-readable draft of every field on the
    submission portal (https://vantage-bench.org/submit), pre-filled from
    what's derivable from the run, with <FILL IN> placeholders for anything
    that needs human judgment. This script never submits anything.

Usage:
    python scripts/run_manifest.py --work-dir ./outputs/<model>/<eval_id>
    python scripts/run_manifest.py --work-dir ./outputs/<model>/<eval_id> --json

Exit code: 0 if at least one task prediction file was found and both files
were written; 1 if --work-dir doesn't exist or no prediction files are found
in it (nothing to manifest).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STATUS_ORDER = {"ok": 0, "warning": 1, "blocker": 2}
SECRET_SUBSTRINGS = ("API_KEY", "TOKEN", "SECRET")
PRED_EXTENSIONS = (".xlsx", ".tsv")  # covers the default (xlsx) and --copy tsv PRED_FORMAT;
# a PRED_FORMAT=json override is not disambiguated from metrics .json files and is out of scope.


def _load_package_submission():
    """Import scripts/package_submission.py by path so TASK_PATTERNS/
    PILLAR_TASKS stay the single source of truth instead of being
    re-derived here (scripts/ has no __init__.py, hence importlib)."""
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
# Git
# ---------------------------------------------------------------------------

def _git_info(report: Report) -> dict:
    def run(args):
        return subprocess.run(["git"] + args, cwd=str(_REPO_ROOT), capture_output=True,
                               text=True, timeout=10)

    try:
        head = run(["rev-parse", "HEAD"])
        if head.returncode != 0:
            report.add("git", "warning", f"not a git repo (or git unavailable) at {_REPO_ROOT}.")
            return {"commit": None, "branch": None, "dirty": None}
        commit = head.stdout.strip()
        branch = run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip() or None
        status = run(["status", "--porcelain"])
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        msg = f"commit={commit[:12]} branch={branch} dirty={dirty}"
        report.add("git", "warning" if dirty else "ok", msg)
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except FileNotFoundError:
        report.add("git", "warning", "git executable not found on PATH.")
        return {"commit": None, "branch": None, "dirty": None}
    except Exception as e:
        report.add("git", "warning", f"git query failed: {type(e).__name__}: {e}")
        return {"commit": None, "branch": None, "dirty": None}


# ---------------------------------------------------------------------------
# Packages / GPU
# ---------------------------------------------------------------------------

def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in ("torch", "transformers", "vllm"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def _gpu_info(report: Report) -> dict:
    """Small, standalone duplicate of preflight_check.py's torch-based GPU
    probe — deliberately not shared code, since this manifest only needs
    the summary (name + count), not the full preflight report shape."""
    try:
        import torch
        if not torch.cuda.is_available():
            report.add("gpu", "warning", "torch installed but no CUDA GPU available.")
            return {"count": 0, "names": []}
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_properties(i).name for i in range(n)]
        report.add("gpu", "ok", f"{n} GPU(s): {', '.join(names)}")
        return {"count": n, "names": names}
    except Exception as e:
        report.add("gpu", "warning", f"could not query GPU info: {type(e).__name__}: {e}")
        return {"count": None, "names": []}


def _secret_env_flags() -> Dict[str, bool]:
    """Boolean-only presence flags for env vars that look like secrets.
    Never writes the actual value."""
    flags = {}
    for key in sorted(os.environ):
        if any(sub in key.upper() for sub in SECRET_SUBSTRINGS):
            flags[key] = bool(os.environ.get(key))
    return flags


# ---------------------------------------------------------------------------
# Work-dir inspection
# ---------------------------------------------------------------------------

def _discover_task_files(work_dir: Path) -> Dict[str, dict]:
    """Map task_key -> {file, dataset_key, model_name, records, mtime} for
    each raw prediction file (.xlsx/.tsv) found directly under work_dir."""
    from vlmeval.smp import load

    found: Dict[str, dict] = {}
    for path in sorted(work_dir.glob("*")):
        if not path.is_file() or path.suffix not in PRED_EXTENSIONS:
            continue
        stem = path.stem
        for task_key, pattern in TASK_PATTERNS:
            idx = stem.lower().find(pattern.lower())
            if idx == -1:
                continue
            dataset_key = stem[idx:]
            model_name = stem[:idx].rstrip("_")
            try:
                records = len(load(str(path)))
            except Exception:
                records = None
            entry = {
                "file": path.name,
                "dataset_key": dataset_key,
                "model_name": model_name,
                "records": records,
                "mtime": datetime.datetime.fromtimestamp(
                    path.stat().st_mtime, tz=datetime.timezone.utc
                ).isoformat(),
            }
            if task_key in found:
                # Multiple variants (e.g. both 8frame and 16frame) matched
                # the same task — keep the most recently modified one,
                # mirroring package_submission.py's "use the most recent"
                # policy for duplicate submission files.
                prev_mtime = (work_dir / found[task_key]["file"]).stat().st_mtime
                if path.stat().st_mtime <= prev_mtime:
                    break
            found[task_key] = entry
            break
    return found


def _infer_model_name(task_files: Dict[str, dict]) -> Optional[str]:
    names = [v["model_name"] for v in task_files.values() if v["model_name"]]
    if not names:
        return None
    # Most common candidate; ties broken by first-seen.
    from collections import Counter
    return Counter(names).most_common(1)[0][0]


def _frame_variant(dataset_key: str, task_key: str) -> Optional[str]:
    """Best-effort extraction of the sampling-variant suffix (e.g. '8frame',
    '4fps') from a dataset key like 'VANTAGE_VQA_8frame'."""
    for _tk, pattern in TASK_PATTERNS:
        if _tk != task_key:
            continue
        if dataset_key.lower().startswith(pattern.lower()):
            rest = dataset_key[len(pattern):].lstrip("_")
            return rest or None
    return None


# ---------------------------------------------------------------------------
# form_metadata.md
# ---------------------------------------------------------------------------

def _find_archive_near(work_dir: Path) -> Optional[Path]:
    """Look only in work_dir itself and its immediate parent (e.g.
    outputs/<model>/) — NOT the process cwd, which could pick up an
    unrelated archive left over from something else entirely."""
    for candidate_dir in (work_dir, work_dir.parent):
        hits = sorted(candidate_dir.glob("*.tar.gz")) + sorted(candidate_dir.glob("*.tgz"))
        if hits:
            return hits[0]
    return None


def _build_form_metadata(work_dir: Path, manifest: dict) -> str:
    task_files = manifest["tasks"]
    model_name = manifest["model_name"] or "<FILL IN>"
    covered_tasks = set(task_files)
    pillars_submitted = []
    for pillar, tasks in PILLAR_TASKS.items():
        if all(t in covered_tasks for t in tasks):
            pillars_submitted.append(f"[x] Pillar {pillar} — {', '.join(tasks)} (all present)")
        elif any(t in covered_tasks for t in tasks):
            have = [t for t in tasks if t in covered_tasks]
            missing = [t for t in tasks if t not in covered_tasks]
            pillars_submitted.append(
                f"[ ] Pillar {pillar} — INCOMPLETE, have {have}, missing {missing}. "
                f"A pillar must be submitted in full or not checked."
            )
        else:
            pillars_submitted.append(f"[ ] Pillar {pillar} — not attempted")

    variants = sorted({
        v for tk, entry in task_files.items()
        if (v := _frame_variant(entry["dataset_key"], tk))
    })
    variants_str = ", ".join(variants) if variants else "<FILL IN — not inferable from filenames>"

    archive = _find_archive_near(work_dir)
    archive_line = (f"`{archive}`" if archive is not None
                     else "<FILL IN — no *.tar.gz found next to work-dir; run "
                          "`python scripts/package_submission.py --work-dir ... --out submission.tar.gz` first>")

    git = manifest["git"]
    commit_line = git["commit"] or "<FILL IN — not resolvable, not a git checkout>"

    lines = []
    lines.append(f"# VANTAGE-Bench submission form draft")
    lines.append("")
    lines.append(f"Generated by `scripts/run_manifest.py` from `{work_dir}` on "
                  f"{manifest['generated_at']}. Fill every `<FILL IN>` before using this as a "
                  f"reference while filling out https://vantage-bench.org/submit — **this script "
                  f"never submits anything**, it only drafts a local reference.")
    lines.append("")
    lines.append("## 01 Identity")
    lines.append("")
    lines.append("- Candidate name: <FILL IN>")
    lines.append("- Organization: <FILL IN>")
    lines.append("- Model card / paper URL: <FILL IN>")
    lines.append("- Contact email: <FILL IN>")
    lines.append("")
    lines.append("## 02 Submission Type")
    lines.append("")
    lines.append("- Evaluation track: <FILL IN — Public / Preview>")
    lines.append("- Pipeline type: <FILL IN — Single model / System pipeline>")
    lines.append("- System access type: <FILL IN — Fully open-weight / Mixed / Proprietary>")
    lines.append("")
    lines.append("## 03 Model Configuration")
    lines.append("")
    lines.append(f"- Primary model / checkpoint: {model_name}")
    lines.append("- Parameter count: <FILL IN>")
    lines.append("- Inference precision: <FILL IN — FP32 / FP16 / BF16 / INT8 / INT4 / Mixed precision / Unknown>")
    lines.append("- Training type: <FILL IN — Zero-shot / Fine-tuned>")
    lines.append("")
    lines.append("## 04 Inference Setup")
    lines.append("")
    lines.append(f"- Primary inference infrastructure: <FILL IN — e.g. local GPU / vLLM / hosted API>")
    lines.append("- Official evaluation harness used: Yes (this repo, VANTAGE-Bench's VLMEvalKit fork, "
                  "is the harness — default assumption; correct if a modified fork was used)")
    lines.append(f"- Frame-sampling variant(s) inferred from output filenames: {variants_str}")
    lines.append("- Temperature: <FILL IN — not recoverable from work-dir filenames/metadata>")
    lines.append("- tensor_parallel_size / other hyperparameters: <FILL IN — not recoverable from "
                  "work-dir filenames/metadata; check your run.py --model config JSON>")
    lines.append(f"- Run git commit: {commit_line} (dirty tree: {git['dirty']})")
    lines.append("")
    lines.append("## 05 Pillars Submitted")
    lines.append("")
    lines.extend(f"- {line}" for line in pillars_submitted)
    lines.append("")
    lines.append("## 06 Predictions File")
    lines.append("")
    lines.append(f"- Archive: {archive_line}")
    lines.append("")
    lines.append("## 07 Acknowledgements")
    lines.append("")
    lines.append("The portal presents five checkbox acknowledgement statements. Their exact current "
                  "wording is out of scope for this script to guess (legal/policy text can change "
                  "independently of this repo) — **re-read them live on the form** before checking "
                  "any of them. Nothing below is pre-checked:")
    lines.append("- [ ] Acknowledgement 1 — <re-read on the live form>")
    lines.append("- [ ] Acknowledgement 2 — <re-read on the live form>")
    lines.append("- [ ] Acknowledgement 3 — <re-read on the live form>")
    lines.append("- [ ] Acknowledgement 4 — <re-read on the live form>")
    lines.append("- [ ] Acknowledgement 5 — <re-read on the live form>")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_manifest.py",
        description="Write run_manifest.json and form_metadata.md for a completed VANTAGE-Bench run.",
    )
    p.add_argument("--work-dir", required=True, type=Path,
                    help="outputs/<model>/<eval_id>/ directory produced by run.py.")
    p.add_argument("--json", action="store_true",
                    help="print the summary as a single JSON object instead of text "
                         "(the two output files are always written either way).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = Report()

    work_dir = args.work_dir.expanduser().resolve()
    if not work_dir.is_dir():
        print(f"Error: --work-dir does not exist: {work_dir}")
        return 1

    task_files = _discover_task_files(work_dir)
    if not task_files:
        report.add("discovery", "blocker", f"no *.xlsx/*.tsv prediction files found under {work_dir}.")
    else:
        report.add("discovery", "ok",
                    f"found {len(task_files)} task prediction file(s): {', '.join(sorted(task_files))}.")

    git = _git_info(report)
    packages = _package_versions()
    gpu = _gpu_info(report)
    secrets = _secret_env_flags()
    model_name = _infer_model_name(task_files)

    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "git": git,
        "python_version": sys.version,
        "packages": packages,
        "gpu": gpu,
        "model_name": model_name,
        "dataset_keys": sorted({v["dataset_key"] for v in task_files.values()}),
        "tasks": task_files,
        "env_secrets_present": secrets,
    }

    manifest_path = work_dir / "run_manifest.json"
    form_path = work_dir / "form_metadata.md"

    if task_files:
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            report.add("manifest_write", "ok", f"wrote {manifest_path}")
        except OSError as e:
            report.add("manifest_write", "blocker", f"failed to write {manifest_path}: {e}")

        try:
            form_path.write_text(_build_form_metadata(work_dir, manifest), encoding="utf-8")
            report.add("form_metadata_write", "ok", f"wrote {form_path}")
        except OSError as e:
            report.add("form_metadata_write", "blocker", f"failed to write {form_path}: {e}")
    else:
        report.add("manifest_write", "blocker", "skipped — no task prediction files to manifest.")
        report.add("form_metadata_write", "blocker", "skipped — no task prediction files to manifest.")

    status = report.worst_status()

    if args.json:
        print(json.dumps({
            "status": status,
            "findings": report.findings,
            "manifest_path": str(manifest_path) if task_files else None,
            "form_metadata_path": str(form_path) if task_files else None,
            "model_name": model_name,
            "dataset_keys": manifest["dataset_keys"],
        }, indent=2))
    else:
        print("=" * 78)
        print("VANTAGE-Bench run manifest")
        print(f"Work dir: {work_dir}")
        print("=" * 78)
        width = max((len(f["check"]) for f in report.findings), default=10) + 2
        for f in report.findings:
            tag = {"ok": "OK", "warning": "WARN", "blocker": "BLOCK"}[f["status"]]
            print(f"[{tag:<5}] {f['check']:<{width}} {f['message']}")
        print("-" * 78)
        if task_files:
            print(f"Model:        {model_name}")
            print(f"Dataset keys: {', '.join(manifest['dataset_keys'])}")
            print(f"Manifest:     {manifest_path}")
            print(f"Form draft:   {form_path}")
        print(f"Overall status: {status.upper()}")
        print("=" * 78)

    return 0 if status != "blocker" else 1


if __name__ == "__main__":
    sys.exit(main())
