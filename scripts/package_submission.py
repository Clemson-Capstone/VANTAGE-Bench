#!/usr/bin/env python3
"""
package_submission.py — Bundle per-task *_submission.jsonl files into the
.tar.gz archive required by the VANTAGE-Bench submission portal.

Usage:
    python scripts/package_submission.py \
        --work-dir ./outputs/<model>/<eval_id> \
        --out submission.tar.gz

The script walks <work-dir>, finds all *_submission.jsonl files, maps each to
its canonical task filename, and creates a .tar.gz.  Only tasks that were
actually run are included.  The submission portal validates that all tasks
within each selected pillar are present.

Canonical task filenames (inside the archive):
    vqa.jsonl               VANTAGE_VQA_*
    event_verification.jsonl VANTAGE_EventVerification_*
    temporal.jsonl          VANTAGE_Temporal_*
    dvc.jsonl               VANTAGE_DVC_*
    sot.jsonl               VANTAGE_SOT*
    grounding.jsonl         VANTAGE_2DGrounding*
    pointing.jsonl          VANTAGE_2DPointing*
    astro.jsonl             Astro2D*
"""

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

# Maps a task key to the dataset-name patterns that produce its submission file.
# Pattern matching is case-insensitive prefix on the filename stem after
# stripping the model name prefix.
TASK_PATTERNS = [
    ("vqa",                "VANTAGE_VQA"),
    ("event_verification", "VANTAGE_EventVerification"),
    ("temporal",           "VANTAGE_Temporal"),
    ("dvc",                "VANTAGE_DVC"),
    ("sot",                "VANTAGE_SOT"),
    ("grounding",          "VANTAGE_2DGrounding"),
    ("pointing",           "VANTAGE_2DPointing"),
    ("astro",              "Astro2D"),
]

PILLAR_TASKS = {
    "I  — Semantic":         ["vqa", "event_verification"],
    "II — Spatial":          ["grounding", "pointing", "astro"],
    "III — Temporal":        ["temporal", "dvc"],
    "IV — Spatio-Temporal":  ["sot"],
}


def find_submission_files(work_dir: Path) -> dict[str, Path]:
    """Return {task_key: path} for all *_submission.jsonl files found."""
    found: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in sorted(work_dir.rglob("*_submission.jsonl")):
        stem = path.stem  # e.g. "GPT4o_VANTAGE_VQA_8frame_submission"
        matched = None
        for task_key, pattern in TASK_PATTERNS:
            # Check if the pattern appears in the stem (case-insensitive)
            if pattern.lower() in stem.lower():
                matched = task_key
                break
        if matched is None:
            print(f"  [warn] unrecognized submission file, skipping: {path.name}")
            continue
        if matched in found:
            duplicates.setdefault(matched, [found[matched]]).append(path)
        else:
            found[matched] = path

    # Warn if multiple files matched the same task (e.g. both 8frame and 16frame runs)
    for task_key, paths in duplicates.items():
        all_paths = [found[task_key]] + paths
        print(f"  [warn] multiple submission files for task '{task_key}':")
        for p in all_paths:
            print(f"         {p}")
        # Keep the last one (most recent by sort order)
        found[task_key] = all_paths[-1]
        print(f"         Using: {found[task_key]}")

    return found


def print_pillar_coverage(found: dict[str, Path]) -> None:
    """Print which pillars are fully covered."""
    print("\nPillar coverage:")
    for pillar, tasks in PILLAR_TASKS.items():
        covered = [t for t in tasks if t in found]
        missing = [t for t in tasks if t not in found]
        status = "COMPLETE" if not missing else f"INCOMPLETE — missing: {', '.join(missing)}"
        print(f"  Pillar {pillar}: {status}")
        for t in covered:
            print(f"    {t}: {found[t].name}")
    print()


def build_archive(found: dict[str, Path], out_path: Path) -> None:
    """Write a .tar.gz with one canonically-named .jsonl per task."""
    if not found:
        print("No submission files found. Run inference first with --mode infer.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        staged = []
        for task_key, src in sorted(found.items()):
            dst = tmp_path / f"{task_key}.jsonl"
            dst.write_bytes(src.read_bytes())
            staged.append((dst, f"{task_key}.jsonl"))

        with tarfile.open(out_path, "w:gz") as tar:
            for staged_path, arcname in staged:
                tar.add(staged_path, arcname=arcname)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Archive written: {out_path}  ({size_mb:.1f} MB)")
    print("Contents:")
    with tarfile.open(out_path, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"  {member.name}  ({member.size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Bundle VANTAGE-Bench submission JSONL files into a .tar.gz archive."
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Output directory produced by run.py (contains *_submission.jsonl files).",
    )
    parser.add_argument(
        "--out",
        default="submission.tar.gz",
        type=Path,
        help="Path for the output .tar.gz archive (default: submission.tar.gz).",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    if not work_dir.is_dir():
        print(f"Error: --work-dir does not exist: {work_dir}")
        sys.exit(1)

    out_path = args.out.expanduser().resolve()
    if out_path.suffix not in (".gz", ".tgz"):
        print(f"Warning: --out does not end in .tar.gz or .tgz: {out_path}")

    print(f"Scanning: {work_dir}")
    found = find_submission_files(work_dir)

    if not found:
        print("No *_submission.jsonl files found. Run inference with --mode infer first.")
        sys.exit(1)

    print_pillar_coverage(found)
    build_archive(found, out_path)
    print(f"\nNext step: upload {out_path} at https://vantage-bench.org/submit")


if __name__ == "__main__":
    main()
