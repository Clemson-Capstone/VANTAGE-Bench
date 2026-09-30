#!/usr/bin/env python3
"""
package_submission.py - Bundle per-task submission JSONL files into the
.tar.gz archive required by the VANTAGE-Bench submission portal.

Usage:
    python scripts/package_submission.py \
        --work-dir ./outputs/<model>/<eval_id> \
        --out submission.tar.gz

The script walks <work-dir>, finds every submission JSONL the harness wrote,
maps each to its canonical task filename, and creates a .tar.gz.  Only tasks
that were actually run are included.  The submission portal validates that
all tasks within each selected pillar are present.

Two file naming conventions are accepted, because run.py writes different
names depending on --mode:
    <model>_<dataset>_submission.jsonl   written by dataset.evaluate()
                                         (--mode all, --mode eval)
    <model>_<dataset>.submission.jsonl   written by the inference loop
                                         (--mode infer)

When several files match one task, the canonical split of the dataset
(e.g. VANTAGE_2DGrounding, VANTAGE_VQA_8frame) is preferred over variant or
debug splits (VANTAGE_2DGrounding_val, VANTAGE_2DGrounding_small,
VANTAGE_VQA_8frame_200); ties are broken by modification time (newest wins).
A variant split is still accepted when it is the only file for that task,
and the chosen file is always printed.

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
import re
import sys
import tarfile
import tempfile
from pathlib import Path

# Maps a task key to the dataset-name prefix that produces its submission file.
# The prefix must appear in the filename stem at an underscore boundary (the
# stem is "<model>_<dataset>" once the submission suffix is stripped), so it
# is matched case-insensitively but not as an arbitrary substring.
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

NO_FILES_MESSAGE = (
    "No submission files found under --work-dir.\n"
    "Expected <model>_<dataset>_submission.jsonl (written by run.py --mode all / --mode eval)\n"
    "or <model>_<dataset>.submission.jsonl (written by run.py --mode infer).\n"
    "Run run.py first, or point --work-dir at the run folder it produced."
)

PILLAR_TASKS = {
    "I  — Semantic":         ["vqa", "event_verification"],
    "II — Spatial":          ["grounding", "pointing", "astro"],
    "III — Temporal":        ["temporal", "dvc"],
    "IV — Spatio-Temporal":  ["sot"],
}


# Filename suffixes that mark a submission JSONL, in both naming conventions.
SUBMISSION_SUFFIXES = ("_submission.jsonl", ".submission.jsonl")

# Frame-count / fps variants are registered dataset keys for the full split
# (VANTAGE_VQA_16frame, VANTAGE_Temporal_0.5fps, VANTAGE_SOT_16f, ...), so a
# dataset component that ends in one of these still counts as canonical.
_FRAME_SUFFIX_RE = re.compile(r"^(?:_\d+frame|_\d+f|_\d+(?:\.\d+)?fps)?$", re.IGNORECASE)


def strip_submission_suffix(filename: str):
    """Return "<model>_<dataset>" for a submission filename, or None."""
    for suffix in SUBMISSION_SUFFIXES:
        if filename.lower().endswith(suffix):
            return filename[: -len(suffix)]
    return None


def classify_dataset_component(component: str, task_key: str):
    """Classify a "<model>_<dataset>" stem against one task's pattern.

    Returns "canonical" when the dataset part is exactly the task pattern
    (optionally followed by a frame/fps suffix), "variant" when the pattern
    is followed by anything else (e.g. "_val", "_small", "_8frame_200"), and
    None when the pattern does not occur at an underscore boundary.
    """
    pattern = dict(TASK_PATTERNS)[task_key]
    m = re.search(r"(?:^|_)" + re.escape(pattern) + r"(?=$|_)", component, re.IGNORECASE)
    if m is None:
        return None
    remainder = component[m.end():]
    return "canonical" if _FRAME_SUFFIX_RE.match(remainder) else "variant"


def find_submission_files(work_dir: Path) -> dict[str, Path]:
    """Return {task_key: path} for the submission JSONL files under work_dir.

    Both "<model>_<dataset>_submission.jsonl" (evaluate) and
    "<model>_<dataset>.submission.jsonl" (inference) are accepted. For each
    task the canonical split is preferred over variant splits, and among
    equally ranked files the most recently modified one wins.
    """
    candidates: dict[str, list[tuple[int, float, Path]]] = {}
    seen: set = set()

    for path in sorted(work_dir.rglob("*.jsonl")):
        component = strip_submission_suffix(path.name)
        if component is None or not path.is_file():
            continue
        real = path.resolve()
        if real in seen:
            continue  # symlink to a file already collected (outputs/<model>/ layout)
        seen.add(real)

        matched = None
        rank = None
        for task_key, _pattern in TASK_PATTERNS:
            kind = classify_dataset_component(component, task_key)
            if kind is not None:
                matched, rank = task_key, (0 if kind == "canonical" else 1)
                break
        if matched is None:
            print(f"  [warn] unrecognized submission file, skipping: {path.name}")
            continue
        candidates.setdefault(matched, []).append((rank, path.stat().st_mtime, path))

    found: dict[str, Path] = {}
    for task_key, entries in candidates.items():
        # Best rank first (canonical before variant), then newest mtime first.
        entries.sort(key=lambda e: (e[0], -e[1]))
        best_rank, _mtime, best = entries[0]
        found[task_key] = best
        if len(entries) > 1:
            print(f"  [warn] multiple submission files for task '{task_key}':")
            for rank, _m, p in entries:
                print(f"         {p}" + ("  (variant split)" if rank else ""))
            print(f"         Using: {best}")
        elif best_rank:
            print(f"  [note] task '{task_key}': only a variant split was found, using: {best}")

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
        print(NO_FILES_MESSAGE)
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
        help="Output directory produced by run.py (contains *_submission.jsonl or *.submission.jsonl files).",
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
        print(NO_FILES_MESSAGE)
        sys.exit(1)

    print_pillar_coverage(found)
    build_archive(found, out_path)
    print(f"\nNext step: upload {out_path} at https://vantage-bench.org/submit")


if __name__ == "__main__":
    main()
