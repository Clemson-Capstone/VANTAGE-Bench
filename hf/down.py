"""
Dataset download helper — for most users, use run_lmudata.py instead.

    python scripts/run_lmudata.py --all --lmu-root ~/LMUData

This script is kept as a low-level alternative for downloading individual
archive files from a private/internal HuggingFace dataset repo using the
snapshot_download API. It is not needed for the public benchmark workflow.

Usage (internal / advanced only):
  1. Set HF_REPO to your dataset repo id.
  2. Set LOCAL_DIR to the target local directory.
  3. Edit TARGET_PATTERNS to match the files you want to pull.
  4. Run: python hf/down.py
"""

from huggingface_hub import snapshot_download
import time

HF_REPO = "nvidia/PhysicalAI-VANTAGE-Bench"
LOCAL_DIR = "~/LMUData"

# Restrict to specific files/folders; use ["*"] to pull everything.
TARGET_PATTERNS = ["*"]

for pattern in TARGET_PATTERNS:
    print(f"--- Syncing: {pattern} ---")
    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=[pattern],
            max_workers=2,
            resume_download=True,
            force_download=False,
        )
        print(f"Finished {pattern}. Waiting 10s ...")
        time.sleep(10)
    except Exception as e:
        print(f"Error on {pattern}: {e}")
        continue

print("Done.")
