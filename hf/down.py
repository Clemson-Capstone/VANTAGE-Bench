from huggingface_hub import snapshot_download
import time

# List your tasks individually
# target_tasks = [
#     "2dbbox/*",
#     "Dense Video Caption/*",
#     "Spatial/*",
#     "Temporal/*",
#     "VQA/*",
#     "event_verification_subset/*"
# ]
target_tasks = ["<dataset_archive>.tar.gz"]

for pattern in target_tasks:
    print(f"--- Starting Sync for: {pattern} ---")
    try:
        snapshot_download(
            repo_id="<your-hf-org>/<your-dataset-repo>",
            repo_type="dataset",
            local_dir="<local_data_dir>",
            local_dir_use_symlinks=False,
            allow_patterns=[pattern], # One pattern at a time
            max_workers=2,            # Slow and steady to avoid 429
            resume_download=True,     # Essential for resuming
            force_download=False      # MUST be False to avoid re-checking everything
        )
        print(f"Finished {pattern}. Waiting 10s to clear API limits...")
        time.sleep(20) # Cooling off period for the HF API
    except Exception as e:
        print(f"Error on {pattern}: {e}")
        continue

print("All targeted folders synced.")
