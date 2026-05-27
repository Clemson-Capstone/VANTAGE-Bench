import os
from huggingface_hub import HfApi
api = HfApi()

LOCAL_FILE = "leaderboard.json"
REPO_ID = "anonymous/vantage-leaderboard"
TARGET_NAME = "data/leaderboard.json"

print(f"--- Uploading {LOCAL_FILE} to {REPO_ID} ---")
    
try:
    # Check if file exists locally first
    if not os.path.exists(LOCAL_FILE):
        print(f"Error: Local file '{LOCAL_FILE}' not found.")

    api.upload_file(
        path_or_fileobj=LOCAL_FILE,
        path_in_repo=TARGET_NAME,
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Update leaderboard.json"
    )
    print("Upload successful!")
except Exception as e:
    print(f"An error occurred: {e}")