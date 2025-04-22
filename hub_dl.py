from huggingface_hub import snapshot_download
import os


repo_id = "thecr7guy/GPT2fromScratch"
repo_type = "model"
local_dir = "/workspace/." 
token = "Enter token here"


if local_dir:
    os.makedirs(local_dir, exist_ok=True)

print(f"Downloading repository '{repo_id}' to '{local_dir or 'cache'}'...")

try:
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False, 
        token=token,
    )
    print(f"Repository downloaded successfully to: {downloaded_path}")

except Exception as e:
    print(f"Error downloading repository: {e}")
