 #!/usr/bin/env python3

import os
from huggingface_hub import HfApi,hf_hub_download

# === Configuration ===
# Your repo in the form "username/repo_name" (must already exist)
REPO_ID = "thecr7guy/GPT2fromScratch"
# Local path to the file you want to upload
LOCAL_FILE_PATH = "../logs/."
# Where the file will live in the repo (can include subfolders)
PATH_IN_REPO = "checkpoints/cc_stories_epoch_9.pth"

# === Get your token ===
TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("Please set your HF_TOKEN environment variable.")

# === Upload ===
api = HfApi()
# api.upload_file(
#     path_or_fileobj=LOCAL_FILE_PATH,   # local file path
#     path_in_repo=PATH_IN_REPO,         # destination path in the repo
#     repo_id=REPO_ID,                   # your repo
#     token=TOKEN                        # your access token
# )

hf_hub_download(
    repo_id=REPO_ID,
    filename=PATH_IN_REPO,   # remote path
    cache_dir=LOCAL_FILE_PATH,     # where to mirror the cache
    token=TOKEN,             # can be None if public
    local_dir=LOCAL_FILE_PATH,     # local copy root
)

print(f"✅ Uploaded {LOCAL_FILE_PATH} to {REPO_ID}/{PATH_IN_REPO}")