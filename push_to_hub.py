import argparse
import os
import sys
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

def push_directory_to_hub(local_dir: str, repo_id: str, commit_message: str = "Push directory contents"):


    os.environ["HF_TOKEN"] = "Enter token here"

    token = HfFolder.get_token()
    api = HfApi()


    api.repo_info(repo_id=repo_id, token=token)
    print("Repository found.")

    print(f"\nUploading contents of '{local_dir}' to '{repo_id}'...")

    repo_url = api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=token,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a local directory to an existing Hugging Face Hub repository.")
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Path to the local directory containing the files to push."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face Hub repository ID (e.g., 'username/repo-name')."
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Push directory contents via script",
        help="Commit message for the upload."
    )

    args = parser.parse_args()

    push_directory_to_hub(args.local_dir, args.repo_id, args.commit_message)