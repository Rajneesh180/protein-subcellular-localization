"""Download HPA dataset from Kaggle."""

import os
import subprocess
import sys


COMPETITION = "human-protein-atlas-image-classification"


def download_hpa(dest_dir: str = "data") -> None:
    os.makedirs(dest_dir, exist_ok=True)

    # check kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)

    print(f"Downloading {COMPETITION} to {dest_dir}/...")
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", COMPETITION,
            "-p", dest_dir,
        ],
        check=True,
    )

    # unzip
    zip_path = os.path.join(dest_dir, f"{COMPETITION}.zip")
    if os.path.exists(zip_path):
        print("Extracting...")
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", dest_dir], check=True)
        os.remove(zip_path)
        print("Done.")
    else:
        print("No zip found — data might already be extracted.")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "data"
    download_hpa(dest)
