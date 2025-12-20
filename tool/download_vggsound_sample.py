"""
Download a small reproducible subset of VGGSound videos.

This script is intended for:
- development
- debugging
- sanity checks

It downloads short (10s) clips from YouTube using metadata
from a lightweight CSV file.

IMPORTANT:
- This script is NOT required to run training or inference
  if the dataset is already present locally.
- Some videos may fail due to removal, privacy, or region blocks.
"""

import os
import subprocess
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CSV_PATH = "data/vggsound_sample/vggsound_sample.csv"
VIDEO_DIR = "data/vggsound_sample/videos"

CLIP_LEN = 10  # VGGSound clips are fixed to 10 seconds

os.makedirs(VIDEO_DIR, exist_ok=True)

# --------------------------------------------------
# Load metadata
# --------------------------------------------------
# CSV format (no header):
# video_id, start_time, label, split
df = pd.read_csv(
    CSV_PATH,
    header=None,
    names=["video_id", "start_time", "label", "split"]
)

print(f"Found {len(df)} video entries")

# --------------------------------------------------
# Download loop
# --------------------------------------------------

for _, row in df.iterrows():
    video_id = row["video_id"]
    start = int(row["start_time"])
    end = start + CLIP_LEN

    output_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    print(f"Downloading {video_id} [{start}-{end}] ({row['label']})")

    # yt-dlp command:
    # - downloads best available video + audio
    # - merges streams into mp4
    # - extracts only the required time segment
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", output_path,
    ]

    subprocess.run(cmd)