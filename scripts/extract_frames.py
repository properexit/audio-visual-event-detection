"""
Extract uniformly sampled video frames from VGGSound clips.

This script:
- Samples a fixed number of frames per video (default: 8)
- Resizes frames to 224×224 (standard for CNN backbones like ResNet)
- Stores frames per video in a dedicated folder

Required by:
- VideoEncoder
- AudioVisualDataset
- Fusion training & robustness experiments

NOTE:
- Requires `ffmpeg` installed and available in PATH
"""

import os
import subprocess

VIDEO_DIR = "data/vggsound_sample/videos"
FRAME_DIR = "data/vggsound_sample/frames"

NUM_FRAMES = 8          # frames per clip
IMG_SIZE = 224          # spatial resolution

os.makedirs(FRAME_DIR, exist_ok=True)

# --------------------------------------------------
# Frame extraction
# --------------------------------------------------

for filename in os.listdir(VIDEO_DIR):
    if not filename.endswith(".mp4"):
        continue

    video_id = filename.replace(".mp4", "")
    video_path = os.path.join(VIDEO_DIR, filename)

    output_dir = os.path.join(FRAME_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # ffmpeg filter explanation:
    # fps = NUM_FRAMES / 10 → uniformly sample frames from a 10s clip
    # scale → resize frames to IMG_SIZE × IMG_SIZE
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"fps={NUM_FRAMES}/10,scale={IMG_SIZE}:{IMG_SIZE}",
        os.path.join(output_dir, "%03d.jpg"),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(f"Extracted {NUM_FRAMES} frames → {output_dir}")