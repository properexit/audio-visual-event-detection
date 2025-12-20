"""
Extract audio tracks from downloaded VGGSound video clips.

This script:
- Converts each `.mp4` video into a `.wav` audio file
- Forces mono audio
- Resamples to 16 kHz (standard for audio ML models)

Required by:
- AudioVisualDataset
- Audio encoder training
- Fusion model training

NOTE:
- Requires `ffmpeg` to be installed and available in PATH
"""

import os
import subprocess

VIDEO_DIR = "data/vggsound_sample/videos"
AUDIO_DIR = "data/vggsound_sample/audio"

os.makedirs(AUDIO_DIR, exist_ok=True)

# --------------------------------------------------
# Audio extraction
# --------------------------------------------------

for filename in os.listdir(VIDEO_DIR):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, filename)
    audio_path = os.path.join(
        AUDIO_DIR,
        filename.replace(".mp4", ".wav")
    )

    # ffmpeg command:
    # -y     : overwrite existing files
    # -ac 1  : mono channel
    # -ar    : sample rate = 16 kHz
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        audio_path,
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(f"Extracted audio → {audio_path}")