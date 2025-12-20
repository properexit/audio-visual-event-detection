"""
Utility functions for loading audio and video inputs at inference time.

These helpers are used by:
- scripts/predict.py

They intentionally mirror the preprocessing used during training,
but are lightweight and inference-focused.
"""

import os
import torch
import soundfile as sf
from torchvision import transforms
from PIL import Image


# --------------------------------------------------
# Audio loading (WAV → Tensor)
# --------------------------------------------------

def load_audio(
    wav_path: str,
    sample_rate: int = 16000
) -> torch.Tensor:
    """
    Load and preprocess audio for inference.

    Steps:
    - Load WAV file
    - Convert stereo → mono
    - Trim or zero-pad to exactly 1 second
    - Return tensor shaped for AudioEncoder

    Args:
        wav_path: path to .wav file
        sample_rate: target sampling rate (Hz)

    Returns:
        audio: Tensor of shape (1, 16000)
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    audio_np, sr = sf.read(wav_path)
    audio = torch.from_numpy(audio_np).float()

    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(dim=1)

    # Add channel dimension: (1, T)
    audio = audio.unsqueeze(0)

    # Trim / pad to fixed length
    audio = audio[:, :sample_rate]
    if audio.shape[1] < sample_rate:
        audio = torch.nn.functional.pad(
            audio, (0, sample_rate - audio.shape[1])
        )

    return audio


# --------------------------------------------------
# Video loading (frames → Tensor)
# --------------------------------------------------

def load_video(
    frame_dir: str,
    num_frames: int = 8
) -> torch.Tensor:
    """
    Load and preprocess video frames for inference.

    Assumes frames were pre-extracted using extract_frames.py.

    Steps:
    - Load up to num_frames JPEGs
    - Resize to 224×224
    - Convert to tensor
    - Stack into (T, 3, 224, 224)

    Args:
        frame_dir: directory containing frame images
        num_frames: number of frames to load

    Returns:
        video: Tensor of shape (T, 3, 224, 224)
    """
    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(frame_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_files = sorted(os.listdir(frame_dir))[:num_frames]
    if len(frame_files) == 0:
        raise RuntimeError("No frames found in directory")

    frames = []
    for fname in frame_files:
        img = Image.open(
            os.path.join(frame_dir, fname)
        ).convert("RGB")
        frames.append(transform(img))

    return torch.stack(frames)