import os
import torch
import pandas as pd
import soundfile as sf

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AudioVisualDataset(Dataset):
    """
    Audio–Visual Dataset for event recognition.

    Each sample consists of:
    - audio waveform (1 second, mono)
    - a fixed number of video frames
    - a class label

    Returned shapes:
        audio: (1, 16000)
        video: (T, 3, 224, 224)
        label: int
    """

    def __init__(
        self,
        root: str = "data/vggsound_sample",
        num_frames: int = 8,
        sample_rate: int = 16000
    ):
        """
        Args:
            root: dataset root directory
            num_frames: number of video frames per sample
            sample_rate: audio sampling rate (Hz)
        """
        self.root = root
        self.num_frames = num_frames
        self.sample_rate = sample_rate

        # Load metadata file
        self.labels = pd.read_csv(
            os.path.join(root, "labels.csv")
        )

        # Map string labels to integer IDs
        self.label_to_idx = {
            label: idx
            for idx, label in enumerate(
                self.labels["label"].unique()
            )
        }

        # Image preprocessing for video frames
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Load a single audio–video sample.

        Steps:
        1. Load audio waveform
        2. Normalize to mono and fixed duration
        3. Load and preprocess video frames
        """

        row = self.labels.iloc[idx]
        video_id = row["video_id"]
        label = self.label_to_idx[row["label"]]

        # =====================
        # Audio loading
        # =====================
        wav_path = os.path.join(
            self.root,
            "audio",
            f"{video_id}.wav"
        )

        if not os.path.exists(wav_path):
            raise FileNotFoundError(
                f"Missing audio file: {wav_path}"
            )

        audio_np, sr = sf.read(wav_path)
        audio = torch.from_numpy(audio_np).float()

        # Convert stereo to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(dim=1)

        # Add channel dimension: (1, T)
        audio = audio.unsqueeze(0)

        # Trim or pad to exactly 1 second
        audio = audio[:, : self.sample_rate]

        if audio.shape[1] < self.sample_rate:
            audio = torch.nn.functional.pad(
                audio,
                (0, self.sample_rate - audio.shape[1])
            )

        # =====================
        # Video loading
        # =====================
        frame_dir = os.path.join(
            self.root,
            "frames",
            video_id
        )

        frame_files = sorted(
            os.listdir(frame_dir)
        )[: self.num_frames]

        frames = []
        for fname in frame_files:
            img = Image.open(
                os.path.join(frame_dir, fname)
            ).convert("RGB")

            frames.append(self.transform(img))

        video = torch.stack(frames)  # (T, 3, 224, 224)

        return audio, video, label