import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Audio encoder for raw waveform input.

    Design goals:
    - Simple and fast (no spectrogram dependency)
    - Works directly on waveforms
    - Produces a fixed-length embedding for fusion

    Input:
        x: (B, C, T)
            B = batch size
            C = channels (1 = mono, 2 = stereo)
            T = number of audio samples

    Output:
        embedding: (B, D)
    """

    def __init__(self):
        super().__init__()

        # Dimensionality of the final audio embedding
        self.output_dim = 128

        # -------------------------
        # Temporal feature extractor
        # -------------------------
        # Conv1D layers progressively downsample time
        # while increasing channel capacity
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # -------------------------
        # Projection layer
        # -------------------------
        # Maps pooled features to a fixed embedding size
        self.proj = nn.Linear(64, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Audio waveform tensor of shape (B, C, T)

        Returns:
            audio_embedding: (B, output_dim)
        """

        # -------------------------
        # Channel handling
        # -------------------------
        # Convert stereo audio to mono if needed
        if x.size(1) == 2:
            x = x.mean(dim=1, keepdim=True)

        # -------------------------
        # Temporal convolution
        # -------------------------
        # Shape preserved as (B, C, T)
        x = self.conv(x)

        # -------------------------
        # Global temporal pooling
        # -------------------------
        # Removes dependence on exact audio length
        x = x.mean(dim=-1)  # (B, 64)

        # -------------------------
        # Final embedding
        # -------------------------
        audio_embedding = self.proj(x)

        return audio_embedding