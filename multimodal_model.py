import torch
import torch.nn as nn

from src.audio.model import AudioEncoder
from src.video.model import VideoEncoder
from src.fusion.attention import ModalityAttention


class AudioVisualEventModel(nn.Module):
    """
    Audio-Visual Event Recognition model.

    This model follows a modular multimodal design:
    1. Encode audio and video independently
    2. Fuse modality representations using attention
    3. Predict event class from the fused embedding

    The architecture is intentionally simple and interpretable,
    making it suitable for robustness and ablation studies.
    """

    def __init__(self, num_classes: int = 10):
        """
        Args:
            num_classes: Number of target event categories
        """
        super().__init__()

        # -------------------------
        # Modality-specific encoders
        # -------------------------
        # AudioEncoder → waveform → fixed-length embedding
        # VideoEncoder → frame sequence → fixed-length embedding
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()

        audio_dim = self.audio_encoder.output_dim
        video_dim = self.video_encoder.output_dim

        # -------------------------
        # Attention-based fusion
        # -------------------------
        # Learns how much to rely on audio vs video
        # for each input sample
        self.fusion = ModalityAttention(
            audio_dim=audio_dim,
            video_dim=video_dim
        )

        # -------------------------
        # Classification head
        # -------------------------
        # Operates on concatenated, attention-weighted embeddings
        self.classifier = nn.Linear(
            audio_dim + video_dim,
            num_classes
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor
    ):
        """
        Forward pass.

        Args:
            audio: Tensor of shape (B, 1, T)
            video: Tensor of shape (B, T, 3, H, W)

        Returns:
            logits: (B, num_classes)
            attn:   (B, 2) attention weights [audio, video]
        """

        # Encode each modality independently
        audio_emb = self.audio_encoder(audio)   # (B, audio_dim)
        video_emb = self.video_encoder(video)   # (B, video_dim)

        # Fuse representations with attention
        fused, attn = self.fusion(audio_emb, video_emb)

        # Final classification
        logits = self.classifier(fused)

        return logits, attn