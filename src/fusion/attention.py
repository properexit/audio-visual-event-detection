import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttention(nn.Module):
    """
    Attention-based fusion module for audio–video representations.

    Purpose:
    --------
    Learn how much the model should rely on each modality
    (audio vs video) on a per-sample basis.

    Instead of assuming both modalities are equally reliable,
    this module lets the network adaptively reweight them.

    Example behavior:
    - Loud, clear sound → rely more on audio
    - Silent or noisy audio → rely more on video
    """

    def __init__(self, audio_dim: int, video_dim: int):
        """
        Args:
            audio_dim: dimensionality of audio embedding
            video_dim: dimensionality of video embedding
        """
        super().__init__()

        # Linear projections that map each modality
        # embedding to a single scalar "importance score"
        self.audio_proj = nn.Linear(audio_dim, 1)
        self.video_proj = nn.Linear(video_dim, 1)

    def forward(
        self,
        audio_emb: torch.Tensor,
        video_emb: torch.Tensor
    ):
        """
        Forward pass.

        Args:
            audio_emb: Audio embeddings (B, Da)
            video_emb: Video embeddings (B, Dv)

        Returns:
            fused: Concatenated, attention-weighted embedding (B, Da + Dv)
            attn: Attention weights over modalities (B, 2)
                  [audio_weight, video_weight]
        """

        # -------------------------
        # 1. Compute modality scores
        # -------------------------
        # Each modality produces a scalar confidence score
        audio_score = self.audio_proj(audio_emb)   # (B, 1)
        video_score = self.video_proj(video_emb)   # (B, 1)

        # Stack scores into a single tensor
        scores = torch.cat(
            [audio_score, video_score],
            dim=1
        )  # (B, 2)

        # -------------------------
        # 2. Normalize with softmax
        # -------------------------
        # Produces a probability distribution over modalities
        attn = F.softmax(scores, dim=1)  # (B, 2)

        # -------------------------
        # 3. Apply attention weights
        # -------------------------
        audio_weighted = audio_emb * attn[:, 0:1]
        video_weighted = video_emb * attn[:, 1:2]

        # -------------------------
        # 4. Fuse representations
        # -------------------------
        fused = torch.cat(
            [audio_weighted, video_weighted],
            dim=1
        )  # (B, Da + Dv)

        return fused, attn