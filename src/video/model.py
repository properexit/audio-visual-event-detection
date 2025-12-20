import torch
import torch.nn as nn
from torchvision.models import resnet18


class VideoEncoder(nn.Module):
    """
    CNN-based video encoder for short video clips.

    Design goals:
    - Reuse strong image features (ImageNet-pretrained CNN)
    - Keep the model lightweight and fast
    - Produce a fixed-size embedding per video clip

    Input:
        x: (B, T, 3, H, W)
            B = batch size
            T = number of frames
            H, W = frame resolution

    Output:
        video_embedding: (B, D)
    """

    def __init__(self):
        super().__init__()

        # Dimensionality of the final video embedding
        self.output_dim = 128

        # -------------------------
        # Frame-level feature extractor
        # -------------------------
        # ResNet-18 pretrained on ImageNet is used as a
        # generic visual backbone.
        backbone = resnet18(pretrained=True)

        # Remove the final classification layer
        # Output per frame: (512, 1, 1)
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # -------------------------
        # Projection layer
        # -------------------------
        # Maps ResNet features to a compact embedding
        self.proj = nn.Linear(512, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Video tensor of shape (B, T, 3, H, W)

        Returns:
            video_embedding: (B, output_dim)
        """

        # -------------------------
        # Frame-wise processing
        # -------------------------
        # Merge batch and time dimensions so that
        # ResNet processes each frame independently
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Extract visual features per frame
        feats = self.feature_extractor(x)  # (B*T, 512, 1, 1)
        feats = feats.view(B, T, 512)       # (B, T, 512)

        # -------------------------
        # Temporal aggregation
        # -------------------------
        # Simple and stable baseline:
        # average features across time
        feats = feats.mean(dim=1)           # (B, 512)

        # -------------------------
        # Final embedding
        # -------------------------
        video_embedding = self.proj(feats)  # (B, output_dim)

        return video_embedding