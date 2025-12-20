import os
import sys
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(PROJECT_ROOT)

from multimodal_model import AudioVisualEventModel


# --------------------------------------------------
# Modality perturbations
# --------------------------------------------------

def mute_audio(audio: torch.Tensor) -> torch.Tensor:
    """
    Simulates complete audio failure.

    Args:
        audio: Tensor of shape (B, 1, T)

    Returns:
        Zeroed audio tensor of the same shape.
    """
    return torch.zeros_like(audio)


def blur_video(
    video: torch.Tensor,
    kernel_size: int = 15
) -> torch.Tensor:
    """
    Simulates visual degradation by spatial blurring.

    Args:
        video: Tensor of shape (B, T, 3, H, W)
        kernel_size: Size of averaging kernel

    Returns:
        Blurred video tensor with the same shape.
    """
    B, T, C, H, W = video.shape

    # Flatten temporal dimension for spatial processing
    video = video.view(B * T, C, H, W)

    video = F.avg_pool2d(
        video,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    return video.view(B, T, C, H, W)


# --------------------------------------------------
# Evaluation utilities
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate_condition(
    name: str,
    model: torch.nn.Module,
    audio: torch.Tensor,
    video: torch.Tensor
):
    """
    Runs the model under a specific perturbation
    and reports average modality attention.
    """
    with torch.no_grad():
        _, attn = model(audio, video)

        # attn shape: (B, 2) → [audio_weight, video_weight]
        mean_attn = attn.mean(dim=0).cpu().tolist()

    print(f"\n{name}")
    print(f"  Audio attention : {mean_attn[0]:.3f}")
    print(f"  Video attention : {mean_attn[1]:.3f}")


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():
    """
    Robustness analysis for audio–visual fusion.

    Evaluates how modality attention shifts under:
    - normal conditions
    - muted audio
    - blurred video
    """

    # -------------------------
    # Load model
    # -------------------------
    model = AudioVisualEventModel(
        num_classes=5
    ).to(DEVICE)

    checkpoint_path = "checkpoints/fusion_model.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=DEVICE
            )
        )
        print("Loaded checkpoint:", checkpoint_path)
    else:
        print("No checkpoint found, using random weights")

    model.eval()

    # -------------------------
    # Dummy batch for testing
    # -------------------------
    audio = torch.randn(
        4, 1, 16000
    ).to(DEVICE)

    video = torch.randn(
        4, 8, 3, 224, 224
    ).to(DEVICE)

    # -------------------------
    # Robustness conditions
    # -------------------------
    evaluate_condition(
        "NORMAL",
        model,
        audio,
        video
    )

    evaluate_condition(
        "MUTED AUDIO",
        model,
        mute_audio(audio),
        video
    )

    evaluate_condition(
        "BLURRED VIDEO",
        model,
        audio,
        blur_video(video)
    )


if __name__ == "__main__":
    main()