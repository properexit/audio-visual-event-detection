import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.av_dataset import AudioVisualDataset
from multimodal_model import AudioVisualEventModel


# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checkpoints"


def main():
    """
    Trains an audio–visual event recognition model with
    attention-based fusion and logs modality usage.
    """

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    dataset = AudioVisualDataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # -------------------------
    # Model
    # -------------------------
    model = AudioVisualEventModel(
        num_classes=len(dataset.label_to_idx)
    ).to(DEVICE)

    # -------------------------
    # Optimizer & Loss
    # -------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(NUM_EPOCHS):
        model.train()

        total_loss = 0.0
        attention_sum = torch.zeros(2)

        for audio, video, labels in loader:
            audio = audio.to(DEVICE)
            video = video.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            logits, attn = model(audio, video)
            loss = criterion(logits, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track mean attention across the batch
            attention_sum += attn.mean(dim=0).detach().cpu()

        mean_attention = (
            attention_sum / len(loader)
        ).tolist()

        print(
            f"Epoch {epoch} | "
            f"Loss: {total_loss:.3f} | "
            f"Mean attention: {mean_attention}"
        )

    # -------------------------
    # Save model & metadata
    # -------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "fusion_model.pt"
        )
    )

    with open(
        os.path.join(
            CHECKPOINT_DIR,
            "label_map.json"
        ),
        "w"
    ) as f:
        json.dump(dataset.label_to_idx, f)

    print(
        f"Model saved to {CHECKPOINT_DIR}/fusion_model.pt"
    )


if __name__ == "__main__":
    main()