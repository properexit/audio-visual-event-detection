import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
from multimodal_model import AudioVisualEventModel
from src.utils.io import load_audio, load_video
import json 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/fusion_model.pt"


def main():

    with open("checkpoints/label_map.json") as f:
        label_to_idx = json.load(f)

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    model = AudioVisualEventModel(num_classes=num_classes)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # -------------------------
    # Test sample
    # -------------------------
    audio_path = "data/vggsound_sample/audio/---g-f_I2yQ.wav"
    frame_dir = "data/vggsound_sample/frames/---g-f_I2yQ"

    # -------------------------
    # Load inputs
    # -------------------------
    audio = load_audio(audio_path).unsqueeze(0).to(DEVICE)   # (1, 1, 16000)
    video = load_video(frame_dir).unsqueeze(0).to(DEVICE)   # (1, T, 3, 224, 224)

    # -------------------------
    # Inference
    # -------------------------
    with torch.no_grad():
        logits, attn = model(audio, video)

    pred = logits.argmax(dim=1).item()

    print("\n=== PREDICTION ===")
    print("Predicted class:", pred)
    print("Fusion attention [audio, video]:", attn.squeeze().tolist())
    pred_idx = logits.argmax(dim=1).item()
    pred_label = idx_to_label[pred_idx]

    print("Predicted event:", pred_label)

if __name__ == "__main__":
    main()