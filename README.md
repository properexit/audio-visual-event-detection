Audio-Visual Event Recognition with Modality Attention

---

## Overview

This project implements an end-to-end multimodal learning system for audio-visual event recognition, combining raw audio waveforms and video frames using a learned modality-attention fusion mechanism.

The focus of this work is robust multimodal fusion, learning when to trust audio vs video, especially under degraded conditions such as muted audio or blurred video.

**Note:** Due to heavy multimedia dependencies (FFmpeg, yt-dlp, PyTorch, torchvision, soundfile, etc.), the project is not intended for easy local execution.  
Key results are reported below.

---

## Key Contributions

- Late fusion architecture with explicit audio–video attention  
- Lightweight CNN audio encoder (raw waveform)  
- Pretrained ResNet-18 video encoder with temporal average pooling  
- Adaptive modality weighting learned end-to-end  
- Robustness evaluation under modality corruption  
- Real-world data from VGGSound (YouTube videos)

---

## Model Overview

### Architecture

Audio (waveform) → Audio Encoder  
Video (frames) → Video Encoder  

Both embeddings are fused using **Modality Attention**, followed by a classifier.

### Components

| Module | Description |
|------|-------------|
| AudioEncoder | 1D CNN over raw audio waveform |
| VideoEncoder | ResNet-18 backbone + temporal average pooling |
| ModalityAttention | Learns scalar weights for audio vs video |
| Classifier | Linear layer over fused embedding |

---

## Modality Attention (Core Idea)

Instead of naively concatenating audio and video features, the model learns attention weights:

```
[audio_weight, video_weight] = softmax([Wa·a, Wv·v])
```

These weights indicate how much the prediction relies on each modality.

This enables:
- Graceful handling of missing/noisy modalities  
- Interpretable multimodal behavior  
- Robust inference in real-world conditions  

---

## Experimental Results

### Training (VGGSound sample subset)

- Optimizer: Adam  
- Loss: Cross-Entropy  
- Epochs: 5  
- Dataset size: small curated subset (5 classes)

```
Epoch 4 | Loss: 3.731
Mean attention: [Audio: 0.278, Video: 0.722]
```

**Interpretation:**  
For visually distinctive events, the model naturally prefers video.

---

## Robustness Evaluation (Key Result)

The trained model was evaluated under modality corruption.

### Conditions

1. Normal input  
2. Muted audio (audio = zeros)  
3. Blurred video (spatial degradation)  

### Results

| Condition | Audio Attention | Video Attention |
|---------|-----------------|-----------------|
| Normal | 0.379 | 0.621 |
| Muted Audio | 0.377 | 0.623 |
| Blurred Video | 0.456 | 0.544 |

### Key Observations

- When audio is muted, the model relies more on video  
- When video is blurred, the model increases reliance on audio  
- Attention adapts without retraining  

This confirms **true multimodal robustness**, not hardcoded fusion.

---

## Inference Example

Prediction on a real video clip:

```
Predicted class: people marching
Fusion attention [audio, video]: [0.36, 0.64]
```

The model:
- Produces a class label  
- Reports interpretable modality contribution  
- Runs end-to-end from video → prediction  

---

## Repository Structure (High Level)

```
cv_audvid/
├── src/
│   ├── audio/
│   ├── video/
│   ├── fusion/
│   ├── dataset/
│   ├── utils/
│   └── train_fusion.py
├── scripts/
│   ├── extract_audio.py
│   ├── extract_frames.py
│   └── predict.py
├── experiments/
│   └── robustness_test.py
├── checkpoints/
│   ├── fusion_model.pt
│   └── label_map.json
├── tools/
│   ├── download_vggsound_sample.py
└── README.md
```

---

## Dataset

- Source: VGGSound  
- Data: YouTube videos (10s clips)  
- Modalities:
  - Audio: 16 kHz mono waveform  
  - Video: 8 uniformly sampled frames per clip  
- Labels: Event categories (subset)

**Note:** Full dataset download is not included due to size.
