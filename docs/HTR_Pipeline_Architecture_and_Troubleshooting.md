---
title: "Legal Records HTR Engine - System Architecture & Engineering Log"
date: 2026-09-21
tags:
  - deep-learning
  - htr
  - computer-vision
  - arch-linux
  - docker
  - pytorch
  - trocr
status: in-progress
hardware:
  gpu: "NVIDIA GeForce GTX 1650 (Mobile/Desktop, 4GB VRAM)"
  driver: "615.71.09"
  cuda_host: "13.4"
  cuda_container: "12.4"
---

# Legal Records HTR Engine: System Architecture & Runbook

> [!abstract] Overview
> An end-to-end deep learning system designed to digitize, transcribe, and index degraded historical legal scans and handwritten court affidavits into a searchable hybrid repository (PostgreSQL + pgvector).

---

## 1. Pipeline Architecture
[Raw Scanned Document]
│
▼
[Stage 1: Preprocessing]
├── MinAreaRect Deskewing
├── Sauvola Adaptive Binarization
└── Median Denoising Filter
│
▼
[Stage 2: Line Segmentation]
├── Horizontal Projection Profiling (HPP)
├── Dynamic Trough Thresholding
└── Line-Level Bounding Box Cropping
│
▼
[Stage 3: HTR Engine]
├── ViT Encoder + Autoregressive BART Decoder (TrOCR-base)
├── FP16 Half-Precision on GTX 1650 (VRAM < 1 GB)
└── Softmax Token Probability Averaging (Confidence Scoring)
│
▼
[Stage 4: Downstream Services]
├── Legal NER (Parties, Dates, Case Numbers)
└── Hybrid Search (BM25 Lexical + Dense Vector Embeddings)


---

## 2. Arch Linux & Docker Troubleshooting Log

> [!bug] Issue 1: Docker CDI Vendor Discovery Failure
> **Symptom:**
> `docker: Error response from daemon: failed to discover GPU vendor from CDI: no known GPU vendor found`
> 
> **Root Cause:**
> 1. Modern Docker releases prioritize the Container Device Interface (CDI) specification.
> 2. The user Docker client context was set to `desktop-linux` instead of the system host daemon (`default`).
> 
> **Resolution:**
> ```bash
> docker context use default
> sudo mkdir -p /etc/cdi /var/run/cdi
> sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
> sudo cp /etc/cdi/nvidia.yaml /var/run/cdi/nvidia.yaml
> sudo systemctl restart docker
> ```

> [!bug] Issue 2: NumPy 2.x C-API ABI Incompatibility
> **Symptom:**
> `UserWarning: Failed to initialize NumPy: _ARRAY_API not found` followed by `[transformers] Disabling PyTorch because PyTorch >= 2.5 is required`.
> 
> **Root Cause:**
> PyTorch 2.2 was built against NumPy 1.x. A fresh `pip install` pulled NumPy 2.2+, breaking C-extensions and disabling PyTorch in Hugging Face.
> 
> **Resolution:**
> - Upgraded Dockerfile base to `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`.
> - Pinned `numpy<2.0.0` in `requirements.txt`.

> [!bug] Issue 3: Missing Tokenizer Dependencies for TrOCR
> **Symptom:**
> `ValueError: Couldn't instantiate the backend tokenizer ... You need to have sentencepiece or tiktoken installed`.
> 
> **Resolution:**
> Appended `sentencepiece>=0.2.0` and `protobuf>=4.25.0` to `requirements.txt`.

---

## 3. Mathematical Foundations

### Sauvola Adaptive Binarization
Standard Otsu thresholding fails on legal parchment due to non-uniform illumination and ink bleed-through. Sauvola thresholding calculates an adaptive threshold $T(x, y)$ per pixel:

$$T(x, y) = m(x, y) \cdot \left(1 + k \cdot \left(\frac{s(x, y)}{R} - 1\right)\right)$$

Where:
- $m(x, y)$ is the local neighborhood mean intensity.
- $s(x, y)$ is the local neighborhood standard deviation.
- $R$ is the dynamic range of standard deviation ($128$ for 8-bit grayscale).
- $k$ is the threshold control coefficient ($0.2$ for faint cursive script).

### Horizontal Projection Profiling (HPP) Line Segmentation
Documents are segmented into lines by integrating inverted pixel intensity horizontally across each row $y$:

$$H(y) = \sum_{x=0}^{W-1} \frac{255 - I(x, y)}{255}$$

Valleys where $H(y) \approx 0$ mark inter-line boundaries.

---

## 4. Hardware Optimization for 4GB VRAM (GTX 1650)

> [!tip] Low-VRAM Inference Rules
> 1. **Half-Precision (FP16):** Load weights with `torch_dtype=torch.float16` to drop memory footprint from ~2.2 GB to ~850 MB.
> 2. **Batch Processing:** Process line crops sequentially or in small mini-batches ($B \le 4$).
> 3. **No Grad:** Wrap inference in `with torch.no_grad():`.

---

## 5. Confidence Score Computation
Sequence-level confidence is calculated from generated token probabilities:

$$\text{Confidence} = \frac{1}{N} \sum_{t=1}^{N} P(y_t \mid y_{<t}, X)$$
