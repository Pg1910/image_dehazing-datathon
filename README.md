# image_dehazing-datathon
this project is part of the solasta26 hackathon series by datathon .PS- image dehazing

Here is a clean, professional `README.md` tailored to your project.

---

# Image Dehazing Pipeline (NTIRE-Style Evaluation + Web Deployment Ready)

## Overview

This project implements a complete **Image Dehazing pipeline** focused on:

* Robust inference under non-homogeneous haze (fog/smog)
* Structural fidelity (no hallucinated textures)
* NTIRE-style quantitative evaluation (PSNR, SSIM)
* Modular architecture for benchmarking multiple models
* Deployment-ready inference pipeline

The system is designed for real-world Indian fog conditions, including:

* Road safety monitoring
* Railway surveillance
* Traffic camera systems
* Drone-based inspection
* Smart-city infrastructure

---

# Implemented Models

## 1️⃣ DehazeFormer (Transformer-Based)

* Swin-based architecture
* Lightweight variants: `t`, `s`, `b`
* Strong balance between performance and VRAM usage
* Best for production deployment

## 2️⃣ Wavelet U-Net

* Haar wavelet-based downsampling (DWT/IDWT)
* No pooling layers
* Efficient and stable
* Low compute footprint
* Good structural preservation

---

# Project Structure

---


## Pipeline Architecture

![Pipeline Architecture](Green%20and%20Peach%20Simple%20Flowchart%20.png)
*Figure: The complete pipeline architecture for our dehazing solution, as illustrated in the combined green and peach flowchart.*

```
dehaze-hack/
│
├── src/
│   ├── models/
│   │   ├── dehazeformer_wrapper.py
│   │   ├── wavelet_unet_wrapper.py
│   │   ├── wavelet_unet_model.py
│   │   └── wavelet_ops.py
│   │
│   ├── utils/
│   │   ├── io.py
│   │   ├── tiling.py
│   │   └── postprocess.py
│   │
│   ├── infer.py
│   └── eval.py
│
├── data/
│   ├── val/
│   │   ├── hazy/
│   │   └── gt/
│
├── weights/
│
└── outputs/
```

---

# Installation

Create virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision
pip install numpy opencv-python scikit-image tqdm
```

---

# Single Image Inference

## DehazeFormer

```bash
python -m src.infer \
  --model dehazeformer \
  --variant s \
  --ckpt weights/dehazeformer_outdoor_s.pth \
  --input data/sample.png \
  --output out.png \
  --tiles 512,384,320 \
  --overlap 96 \
  --post
```

## Wavelet U-Net

```bash
python -m src.infer \
  --model wavelet-unet \
  --ckpt weights/wavelet_unet.pth \
  --input data/sample.png \
  --output out_wavelet.png \
  --tiles 512,384,320 \
  --overlap 96
```

---

# Batch Evaluation (PSNR + SSIM)

Dataset must be structured as:

```
data/val/
    hazy/
    gt/
```

Filenames must match.

Run evaluation:

```bash
python -m src.eval \
  --model dehazeformer \
  --variant s \
  --ckpt weights/dehazeformer_outdoor_s.pth \
  --hazy_dir data/val/hazy \
  --gt_dir data/val/gt \
  --out_dir outputs/dehazeformer_eval \
  --tiles 512,384,320 \
  --overlap 96 \
  --post
```

For Wavelet U-Net:

```bash
python -m src.eval \
  --model wavelet-unet \
  --ckpt weights/wavelet_unet.pth \
  --hazy_dir data/val/hazy \
  --gt_dir data/val/gt \
  --out_dir outputs/wavelet_eval
```

---

# Metrics

The evaluation script computes:

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

Results are saved in:

```
outputs/<run_name>/metrics.csv
```

Including:

* Per-image PSNR
* Per-image SSIM
* Mean PSNR
* Mean SSIM

---

# Post-Processing (Optional)

A lightweight, safe post-processing module is available:

* Gray-world white balance
* Mild gamma correction
* Soft CLAHE contrast enhancement

Enabled via:

```
--post
```

This avoids hallucinated detail while improving visual clarity.

---

# Design Philosophy

* No generative hallucination
* No artificial sharpening
* No texture invention
* Strict structural fidelity
* NTIRE-style reproducibility

---

# Hardware Profile

Tested on:

* RTX 3050 (4GB VRAM)
* Intel i5-12500H

Inference uses:

* Mixed precision (AMP)
* Tiled processing for large images
* Automatic fallback tile sizing

---

# Future Extensions

* SDCNet integration (optional high-performance model)
* Real-time web dashboard
* Synthetic haze generation pipeline
* Cluster-wise ensemble inference
* Quantitative NTIRE benchmark submission support

---

# Author Notes

This implementation prioritizes:

* Practical deployment
* Clean modular design
* Fast experimentation
* Low VRAM compatibility
* Quantitative benchmarking

The system is suitable for hackathon-level demonstration as well as research benchmarking.
