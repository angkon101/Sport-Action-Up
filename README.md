# Fine-Grained Sports Action Recognition with Video Foundation Models (VideoMAE)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-VideoMAE-yellow.svg)](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Compute](https://img.shields.io/badge/Compute-Kaggle_T4_|_Colab-blue.svg)](https://kaggle.com)

A state-of-the-art Video Transformer framework for fine-grained sports action recognition across **82 Sports Action Categories** (~9,151 video clips), powered by **VideoMAE ViT-B** (self-supervised masked video autoencoding pre-trained on Kinetics-400) and evaluated against 3D CNN baselines (X3D, SlowFast) and Video Swin Transformers.

---

## 🚀 Architectural Overview

```
              Raw Sports Video Clip (16 sampled frames @ 224x224)
                                     │
                                     ▼
                [Spatio-Temporal Tubelet Patch Embedding]
                               (16x16x2)
                                     │
                                     ▼
                  [VideoMAE ViT-B Transformer Encoder]
                      (12 Layers, 768 Embedding Dim)
                                     │
                                     ▼
                         [Temporal Average Pooling]
                                     │
                                     ▼
                         [LayerNorm + Dropout 0.5]
                                     │
                                     ▼
                     [82-Class Action Classifier Head]
```

---

## 📊 Compute Partitioning Strategy

| Platform | Compute Role | Optimization Details |
| :--- | :--- | :--- |
| **⚡ Kaggle** (Primary, 2× T4 16 GB) | Full End-to-End Pipeline & Main Runs | Multi-GPU DDP via `torchrun`, fast `decord` video sampling, dynamic group-wise splits, fp16 GradScaler, auto-resume |
| **☁️ Google Colab** (Backup, 1× T4 16 GB) | Fallback & Auxiliary Benchmarking | Google Drive auto-mounting for persistent checkpoints, fast VM disk staging, auto-resume |
| **💻 Local Machine** | Code Management & Git Sync | Zero heavy compute required; all data engineering and training run in cloud |

---

## 📁 Repository Structure

```text
├── Archive/                        # Preserved previous HOI-BiLSTM work & raw datasets
│   ├── Model Data/                 # 9,151 RAW SPORTS VIDEO CLIPS (Preserved)
│   ├── checkpoints/                # Archived M7 checkpoints
│   ├── configs/                    # Archived YAML configurations
│   ├── data/                       # Archived feature tensors & old splits
│   ├── notebooks/                  # Archived HOI BiLSTM notebooks
│   ├── src/                        # Archived HOI extractor & BiLSTM code
│   └── README_HOI_BiLSTM.md        # Archived previous README
├── configs/                        # YAML configuration files
│   ├── videomae_base.yaml          # VideoMAE ViT-B training configuration
│   └── x3d_baseline.yaml           # X3D CNN baseline configuration
├── data/
│   ├── splits/                     # Leak-free group-wise CSV splits (train/val/test)
│   └── processed/                  # Transcoded video clips (224px / 160px)
├── notebooks/
│   ├── 01_kaggle_videomae_train.ipynb   # Master Kaggle Training Notebook (Auto-Resume)
│   ├── 02_colab_videomae_backup.ipynb  # Google Colab Backup Notebook
│   └── 03_local_subset_debug.ipynb     # Local 4GB GPU smoke test notebook
├── scripts/
│   ├── 01_audit_dataset.py         # Local dataset audit & integrity checker
│   ├── 02_create_group_splits.py   # Group-wise 70/15/15 CSV partitioner
│   ├── 03_transcode_videos.py      # Multi-threaded 224/160px offline transcoder
│   └── 04_package_kaggle_shards.py # Zip sharding for Kaggle dataset upload
├── src/
│   ├── models/                     # VideoMAE, X3D, and Video Swin models
│   ├── datasets/                   # PyTorch VideoDataset with Decord/PyAV
│   ├── engine/                     # Trainer, Auto-Resume, AMP Scaler, Evaluator
│   └── utils/                      # Metrics, Logging, and Attention Visualizers
├── WORKING_PLAN.md                 # Complete 9-week thesis execution plan & specification
├── requirements.txt                # Python dependencies
└── README.md                       # This document
```

---

## 📖 Complete Master Working Plan

For the comprehensive 9-week thesis research specification, hardware memory budgets, ablation protocols, and evaluation plans, please refer to:

👉 **[WORKING_PLAN.md](WORKING_PLAN.md)**
