# 🏅 Fine-Grained Sports Action Recognition with Video Transformers (VideoMAE)

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-VideoMAE_ViT--B-yellow.svg)](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Compute: Kaggle 2x T4](https://img.shields.io/badge/Kaggle-2x_NVIDIA_T4_DDP-20BEFF.svg?logo=kaggle&logoColor=white)](https://kaggle.com)
[![Compute: Google Colab](https://img.shields.io/badge/Colab-1x_T4_Auto--Resume-F9AB00.svg?logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)

**A Deep Learning Research Framework for Fine-Grained Sports Action Recognition across 82 Categories (9,151 Videos) using Video Masked Autoencoders (VideoMAE ViT-B), evaluated against 3D CNNs (X3D, SlowFast) and Video Swin Transformers.**

[📖 Working Plan](WORKING_PLAN.md) • [🚀 Run Guide](RUN_GUIDE.md) • [⚡ Kaggle Notebook](notebooks/01_Kaggle_Sports_Action_VideoMAE.ipynb) • [☁️ Colab Notebook](notebooks/02_Colab_Sports_Action_VideoMAE.ipynb)

</div>

---

## 📌 Table of Contents
- [1. Overview & Research Motivation](#1-overview--research-motivation)
- [2. System Architecture](#2-system-architecture)
- [3. The 82-Class Sports Dataset](#3-the-82-class-sports-dataset)
- [4. Zero-Leakage Group-Wise Splitting](#4-zero-leakage-group-wise-splitting)
- [5. Cloud-First Execution & Memory Strategy](#5-cloud-first-execution--memory-strategy)
- [6. Quickstart: Running on Kaggle (Primary Cloud)](#6-quickstart-running-on-kaggle-primary-cloud)
- [7. Quickstart: Running on Google Colab (Backup Cloud)](#7-quickstart-running-on-google-colab-backup-cloud)
- [8. Local Machine Workflow (Data Audit & Debugging)](#8-local-machine-workflow-data-audit--debugging)
- [9. Model Hierarchy & Baseline Benchmarks](#9-model-hierarchy--baseline-benchmarks)
- [10. Expected Outputs & Thesis Deliverables](#10-expected-outputs--thesis-deliverables)
- [11. Repository Structure](#11-repository-structure)
- [12. Troubleshooting & FAQ](#12-troubleshooting--faq)
- [13. License & Acknowledgements](#13-license--acknowledgements)

---

## 1. Overview & Research Motivation

Fine-grained sports action recognition poses distinct computer vision challenges:
1. **High Inter-Class Visual Similarity**: Actions like `BaseballPitch` vs. `CricketBowling`, or `TennisSwing` vs. `TableTennisShot` share similar athletic body poses, lighting, and grass/court backgrounds.
2. **Fast Non-Linear Dynamics**: Critical motion cues occur in small temporal windows (e.g. racket contact, ball release, gymnastics apparatus dismount).
3. **Severe Hardware Boundaries**: Pre-training and fine-tuning 3D video models typically requires massive compute clusters. This framework is engineered to run **100% within free-tier cloud environments (Kaggle 2× T4 GPUs and Google Colab T4)** without out-of-memory (OOM) crashes or session disconnect data loss.

### 🌟 Why VideoMAE ViT-B?
Traditional pipelines rely on multi-stage CNN-LSTM pipelines or separate feature extractors (ResNet + YOLO + Keypoints) that lose joint spatio-temporal representations. **VideoMAE** (Video Masked Autoencoder) treats video as a unified 3D volume:
- Uses a **90% masking ratio** during self-supervised pre-training, forcing the Transformer to learn motion trajectory and temporal dynamics rather than static appearance shortcuts.
- Processes video via **spatio-temporal tubelet embeddings ($16 \times 16 \times 2$)**, allowing full self-attention across both space and time.
- Pre-trained on Kinetics-400 and adapted here with a custom classification head for **82 sports categories**.

---

## 2. System Architecture

```
                    Raw Sports Video Clip (16 sampled frames @ 224×224)
                                           │
                                           ▼
                      [In-Memory C++ Fast Decoding via Decord]
                       Uniform temporal sampling across duration
                                           │
                                           ▼
                      [Spatio-Temporal Tubelet Patch Embedding]
                                    (16 × 16 × 2)
                                           │
                                           ▼
                     [VideoMAE ViT-B Spatio-Temporal Encoder]
                     - 12 Transformer Layers, 768 Embedding Dim
                     - 12 Multi-Head Self-Attention Heads
                     - Gradient Checkpointing (60-70% VRAM saving)
                                           │
                                           ▼
                            [Temporal Average Pooling]
                                           │
                                           ▼
                            [LayerNorm + Dropout (p=0.5)]
                                           │
                                           ▼
                        [82-Class Sports Classifier Head]
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
          [Softmax Cross-Entropy Loss]                [Predicted Top-k Sports Action]
```

---

## 3. The 82-Class Sports Dataset

The dataset encompasses **9,151 video clips** spanning **82 distinct sports disciplines**:

<details>
<summary><b>Click to expand the full list of 82 Sports Categories</b></summary>

| Category Type | Included Sports Actions |
| :--- | :--- |
| **Racket & Net Sports** | TennisSwing, TableTennisShot, Badminton, VolleyballSpiking |
| **Athletics & Track/Field** | HighJump, LongJump, PoleVault, JavelinThrow, Shotput, HammerThrow, ThrowDiscus, Run-Side, Walk-Front |
| **Gymnastics & Acrobatic** | BalanceBeam, FloorGymnastics, ParallelBars, UnevenBars, PommelHorse, StillRings, TrampolineJumping, HandstandPushups, HandstandWalking |
| **Aquatic Sports** | BreastStroke, FrontCrawl, Diving, Diving-Side, CliffDiving, Surfing, Rafting, Skijet, Rowing |
| **Combat & Martial Arts** | BoxingPunchingBag, BoxingSpeedBag, Punch, SumoWrestling, Fencing, TaiChi, Nunchucks |
| **Team & Ball Sports** | Basketball, BasketballDunk, SoccerPenalty, SoccerJuggling, CricketShot, CricketBowling, BaseballPitch, FieldHockeyPenalty |
| **Strength & Fitness** | BenchPress, CleanAndJerk, PullUps, PushUps, WallPushups, BodyWeightSquats, Lunges, Lifting, RopeClimbing |
| **Equestrian & Animal Sports** | HorseRace, HorseRiding, Riding-Horse |
| **Recreational & Target Sports** | Archery, Billiards, Bowling, GolfSwing, Golf-Swing-Front, Golf-Swing-Back, Golf-Swing-Side, FrisbeeCatch, HulaHoop, SkateBoarding, SkateBoarding-Front, Skiing, SkyDiving, IceDancing, Swing, Swing-Bench, Swing-SideAngle |

</details>

---

## 4. Zero-Leakage Group-Wise Splitting

In the UCF dataset family, video filenames follow a structured convention:
$$\mathbf{v\_[ActionName]\_g[GroupID]\_c[ClipID].avi}$$
* Example: `v_Archery_g01_c01.avi` and `v_Archery_g01_c02.avi` are filmed from the **same camera session**, featuring the **same athlete**, **same background**, and **same equipment**.
* **The Leakage Danger**: Naive random train/test splitting distributes clips from `g01` into both sets, causing visual contamination and producing falsely inflated test metrics.
* **Our Solution**: Strict **Group-Wise Partitioning** (implemented in [`src/datasets/splitter.py`](src/datasets/splitter.py)):
  * All clips sharing the same `(class_name, group_id)` are assigned atomically to **either** Train, Validation, or Test.
  * Target distribution: **70% Train** (~6,400 clips), **15% Validation** (~1,375 clips), and **15% Test** (~1,375 clips).
  * Mathematically verified: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

---

## 5. Cloud-First Execution & Memory Strategy

Video Transformer training is computationally demanding. We designed this framework for **100% cloud execution on free-tier platforms**:

| Strategy Layer | Implementation | Benefit |
| :--- | :--- | :--- |
| **Multi-GPU Parallelism (Kaggle)** | PyTorch DDP via `torchrun --nproc_per_node=2` | Leverages both 16 GB T4 GPUs simultaneously, cutting 30 epochs from ~8 hours to **~4 hours**. |
| **fp16 Mixed Precision** | `torch.cuda.amp.autocast(dtype=torch.float16)` + `GradScaler` | Reduces activation memory by 50% without numerical instability. |
| **Gradient Checkpointing** | `model.gradient_checkpointing_enable()` | Traded ~15% compute time to save **60–70% activation VRAM**. |
| **Micro-Batch Accumulation** | Micro-batch 2–4 per GPU; Accumulation 4–16 steps | Decouples memory limits from training dynamics to achieve an **Effective Batch Size of 32–64**. |
| **Session Cut Auto-Resume** | Atomic checkpointing with RNG state persistence | If a cloud session terminates, re-running the script resumes from the exact epoch and step. |
| **In-Memory C++ Decoding** | `decord.VideoReader` | Zero offline disk transcoding required; samples 16 uniform frames in RAM in milliseconds. |

---

## 6. Quickstart: Running on Kaggle (Primary Cloud)

Kaggle is the recommended main platform because it provides **2× NVIDIA T4 GPUs (32 GB total VRAM)** and 30 GPU hours per week.

```
Kaggle Notebook -> Accelerator: GPU T4 x2 -> Add Input (Dataset) -> Run Cells
```

### Step 1: Attach Your Dataset in Kaggle
1. Upload your dataset archive `Model Data.zip` to Kaggle as a new private dataset:
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) $\to$ **New Dataset**.
   - Drag & drop `Archive/Model Data.zip` and give it a title (e.g. `sport-action-up`).
2. Open or create a Kaggle notebook.
3. On the right-hand sidebar under **Input**, click **`+ Add Input`** and select your uploaded dataset.

### Step 2: Open Master Notebook
Upload and open [`notebooks/01_Kaggle_Sports_Action_VideoMAE.ipynb`](notebooks/01_Kaggle_Sports_Action_VideoMAE.ipynb) in Kaggle.

### Step 3: Run the Cells
1. **Set Accelerator**: In the right sidebar under **Settings**, select **Accelerator $\to$ GPU T4 x2**.
2. **Install & Sync Code**:
   ```python
   !pip install -q decord av transformers accelerate timm pandas pyyaml
   %cd /kaggle/working
   !rm -rf /kaggle/working/Sport-Action-Up
   !git clone https://github.com/angkon101/Sport-Action-Up.git
   %cd /kaggle/working/Sport-Action-Up
   ```
3. **Launch Dual-GPU DDP Training**:
   ```python
   !torchrun --nproc_per_node=2 train_cloud.py --config configs/videomae_kaggle_dual_t4.yaml
   ```
4. **Auto-Resume**: If the session disconnects, simply re-run the training cell. The script automatically detects `/kaggle/working/checkpoints/last_checkpoint.pth` and continues training seamlessly!

---

## 7. Quickstart: Running on Google Colab (Backup Cloud)

Google Colab provides a free single **NVIDIA T4 (16 GB)**. We use Google Drive to persist all checkpoints and logs across disconnects.

### Step 1: Open Master Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File $\to$ Upload Notebook**, and select [`notebooks/02_Colab_Sports_Action_VideoMAE.ipynb`](notebooks/02_Colab_Sports_Action_VideoMAE.ipynb).
3. Select **Runtime $\to$ Change runtime type $\to$ T4 GPU**.

### Step 2: Mount Google Drive & Clone Code
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf /content/Sport-Action-Up
!git clone https://github.com/angkon101/Sport-Action-Up.git
%cd /content/Sport-Action-Up
```

### Step 3: Extract Dataset to Fast Local SSD
```python
# Unzips from Drive to fast local VM disk (/content/dataset)
!unzip -q "/content/drive/MyDrive/Model Data.zip" -d /content/dataset
```

### Step 4: Launch Training
```python
!python train_cloud.py --config configs/videomae_colab_t4.yaml --data-dir /content/dataset
```
* Checkpoints are automatically synced to `/content/drive/MyDrive/SportsAction_Checkpoints/`.

---

## 8. Local Machine Workflow (Data Audit & Debugging)

Your local machine (even with a 4 GB GPU or CPU only) handles data integrity auditing and micro-subset validation:

### 1. Audit Dataset Integrity
Scans video clips, flags corrupt files, and verifies resolutions and frame counts:
```bash
python scripts/01_audit_dataset.py --data-dir "Archive/Model Data" --output-dir "data/audit"
```

### 2. Generate Leakage-Free Group Splits Locally
```bash
python scripts/02_create_group_splits.py --data-dir "Archive/Model Data" --output-dir "data/splits"
```

### 3. Optional: Transcode Videos to 224px @ 25 FPS
```bash
python scripts/03_transcode_videos.py --src-dir "Archive/Model Data" --dst-dir "data/processed/videos_224px" --shorter-side 224 --fps 25.0
```

---

## 9. Model Hierarchy & Baseline Benchmarks

To establish rigorous academic credibility, VideoMAE is evaluated against a 4-tier baseline hierarchy:

```
                            [Evaluation Benchmark Hierarchy]
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
[1. Frozen-Feature Baseline]    [2. Efficient 3D CNNs]             [3. Video Transformers]
- VideoMAE ViT-B (Frozen)       - X3D-Small (PyTorchVideo)         - VideoMAE ViT-B (End-to-End)
- Train MLP Head on Features    - SlowFast R50 (PyTorchVideo)      - Video Swin-Transformer (Swin-T)
- Runtime: ~15 minutes          - Runtime: ~1.5 hours / 30 epochs  - Runtime: ~3.5 hours / 30 epochs
- Purpose: Representation floor - Purpose: Real-time efficiency    - Purpose: SOTA action modeling
```

| Model | Architecture Family | Pre-training Source | Parameters | Role in Thesis | Kaggle GPU Runtime |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **VideoMAE Frozen Head** | Linear Probe | Kinetics-400 (Frozen) | ~0.1M | Representation floor | **~15 min** |
| **X3D-S / X3D-M** | Efficient 3D CNN | Kinetics-400 | 3.8M | Lightweight edge baseline | **~1.5 hrs** |
| **SlowFast (ResNet-50)** | Dual-Stream 3D CNN | Kinetics-400 | 34.0M | Competitive 3D CNN baseline | **~2.5 hrs** |
| **Video Swin-T** | Hierarchical Transformer| Kinetics-400 | 28.2M | Transformer architecture baseline| **~3.5 hrs** |
| **VideoMAE ViT-B (Proposed)**| **Spatio-Temporal ViT** | **Kinetics-400 (Self-Supervised)**| **86.0M** | **Proposed Primary Model** | **~4.0 hrs (Dual T4 DDP)** |

---

## 10. Expected Outputs & Thesis Deliverables

Every training run produces publication-ready artifacts:

1. **Model Weights**:
   - `best_checkpoint.pth`: Weights achieving highest validation accuracy on 82 classes.
   - `last_checkpoint.pth`: Resumable state (weights, optimizer, scheduler, AMP scaler, RNG state).
2. **Telemetry Logs**:
   - `training_history.csv`: Per-epoch record of Train Loss, Val Loss, Top-1 Acc, Top-5 Acc, and Learning Rate.
3. **Academic Evaluation Visualizations**:
   - **82-Class Confusion Matrix Heatmap**: Visualizes fine-grained misclassifications across similar sports.
   - **Convergence Curves**: Loss and Accuracy vs. Epoch.
   - **Spatio-Temporal Attention Maps**: Shows what motion dynamics and athlete joints the transformer focuses on.
   - **Spatio-Temporal Grad-CAM**: Heatmap overlays confirming the model tracks athletic equipment (rackets, balls, bars) rather than background context.

---

## 11. Repository Structure

```text
D:\AKHI THESIS UP\
├── Archive/                        # Complete preservation of previous HOI-BiLSTM work
│   ├── Model Data/                 # 9,151 RAW SPORTS VIDEOS (Preserved)
│   ├── Model Data.zip              # Pre-packaged 4.48 GB dataset archive ready for upload
│   ├── checkpoints/                # Archived M7 checkpoints
│   ├── configs/                    # Archived YAML configurations
│   ├── notebooks/                  # Archived HOI notebooks
│   ├── src/                        # Archived HOI extractor & BiLSTM models
│   └── README_HOI_BiLSTM.md        # Archived previous README
├── configs/                        # Production YAML configurations
│   ├── videomae_base.yaml          # Base VideoMAE configuration
│   ├── videomae_kaggle_dual_t4.yaml# Kaggle Dual-T4 DDP configuration
│   └── videomae_colab_t4.yaml      # Google Colab single-T4 configuration
├── data/
│   ├── audit/                      # Data integrity reports & statistics
│   └── splits/                     # Leakage-free group-wise CSV splits (train/val/test)
├── notebooks/
│   ├── 01_Kaggle_Sports_Action_VideoMAE.ipynb # Master Kaggle Dual-GPU Notebook
│   └── 02_Colab_Sports_Action_VideoMAE.ipynb  # Master Colab Google Drive Notebook
├── scripts/
│   ├── 01_audit_dataset.py         # Dataset auditor & corrupt file detector
│   ├── 02_create_group_splits.py   # Group-wise 70/15/15 CSV partitioner
│   ├── 03_transcode_videos.py      # Multi-threaded 224/160px video transcoder
│   └── generate_cloud_notebooks.py # Programmatic notebook generator
├── src/
│   ├── datasets/                   # PyTorch VideoDataset with Decord & Splitter
│   │   ├── splitter.py             # Dynamic zero-leakage group-wise splitter
│   │   └── video_dataset.py        # Fast C++ Decord reader & spatio-temporal transforms
│   ├── engine/                     # Core training & validation machinery
│   │   ├── checkpoint.py           # Atomic state save & auto-resume engine
│   │   └── trainer.py              # fp16 AMP, gradient accumulation & evaluation
│   ├── models/                     # Model architecture definitions
│   │   └── videomae.py             # VideoMAE ViT-B with layer-wise decay & grad checkpointing
│   └── utils/                      # Evaluation metrics & telemetry
│       └── metrics.py              # Top-1/Top-5 accuracy & AverageMeter
├── train_cloud.py                  # Master CLI training runner (Single GPU & DDP)
├── WORKING_PLAN.md                 # Complete 9-week thesis research specification
├── RUN_GUIDE.md                    # Step-by-step handbook for Kaggle, Colab, and Local
├── requirements.txt                # Python dependencies
└── README.md                       # This master documentation
```

---

## 12. Troubleshooting & FAQ

<details>
<summary><b>Q1: Kaggle says "FileNotFoundError: No video files found at '/kaggle/input/sport-action-up'"</b></summary>

**Cause**: You have not attached your dataset to the Kaggle notebook.  
**Fix**: On the right sidebar of your Kaggle notebook, click **`+ Add Input`** under "Input". Search for or upload your dataset (`Model Data.zip`). `train_cloud.py` will automatically detect any mounted dataset regardless of the title you give it!
</details>

<details>
<summary><b>Q2: Colab disconnects mid-training. Did I lose my progress?</b></summary>

**No!** Checkpoints are saved automatically after every single epoch to `/content/drive/MyDrive/SportsAction_Checkpoints/last_checkpoint.pth`. When you reconnect Colab, simply run the training cell again:
```bash
!python train_cloud.py --config configs/videomae_colab_t4.yaml --data-dir /content/dataset
```
The script will print:
`[🔄] Resuming from detected checkpoint: .../last_checkpoint.pth`
and continue from the exact epoch it left off.
</details>

<details>
<summary><b>Q3: What if I encounter a CUDA Out of Memory (OOM) error?</b></summary>

**Fix**: The configurations are pre-calibrated for 16 GB GPUs. However, if using larger frame counts:
- Lower `micro_batch_size` from 4 to 2 (or 1) in your config YAML.
- Increase `gradient_accumulation_steps` proportionally (e.g., from 4 to 8) to maintain the identical effective batch size of 32.
- Ensure `gradient_checkpointing: true` is enabled in your YAML config.
</details>

---

## 13. License & Acknowledgements

This research project is distributed under the **MIT License**.

### Acknowledgements
* Pre-trained VideoMAE models courtesy of [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [Hugging Face Transformers](https://huggingface.co/).
* UCF101 dataset provided by the Center for Research in Computer Vision (CRCV) at the University of Central Florida.
* Video decoding powered by the [Decord](https://github.com/dmlc/decord) C++ framework.
