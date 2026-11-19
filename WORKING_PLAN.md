# 📋 Master Working Plan: Video Transformer & VideoMAE Framework for Sports Action Recognition

> **Research Thesis**: Fine-Grained Sports Action Recognition across 82 Sports Categories  
> **Primary Compute**: Kaggle (2× NVIDIA T4 16GB / Tesla P100 16GB, ~30 GPU hrs/week)  
> **Secondary / Backup Compute**: Google Colab Free Tier (1× NVIDIA T4 16GB)  
> **Local Machine**: 4 GB NVIDIA GPU (GTX 1650/RTX 3050 mobile) — Data Audit, Group-Wise Splitting, Video Preprocessing, & Sub-Batch Smoke Tests  
> **Date Updated**: November 2026  

---

## 📑 Table of Contents
1. [Executive Summary & Architectural Pivot](#1-executive-summary--architectural-pivot)
2. [Hardware & Free-Tier Quota Engineering](#2-hardware--free-tier-quota-engineering)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Phase 0: Local Preprocessing & Leak-Free Splitting (Week 1)](#phase-0-local-preprocessing--leak-free-splitting-week-1)
5. [Phase 1: Environment & Resume-Safe Training Engine (Week 1)](#phase-1-environment--resume-safe-training-engine-week-1)
6. [Phase 2: VRAM Optimization & Memory Budgets (Week 2)](#phase-2-vram-optimization--memory-budgets-week-2)
7. [Phase 3: Baseline Hierarchy (Weeks 2–3)](#phase-3-baseline-hierarchy-weeks-23)
8. [Phase 4: Main VideoMAE Fine-Tuning Runs (Weeks 3–5)](#phase-4-main-videomae-fine-tuning-runs-weeks-35)
9. [Phase 5: Ablation Studies (Weeks 5–6)](#phase-5-ablation-studies-weeks-56)
10. [Phase 6: Evaluation, Visualization & Thesis Writing (Weeks 7–9)](#phase-6-evaluation-visualization--thesis-writing-weeks-79)
11. [Repository Layout](#11-repository-layout)
12. [Immediate Next Actions](#12-immediate-next-actions)

---

## 1. Executive Summary & Architectural Pivot

### 1.1 The Shift from Multimodal CNN-LSTM to Video Transformers
* **Previous Plan (Archived in `Archive/`)**: Extracted offline multi-branch features (ResNet-50 visual, Keypoint R-CNN 17 pose joints, Faster R-CNN object bounding boxes, and custom HOI geometric kinematics) feeding into a 2-layer BiLSTM with temporal attention.
  * *Limitation*: Heavy preprocessing dependencies, fragmented feature representations, inability to jointly optimize spatio-temporal representations end-to-end, and high inference latency.
* **New Plan**: End-to-end Video Foundation Models centered on **VideoMAE ViT-B** (Self-supervised masked autoencoder pre-trained on Kinetics-400), evaluated against strong lightweight CNN baselines (X3D, SlowFast) and Video Swin Transformers.
  * *Advantage*: Unified spatio-temporal self-attention, state-of-the-art representation learning, scalable fine-tuning, and direct comparison against modern action recognition benchmarks.

---

## 2. Hardware & Free-Tier Quota Engineering

Because free-tier GPU instances impose strict session timeouts and memory boundaries, our training pipeline is built around **strict resource isolation**:

| Resource Attribute | Local Machine (Local Host) | Kaggle (Primary Cloud) | Google Colab (Secondary Backup) |
| :--- | :--- | :--- | :--- |
| **GPU Model** | 4 GB VRAM (e.g., GTX 1650) | 2× NVIDIA T4 (16 GB each) or P100 (16 GB) | 1× NVIDIA T4 (16 GB, non-guaranteed) |
| **CPU / RAM** | Local Multi-core CPU / 16–32 GB RAM | 4 vCPUs / ~30 GB RAM | 2 vCPUs / ~12.7 GB RAM |
| **Session Lifetime** | Persistent / Local | 9–12 hours max per session | Disconnects unpredictably (often 2–6 hrs) |
| **Weekly Quota** | Unlimited | ~30 GPU Hours / week (refreshes weekly) | Dynamic / opaque compute units |
| **Storage Persistence** | Local SSD (Permanent) | `/kaggle/working` ephemeral; Kaggle Datasets | Drive mount (15 GB free, high I/O latency) |
| **Assigned Role** | **Data Audit, Group Splitting, Video Transcoding & Micro-Subset Tests** | **Main VideoMAE Fine-Tuning & Multi-Epoch Benchmarks** | **Fallback Debugging & Baseline Runs** |

### Free-Tier Failure Defense Protocols
1. **Zero Decoding on Cloud Workers**: Cloud VMs have only 2–4 CPU cores. Real-time video decoding during training starves the GPU, causing 70–80% GPU idle time. We transcode all clips to 224px (or 160px) @ 25 FPS locally, and optionally pre-extract uniform frame tensors.
2. **Atomic State Checkpointing**: Every epoch saves `last_checkpoint.pth` containing:
   * `model_state_dict`
   * `optimizer_state_dict`
   * `scheduler_state_dict`
   * `scaler_state_dict` (AMP GradScaler)
   * `epoch` and `global_step`
   * `best_val_acc` and `metrics_history`
   * `rng_states` (PyTorch, CUDA, NumPy, Python standard random)
3. **Session Interruption Recovery**: Scripts check for existing checkpoint files at launch and automatically resume training without human intervention.
4. **Bandwidth & Storage Optimization**: Kaggle dataset uploads are zipped into multi-part shards to avoid upload throttling on thousands of small files.

---

## 3. End-to-End System Architecture

```
                               Raw Videos (82 Sports Classes, 9,151 clips)
                                    [Archive/Model Data/*.avi]
                                                │
                                                ▼
                             [Phase 0: Local Preprocessing Engine]
                                                │
                   ┌────────────────────────────┴────────────────────────────┐
                   ▼                                                         ▼
    [Group-Wise Splitter (70/15/15)]                           [Video Preprocessing / Resize]
    - Parse: v_<class>_g<group>_c<clip>.avi                   - Transcode shorter side to 224px @ 25 FPS
    - Group-isolated partitioning                              - Sub-160px option for fast ablation
    - Exports: train.csv, val.csv, test.csv                    - Pack into Kaggle Dataset shards
                   │                                                         │
                   └────────────────────────────┬────────────────────────────┘
                                                ▼
                              Private Kaggle Dataset (Mounted in /kaggle/input)
                                                │
                                                ▼
                             [Phase 1 & 2: Cloud Training Engine]
                     ┌──────────────────────────────────────────────────────┐
                     │ Precision: fp16 Mixed Precision (torch.cuda.amp)     │
                     │ Micro-Batch: 2 to 4 per T4 GPU                       │
                     │ Grad Accumulation: 8 to 16 steps (Effective Batch 32)│
                     │ Gradient Checkpointing: Enabled                      │
                     │ Optimizer: AdamW + Layer-wise Decay (0.75) + Cosine  │
                     └──────────────────────────────────────────────────────┘
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
         [Branch A: Video Transformers]                   [Branch B: Reference Baselines]
         - VideoMAE ViT-B (MCG-NJU K400)                  - X3D-S / X3D-M (PyTorchVideo)
         - VideoMAE V2 (Optional OpenGVLab)               - SlowFast R50 (Kinetics-400)
         - Video Swin-T (MMAction2)                       - Frozen-Feature Linear Head
                        │                                               │
                        └───────────────────────┬───────────────────────┘
                                                ▼
                             [Phase 5: Systematic Ablations]
                             - Temporal resolution: 8 vs 16 frames
                             - Fine-tuning: Frozen vs Top-K vs Full
                             - Data fraction: 10% / 25% / 50% / 100%
                             - Pose fusion (RTMPose + PoseC3D)
                                                │
                                                ▼
                             [Phase 6: Evaluation & Reporting]
                             - Top-1 & Top-5 accuracy across 82 categories
                             - Confusion matrix & error category taxonomy
                             - Spatio-temporal Attention Rollout & Grad-CAM
```

---

## 4. Phase 0: Local Preprocessing & Leak-Free Splitting (Week 1)

### 4.1 Data Integrity Audit
* **Input Directory**: `Archive/Model Data` (holding all 9,151 `.avi` videos).
* **Checks**:
  * Verify video header validity (OpenCV / PyAV / decord).
  * Identify zero-byte or corrupt files.
  * Extract metadata: FPS, frame count, duration, resolution, aspect ratio.
  * Compute class distribution histograms to identify class imbalances across the 82 sports actions.

### 4.2 Group-Wise Leakage-Free Splitting
* **The Problem**: In UCF datasets, clips named `v_<Action>_g01_c01.avi` and `v_<Action>_g01_c02.avi` originate from the exact same source video (identical athlete, background, lighting, and camera angle). Random splitting leaks test scenes into training data, producing falsely inflated test metrics.
* **The Solution**: Group-wise partitioning by source video group ID:
  $$\text{Group ID} = \text{regex}(\text{"g[0-9]+"}, \text{filename})$$
* **Target Ratio**:
  * **Train**: ~70% of groups (~6,400 clips)
  * **Validation**: ~15% of groups (~1,375 clips) — used for early stopping and hyperparameter selection.
  * **Test**: ~15% of groups (~1,375 clips) — held out strictly for final thesis benchmarks.
* **Output Format**: Clean, canonical CSV files in `data/splits/`:
  * `train.csv` (`video_path, class_name, class_id, group_id, frame_count, duration`)
  * `val.csv`
  * `test.csv`

### 4.3 Video Transcoding & Resolution Normalization
* Re-encode all videos to reduce disk size and cloud CPU decode latency:
  * Minimum side: **224 px** (standard for ViT-B patch size $16 \times 16$).
  * Secondary ablation size: **160 px** (for rapid experimentation).
  * Frame rate: Fixed **25 FPS**.
  * Codec: H.264 (`libx264`), CRF 20, fast preset.
* **Storage Output**: Zipped into 2 GB shards (`sports_dataset_224px.zip.001`, etc.) for seamless upload to a private Kaggle Dataset via the Kaggle CLI.

---

## 5. Phase 1: Environment & Resume-Safe Training Engine (Week 1)

### 5.1 Training Loop Architecture & State Machine
The trainer is implemented as an interrupt-resilient state machine:

```python
# Checkpoint State Structure
checkpoint = {
    'epoch': current_epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_val_acc': best_val_acc,
    'rng_states': {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
        'numpy': np.random.get_state(),
        'python': random.getstate()
    }
}
```

### 5.2 Auto-Resume Protocol
1. On startup, check `checkpoint_dir` for `last_checkpoint.pth`.
2. If found:
   * Load model weights and transfer optimizer state to GPU.
   * Restore learning rate schedule step.
   * Restore PyTorch AMP `GradScaler`.
   * Restore all RNG states to preserve deterministic data loader shuffling.
   * Resume loop from `checkpoint['epoch'] + 1`.
3. If not found, download pre-trained Kinetics weights and initialize fresh head for 82 classes.

### 5.3 Logging Strategy
* **Local CSV Logger**: Append-only `training_history.csv` saved after every epoch (survives network disconnects).
* **Weights & Biases (W&B)**: Real-time telemetry (train loss, val loss, top-1 accuracy, top-5 accuracy, learning rate, GPU memory).

### 5.4 Phase 1 Smoke Test Milestone
* Execute a 200-clip run for 2 epochs on Kaggle / Colab to verify:
  * Data loading speed and GPU utilization $>90\%$.
  * AMP mixed precision stability (zero NaN loss explosions).
  * Manual kill and auto-resume test.

---

## 6. Phase 2: VRAM Optimization & Memory Budgets (Week 2)

### 6.1 Memory Budget for NVIDIA T4 (16 GB)
The base VideoMAE model has ~86M parameters (~345 MB fp32, ~172 MB fp16). However, self-attention activations across $T=16$ frames with $H \times W = 224 \times 224$ (1,568 spatio-temporal tokens per clip) scale quadratically without optimization.

| Optimization Layer | Impact on VRAM | Implementation |
| :--- | :--- | :--- |
| **fp16 Mixed Precision** | 50% activation memory savings | `torch.cuda.amp.autocast(dtype=torch.float16)` + `GradScaler` |
| **Gradient Checkpointing** | 60–70% activation savings | `model.gradient_checkpointing_enable()` |
| **Micro-Batch Size** | Limits peak instantaneous tensor size | `per_device_batch_size = 2` (or 4 at 160px) |
| **Gradient Accumulation** | Decouples memory from batch statistics | `accumulation_steps = 16` $\to$ Effective batch = 32 |
| **PyTorch 2.0 SDPA** | $O(N)$ flash memory attention | `torch.nn.functional.scaled_dot_product_attention` |

### 6.2 Hyperparameter Specification (VideoMAE ViT-B)
* **Backbone**: `MCG-NJU/videomae-base-finetuned-kinetics` (Hugging Face / PyTorch).
* **Input Resolution**: $16 \text{ frames} \times 224 \times 224$ (sample stride $\tau = 2$ or 4).
* **Optimizer**: `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0.05).
* **Base Learning Rate**: $1 \times 10^{-3}$ with linear scaling:
  $$\text{Effective LR} = \frac{\text{Effective Batch Size}}{256} \times \text{Base LR}$$
* **Layer-wise LR Decay**: $0.75$ (lower layers adapt slower than higher classification layers).
* **Learning Rate Schedule**: Cosine annealing with 5 epochs linear warmup.
* **Regularization / Augmentation**:
  * Random resized crop, horizontal flip ($p=0.5$).
  * AutoAugment / RandAugment for video.
  * Label smoothing ($\epsilon = 0.1$).
  * Mixup ($\alpha = 0.8$) and CutMix ($\alpha = 1.0$).

---

## 7. Phase 3: Baseline Hierarchy (Weeks 2–3)

To ensure academic rigor for the thesis, VideoMAE must be benchmarked against standard architectures:

```
                            [Evaluation Benchmark Hierarchy]
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
[1. Frozen-Feature Baseline]    [2. Efficient 3D CNNs]             [3. Video Transformers]
- VideoMAE ViT-B (Frozen)       - X3D-Small (PyTorchVideo)         - VideoMAE ViT-B (End-to-End)
- Train MLP Head on Features    - SlowFast R50 (PyTorchVideo)      - Video Swin-Transformer (Swin-T)
- Runtime: ~10 minutes          - Runtime: ~2 hours / 30 epochs    - Runtime: ~5-7 hours / 30 epochs
- Purpose: Representation floor - Purpose: Real-time efficiency    - Purpose: SOTA action modeling
```

1. **Frozen VideoMAE Feature Baseline**:
   * Pass all training clips through the frozen pre-trained VideoMAE backbone.
   * Extract pooled 768-d embeddings.
   * Train a 2-layer MLP classifier on Kaggle/local GPU.
   * *Purpose*: Fast lower bound benchmark completed in $<15$ minutes.
2. **X3D-S / X3D-M**:
   * State-of-the-art lightweight CNN for action recognition.
   * Extremely low FLOPs; acts as the efficiency benchmark.
3. **SlowFast (ResNet-50)**:
   * Two-stream 3D CNN (Slow path for spatial semantics, Fast path for temporal dynamics).
   * Gold-standard CNN baseline.
4. **Video Swin-T**:
   * Shifted-window spatio-temporal self-attention.
   * Demonstrates performance difference between hierarchical (Swin) and non-hierarchical (ViT) video transformers.

---

## 8. Phase 4: Main VideoMAE Fine-Tuning Runs (Weeks 3–5)

### 8.1 Compute Allocation & Execution Plan
* **Dataset Scale**: ~6,400 training clips across 82 categories.
* **Epoch Duration on Kaggle T4**:
  * With 16 frames @ 224px and pre-extracted/transcoded videos: ~12–15 minutes / epoch.
  * 30 epochs = ~6.5 to 7.5 GPU hours.
* **Execution Strategy**:
  * Run 1 (Epochs 1–15): Session 1 (~3.5 hours) $\to$ Save `checkpoint_epoch_15.pth`.
  * Run 2 (Epochs 16–30): Session 2 (~3.5 hours) $\to$ Auto-resume from epoch 15 to completion.
* **Multi-GPU Acceleration**:
  * Utilize Kaggle's dual T4 setup with PyTorch Distributed Data Parallel (`torch.distributed` / `torchrun`).
  * 2 GPUs cut epoch duration in half (~7 minutes / epoch).

### 8.2 Quota Conservation Rules
* Perform hyperparameter exploration (learning rate tuning, weight decay sweeps) on a **25% data subset** or at **8 frames @ 160px**.
* Reserve full 30-epoch runs exclusively for the primary model configuration.
* Run 3 random seeds only on the final winning configuration to report mean $\pm$ standard deviation for thesis tables.

---

## 9. Phase 5: Ablation Studies (Weeks 5–6)

All thesis contributions require controlled ablation experiments:

| Experiment ID | Ablation Dimension | Configurations | Scientific Question | GPU Quota Budget |
| :--- | :--- | :--- | :--- | :--- |
| **ABL-1** | **Temporal Frame Count** | 8 frames vs. 16 frames | How does temporal sampling density affect fine-grained sports recognition? | ~4 hours |
| **ABL-2** | **Spatial Resolution** | 160×160 vs. 224×224 | What is the accuracy vs. compute tradeoff between low and high resolution? | ~3 hours |
| **ABL-3** | **Fine-Tuning Depth** | (1) Linear Probe (Frozen) <br>(2) Top-3 Blocks Only <br>(3) Full End-to-End | Does pre-trained Kinetics VideoMAE require full adaptation for sports? | ~4 hours |
| **ABL-4** | **Data Scaling Curve** | 10%, 25%, 50%, 100% data | How sample-efficient is self-supervised VideoMAE on small sports domains? | ~5 hours |
| **ABL-5 (Bonus)**| **Pose Hybridization** | VideoMAE + RTMPose/PoseC3D | Does explicit skeletal topology provide complementary cues to self-attention? | Optional / If quota permits |

---

## 10. Phase 6: Evaluation, Visualization & Thesis Writing (Weeks 7–9)

### 10.1 Quantitative Benchmarking
* Metrics: Top-1 Accuracy, Top-5 Accuracy, Macro F1-score, Mean Per-Class Accuracy.
* Statistical significance: Paired t-test across 3 seeds versus CNN baselines (X3D, SlowFast).

### 10.2 Error Taxonomy & Qualitative Analysis
* **Confusion Matrix**: High-resolution 82-class confusion matrix identifying clusters of sports actions with visual similarities (e.g., `BaseballPitch` vs. `CricketBowling`, or `TennisSwing` vs. `TableTennisShot`).
* **Failure Analysis**: Systematic review of misclassified clips grouped by:
  * Camera distance (close-up vs. aerial view).
  * Equipment occlusion.
  * Motion blur and rapid camera panning.

### 10.3 Interpretability & Visualizations (Cheap Inference)
* **Spatio-Temporal Attention Rollout**: Visualize what spatial patches and temporal frames the self-attention heads focus on during peak action moments.
* **Grad-CAM on VideoMAE**: Spatio-temporal heatmaps showing discriminative regions (e.g., ball trajectory, racket contact, gymnastics apparatus).

---

## 11. Repository Layout

```text
D:\AKHI THESIS UP\
├── Archive/                        # Complete archive of previous HOI-BiLSTM work
│   ├── checkpoints/                # Archived M7 checkpoints
│   ├── configs/                    # Archived YAML configurations
│   ├── data/                       # Archived feature tensors & old splits
│   ├── diagrams/                   # Archived confusion matrices & loss curves
│   ├── notebooks/                  # Archived HOI BiLSTM notebooks
│   ├── src/                        # Archived HOI extractor & BiLSTM code
│   ├── Model Data/                 # 9,151 RAW SPORTS VIDEO CLIPS (Preserved)
│   ├── ucf action/                 # Archived legacy UCF Sports dataset
│   ├── README_HOI_BiLSTM.md        # Archived previous README
│   └── RUN_GUIDE_HOI_BiLSTM.md     # Archived previous execution guide
├── configs/                        # New configuration files
│   ├── videomae_base.yaml          # VideoMAE ViT-B configuration
│   ├── x3d_baseline.yaml           # X3D baseline configuration
│   └── ablation_configs.yaml       # Ablation sweep configurations
├── data/
│   ├── splits/                     # Group-wise leak-free CSV splits (train/val/test)
│   └── processed/                  # Transcoded video clips (224px / 160px)
├── notebooks/
│   ├── 01_kaggle_videomae_train.ipynb   # Kaggle primary training notebook (Auto-resume)
│   ├── 02_colab_videomae_backup.ipynb  # Colab backup notebook
│   └── 03_local_subset_debug.ipynb     # Local 4GB GPU smoke test notebook
├── scripts/
│   ├── 01_audit_dataset.py         # Local data audit & integrity verification
│   ├── 02_create_group_splits.py   # Group-wise 70/15/15 CSV partitioner
│   ├── 03_transcode_videos.py      # Multi-threaded 224/160px offline transcoder
│   └── 04_package_kaggle_shards.py # Zip sharding for Kaggle dataset upload
├── src/
│   ├── models/                     # VideoMAE, X3D, and Baseline model wrappers
│   ├── datasets/                   # PyTorch VideoDataset with Decord/PyAV
│   ├── engine/                     # Trainer, Auto-Resume, AMP Scaler, Evaluator
│   └── utils/                      # Metrics, Logging, and Attention Visualizers
├── README.md                       # Master project overview
├── WORKING_PLAN.md                 # This living specification document
└── requirements.txt                # Lightweight requirements for local & cloud
```

---

## 12. Immediate Next Actions

1. **Step 1 (Local)**: Run `scripts/01_audit_dataset.py` on `Archive/Model Data` to produce the full dataset audit report (class counts, video counts, resolution distribution, corrupt file check).
2. **Step 2 (Local)**: Execute `scripts/02_create_group_splits.py` to generate the strict group-wise 70/15/15 train/val/test CSV splits.
3. **Step 3 (Local)**: Execute `scripts/03_transcode_videos.py` to downscale clips to 224px (and 160px) at 25 FPS using multi-threaded ffmpeg.
4. **Step 4 (Cloud Setup)**: Package into shards and upload as a private Kaggle Dataset.
5. **Step 5 (Kaggle)**: Launch `notebooks/01_kaggle_videomae_train.ipynb` for the 200-clip Phase 1 smoke test.
