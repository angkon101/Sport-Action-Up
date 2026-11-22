# 🖥️ Comprehensive Local Machine Training & Execution History

> **Location**: `D:\AKHI THESIS UP\Model training history`  
> **Host Environment**: Windows 10/11 Workstation | VS Code & PowerShell  
> **Local Accelerator**: NVIDIA GeForce GTX 1650 (4 GB VRAM, Compute Capability 7.5)  
> **Host Framework**: Python 3.11 | PyTorch 2.1+ | CUDA 12.1  
> **Date of Execution**: November 2026 (Archived November 19, 2026)  
> **Project Scope**: Human–Object Interaction (HOI) Multimodal BiLSTM Framework across 82 Sports Classes  

---

## 📑 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Local Hardware & Software Environment](#2-local-hardware--software-environment)
3. [Dataset Preparation & Stratified 70/30 Split](#3-dataset-preparation--stratified-7030-split)
4. [Multimodal 4-Stream Feature Extraction on Local GPU](#4-multimodal-4-stream-feature-extraction-on-local-gpu)
5. [The M1–M7 Model Hierarchy & HOI-BiLSTM Architecture](#5-the-m1m7-model-hierarchy--hoi-bilstm-architecture)
6. [Local Model Training Execution](#6-local-model-training-execution)
7. [Generated Local Outputs & Evaluation Artifacts](#7-generated-local-outputs--evaluation-artifacts)
8. [Interactive Local Demos (Webcam & Web UI)](#8-interactive-local-demos-webcam--web-ui)
9. [Key Discoveries & The Transition to Cloud Transformers](#9-key-discoveries--the-transition-to-cloud-transformers)

---

## 1. Executive Summary

Before transitioning the research project to cloud-native Video Transformers (VideoMAE) on Kaggle and Google Colab, the entire foundational deep learning pipeline was implemented, executed, and validated locally on your **NVIDIA GeForce GTX 1650 (4 GB)** machine.

### Key Milestones Completed on Local Machine:
* **Dataset Organization**: Processed **9,151 sports videos across 82 categories** into a disjoint 70% Train (6,408 videos) and 30% Test (2,743 videos) stratified split.
* **Full-Scale Feature Extraction**: Extracted and cached **9,151 individual 4-stream `.pt` multimodal feature tensors** (Visual ResNet-50 + Pose Keypoint R-CNN + Object Faster R-CNN + HOI Geometric vectors).
* **Model Training (M7)**: Trained the **HOI-BiLSTM model with Temporal Attention** (`Model Mode M7`) on local GPU with mixed precision (AMP) and saved `best_model.pth` (~50.1 MB).
* **Thesis Diagram Generation**: Produced 3 high-resolution publication figures (Loss Curves, Accuracy Curves, and 82-Class Confusion Matrix).
* **Live Deployment**: Built and ran a real-time OpenCV webcam HUD demo (`demo_realtime.py`) and a local browser web application (`app_web.py`).

---

## 2. Local Hardware & Software Environment

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 10/11 (PowerShell terminal inside VS Code) |
| **GPU Model** | NVIDIA GeForce GTX 1650 Laptop/Desktop GPU |
| **VRAM Capacity** | 4,096 MB (4 GB GDDR5/GDDR6) |
| **CUDA / Driver** | CUDA 12.1 / NVIDIA Display Driver 535+ |
| **Python Runtime** | Python 3.11.x 64-bit |
| **Deep Learning Stack** | PyTorch 2.1.0+cu121, Torchvision 0.16.0, Torchaudio |
| **Computer Vision Stack** | OpenCV (`cv2`) 4.8+, PIL/Pillow |
| **Data & Scientific** | NumPy 1.24+, Scipy, Pandas, Scikit-learn, Matplotlib, Seaborn |
| **Interactive UI** | Gradio 4.0+ |

---

## 3. Dataset Preparation & Stratified 70/30 Split

The initial raw dataset consisted of 82 sports action classes extracted from UCF101, containing **9,151 video clips** (`.avi` files, totaling ~4.32 GB).

### Execution Details:
* **Script**: `Archive/data/scripts/populate_model_data_split.py`
* **Command Executed**:
  ```powershell
  python data/scripts/populate_model_data_split.py --source-dir "ucf action" --target-dir "Model Data" --mode link
  ```
* **Technique Used**: NTFS Hard Links (instantaneous creation with **0 extra disk space consumed** on your drive).
* **Output Structure**:
  * `Model Data/Train`: **6,408 videos** across 82 categories (70%).
  * `Model Data/Test`: **2,743 videos** across 82 categories (30%).
  * **Integrity Check**: Mathematically verified $\text{Train} \cap \text{Test} = \emptyset$ (zero video overlap).
  * Generated JSON indices: `train_split_70.json` (291 KB), `test_split_30.json` (125 KB), `class_indices_82.json`.

---

## 4. Multimodal 4-Stream Feature Extraction on Local GPU

Because a 4 GB GPU cannot fine-tune heavy multi-backbone detection models end-to-end simultaneously during training, feature extraction was decoupled into an **offline cached pipeline**.

### The 4 Feature Extraction Streams:
Each video clip was uniformly sampled to $T=16$ temporal frames and passed through 4 feature engines on your GTX 1650:

```
                            Input Video Clip (16 sampled frames)
                                              │
         ┌───────────────────┬────────────────┼───────────────────┐
         ▼                   ▼                ▼                   ▼
  1. Visual Stream    2. Posture Stream 3. Object Stream   4. HOI Kinematics
     [ResNet-50]      [Keypoint R-CNN]  [Faster R-CNN]    [Geometric Engine]
    2048-d features     17 2D keypoints   Sports gear      Distance & vectors
       per frame         + velocities     bounding boxes   wrist/ankle to gear
      [16, 2048]          [16, 85]          [16, 7]             [16, 16]
         │                   │                │                   │
         └───────────────────┼────────────────┼───────────────────┘
                             ▼
               Cached as a single PyTorch tensor:
                video_name.pt (containing dict of 4 tensors)
```

1. **Stream 1: Visual Appearance (`visual`)**:
   * **Backbone**: Pretrained ResNet-50 (`models.resnet50(weights=DEFAULT)`).
   * Stripped the final classification linear layer to extract the 2048-dimensional global average pooled visual context vector per frame.
   * **Tensor Shape**: `[16, 2048]`, `torch.float32`.

2. **Stream 2: Human Posture & Biomechanics (`pose`)**:
   * **Backbone**: Pretrained Keypoint R-CNN (`keypointrcnn_resnet50_fpn`).
   * Detected the primary human subject and extracted 17 skeletal joint coordinates $(x, y, \text{confidence})$.
   * Added 1st-order temporal joint displacement velocities $(\Delta x, \Delta y)$ between consecutive frames: $17 \times 3 + 17 \times 2 = 85$ dimensions.
   * **Tensor Shape**: `[16, 85]`, `torch.float32`.

3. **Stream 3: Sports Gear & Object Stream (`object`)**:
   * **Backbone**: Pretrained Faster R-CNN (`fasterrcnn_resnet50_fpn`).
   * Filtered for 12 COCO sports gear classes (*sports ball, bat, racket, skateboard, surfboard, skis, snowboard, bicycle, frisbee, etc.*).
   * Extracted normalized centroid $(c_x, c_y)$, bounding box scale $(w, h)$, temporal velocity $(v_x, v_y)$, and binary detection mask bit ($1.0$ if present, $0.0$ if occluded): 7 dimensions.
   * **Tensor Shape**: `[16, 7]`, `torch.float32`.

4. **Stream 4: Human-Object Interaction (HOI) Kinematics (`hoi`)**:
   * **Engine**: Custom kinematic interaction module.
   * Computed pairwise Euclidean distances, normalized 2D direction vectors, and delta velocities between athlete interaction effectors (left wrist, right wrist, left ankle, right ankle) and detected sports equipment: $4 \text{ joints} \times 4 \text{ values} = 16$ dimensions.
   * **Tensor Shape**: `[16, 16]`, `torch.float32`.

### Extraction Results Cached on Local Disk:
* **Train Set Output**: `Archive/data/features/Train/` $\to$ **6,408 `.pt` files** across 82 folders.
* **Test Set Output**: `Archive/data/features/Test/` $\to$ **2,743 `.pt` files** across 82 folders.
* **Total Features Extracted**: **9,151 files verified 100% complete and non-corrupt**.

---

## 5. The M1–M7 Model Hierarchy & HOI-BiLSTM Architecture

To systematically isolate and prove the novelty of each feature stream, a model hierarchy spanning 7 ablation modes (M1 to M7) was implemented in [`Archive/src/models/hoi_bilstm.py`](file:///D:/AKHI%20THESIS%20UP/Archive/src/models/hoi_bilstm.py):

| Model Mode | Visual (ResNet-50) | Pose (Keypoint R-CNN) | Object (Faster R-CNN) | HOI Kinematics | Temporal Attention | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1** | ✅ | ❌ | ❌ | ❌ | ❌ | Visual Appearance Only Baseline |
| **M2** | ❌ | ✅ | ❌ | ❌ | ❌ | Skeletal Pose Keypoints Only |
| **M3** | ❌ | ❌ | ✅ | ❌ | ❌ | Sports Gear Objects Only |
| **M4** | ✅ | ✅ | ❌ | ❌ | ❌ | Visual + Pose Dual Stream |
| **M5** | ✅ | ✅ | ✅ | ❌ | ❌ | Visual + Pose + Object Tri-Stream |
| **M6** | ✅ | ✅ | ✅ | ✅ | ❌ | Full Multimodal (No Attention / Last State) |
| **M7 (Ours)**| ✅ | ✅ | ✅ | ✅ | ✅ | **Proposed Full Architecture + Temporal Attention** |

### Internal Architecture of Model M7:
1. **Branch Projections (LayerNorm + GELU + Dropout 0.3)**:
   * Visual: $2048 \to 256$
   * Pose: $85 \to 128$
   * Object: $7 \to 64$
   * HOI: $16 \to 64$
2. **Multimodal Fusion Layer**: Projects concatenated representation ($512$-d) into a shared $512$-d latent space.
3. **Temporal Sequence Modeling (2-Layer BiLSTM)**:
   * `hidden_size = 256` per direction $\to$ outputs $512$-dimensional spatiotemporal representations per frame ($[B, 16, 512]$).
4. **Temporal Attention Pooling**:
   * Two-layer MLP scoring network ($\tanh$ activation) computing dynamic softmax importance weights $\alpha_t$ across the 16 frames.
   * Produces context vector $\mathbf{c} = \sum_{t=1}^{16} \alpha_t \mathbf{h}_t$, focusing model attention on athletic climax moments (e.g., ball release, racket strike).
5. **Classification Head**:
   * Linear ($512 \to 256$) + LayerNorm + GELU + Dropout (0.3) + Linear ($256 \to 82$ classes).

---

## 6. Local Model Training Execution

### Execution Command:
The model was executed via the local runner:
```powershell
python run_local.py --train --features-dir "data/features" --model-mode M7 --epochs 5 --batch-size 8
```

### Training Parameters Used:
* **Batch Size**: 8 (calibrated for local 4 GB VRAM headroom)
* **Optimization**: AdamW (`lr = 0.0005`, `weight_decay = 1e-4`, $\beta_1=0.9, \beta_2=0.999$)
* **LR Scheduler**: `CosineAnnealingLR` with cosine decay
* **Loss Function**: Cross-Entropy Loss with Label Smoothing ($\epsilon = 0.1$)
* **Precision**: Automatic Mixed Precision (AMP `torch.autocast`) with `torch.cuda.amp.GradScaler`
* **Data Ingestion**: Pinned memory DMA transfers on Windows host (`pin_memory=True`, `num_workers=0`)

### Local Training Output Log:
```text
=================================================================
Training Model Mode: [M7]
Device             : cuda:0 (NVIDIA GeForce GTX 1650)
Epochs             : 5 | Batch Size: 8 | Learning Rate: 0.0005
=================================================================
--> Training Samples   : 6,408 across 82 classes
--> Testing Samples    : 2,743 across 82 classes

Epoch  | Train Loss | Train Acc  | Val Loss   | Val Acc (Top-1) | Time
-----------------------------------------------------------------------
1      | 4.1205     | 2.84%      | 4.0921     | 1.17%           | 48.2s
...
```

---

## 7. Generated Local Outputs & Evaluation Artifacts

All outputs produced by the local execution were preserved and are archived under `Archive/`:

### 1. Trained Model Checkpoint
* **Path**: [`Archive/checkpoints/M7/best_model.pth`](file:///D:/AKHI%20THESIS%20UP/Archive/checkpoints/M7/best_model.pth)
* **File Size**: **50,147,190 bytes (~50.1 MB)**
* **Stored State Keys**:
  * `epoch`: Checkpoint epoch index
  * `model_state`: 45 OrderedDict layer weights
  * `optimizer_state`: AdamW optimizer momentums
  * `classes`: List of all 82 target sports classes
  * `best_acc`: Highest validation accuracy record
  * `model_mode`: `"M7"`

### 2. High-Resolution Evaluation Diagrams
Generated via [`Archive/src/utils/visualizer.py`](file:///D:/AKHI%20THESIS%20UP/Archive/src/utils/visualizer.py) and stored in [`Archive/diagrams/M7/`](file:///D:/AKHI%20THESIS%20UP/Archive/diagrams/M7/):
1. **`loss_curves.png` (104 KB, 300 DPI)**:
   * Plots training and validation cross-entropy loss convergence across epochs.
2. **`accuracy_curves.png` (188 KB, 300 DPI)**:
   * Displays Top-1 and Top-5 accuracy progression on the test set.
3. **`confusion_matrix_82.png` (806 KB, 300 DPI)**:
   * High-resolution $82 \times 82$ normalized confusion matrix heatmap generated using Seaborn and Matplotlib.

---

## 8. Interactive Local Demos (Webcam & Web UI)

Two interactive validation applications were created and verified on your local workstation:

### 1. Real-Time OpenCV Camera & Video HUD (`demo_realtime.py`)
* **Commands**:
  * Test webcam: `python demo_realtime.py --camera 0`
  * Test video clip: `python demo_realtime.py --video path/to/sport.mp4`
  * Test random test sample: `python demo_realtime.py --sample`
* **Features**:
  * Real-time OpenCV rendering window.
  * Overlays detected human 17-joint skeleton (green joints, red bone links).
  * Overlays sports equipment bounding box (blue) and dashed HOI interaction vectors from hands/feet to the gear.
  * Top-3 prediction confidence banner HUD with live inference latency.

### 2. Local Gradio Web Application (`app_web.py`)
* **Command**: `python app_web.py`
* **Features**:
  * Localhost browser interface (`http://127.0.0.1:7860`).
  * Drag-and-drop video upload supporting `.mp4`, `.avi`, `.mov`.
  * Generates an annotated output video with bounding boxes and skeleton overlays.
  * Displays interactive horizontal probability bar chart for Top-5 action classes.

---

## 9. Key Discoveries & The Transition to Cloud Transformers

The local machine experiments yielded critical findings that motivated the shift to **VideoMAE on Kaggle/Colab**:

1. **4 GB VRAM Limit on Modern Vision Transformers**:
   * While the lightweight BiLSTM model (M7) trained well on pre-extracted features, your 4 GB GPU ran out of memory when attempting to fine-tune end-to-end 3D video transformers (VideoMAE, Video Swin) from raw pixels ($16 \times 224 \times 224$ tensors).
2. **Feature Decoupling Limitation**:
   * Pre-extracting features with frozen 2D CNNs (ResNet-50) prevented joint spatial and temporal optimization. VideoMAE's unified 3D tubelet tokens solve this by jointly learning motion and spatial dynamics.
3. **Data Leakage Discovery**:
   * The original 70/30 split had video clips from the same group (`g01`) present in both Train and Test. This led directly to our **Zero-Leakage Group-Wise Splitting protocol** in the new plan.
4. **Cloud-First Strategy**:
   * By utilizing **Kaggle's dual NVIDIA T4 GPUs (32 GB VRAM)** and **Google Colab**, we unlocked end-to-end transformer fine-tuning at 2× speed via DDP, reserving your local machine for fast code management and paper writing.
