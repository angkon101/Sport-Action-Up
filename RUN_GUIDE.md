# 🚀 Execution & Operational Run Guide: Sports Action VideoMAE

This operational handbook provides step-by-step instructions for reproducing, training, and evaluating the **82-Class Sports Action Recognition framework** across **Kaggle**, **Google Colab**, and **Local Workstations**.

---

## 📑 Guide Sections
1. [Prerequisites & Dataset Preparation](#1-prerequisites--dataset-preparation)
2. [Track A: Running on Kaggle (Primary — 2× T4 DDP)](#2-track-a-running-on-kaggle-primary--2-t4-ddp)
3. [Track B: Running on Google Colab (Backup — 1× T4 Auto-Resume)](#3-track-b-running-on-google-colab-backup--1-t4-auto-resume)
4. [Track C: Local Machine Workflow (Audit & Sub-Batch Smoke Test)](#4-track-c-local-machine-workflow-audit--sub-batch-smoke-test)
5. [Monitoring, Telemetry & Training Curves](#5-monitoring-telemetry--training-curves)
6. [Evaluation & Metric Generation](#6-evaluation--metric-generation)
7. [Inference on Custom Videos](#7-inference-on-custom-videos)

---

## 1. Prerequisites & Dataset Preparation

The dataset comprises **9,151 video clips across 82 categories**. A pre-packaged, ready-to-upload zip archive is located in your workspace:
* **Local Path**: `D:\AKHI THESIS UP\Archive\Model Data.zip` (~4.48 GB).

### Preparing the Dataset on Cloud Platforms:

#### For Kaggle:
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click **"New Dataset"**.
2. Drag and drop `Model Data.zip` (or upload via Kaggle CLI: `kaggle datasets create -p ...`).
3. Title it `sport-action-up` (or any title).
4. Once created, Kaggle automatically extracts the videos into a high-speed read-only NVMe mount at `/kaggle/input/<dataset-name>/`.

#### For Google Colab:
1. Upload `Model Data.zip` directly to your Google Drive root directory:
   `Google Drive -> MyDrive / Model Data.zip`
2. Ensure at least **6 GB of free storage** remains on your Google Drive for checkpoints and unzipped data.

---

## 2. Track A: Running on Kaggle (Primary — 2× T4 DDP)

Kaggle is the primary training platform because it provides **two NVIDIA T4 GPUs (32 GB total VRAM)** with PyTorch Distributed Data Parallel (DDP).

### Step-by-Step Instructions:

1. **Create or Open Notebook on Kaggle**:
   * Create a new Notebook on Kaggle or import [`notebooks/01_Kaggle_Sports_Action_VideoMAE.ipynb`](notebooks/01_Kaggle_Sports_Action_VideoMAE.ipynb).
2. **Enable Dual T4 GPUs**:
   * On the right-hand settings panel:
     * **Accelerator**: Select **GPU T4 x2**.
     * **Internet**: Turn **ON**.
3. **Attach Your Dataset**:
   * On the right panel under **Input**, click **`+ Add Input`**.
   * Go to **"Your Datasets"** and attach your uploaded sports dataset.
4. **Execute Initialization Cells**:
   ```python
   # Cell 1: Verify 2x T4 GPUs
   !nvidia-smi

   # Cell 2: Install fast decoders & transformers
   !pip install -q decord av transformers accelerate timm pandas pyyaml

   # Cell 3: Clone fresh repository cleanly
   %cd /kaggle/working
   !rm -rf /kaggle/working/Sport-Action-Up
   !git clone https://github.com/angkon101/Sport-Action-Up.git
   %cd /kaggle/working/Sport-Action-Up
   ```
5. **Verify Dataset Mounting**:
   ```python
   # Cell 4: Verify clips are detected
   from pathlib import Path
   clips = list(Path('/kaggle/input').rglob('*.avi')) + list(Path('/kaggle/input').rglob('*.mp4'))
   print(f"✓ Total videos detected: {len(clips):,}")
   ```
6. **Launch Multi-GPU DDP Training**:
   ```python
   # Cell 5A: Launch across both T4 GPUs (Effective Batch Size = 32)
   !torchrun --nproc_per_node=2 train_cloud.py --config configs/videomae_kaggle_dual_t4.yaml
   ```
7. **Session Timeout Recovery**:
   * If Kaggle disconnects or hits session limits, simply re-run Cell 5A.
   * `train_cloud.py` automatically checks `/kaggle/working/checkpoints/last_checkpoint.pth` and resumes from the exact epoch.

---

## 3. Track B: Running on Google Colab (Backup — 1× T4 Auto-Resume)

Google Colab provides a free single **NVIDIA T4 (16 GB)**. Google Drive is used as persistent storage.

### Step-by-Step Instructions:

1. **Open Colab**:
   * Navigate to [Google Colab](https://colab.research.google.com/) $\to$ Upload [`notebooks/02_Colab_Sports_Action_VideoMAE.ipynb`](notebooks/02_Colab_Sports_Action_VideoMAE.ipynb).
2. **Set GPU**:
   * Select **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
3. **Mount Drive & Clone Code**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   %cd /content
   !rm -rf /content/Sport-Action-Up
   !git clone https://github.com/angkon101/Sport-Action-Up.git
   %cd /content/Sport-Action-Up
   ```
4. **Unpack Dataset to Fast Colab Local Disk**:
   ```python
   # Unzip to local NVMe SSD (/content/dataset) to avoid slow Drive API reads
   !unzip -q "/content/drive/MyDrive/Model Data.zip" -d /content/dataset
   ```
5. **Launch Training**:
   ```python
   !python train_cloud.py --config configs/videomae_colab_t4.yaml --data-dir /content/dataset
   ```
6. **Auto-Resume Protection**:
   * Checkpoints save continuously to:
     `/content/drive/MyDrive/SportsAction_Checkpoints/last_checkpoint.pth`
   * If Colab disconnects, restart the runtime, re-mount Drive, and re-execute the training command. It picks up automatically!

---

## 4. Track C: Local Machine Workflow (Audit & Sub-Batch Smoke Test)

Your local machine handles validation, data auditing, and pipeline testing:

### 1. Run Data Integrity Audit
```powershell
python scripts/01_audit_dataset.py --data-dir "Archive/Model Data" --output-dir "data/audit"
```
* Generates `dataset_audit_all_clips.csv` and `audit_summary.json`.
* Flags zero-byte files, unreadable headers, and class distribution histograms.

### 2. Generate Leakage-Free Group Splits
```powershell
python scripts/02_create_group_splits.py --data-dir "Archive/Model Data" --output-dir "data/splits" --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```
* Generates `train.csv`, `val.csv`, and `test.csv` with zero group leakage.

### 3. Quick 2-Epoch Local Smoke Test (Debugging)
```powershell
python train_cloud.py --config configs/videomae_base.yaml --data-dir "Archive/Model Data" --epochs 2 --micro-batch 1
```

---

## 5. Monitoring, Telemetry & Training Curves

### Viewing Curves in Notebooks
After (or during) training, plot real-time curves directly in Kaggle/Colab:

```python
import pandas as pd
import matplotlib.pyplot as plt

log_file = "/kaggle/working/training_history.csv" # or /content/drive/MyDrive/SportsAction_Checkpoints/training_history.csv
df = pd.read_csv(log_file)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:blue', lw=2)
ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:orange', lw=2)
ax1.set_title("Cross-Entropy Loss vs. Epoch", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Accuracy
ax2.plot(df['epoch'], df['train_top1'], label='Train Top-1 Acc', color='tab:green', lw=2)
ax2.plot(df['epoch'], df['val_top1'], label='Val Top-1 Acc', color='tab:red', lw=2)
ax2.set_title("Top-1 Accuracy (%) vs. Epoch", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("convergence_curves.png", dpi=300)
plt.show()
```

---

## 6. Evaluation & Metric Generation

### 1. Confusion Matrix Generation (82 Classes)
Evaluate the best checkpoint on the held-out test split:
```python
import torch
from src.models.videomae import SportsVideoMAE
from src.datasets.video_dataset import VideoClipsDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoints/videomae/best_checkpoint.pth", map_location=device)

model = SportsVideoMAE(num_classes=82).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load test dataset
df_test = pd.read_csv("data/splits/test.csv")
test_dataset = VideoClipsDataset(df_test, num_frames=16, is_training=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for pixel_values, labels, _ in test_loader:
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)
        preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(all_labels, all_preds, digits=4))
```

---

## 7. Inference on Custom Videos

To classify any arbitrary sports video clip:

```python
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F
from src.models.videomae import SportsVideoMAE

def predict_video(video_path: str, checkpoint_path: str, num_frames: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SportsVideoMAE(num_classes=82).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Read 16 frames uniformly
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            tensor_f = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            # Normalize ImageNet
            tensor_f = F.normalize(tensor_f, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            frames.append(tensor_f)
        idx += 1
    cap.release()

    video_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(device) # (1, 16, 3, 224, 224)
    
    with torch.no_grad():
        logits = model(pixel_values=video_tensor)["logits"]
        probs = torch.softmax(logits, dim=-1)
        top3_prob, top3_idx = torch.topk(probs, 3)

    return top3_prob.cpu().numpy()[0], top3_idx.cpu().numpy()[0]
```
