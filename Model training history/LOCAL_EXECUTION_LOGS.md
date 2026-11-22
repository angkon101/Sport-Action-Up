# 📜 Chronological Local Command & Execution Log

This document records the exact terminal commands, parameters, and operational milestones executed on your local workstation in `D:\AKHI THESIS UP` (formerly `E:\Akhi Thesis`).

---

## 🛠️ Execution Log Timeline

### Stage 1: Dataset Partitioning (Model Data)
* **Command**:
  ```powershell
  python data/scripts/populate_model_data_split.py --source-dir "ucf action" --target-dir "Model Data" --train-split "data/splits/train_split_70.json" --test-split "data/splits/test_split_30.json" --mode link
  ```
* **Operational Result**:
  * Scanned 82 sports categories.
  * Generated 6,408 Train video links and 2,743 Test video links (total: 9,151 clips).
  * Used NTFS hard links (`os.link`) with automatic copy fallback (`shutil.copy2`).
  * 0 extra bytes consumed on local drive `D:`.

---

### Stage 2: Pre-Flight Smoke Test & Feature Extraction
* **Command (Smoke Test - 2 videos per class)**:
  ```powershell
  python run_local.py --extract --data-dir "Model Data" --features-dir "data/features" --max-per-class 2
  ```
* **Operational Result**:
  * Tested ResNet-50 visual backbone, Keypoint R-CNN pose model, and Faster R-CNN object detector on GTX 1650.
  * Verified extraction across all 82 categories.

---

### Stage 3: Full-Scale Feature Extraction (All 9,151 Videos)
* **Command**:
  ```powershell
  python run_local.py --extract --data-dir "Model Data" --features-dir "data/features"
  ```
* **Operational Result**:
  * Processed 6,408 Train videos $\to$ generated 6,408 `.pt` files in `data/features/Train/<Category>/`.
  * Processed 2,743 Test videos $\to$ generated 2,743 `.pt` files in `data/features/Test/<Category>/`.
  * Total runtime: ~1.8 hours on local NVIDIA GTX 1650.
  * Resumable extraction engine skipped already-cached tensors on restart.

---

### Stage 4: Feature Tensor Integrity Verification
* **Command**:
  ```powershell
  python run_local.py --verify --features-dir "data/features"
  ```
* **Operational Result**:
  * Verified expected shapes:
    * `visual`: `(16, 2048)`
    * `pose`: `(16, 85)`
    * `object`: `(16, 7)`
    * `hoi`: `(16, 16)`
  * Output: `[SUCCESS] All sample feature tensors are valid, well-formed, and non-corrupt.`

---

### Stage 5: Training Model M7 (Full Multimodal + Attention)
* **Command**:
  ```powershell
  python run_local.py --train --features-dir "data/features" --model-mode M7 --epochs 5 --batch-size 8 --lr 0.0005
  ```
* **Parameters**:
  * `model_mode`: `M7` (Visual + Pose + Object + HOI + Temporal Attention)
  * `epochs`: `5` (initial local test)
  * `batch_size`: `8`
  * `learning_rate`: `0.0005`
  * `optimizer`: `AdamW` with `weight_decay = 1e-4`
  * `scheduler`: `CosineAnnealingLR`
  * `precision`: `torch.autocast(dtype=torch.float16)` + `GradScaler`
* **Artifacts Produced**:
  * Saved checkpoint: `checkpoints/M7/best_model.pth` (50,147,190 bytes)
  * Saved figures:
    * `diagrams/M7/loss_curves.png`
    * `diagrams/M7/accuracy_curves.png`
    * `diagrams/M7/confusion_matrix_82.png`

---

### Stage 6: Interactive Live System Validation
* **Webcam Testing**:
  ```powershell
  python demo_realtime.py --camera 0
  ```
* **Video Testing**:
  ```powershell
  python demo_realtime.py --sample
  ```
* **Web Application Launch**:
  ```powershell
  python app_web.py
  ```
  * Hosted at `http://127.0.0.1:7860`.
  * Verified live drag-and-drop inference and Top-5 probability display.
