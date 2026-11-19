"""Generates master Kaggle and Google Colab Jupyter Notebooks."""

import json
from pathlib import Path


def make_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {filepath}")


kaggle_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏆 Fine-Grained Sports Action Recognition with VideoMAE ViT-B\n",
            "### Master Kaggle Pipeline (Optimized for 2x T4 GPUs with DDP)\n",
            "\n",
            "This notebook executes the complete end-to-end training and evaluation pipeline on Kaggle:\n",
            "- **Hardware**: 2x NVIDIA T4 (16 GB each) or 1x Tesla P100 (16 GB)\n",
            "- **Framework**: VideoMAE ViT-B (`MCG-NJU/videomae-base-finetuned-kinetics`)\n",
            "- **Acceleration**: PyTorch Distributed Data Parallel (DDP) via `torchrun`\n",
            "- **Precision**: fp16 Mixed Precision with GradScaler\n",
            "- **Reliability**: Atomic state auto-resume across session interruptions\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Environment & Hardware Verification\n",
            "!nvidia-smi\n",
            "import torch\n",
            "print('CUDA Available:', torch.cuda.is_available())\n",
            "print('Device Count  :', torch.cuda.device_count())\n",
            "for i in range(torch.cuda.device_count()):\n",
            "    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Install Fast Video Decoding & Transformer Dependencies\n",
            "!pip install -q decord av transformers accelerate timm pandas pyyaml\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Clone or Sync Latest Codebase from GitHub\n",
            "import os\n",
            "if not os.path.exists('Sport-Action-Up'):\n",
            "    !git clone https://github.com/angkon101/Sport-Action-Up.git\n",
            "    %cd Sport-Action-Up\n",
            "else:\n",
            "    %cd Sport-Action-Up\n",
            "    !git pull origin main\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Verify Dataset Location\n",
            "from pathlib import Path\n",
            "\n",
            "# Check mounted Kaggle inputs\n",
            "inputs = list(Path('/kaggle/input').glob('*'))\n",
            "print('Mounted inputs on Kaggle:', [p.name for p in inputs])\n",
            "\n",
            "# Find raw video clips\n",
            "sample_videos = list(Path('/kaggle/input').rglob('*.avi')) + list(Path('/kaggle/input').rglob('*.mp4'))\n",
            "print(f'Total video files detected: {len(sample_videos):,}')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🚀 5. Launch Training\n",
            "If you have **2x T4 GPUs** selected in Kaggle settings (*Accelerator -> GPU T4 x2*), run the cell below with `torchrun` to train with **Distributed Data Parallel (DDP)** at 2x speed!\n",
            "If using a single P100 or single T4, use cell 5B.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5A. Multi-GPU DDP Training (Dual T4 - Recommended)\n",
            "!torchrun --nproc_per_node=2 train_cloud.py --config configs/videomae_kaggle_dual_t4.yaml\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5B. Single-GPU Training (Fallback for Tesla P100 / Single T4)\n",
            "# !python train_cloud.py --config configs/videomae_kaggle_dual_t4.yaml --micro-batch 2\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Plot Loss & Accuracy Curves\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "log_path = Path('/kaggle/working/training_history.csv')\n",
            "if log_path.exists():\n",
            "    df = pd.read_csv(log_path)\n",
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
            "    \n",
            "    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:blue')\n",
            "    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:orange')\n",
            "    ax1.set_title('Cross-Entropy Loss')\n",
            "    ax1.set_xlabel('Epoch')\n",
            "    ax1.legend()\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    \n",
            "    ax2.plot(df['epoch'], df['train_top1'], label='Train Top-1', color='tab:green')\n",
            "    ax2.plot(df['epoch'], df['val_top1'], label='Val Top-1', color='tab:red')\n",
            "    ax2.set_title('Top-1 Accuracy (%)')\n",
            "    ax2.set_xlabel('Epoch')\n",
            "    ax2.legend()\n",
            "    ax2.grid(True, alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig('/kaggle/working/training_curves.png', dpi=300)\n",
            "    plt.show()\n",
            "else:\n",
            "    print('Log file not yet generated.')\n",
        ],
    },
]

colab_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚡ Fine-Grained Sports Action Recognition with VideoMAE ViT-B\n",
            "### Master Google Colab Pipeline (Persistent Google Drive Checkpointing)\n",
            "\n",
            "Optimized for Google Colab Free Tier (1x T4 16GB):\n",
            "- Automatically mounts Google Drive to preserve checkpoints across random disconnects\n",
            "- Fast NVMe local staging to avoid slow Drive I/O during training\n",
            "- Auto-resumes from `/content/drive/MyDrive/SportsAction_Checkpoints/last_checkpoint.pth`\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Mount Google Drive for Persistent Storage\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "import os\n",
            "os.makedirs('/content/drive/MyDrive/SportsAction_Checkpoints', exist_ok=True)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. GPU Check\n",
            "!nvidia-smi\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Install Video Transformers & Fast Video Decoding\n",
            "!pip install -q decord av transformers accelerate timm pandas pyyaml\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Clone Latest Repository from GitHub\n",
            "import os\n",
            "if not os.path.exists('Sport-Action-Up'):\n",
            "    !git clone https://github.com/angkon101/Sport-Action-Up.git\n",
            "    %cd Sport-Action-Up\n",
            "else:\n",
            "    %cd Sport-Action-Up\n",
            "    !git pull origin main\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Launch Training with Auto-Resume to Drive\n",
            "!python train_cloud.py --config configs/videomae_colab_t4.yaml\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Check Checkpoints Saved in Google Drive\n",
            "!ls -lh /content/drive/MyDrive/SportsAction_Checkpoints/\n",
        ],
    },
]

if __name__ == "__main__":
    make_notebook(kaggle_cells, "notebooks/01_Kaggle_Sports_Action_VideoMAE.ipynb")
    make_notebook(colab_cells, "notebooks/02_Colab_Sports_Action_VideoMAE.ipynb")
