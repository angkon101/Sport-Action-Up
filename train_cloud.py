"""Unified Cloud Training CLI Runner for Video Transformers (Kaggle & Colab).

Supports:
- Single GPU execution (Colab / Kaggle P100)
- Multi-GPU Distributed Data Parallel (DDP) via torchrun (Kaggle 2x T4)
- Automatic dataset discovery on /kaggle/input and /content
- Dynamic group-wise splitting (or loading pre-generated CSVs)
- Automatic state resume from interruptions
"""

import argparse
import os
from pathlib import Path
import sys
import yaml

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Ensure root is on path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.datasets.splitter import build_group_splits
from src.datasets.video_dataset import VideoClipsDataset
from src.engine.trainer import run_training_loop
from src.models.videomae import SportsVideoMAE


def setup_ddp() -> Tuple[bool, int, int, torch.device]:
    """Initialize Distributed Data Parallel if launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return is_distributed, rank, local_rank, device


def resolve_dataset_dir(configured_path: str) -> Path:
    """Intelligently search common Kaggle and Colab input locations."""
    candidate_paths = [
        Path(configured_path),
        Path("/kaggle/input/sport-action-up"),
        Path("/kaggle/input/sports-action-up"),
        Path("/kaggle/input/model-data"),
        Path("/kaggle/input/ucf101"),
        Path("/content/dataset"),
        Path("Archive/Model Data"),
        Path("Model Data"),
    ]
    for p in candidate_paths:
        if p.exists() and any(p.rglob("*.avi")) or any(p.rglob("*.mp4")):
            return p
    # Fallback to configured path
    return Path(configured_path)


def main():
    parser = argparse.ArgumentParser(description="Cloud VideoMAE Training Runner")
    parser.add_argument("--config", type=str, default="configs/videomae_base.yaml", help="Path to YAML configuration")
    parser.add_argument("--data-dir", type=str, default=None, help="Override source video directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--micro-batch", type=int, default=None, help="Override micro batch size")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto resume from checkpoint")
    args = parser.parse_args()

    # Load Config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # DDP Initialization
    is_distributed, rank, local_rank, device = setup_ddp()
    is_main_process = (rank == 0)

    if is_main_process:
        print("=" * 70)
        print("⚡ CLOUD TRAINING RUNNER: VIDEOMAE SPORTS ACTION")
        print(f"Device        : {device} (Distributed={is_distributed}, WorldSize={dist.get_world_size() if is_distributed else 1})")
        print(f"Config File   : {config_path}")
        print("=" * 70)

    # Resolve Data Path & Splits
    raw_data_dir = resolve_dataset_dir(args.data_dir or cfg["data"].get("source_dir", "Archive/Model Data"))
    splits_dir = Path("data/splits")
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if train_csv.exists() and val_csv.exists():
        if is_main_process:
            print(f"[✓] Loading existing splits from: {splits_dir}")
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
    else:
        if is_main_process:
            print(f"[!] Generating dynamic group-wise splits from: {raw_data_dir}")
        df_train, df_val, _, _ = build_group_splits(
            data_dir=raw_data_dir,
            output_dir=splits_dir if is_main_process else None,
            seed=42,
        )

    num_classes = cfg["model"].get("num_classes", 82)
    num_frames = cfg["data"].get("num_frames", 16)
    image_size = cfg["data"].get("input_size", 224)
    micro_batch = args.micro_batch or cfg["training"].get("micro_batch_size", 2)
    accum_steps = cfg["training"].get("gradient_accumulation_steps", 16)
    epochs = args.epochs or cfg["training"].get("epochs", 30)
    base_lr = float(cfg["training"].get("base_lr", 1e-3))
    layer_decay = float(cfg["training"].get("layer_decay", 0.75))
    checkpoint_dir = Path(cfg["checkpoint"].get("save_dir", "checkpoints/videomae"))
    log_csv = Path(cfg["logging"].get("csv_log_file", "training_history.csv"))

    # Datasets & Loaders
    train_dataset = VideoClipsDataset(
        dataframe=df_train,
        num_frames=num_frames,
        image_size=image_size,
        is_training=True,
    )
    val_dataset = VideoClipsDataset(
        dataframe=df_val,
        num_frames=num_frames,
        image_size=image_size,
        is_training=False,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    num_workers = cfg["data"].get("num_workers", 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model
    model = SportsVideoMAE(
        pretrained_name=cfg["model"].get("pretrained_checkpoint", "MCG-NJU/videomae-base-finetuned-kinetics"),
        num_classes=num_classes,
        dropout=cfg["model"].get("dropout", 0.5),
        gradient_checkpointing=cfg["model"].get("gradient_checkpointing", True),
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Optimizer with Layer-wise Decay
    target_model = model.module if is_distributed else model
    param_groups = target_model.get_parameter_groups_with_decay(
        base_lr=base_lr,
        weight_decay=cfg["training"].get("weight_decay", 0.05),
        layer_decay=layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Run Training
    results = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        total_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        accumulation_steps=accum_steps,
        log_csv_path=log_csv if is_main_process else None,
        auto_resume=args.resume,
        is_main_process=is_main_process,
    )

    if is_main_process:
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE")
        print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
        print(f"Checkpoints directory   : {checkpoint_dir.resolve()}")
        print("=" * 70)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
