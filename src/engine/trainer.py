"""Production-Grade Trainer Engine for Video Transformers on Cloud Platforms.

Features:
- fp16 Mixed Precision via PyTorch AMP (GradScaler)
- Gradient Accumulation with Micro-Batching for T4 16 GB VRAM budget
- DDP Multi-GPU scaling (Kaggle 2x T4 support)
- Deterministic auto-resume from interruptions
- Append-only CSV telemetry surviving session cuts
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.utils.metrics import AverageMeter, compute_topk_accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    accumulation_steps: int = 16,
    is_main_process: bool = True,
) -> Dict[str, float]:
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    optimizer.zero_grad()
    t0 = time.time()

    total_batches = len(dataloader)
    pbar = tqdm(
        dataloader,
        desc=f"Train Epoch {epoch}",
        disable=not is_main_process,
        leave=False,
    )

    for step, (pixel_values, labels, _) in enumerate(pbar):
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        # Mixed Precision Forward
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward with scaled gradients
        scaler.scale(loss).backward()

        # Step optimizer and update scaler every accumulation_steps or at epoch end
        if (step + 1) % accumulation_steps == 0 or (step + 1) == total_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics
        loss_val = loss.item() * accumulation_steps
        top1, top5 = compute_topk_accuracy(logits, labels, topk=(1, 5))

        loss_meter.update(loss_val, batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

        if is_main_process and (step + 1) % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "top1": f"{top1_meter.avg:.2f}%",
                "top5": f"{top5_meter.avg:.2f}%",
            })

    elapsed = time.time() - t0
    return {
        "train_loss": round(loss_meter.avg, 4),
        "train_top1": round(top1_meter.avg, 2),
        "train_top5": round(top5_meter.avg, 2),
        "train_time_sec": round(elapsed, 1),
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    is_main_process: bool = True,
) -> Dict[str, float]:
    """Evaluate model on validation or test set."""
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    t0 = time.time()
    pbar = tqdm(
        dataloader,
        desc=f"Eval Epoch {epoch}",
        disable=not is_main_process,
        leave=False,
    )

    for pixel_values, labels, _ in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

        top1, top5 = compute_topk_accuracy(logits, labels, topk=(1, 5))

        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

    elapsed = time.time() - t0
    return {
        "val_loss": round(loss_meter.avg, 4),
        "val_top1": round(top1_meter.avg, 2),
        "val_top5": round(top5_meter.avg, 2),
        "val_time_sec": round(elapsed, 1),
    }


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    total_epochs: int,
    checkpoint_dir: Path,
    accumulation_steps: int = 16,
    log_csv_path: Optional[Path] = None,
    auto_resume: bool = True,
    is_main_process: bool = True,
) -> Dict[str, Any]:
    """Execute complete multi-epoch training pipeline with auto-resume."""
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 1
    best_val_acc = 0.0
    history = []

    checkpoint_dir = Path(checkpoint_dir)
    last_ckpt = checkpoint_dir / "last_checkpoint.pth"

    # Auto-resume check
    if auto_resume and last_ckpt.exists():
        if is_main_process:
            print(f"\n[🔄] Resuming from detected checkpoint: {last_ckpt}")
        start_epoch, best_val_acc, _ = load_checkpoint(
            checkpoint_path=last_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if is_main_process:
            print(f"[✓] Successfully resumed at Epoch {start_epoch} (Prior Best Val Acc: {best_val_acc:.2f}%)")

    # Load existing CSV history if resuming
    if log_csv_path and log_csv_path.exists():
        try:
            history = pd.read_csv(log_csv_path).to_dict("records")
        except Exception:
            pass

    for epoch in range(start_epoch, total_epochs + 1):
        if is_main_process:
            print(f"\n{'='*65}\n📍 Epoch [{epoch}/{total_epochs}]\n{'='*65}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            accumulation_steps=accumulation_steps,
            is_main_process=is_main_process,
        )

        val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            is_main_process=is_main_process,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_record)

        is_best = val_metrics["val_top1"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["val_top1"]

        if is_main_process:
            print(
                f"[Summary Epoch {epoch}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_top1']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_top1']:.2f}% (Best: {best_val_acc:.2f}%)"
            )

            # Save state
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                is_best=is_best,
                checkpoint_dir=checkpoint_dir,
            )

            # Append to CSV
            if log_csv_path:
                pd.DataFrame(history).to_csv(log_csv_path, index=False)

    return {"best_val_acc": best_val_acc, "history": history}
