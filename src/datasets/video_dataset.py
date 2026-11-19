"""High-Throughput PyTorch Video Dataset for Video Transformers.

Optimized for Kaggle and Google Colab runtimes:
- Fast multi-threaded C++ video decoding via decord (with OpenCV fallback)
- Uniform temporal sampling across video duration (16 or 8 frames)
- Spatial augmentations (RandomResizedCrop, HorizontalFlip, Normalization)
- Outputs (T, C, H, W) format expected by VideoMAE and Video Swin
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


class VideoClipsDataset(Dataset):
    """PyTorch Dataset for Sports Action Video Clips."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_frames: int = 16,
        sampling_rate: int = 4,
        image_size: int = 224,
        is_training: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.df = dataframe.reset_index(drop=True)
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.is_training = is_training
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def _get_frame_indices(self, total_frames: int) -> List[int]:
        """Compute uniform or strided temporal frame indices."""
        if total_frames <= self.num_frames:
            # Repeat frames if video is shorter than required frames
            indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int)
            return indices.tolist()

        if self.is_training:
            # Random uniform start with fixed stride or uniform sampling
            max_start = max(0, total_frames - self.num_frames)
            start_idx = np.random.randint(0, max_start + 1)
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Deterministic uniform sampling for validation/test
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        return indices.tolist()

    def _read_video_decord(self, video_path: str) -> Optional[torch.Tensor]:
        """Read frames using decord fast C++ backend."""
        if not HAS_DECORD:
            return None
        try:
            vr = decord.VideoReader(video_path, num_threads=2)
            total_frames = len(vr)
            if total_frames == 0:
                return None
            indices = self._get_frame_indices(total_frames)
            # Returns shape (T, H, W, C) uint8 tensor
            frames = vr.get_batch(indices)
            # Permute to (T, C, H, W) and convert to float32 [0, 1]
            frames = frames.permute(0, 3, 1, 2).float() / 255.0
            return frames
        except Exception:
            return None

    def _read_video_opencv(self, video_path: str) -> torch.Tensor:
        """Fallback frame reader using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.num_frames

        indices = set(self._get_frame_indices(total_frames))
        frames_dict = {}
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if idx in indices:
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_dict[idx] = frame_rgb
            idx += 1
            if len(frames_dict) >= len(indices):
                break

        cap.release()

        # Handle missing frames by padding
        ordered_indices = self._get_frame_indices(max(idx, 1))
        collected = []
        last_valid = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        for i in ordered_indices:
            if i in frames_dict:
                last_valid = frames_dict[i]
                collected.append(last_valid)
            else:
                collected.append(last_valid)

        # Shape (T, H, W, C) -> (T, C, H, W)
        arr = np.stack(collected, axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        return tensor

    def _apply_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply spatio-temporal video augmentations."""
        # frames is (T, C, H, W) in [0, 1]
        T_len, C, H, W = frames.shape

        if self.is_training:
            # Consistent spatial cropping parameters across all temporal frames
            i, j, h, w = T.RandomResizedCrop.get_params(
                frames[0], scale=(0.5, 1.0), ratio=(0.75, 1.33)
            )
            flip = np.random.rand() > 0.5

            transformed = []
            for t in range(T_len):
                frame = frames[t]
                frame = F.crop(frame, i, j, h, w)
                frame = F.resize(frame, [self.image_size, self.image_size], antialias=True)
                if flip:
                    frame = F.hflip(frame)
                frame = (frame - self.mean) / self.std
                transformed.append(frame)
            return torch.stack(transformed, dim=0)
        else:
            # Center crop validation pipeline
            transformed = []
            for t in range(T_len):
                frame = frames[t]
                frame = F.resize(frame, [self.image_size, self.image_size], antialias=True)
                frame = (frame - self.mean) / self.std
                transformed.append(frame)
            return torch.stack(transformed, dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        class_id = int(row["class_id"])
        class_name = row["class_name"]

        frames = self._read_video_decord(video_path)
        if frames is None:
            frames = self._read_video_opencv(video_path)

        frames = self._apply_transforms(frames)
        # Returns (T, C, H, W) tensor, label id, and class name
        return frames, class_id, class_name
