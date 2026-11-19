"""Dynamic Group-Wise Dataset Splitter for Kaggle and Google Colab.

Automatically discovers sports video clips in cloud environments, groups clips
by source video ID (v_<class>_g<group>_c<clip>.avi) to guarantee zero leakage,
and returns or saves balanced 70% Train / 15% Val / 15% Test splits.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

UCF_PATTERN = re.compile(r"v_([A-Za-z0-9_-]+)_g([0-9]+)_c([0-9]+)\.([a-zA-Z0-9]+)")


def parse_video_metadata(filepath: Path, fallback_class: str) -> Tuple[str, str, str]:
    """Extract (class_name, group_id, clip_id) from filename or folder."""
    match = UCF_PATTERN.match(filepath.name)
    if match:
        class_name, group_id, clip_id, _ = match.groups()
        return class_name, group_id, clip_id
    stem = filepath.stem
    return fallback_class, stem, stem


def build_group_splits(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Scan data_dir, perform group-wise split, and optionally save CSVs."""
    random.seed(seed)
    data_dir = Path(data_dir)

    video_files: List[Path] = []
    for ext in ("*.avi", "*.mp4", "*.mkv", "*.mov", "*.webm"):
        video_files.extend(data_dir.rglob(ext))

    video_files = sorted(list(set(video_files)))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    # Build Class -> Group -> List of video paths
    class_groups: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for v_path in video_files:
        fallback_class = v_path.parent.name
        class_name, group_id, _ = parse_video_metadata(v_path, fallback_class)
        class_groups[class_name][group_id].append(v_path)

    classes = sorted(class_groups.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_rows, val_rows, test_rows = [], [], []

    for cls in classes:
        groups = sorted(list(class_groups[cls].keys()))
        random.shuffle(groups)
        n_groups = len(groups)

        n_train = max(1, math.floor(n_groups * train_ratio))
        n_val = max(1, math.floor(n_groups * val_ratio))
        n_test = n_groups - n_train - n_val
        if n_test <= 0:
            if n_val > 1:
                n_val -= 1
                n_test = 1
            else:
                n_test = 1

        train_grp = set(groups[:n_train])
        val_grp = set(groups[n_train : n_train + n_val])
        test_grp = set(groups[n_train + n_val :])

        for grp in train_grp:
            for p in class_groups[cls][grp]:
                train_rows.append({
                    "video_path": str(p.resolve()),
                    "filename": p.name,
                    "class_name": cls,
                    "class_id": class_to_idx[cls],
                    "group_id": grp,
                    "split": "train",
                })

        for grp in val_grp:
            for p in class_groups[cls][grp]:
                val_rows.append({
                    "video_path": str(p.resolve()),
                    "filename": p.name,
                    "class_name": cls,
                    "class_id": class_to_idx[cls],
                    "group_id": grp,
                    "split": "val",
                })

        for grp in test_grp:
            for p in class_groups[cls][grp]:
                test_rows.append({
                    "video_path": str(p.resolve()),
                    "filename": p.name,
                    "class_name": cls,
                    "class_id": class_to_idx[cls],
                    "group_id": grp,
                    "split": "test",
                })

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(output_dir / "train.csv", index=False)
        df_val.to_csv(output_dir / "val.csv", index=False)
        df_test.to_csv(output_dir / "test.csv", index=False)
        with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)

    return df_train, df_val, df_test, class_to_idx
