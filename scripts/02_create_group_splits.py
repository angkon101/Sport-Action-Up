"""Leak-Free Group-Wise Dataset Splitter for Sports Action Recognition.

Generates 70 / 15 / 15 Train, Validation, and Test splits partitioned strictly
by source video group IDs (e.g., 'g01', 'g02') to prevent test scene/actor leakage.

Outputs:
- data/splits/train.csv
- data/splits/val.csv
- data/splits/test.csv
- data/splits/split_summary.json
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
from typing import Dict, List, Set, Tuple

import pandas as pd


UCF_PATTERN = re.compile(r"v_([A-Za-z0-9_-]+)_g([0-9]+)_c([0-9]+)\.([a-zA-Z0-9]+)")


def parse_group(filename: str, fallback_class: str) -> Tuple[str, str]:
    match = UCF_PATTERN.match(filename)
    if match:
        class_name, group_id, _, _ = match.groups()
        return class_name, group_id
    stem = Path(filename).stem
    return fallback_class, stem


def create_group_splits(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    print("=" * 70)
    print("✂️  LEAK-FREE GROUP-WISE DATASET SPLITTER")
    print(f"Data Source   : {data_dir.resolve()}")
    print(f"Output Split  : {output_dir.resolve()}")
    print(f"Split Target  : {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test")
    print(f"RNG Seed      : {seed}")
    print("=" * 70)

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video files
    video_files: List[Path] = []
    for ext in ("*.avi", "*.mp4", "*.mkv", "*.mov"):
        video_files.extend(data_dir.rglob(ext))

    video_files = sorted(list(set(video_files)))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    # Build Class -> Group -> List of relative video paths
    class_group_clips: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for v_path in video_files:
        cls_fallback = v_path.parent.name
        class_name, group_id = parse_group(v_path.name, cls_fallback)
        class_group_clips[class_name][group_id].append(v_path)

    classes = sorted(class_group_clips.keys())
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}

    train_rows, val_rows, test_rows = [], [], []
    train_groups, val_groups, test_groups = set(), set(), set()

    for cls_name in classes:
        groups = sorted(list(class_group_clips[cls_name].keys()))
        random.shuffle(groups)
        n_groups = len(groups)

        # Calculate partition sizes
        n_train = max(1, math.floor(n_groups * train_ratio))
        n_val = max(1, math.floor(n_groups * val_ratio))
        # Remainder goes to test to guarantee all groups are utilized
        n_test = n_groups - n_train - n_val

        # Handle edge cases with very few groups
        if n_test <= 0:
            if n_val > 1:
                n_val -= 1
                n_test = 1
            else:
                n_test = 1

        train_grp_list = groups[:n_train]
        val_grp_list = groups[n_train : n_train + n_val]
        test_grp_list = groups[n_train + n_val :]

        for grp in train_grp_list:
            train_groups.add((cls_name, grp))
            for path in class_group_clips[cls_name][grp]:
                train_rows.append({
                    "video_path": str(path.resolve()),
                    "rel_path": str(path.relative_to(data_dir)),
                    "filename": path.name,
                    "class_name": cls_name,
                    "class_id": class_to_id[cls_name],
                    "group_id": grp,
                    "split": "train",
                })

        for grp in val_grp_list:
            val_groups.add((cls_name, grp))
            for path in class_group_clips[cls_name][grp]:
                val_rows.append({
                    "video_path": str(path.resolve()),
                    "rel_path": str(path.relative_to(data_dir)),
                    "filename": path.name,
                    "class_name": cls_name,
                    "class_id": class_to_id[cls_name],
                    "group_id": grp,
                    "split": "val",
                })

        for grp in test_grp_list:
            test_groups.add((cls_name, grp))
            for path in class_group_clips[cls_name][grp]:
                test_rows.append({
                    "video_path": str(path.resolve()),
                    "rel_path": str(path.relative_to(data_dir)),
                    "filename": path.name,
                    "class_name": cls_name,
                    "class_id": class_to_id[cls_name],
                    "group_id": grp,
                    "split": "test",
                })

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Verification: Zero Overlap
    train_clip_names = set(df_train["filename"])
    val_clip_names = set(df_val["filename"])
    test_clip_names = set(df_test["filename"])

    clip_overlap = (train_clip_names & val_clip_names) | (train_clip_names & test_clip_names) | (val_clip_names & test_clip_names)
    group_overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)

    if clip_overlap or group_overlap:
        raise ValueError(f"CRITICAL ERROR: Data leakage detected! Overlapping clips: {len(clip_overlap)}, Overlapping groups: {len(group_overlap)}")

    # Class Index Map
    class_map_file = output_dir / "class_indices.json"
    with open(class_map_file, "w", encoding="utf-8") as f:
        json.dump(class_to_id, f, indent=2)

    # Summary
    summary = {
        "classes_count": len(classes),
        "total_clips": len(video_files),
        "train_clips": len(df_train),
        "val_clips": len(df_val),
        "test_clips": len(df_test),
        "train_ratio_actual": round(len(df_train) / len(video_files), 3),
        "val_ratio_actual": round(len(df_val) / len(video_files), 3),
        "test_ratio_actual": round(len(df_test) / len(video_files), 3),
        "total_groups": len(train_groups) + len(val_groups) + len(test_groups),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "leakage_verification": "PASSED (0 overlap between Train, Val, and Test)",
    }

    summary_file = output_dir / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ GROUP-WISE SPLIT GENERATED & VERIFIED")
    print("=" * 70)
    print(f"Categories  : {summary['classes_count']}")
    print(f"Train Clips : {summary['train_clips']:,} ({summary['train_ratio_actual']*100:.1f}%) | Groups: {summary['train_groups']}")
    print(f"Val Clips   : {summary['val_clips']:,} ({summary['val_ratio_actual']*100:.1f}%) | Groups: {summary['val_groups']}")
    print(f"Test Clips  : {summary['test_clips']:,} ({summary['test_ratio_actual']*100:.1f}%) | Groups: {summary['test_groups']}")
    print(f"Total Clips : {summary['total_clips']:,}")
    print(f"Integrity   : {summary['leakage_verification']}")
    print(f"\nFiles saved to:\n  - {train_csv}\n  - {val_csv}\n  - {test_csv}\n  - {class_map_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Create Group-Wise Train/Val/Test Splits")
    default_data_dir = Path("Archive/Model Data")
    if not default_data_dir.exists():
        default_data_dir = Path("data/raw")

    parser.add_argument("--data-dir", type=str, default=str(default_data_dir), help="Source videos path")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Target splits folder")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_group_splits(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
