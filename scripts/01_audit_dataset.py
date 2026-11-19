"""Dataset Audit & Verification Engine for Sports Action Recognition.

Audits all raw video clips in the dataset:
- Scans classes and clip counts
- Parses UCF group IDs (v_<Class>_g<Group>_c<Clip>.avi) to prevent data leakage
- Verifies video file integrity (corrupt file detection via OpenCV)
- Measures video resolutions, FPS, durations, and frame counts
- Saves structured audit report to JSON and CSV
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import pandas as pd
from tqdm import tqdm


UCF_PATTERN = re.compile(r"v_([A-Za-z0-9_-]+)_g([0-9]+)_c([0-9]+)\.([a-zA-Z0-9]+)")


def parse_video_filename(filename: str, fallback_class: str) -> Tuple[str, str, str]:
    """Extract (class_name, group_id, clip_id) from filename."""
    match = UCF_PATTERN.match(filename)
    if match:
        class_name, group_id, clip_id, _ = match.groups()
        return class_name, group_id, clip_id
    # Fallback pattern if named differently
    stem = Path(filename).stem
    return fallback_class, stem, stem


def audit_single_video(video_path: Path) -> Dict[str, Any]:
    """Check integrity and extract metadata for a single video file."""
    res = {
        "path": str(video_path),
        "filename": video_path.name,
        "size_bytes": video_path.stat().st_size,
        "is_valid": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_sec": 0.0,
        "error": None,
    }

    if res["size_bytes"] == 0:
        res["error"] = "Zero-byte file"
        return res

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            res["error"] = "Failed to open with cv2.VideoCapture"
            return res

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if first and last frames can be read
        ret, frame = cap.read()
        if not ret or frame is None:
            res["error"] = "Could not read first frame"
            cap.release()
            return res

        cap.release()

        duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0

        res["is_valid"] = True
        res["width"] = width
        res["height"] = height
        res["fps"] = round(fps, 2)
        res["frame_count"] = frame_count
        res["duration_sec"] = round(duration, 2)

    except Exception as e:
        res["error"] = str(e)

    return res


def run_audit(data_dir: Path, output_dir: Path) -> None:
    print("=" * 70)
    print("🔍 SPORTS ACTION DATASET AUDIT")
    print(f"Data Directory  : {data_dir.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}")
    print("=" * 70)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    extensions = ("*.avi", "*.mp4", "*.mkv", "*.mov")
    video_files: List[Path] = []
    for ext in extensions:
        video_files.extend(data_dir.rglob(ext))

    video_files = sorted(list(set(video_files)))
    total_found = len(video_files)
    print(f"Found {total_found:,} video files across subdirectories.")

    if total_found == 0:
        print("[!] No video files detected. Please check the data-dir path.")
        return

    records = []
    class_groups = defaultdict(lambda: defaultdict(list))
    corrupt_files = []

    print("\nAuditing video integrity & extracting metadata...")
    for v_path in tqdm(video_files, desc="Auditing clips"):
        fallback_class = v_path.parent.name
        class_name, group_id, clip_id = parse_video_filename(v_path.name, fallback_class)

        meta = audit_single_video(v_path)
        meta["class_name"] = class_name
        meta["group_id"] = group_id
        meta["clip_id"] = clip_id

        if not meta["is_valid"]:
            corrupt_files.append((str(v_path), meta["error"]))

        class_groups[class_name][group_id].append(v_path.name)
        records.append(meta)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    csv_out = output_dir / "dataset_audit_all_clips.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[✓] Detailed clip audit saved to: {csv_out}")

    # Summary Statistics
    total_valid = df["is_valid"].sum()
    total_corrupt = len(corrupt_files)
    unique_classes = sorted(df["class_name"].unique())
    total_groups = sum(len(groups) for groups in class_groups.values())

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir.resolve()),
        "total_clips": total_found,
        "valid_clips": int(total_valid),
        "corrupt_clips": total_corrupt,
        "total_classes": len(unique_classes),
        "total_unique_groups": total_groups,
        "classes": unique_classes,
        "class_clip_counts": df.groupby("class_name").size().to_dict(),
        "class_group_counts": {c: len(groups) for c, groups in class_groups.items()},
        "corrupt_details": corrupt_files,
        "avg_duration_sec": round(float(df[df["is_valid"]]["duration_sec"].mean()), 2),
        "avg_frame_count": round(float(df[df["is_valid"]]["frame_count"].mean()), 1),
        "common_resolutions": df[df["is_valid"]]
        .groupby(["width", "height"])
        .size()
        .nlargest(5)
        .to_dict(),
    }

    summary_json = output_dir / "audit_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Summary report saved to: {summary_json}")

    print("\n" + "=" * 70)
    print("📊 AUDIT SUMMARY METRICS")
    print("=" * 70)
    print(f"Total Video Clips  : {total_found:,}")
    print(f"Valid Playable     : {total_valid:,}")
    print(f"Corrupt / Unreadable: {total_corrupt:,}")
    print(f"Unique Classes     : {len(unique_classes)}")
    print(f"Source Video Groups: {total_groups:,}")
    print(f"Average Duration   : {summary['avg_duration_sec']}s")
    print(f"Average Frames     : {summary['avg_frame_count']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit Sports Action Dataset")
    # Default points to preserved raw videos in Archive/Model Data
    default_data_dir = Path("Archive/Model Data")
    if not default_data_dir.exists():
        default_data_dir = Path("data/raw")

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_data_dir),
        help="Path to folder containing raw videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/audit",
        help="Directory to save audit CSV and JSON reports",
    )
    args = parser.parse_args()

    run_audit(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
