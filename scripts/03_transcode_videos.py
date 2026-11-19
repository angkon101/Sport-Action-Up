"""Multi-Threaded Video Transcoding & Resolution Normalization.

Resizes raw video clips so the shorter side is normalized (e.g., 224px or 160px)
at a fixed 25 FPS. Dramatically reduces disk space and CPU decoding latency
during training on free-tier cloud instances (Kaggle / Colab).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import cv2
from tqdm import tqdm


def transcode_clip(
    src_path: Path,
    dst_path: Path,
    target_shorter_side: int = 224,
    target_fps: float = 25.0,
) -> Tuple[bool, str]:
    """Transcodes a single video clip to target shorter side and fixed FPS."""
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return True, "already_exists"

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            return False, f"Failed to open {src_path.name}"

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)

        if orig_w == 0 or orig_h == 0:
            cap.release()
            return False, "Invalid dimensions (0x0)"

        # Compute new dimensions keeping aspect ratio
        if orig_w < orig_h:
            new_w = target_shorter_side
            new_h = int(round(orig_h * (target_shorter_side / orig_w)))
        else:
            new_h = target_shorter_side
            new_w = int(round(orig_w * (target_shorter_side / orig_h)))

        # Ensure dimensions are even numbers (required by many video encoders)
        new_w = new_w + (new_w % 2)
        new_h = new_h + (new_h % 2)

        # Video writer using MP4V or XVID
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_to_use = target_fps if target_fps > 0 else orig_fps
        out = cv2.VideoWriter(str(dst_path), fourcc, fps_to_use, (new_w, new_h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out.write(resized_frame)
            frame_idx += 1

        cap.release()
        out.release()

        if frame_idx == 0:
            if dst_path.exists():
                dst_path.unlink()
            return False, "0 frames written"

        return True, "success"

    except Exception as e:
        if dst_path.exists():
            try:
                dst_path.unlink()
            except Exception:
                pass
        return False, str(e)


def process_wrapper(args):
    src, dst, target_size, target_fps = args
    success, msg = transcode_clip(src, dst, target_size, target_fps)
    return success, msg, src.name


def main():
    parser = argparse.ArgumentParser(description="Multi-Threaded Video Transcoder")
    default_src = Path("Archive/Model Data")
    if not default_src.exists():
        default_src = Path("data/raw")

    parser.add_argument("--src-dir", type=str, default=str(default_src), help="Source videos path")
    parser.add_argument("--dst-dir", type=str, default="data/processed/videos_224px", help="Destination path")
    parser.add_argument("--shorter-side", type=int, default=224, help="Target shorter side (224 or 160)")
    parser.add_argument("--fps", type=float, default=25.0, help="Target FPS (default: 25.0)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional clip limit for testing")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    print("=" * 70)
    print("🎬 MULTI-THREADED VIDEO TRANSCODER")
    print(f"Source Directory : {src_dir.resolve()}")
    print(f"Destination      : {dst_dir.resolve()}")
    print(f"Target Size      : Min side {args.shorter_side}px @ {args.fps} FPS")
    print(f"CPU Workers      : {args.workers}")
    print("=" * 70)

    # Collect source videos
    all_videos = []
    for ext in ("*.avi", "*.mp4", "*.mkv", "*.mov"):
        all_videos.extend(src_dir.rglob(ext))

    all_videos = sorted(list(set(all_videos)))
    if args.max_clips:
        all_videos = all_videos[: args.max_clips]

    total_clips = len(all_videos)
    print(f"Total clips to process: {total_clips:,}\n")

    # Prepare jobs
    jobs = []
    for src_p in all_videos:
        rel = src_p.relative_to(src_dir)
        # Save as mp4 in destination
        dst_p = dst_dir / rel.with_suffix(".mp4")
        jobs.append((src_p, dst_p, args.shorter_side, args.fps))

    t0 = time.time()
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_wrapper, job) for job in jobs]
        with tqdm(total=total_clips, desc="Transcoding") as pbar:
            for fut in as_completed(futures):
                success, msg, name = fut.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    tqdm.write(f"  [!] Failed: {name} ({msg})")
                pbar.update(1)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("🏁 TRANSCODING FINISHED")
    print(f"Processed  : {total_clips:,} clips in {elapsed:.1f}s ({total_clips/max(1,elapsed):.1f} clips/s)")
    print(f"Successful : {success_count:,}")
    print(f"Failed     : {fail_count:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
