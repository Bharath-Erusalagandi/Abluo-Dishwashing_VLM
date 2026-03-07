"""Video scraping and frame extraction for data collection.

Downloads robot failure videos from YouTube and extracts
key frames for annotation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import DATA_DIR
from src.utils.logging import get_logger

log = get_logger(__name__)

# Search queries for YouTube scraping
SEARCH_QUERIES = [
    "dishwashing robot failure",
    "kitchen robot arm dropping dishes",
    "restaurant robot fail",
    "robotic dishwasher malfunction",
    "robot dropping plate",
    "robot manipulation failure kitchen",
    "robot grasp failure glass",
    "robot arm wet dishes",
    "commercial kitchen robot",
    "robot dish rack loading",
]

RAW_DIR = DATA_DIR / "raw" / "videos"
FRAMES_DIR = DATA_DIR / "raw" / "frames"


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    url: str
    duration_s: float
    query: str
    frame_count: int = 0
    local_path: str = ""


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def search_youtube(
    query: str,
    max_results: int = 50,
) -> list[dict]:
    """Search YouTube for videos matching query.

    Uses yt-dlp for metadata extraction (no download yet).
    Returns list of video metadata dicts.
    """
    import subprocess
    _ensure_dirs()

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        f"ytsearch{max_results}:{query}",
    ]

    log.info("youtube_search", query=query, max_results=max_results)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        videos = []
        for line in result.stdout.strip().splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    videos.append({
                        "video_id": data.get("id", ""),
                        "title": data.get("title", ""),
                        "url": data.get("url", data.get("webpage_url", "")),
                        "duration": data.get("duration", 0),
                    })
                except json.JSONDecodeError:
                    continue
        log.info("youtube_search_results", query=query, found=len(videos))
        return videos
    except FileNotFoundError:
        log.warning("yt-dlp not installed. Install with: pip install yt-dlp")
        return []
    except subprocess.TimeoutExpired:
        log.warning("youtube_search_timeout", query=query)
        return []


def download_video(video_url: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Download a single video using yt-dlp.

    Returns path to downloaded file, or None if failed.
    """
    import subprocess

    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "--merge-output-format", "mp4",
        "-o", str(output_dir / "%(id)s.%(ext)s"),
        "--no-playlist",
        video_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Find the output file
            for f in output_dir.iterdir():
                if f.suffix == ".mp4":
                    log.info("video_downloaded", path=str(f))
                    return f
        log.warning("video_download_failed", url=video_url, stderr=result.stderr[:200])
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("video_download_error", error=str(e))
        return None


def extract_frames_scene_change(
    video_path: Path,
    threshold: float = 0.3,
    max_frames: int = 50,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Extract key frames from video using scene-change detection.

    Uses OpenCV to detect significant scene changes, which often
    correspond to grasp attempts / failures.

    Args:
        video_path: Path to video file.
        threshold: Scene change threshold (0-1). Lower = more sensitive.
        max_frames: Maximum frames to extract.
        output_dir: Where to save extracted frames.

    Returns:
        List of paths to extracted frame images.
    """
    import cv2

    output_dir = output_dir or FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("video_open_failed", path=str(video_path))
        return []

    frames_saved: list[Path] = []
    prev_gray = None
    frame_idx = 0

    while cap.isOpened() and len(frames_saved) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Compute frame difference
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff) / 255.0

            if score > threshold:
                frame_path = output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                frames_saved.append(frame_path)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    log.info(
        "frames_extracted",
        video=video_path.name,
        total_frames=frame_idx,
        key_frames=len(frames_saved),
    )
    return frames_saved


def extract_frames_uniform(
    video_path: Path,
    interval_sec: float = 2.0,
    max_frames: int = 100,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Extract frames at uniform time intervals.

    Simpler than scene change detection — good for systematic coverage.
    """
    import cv2

    output_dir = output_dir or FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("video_open_failed", path=str(video_path))
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval_frames = int(fps * interval_sec)
    frames_saved: list[Path] = []
    frame_idx = 0

    while cap.isOpened() and len(frames_saved) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            frame_path = output_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frames_saved.append(frame_path)

        frame_idx += 1

    cap.release()
    log.info(
        "frames_extracted_uniform",
        video=video_path.name,
        interval_sec=interval_sec,
        frames=len(frames_saved),
    )
    return frames_saved


def build_video_manifest(
    queries: Optional[list[str]] = None,
    max_per_query: int = 20,
) -> list[VideoMetadata]:
    """Search YouTube for all queries and build a manifest of videos to download.

    Does NOT download — just creates the metadata list.
    """
    queries = queries or SEARCH_QUERIES
    manifest: list[VideoMetadata] = []
    seen_ids: set[str] = set()

    for query in queries:
        results = search_youtube(query, max_results=max_per_query)
        for r in results:
            vid = r["video_id"]
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                manifest.append(VideoMetadata(
                    video_id=vid,
                    title=r["title"],
                    url=r["url"],
                    duration_s=r.get("duration", 0),
                    query=query,
                ))

    log.info("video_manifest_built", total_unique=len(manifest))

    # Save manifest
    _ensure_dirs()
    manifest_path = DATA_DIR / "raw" / "video_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump([{
            "video_id": v.video_id,
            "title": v.title,
            "url": v.url,
            "duration_s": v.duration_s,
            "query": v.query,
        } for v in manifest], f, indent=2)
    log.info("manifest_saved", path=str(manifest_path))

    return manifest
