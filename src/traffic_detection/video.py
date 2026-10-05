"""OpenCV media helpers and optional browser-compatible MP4 conversion."""
import math
from pathlib import Path
import shutil
import subprocess

import cv2


def open_video(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    ok, frame = capture.read()
    if not ok or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video has no decodable first frame or valid FPS")
    return capture, frame, fps


def preview_frame(path):
    capture, frame, _ = open_video(path)
    capture.release()
    return frame


def make_writer(path, fps, width, height):
    # MPEG-4 codecs need even dimensions. Pad the right/bottom by at most one pixel.
    size = (width + width % 2, height + height % 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("MP4 encoder unavailable in this OpenCV installation")
    return writer


def pad_frame(frame):
    h, w = frame.shape[:2]
    return cv2.copyMakeBorder(frame, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT)


def browser_video(source, target):
    """Convert to H.264. Retain the original if conversion fails."""
    source, target = Path(source), Path(target)
    if target.exists() or source.resolve() == target.resolve():
        raise FileExistsError(f"Choose a new output path: {target}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for browser conversion; download annotated.mp4 instead")
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(source), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
    return target
