
"""Streaming pipeline: one tracker and one counter per run."""
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import cv2

from .counter import LineCounter
from .tracker import Tracker
from .video import open_video, make_writer, pad_frame
from .visualization import draw_frame

EVENT_FIELDS = ["track_id", "class_name", "frame_number", "timestamp_seconds", "direction"]


def run_video(source, output_dir, config, tracker=None, progress=None):
    """Output directory must be new. Optional tracker injection supports offline tests."""
    config.validate()
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    capture, frame, fps = open_video(source)
    writer = None
    started = time.perf_counter()
    frame_index = 0
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        height, width = frame.shape[:2]
        line = config.pixel_line(width, height)
        counter = LineCounter(*line, config.count_classes, config.line_margin_px, config.max_gap_frames)
        tracker = tracker if tracker is not None else Tracker(config)
        video_path = output_dir / "annotated.mp4"
        writer = make_writer(video_path, fps, width, height)
        with ExitStack() as stack:
            events_file = stack.enter_context((output_dir / "crossings.csv").open("w", newline=""))
            events_writer = csv.DictWriter(events_file, fieldnames=EVENT_FIELDS)
            events_writer.writeheader()
            predictions = stack.enter_context((output_dir / "predictions.jsonl").open("w")) if config.save_predictions else None
            while True:
                if frame.shape[:2] != (height, width):
                    raise ValueError("Frame dimensions changed during the video")
                objects = tracker.update(frame)
                events_writer.writerows(counter.update(objects, frame_index, frame_index / fps))
                if predictions:
                    predictions.write(json.dumps(dict(frame_number=frame_index, timestamp_seconds=frame_index / fps, objects=objects)) + "\n")
                rendered = draw_frame(frame, objects, line, counter.summary()["total"])
                writer.write(pad_frame(rendered))
                if frame_index == 0 and not cv2.imwrite(str(output_dir / "preview.jpg"), rendered):
                    raise OSError("Could not write preview image")
                frame_index += 1
                if progress and frame_index % 50 == 0:
                    progress(frame_index)
                if config.max_frames is not None and frame_index >= config.max_frames:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
        elapsed = time.perf_counter() - started
        limited = config.max_frames is not None and frame_index >= config.max_frames
        warnings = []
        if expected_frames > 0 and frame_index < expected_frames and not limited:
            warnings.append("Reading stopped before the container's declared frame count; check for truncated/corrupt input.")
        summary = dict(source=str(source), created_utc=datetime.now(timezone.utc).isoformat(),
                       frames_processed=frame_index, source_fps=fps, source_frame_count=expected_frames,
                       source_size=[width, height], output_size=[width + width % 2, height + height % 2],
                       elapsed_seconds=round(elapsed, 3), processing_fps=round(frame_index / elapsed, 2),
                       stopped_at_frame_limit=limited, counts=counter.summary(), config=config.to_dict(),
                       versions={name: version(name) for name in ["ultralytics", "opencv-python", "PyYAML"]},
                       tracker="ByteTrack", warnings=warnings,
                       note="One crossing per track ID; ID switches can cause missed or duplicate physical vehicles. No audio. Timestamps assume constant FPS.")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
