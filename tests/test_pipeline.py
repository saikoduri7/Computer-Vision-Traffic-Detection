import csv
import json

import cv2
import numpy as np
import pytest

from traffic_detection.config import Config
from traffic_detection.pipeline import run_video


class FakeTracker:
    def __init__(self):
        self.frame = 0

    def update(self, frame):
        y = 30 + self.frame * 10
        self.frame += 1
        return [dict(track_id=1, class_id=2, class_name="car", confidence=.9, bbox=[40, y - 10, 60, y])]


def video(tmp_path):
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    return path


def test_complete_pipeline(tmp_path):
    source = video(tmp_path)
    cfg = Config(max_frames=5, line_start=[.1, .5], line_end=[.9, .5], line_margin_px=2)
    out = tmp_path / "out"
    summary = run_video(source, out, cfg, tracker=FakeTracker())
    assert summary["frames_processed"] == 5
    assert summary["counts"]["total"] == 1
    assert summary["stopped_at_frame_limit"]
    with (out / "crossings.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["class_name"] == "car"
    records = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert [r["timestamp_seconds"] for r in records] == [0, .1, .2, .3, .4]
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 5
    assert cap.get(cv2.CAP_PROP_FPS) == 10
    assert cap.read()[0]
    cap.release()
    assert cv2.imread(str(out / "preview.jpg")).any()
    with pytest.raises(FileExistsError):
        run_video(source, out, cfg, tracker=FakeTracker())


def test_end_of_video_without_predictions(tmp_path):
    summary = run_video(video(tmp_path), tmp_path / "out", Config(save_predictions=False), FakeTracker())
    assert summary["frames_processed"] == 8
    assert not (tmp_path / "out/predictions.jsonl").exists()


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_video(tmp_path / "missing.mp4", tmp_path / "out", Config(), FakeTracker())
    assert not (tmp_path / "out").exists()
