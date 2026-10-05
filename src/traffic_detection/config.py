"""Validated run configuration; importing this module does not load a model."""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import yaml

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Config:
    model: str = "yolo11n.pt"
    confidence: float = 0.25
    image_size: int = 640
    device: str = "auto"
    detect_classes: list[int] = field(default_factory=lambda: list(CLASSES))
    count_classes: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])
    line_start: list[float] = field(default_factory=lambda: [0.1, 0.6])
    line_end: list[float] = field(default_factory=lambda: [0.9, 0.6])
    line_margin_px: float = 5
    max_gap_frames: int = 15
    max_frames: int | None = None
    save_predictions: bool = True

    def validate(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a model name or path")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a string such as auto, cpu, or '0'")
        for name, value in [("confidence", self.confidence), ("line_margin_px", self.line_margin_px)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if self.line_margin_px < 0:
            raise ValueError("line_margin_px must be nonnegative")
        for name in ("image_size", "max_gap_frames", "max_frames"):
            value = getattr(self, name)
            if name == "max_frames" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.save_predictions) is not bool:
            raise ValueError("save_predictions must be true or false")
        for name in ("detect_classes", "count_classes"):
            ids = getattr(self, name)
            if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in CLASSES for i in ids):
                raise ValueError(f"{name} must contain supported COCO IDs: {list(CLASSES)}")
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} contains duplicate IDs")
        if not set(self.count_classes).issubset(self.detect_classes):
            raise ValueError("count_classes must be included in detect_classes")
        for point in (self.line_start, self.line_end):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Line endpoints must each have two normalized coordinates")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                raise ValueError("Line coordinates must be finite numbers between 0 and 1")
        if list(self.line_start) == list(self.line_end):
            raise ValueError("Line endpoints must differ")
        return self

    def to_dict(self):
        return asdict(self)

    def pixel_line(self, width, height):
        return tuple((float(p[0]) * (width - 1), float(p[1]) * (height - 1))
                     for p in (self.line_start, self.line_end))


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    try:
        return Config(**data).validate()
    except TypeError as exc:
        raise ValueError(f"Unknown configuration field: {exc}") from exc
