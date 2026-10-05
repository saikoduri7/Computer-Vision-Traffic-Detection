"""Finite-segment crossing with a dead band and once-per-track counting."""
from collections import Counter
from dataclasses import dataclass, field
import math


@dataclass
class TrackState:
    point: tuple
    distance: float
    last_seen: int
    votes: Counter = field(default_factory=Counter)


class LineCounter:
    def __init__(self, start, end, class_ids, margin=5, max_gap=15):
        self.start, self.end = start, end
        self.dx, self.dy = end[0] - start[0], end[1] - start[1]
        self.length = math.hypot(self.dx, self.dy)
        if self.length == 0:
            raise ValueError("Counting line must have nonzero length")
        self.classes = set(class_ids)
        self.margin, self.max_gap = margin, max_gap
        self.states = {}
        self.counted = set()
        self.counts = Counter()
        self.last_frame = -1

    def distance(self, point):
        return (self.dx * (point[1] - self.start[1]) - self.dy * (point[0] - self.start[0])) / self.length

    def update(self, objects, frame_number, timestamp):
        if frame_number <= self.last_frame:
            raise ValueError("Frames must be processed in increasing order")
        self.last_frame = frame_number
        self.states = {k: v for k, v in self.states.items() if frame_number - v.last_seen <= self.max_gap}
        events = []
        seen = set()
        for obj in objects:
            track_id = obj.get("track_id")
            if track_id is None or track_id in self.counted or track_id in seen:
                continue
            seen.add(track_id)
            x1, _, x2, y2 = obj["bbox"]
            point = ((x1 + x2) / 2, y2)
            distance = self.distance(point)
            previous = self.states.get(track_id)
            if previous is not None:
                previous.last_seen = frame_number
                previous.votes[obj["class_id"]] += 1
            # Inside the dead band, keep the last stable position, but track visibility.
            if abs(distance) <= self.margin:
                continue
            if previous is None:
                self.states[track_id] = TrackState(point, distance, frame_number, Counter({obj["class_id"]: 1}))
                continue
            class_id = previous.votes.most_common(1)[0][0]
            if previous.distance * distance < 0:
                fraction = previous.distance / (previous.distance - distance)
                intersection = tuple(previous.point[i] + fraction * (point[i] - previous.point[i]) for i in (0, 1))
                projection = ((intersection[0] - self.start[0]) * self.dx + (intersection[1] - self.start[1]) * self.dy) / self.length**2
                if 0 <= projection <= 1 and class_id in self.classes:
                    from .config import CLASSES
                    direction = "negative_to_positive" if distance > 0 else "positive_to_negative"
                    name = CLASSES[class_id]
                    events.append(dict(track_id=track_id, class_name=name, frame_number=frame_number,
                                       timestamp_seconds=round(timestamp, 4), direction=direction))
                    self.counted.add(track_id)
                    self.counts[(name, direction)] += 1
            previous.point, previous.distance = point, distance
        return events

    def summary(self):
        totals = {}
        for (name, direction), count in sorted(self.counts.items()):
            totals.setdefault(name, {})[direction] = count
        return {"total": sum(self.counts.values()), "by_class_and_direction": totals}
