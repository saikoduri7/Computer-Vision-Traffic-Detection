"""One fresh YOLO/ByteTrack instance per video prevents state leaking between runs."""
from .detector import Detector


class Tracker(Detector):
    def update(self, frame):
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **self.options())[0]
        return self.records(result)
