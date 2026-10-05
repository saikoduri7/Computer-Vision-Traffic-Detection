import cv2


def draw_frame(frame, objects, line=None, total=0):
    output = frame.copy()
    height, width = frame.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = [round(v) for v in obj["bbox"]]
        color = (90, 220, 90) if obj.get("track_id") is not None else (0, 190, 255)
        track = f"#{obj['track_id']} " if obj.get("track_id") is not None else ""
        label = f"{track}{obj['class_name']} {obj['confidence']:.0%}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        tx = max(0, min(x1, width - text_width - 2))
        ty = max(18, min(height - 4, y1 - 7))
        cv2.putText(output, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if line is not None:
        start, end = [tuple(round(v) for v in point) for point in line]
        cv2.arrowedLine(output, start, end, (255, 190, 0), 3, tipLength=0.03)
        cv2.rectangle(output, (0, 0), (min(width - 1, 330), 38), (25, 25, 25), -1)
        cv2.putText(output, f"Vehicle crossings: {total}", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output
