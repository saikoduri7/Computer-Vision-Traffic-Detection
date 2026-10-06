
"""Generate the committed Colab notebook; --check verifies it is up to date."""
import argparse
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def cell(kind, source):
    body = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip().splitlines(keepends=True),
                id=f"traffic-{len(CELLS):02d}")
    if kind == "code":
        body.update(execution_count=None, outputs=[])
    CELLS.append(body)


cell("markdown", '''
# Traffic Detection, Tracking & Counting

Detect, track, and count vehicles in fixed-camera video using YOLO11, ByteTrack, and OpenCV.
''')
cell("markdown", '''
## Project Setup

Load the project from GitHub or a ZIP and install its dependencies.
''')
cell("code", '''
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

workspace = Path(tempfile.mkdtemp(prefix="traffic-project-", dir="/content"))
repo_url = "https://github.com/saikoduri7/Computer-Vision-Traffic-Detection.git"
if repo_url:
    if not repo_url.startswith("https://github.com/") or any(c.isspace() for c in repo_url):
        raise ValueError("Use an https://github.com/OWNER/REPOSITORY URL")
    REPO = workspace / "repository"
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(REPO)], check=True)
else:
    from google.colab import files
    uploaded = files.upload()
    archives = [name for name in uploaded if name.lower().endswith(".zip")]
    if len(archives) != 1:
        raise ValueError("Upload exactly one project ZIP")
    archive = Path(archives[0])
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (workspace / entry.filename).resolve()
            if not target.is_relative_to(workspace.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(workspace)
    candidates = [p.parent for p in workspace.rglob("pyproject.toml")
                  if (p.parent / "src/traffic_detection/pipeline.py").is_file()]
    if len(candidates) != 1:
        raise ValueError("Could not identify the traffic-detection project in the ZIP")
    REPO = candidates[0]

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
# Makes src visible immediately after an editable install in a running kernel.
sys.path.insert(0, str(REPO / "src"))
import cv2
import torch
from traffic_detection import load_config
from traffic_detection.pipeline import run_video
from traffic_detection.video import preview_frame, browser_video
from traffic_detection.visualization import draw_frame
from traffic_detection.detector import Detector
from google.colab.patches import cv2_imshow

print("Project:", REPO)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
''')
cell("markdown", '''
## Video Input

Connect Google Drive, select your video with `VIDEO_PATH`, and preview its first frame.
''')
cell("code", '''
from google.colab import drive
import shutil

drive.mount("/content/drive")
PROJECT = Path("/content/drive/MyDrive/traffic-detection")
VIDEO_PATH = PROJECT / "data/videos/traffic.mp4"
if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}. Copy its actual path from the Files panel.")

local_input_dir = Path(tempfile.mkdtemp(prefix="traffic-input-", dir="/content"))
LOCAL_VIDEO = local_input_dir / VIDEO_PATH.name
shutil.copy2(VIDEO_PATH, LOCAL_VIDEO)
frame = preview_frame(LOCAL_VIDEO)
print("Original frame size:", frame.shape[1], "x", frame.shape[0])
cv2_imshow(frame)
''')
cell("markdown", '''
## Counting Line Configuration

Set the confidence threshold and preview the counting line using coordinates between 0 and 1.
''')
cell("code", '''
config = load_config(REPO / "configs/default.yaml")
config.line_start = [0.1, 0.6]
config.line_end = [0.9, 0.6]
config.confidence = 0.25

(PROJECT / "models").mkdir(parents=True, exist_ok=True)
config.model = str(PROJECT / "models/yolo11n.pt")
config.validate()
line = config.pixel_line(frame.shape[1], frame.shape[0])
cv2_imshow(draw_frame(frame, [], line))
''')
cell("markdown", '''
## Object Detection

Run the pretrained detector on the first frame and display its labels and confidence scores.
''')
cell("code", '''
detector = Detector(config)
objects = detector.detect(frame)
cv2_imshow(draw_frame(frame, objects, line))
print([(item["class_name"], item["confidence"]) for item in objects])
del detector
if torch.cuda.is_available():
    torch.cuda.empty_cache()
''')
cell("markdown", '''
## Video Tracking and Counting

Process `MAX_FRAMES` frames (or the full video with `None`) and save completed results to Drive.
''')
cell("code", '''
from datetime import datetime, timezone
import uuid

MAX_FRAMES = 100  # Change to None for the full video.
config.max_frames = MAX_FRAMES
config.validate()
run_name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
RUN_DIR = Path("/content/traffic-runs") / run_name
summary = run_video(LOCAL_VIDEO, RUN_DIR, config,
                    progress=lambda n: print(f"Processed {n} frames", flush=True))
DRIVE_RUN = PROJECT / "outputs" / run_name
shutil.copytree(RUN_DIR, DRIVE_RUN)
print("Saved to:", DRIVE_RUN)
print("Counts:", summary["counts"])
print("Processing FPS:", summary["processing_fps"])
for warning in summary["warnings"]:
    print("Warning:", warning)
''')
cell("markdown", '''
## Video Preview

Convert the annotated video to browser-compatible H.264 and preview it when under 20 MB.
''')
cell("code", '''
from IPython.display import Video, display

browser_path = RUN_DIR / "browser.mp4"
if not browser_path.exists():
    browser_video(RUN_DIR / "annotated.mp4", browser_path)
drive_browser = DRIVE_RUN / "browser.mp4"
if not drive_browser.exists():
    shutil.copy2(browser_path, drive_browser)
if browser_path.stat().st_size <= 20 * 1024 * 1024:
    display(Video(str(browser_path), embed=True, width=800))
else:
    print("Video is larger than 20 MB. Open it in Drive:", drive_browser)
''')
cell("markdown", '''
## Count Evaluation

Inspect crossing events and optionally compare the predicted total with your manual count.
''')
cell("code", '''
import csv
import json

with (RUN_DIR / "crossings.csv").open() as handle:
    events = list(csv.DictReader(handle))
print(json.dumps(summary["counts"], indent=2))
print("First 10 crossing events:")
for event in events[:10]:
    print(event)

MANUAL_COUNT = None  # Enter your independently counted total for this exact run.
if MANUAL_COUNT is not None:
    predicted = summary["counts"]["total"]
    error = abs(predicted - MANUAL_COUNT)
    print("Absolute error:", error)
    print("Relative error:", f"{100 * error / MANUAL_COUNT:.1f}%" if MANUAL_COUNT > 0 else "undefined (manual count is zero)")
''')
cell("markdown", '''
## Results Export

Download a ZIP of the preview, crossing events, configuration, and summary for GitHub.
''')
cell("code", '''
import yaml
from google.colab import files

example_dir = Path(tempfile.mkdtemp(prefix="traffic-example-", dir="/content"))
shutil.copy2(RUN_DIR / "preview.jpg", example_dir / "detection_preview.jpg")
shutil.copy2(RUN_DIR / "crossings.csv", example_dir / "crossings.csv")
public_summary = json.loads(json.dumps(summary))
public_summary["source"] = VIDEO_PATH.name
public_summary["config"]["model"] = Path(config.model).name
(example_dir / "summary.json").write_text(json.dumps(public_summary, indent=2))
(example_dir / "config.yaml").write_text(yaml.safe_dump(public_summary["config"]))
archive = shutil.make_archive(str(example_dir), "zip", example_dir)
files.download(archive)
''')
cell("markdown", '''
## GitHub Publishing

Save a copy of the notebook to GitHub after clearing large outputs, and commit supporting files separately.
''')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebook = dict(cells=CELLS, metadata={"colab": {"name": "traffic_detection_colab.ipynb"},
                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                    "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)
    path = ROOT / "notebooks/traffic_detection_colab.ipynb"
    content = json.dumps(notebook, indent=2) + "\n"
    if args.check:
        import nbformat
        nbformat.validate(notebook)
        if not path.is_file() or path.read_text() != content:
            raise SystemExit("Notebook differs from generator; run python scripts/build_notebook.py")
        for item in CELLS:
            if item["cell_type"] == "code":
                compile("".join(item["source"]), "<notebook>", "exec")
        print("Notebook matches generator; all code cells compile.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
