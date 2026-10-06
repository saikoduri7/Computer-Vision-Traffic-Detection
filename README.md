


# Traffic Detection, Tracking & Counting

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saikoduri7/Computer-Vision-Traffic-Detection/blob/main/notebooks/traffic_detection_colab.ipynb)

A Google Colab project that reads a fixed-camera traffic video, detects people
and common vehicles, tracks their IDs, and counts vehicles crossing a configured
line. Built with OpenCV, pretrained YOLO11, and ByteTrack.

## Demo

*(https://github.com/user-attachments/assets/2284ca47-e9ed-4aa4-b78d-ead7cd74a04b
)*

## Run in Colab

1. Open `notebooks/traffic_detection_colab.ipynb` in
   [Google Colab](https://colab.research.google.com/) using **File → Open notebook → GitHub**
   and paste this repository's URL. Alternatively upload the notebook file.
2. Select a GPU runtime if available; CPU is supported.
3. Run the setup cell; this repository's URL is already configured. If you have
   the ZIP instead, set `repo_url = ""` and upload the ZIP in the setup cell.
4. Mount Drive. Select your video, normally
   `/content/drive/MyDrive/traffic-detection/data/videos/traffic.mp4`.
5. Preview the first frame, adjust the counting line, and run a short check.
6. Run the full video, export to Drive, preview the result, and inspect counts.

The notebook contains all setup steps. It imports reusable code from `src/`,
so you need the repository or ZIP as well as the notebook. A private repository
can be used through the ZIP option without putting a token in a notebook.

## Features

- Detects person, bicycle, car, motorcycle, bus, truck.
- Counts bicycles and motor vehicles by default; people remain visible but uncounted.
- Uses normalized line coordinates, persistent track IDs, a dead band, and finite-segment crossing checks.
- Exports annotated MP4, preview JPEG, per-frame JSONL, crossing CSV, and a JSON run summary.
- Streams frames; protects existing run directories; supports CPU or CUDA.

This is a prototype for uploaded videos, not a continuously deployed traffic service.
It does not estimate speed, detect lanes, classify signal colors, or infer violations.

## Repository layout

```text
notebooks/                  Colab setup, configuration, execution, preview
src/traffic_detection/      detector, tracker, counter, video I/O, pipeline, CLI
configs/default.yaml        model, classes, line geometry, run options
tests/                      deterministic geometry/config/video tests
docs/architecture.md        how the modules connect
docs/results.md             evaluation instructions and limitations
examples/                   explicitly synthetic sample outputs
data/README.md              input data instructions and sources
models/README.md            pretrained model provenance
scripts/build_notebook.py   reproducible notebook generation/validation
```

## Configuration

Line coordinates range from 0 to 1 relative to the original frame. Example:
`line_start: [0.1, 0.6]` and `line_end: [0.9, 0.6]` draws a horizontal line 60%
down the image. Adjust it to each camera. For a left-to-right line,
`negative_to_positive` means above-to-below; the reverse direction means below-to-above.

A track counts at most once per run. Counts are estimated crossings, not the
number of detections and not a guaranteed count of unique physical vehicles.

## Optional local/terminal usage

```bash
python -m pip install -e '.[dev]'
traffic-detect --input path/to/traffic.mp4 --output outputs/run_001 --config configs/default.yaml
python -m pytest -q
```

No custom training is required. Weights download on first inference.
Colab's preinstalled PyTorch is used; package versions are saved in each summary.

## Outputs and reproducibility

Each run has `annotated.mp4`, `preview.jpg`, `crossings.csv`, `summary.json`, and
optionally `predictions.jsonl`. The notebook can also create `browser.mp4`.
Frame numbering starts at zero. Input should have constant FPS. Output is silent.
The summary records the configuration, versions, source, counts, and processing FPS.

When finished, save a copy of the notebook to GitHub and commit supporting source
files separately. Clear embedded video outputs before saving notebooks.
Keep datasets, videos, weights, credentials, and full predictions out of Git.
Only small, intentional examples belong in `examples/`.

## Verification

Run `pytest -q` and `python scripts/build_notebook.py --check`. CI runs both.
Tests use synthetic tracks and video and require no model download. For actual
accuracy, manually count a clip and report the comparison in `docs/results.md`.

## Troubleshooting

- **Video not found:** copy the actual file path from Colab's Files panel; filenames are case-sensitive.
- **Run directory exists:** choose a new directory; the notebook creates unique names automatically.
- **No GPU:** CPU works, more slowly. The notebook selects CUDA only when available.
- **MP4 will not play:** use the notebook's H.264 conversion cell or download the original.
- **Counts are zero:** check the line crosses vehicle paths and that confirmed IDs are visible.
- **Too many counts:** inspect ID switches and class errors; reposition the line away from occlusion.
- **Interrupted runtime:** rerun setup and mount Drive; inputs and completed exports persist there.

## Sources and licensing

- [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)

Project code is licensed under AGPL-3.0 (see `LICENSE`), matching the open-source
Ultralytics integration. Third-party code, pretrained weights, and input media
retain their own applicable terms. See [Ultralytics licensing](https://www.ultralytics.com/license).
