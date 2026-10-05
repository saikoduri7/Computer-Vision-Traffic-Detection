# Validation and results

## What is established

Automated tests exercise line crossings, directions, finite endpoints, jitter,
missing IDs, excluded people, gaps, duplicate prevention, configuration, and a
synthetic video through the output pipeline. Tests do not establish real-road
accuracy. The synthetic example is explicitly labeled in `examples/README.md`.

Local verification on October 5, 2026:

- 28 automated tests passed.
- Notebook format validated; all code cells compiled; generated notebook matched its source.
- YOLO11 nano detected a bus and people in the
  [Ultralytics sample image](https://ultralytics.com/images/bus.jpg).
- An eight-frame video made by repeating that image ran through actual ByteTrack
  inference, produced confirmed track IDs, and correctly produced zero stationary crossings.
- Test environment: Python 3.12, CPU, Ultralytics 8.4.173, OpenCV 4.14.0.94, lap 0.5.13.

The repeated-image check validates model integration, not tracking accuracy on
moving traffic. Drive authorization and Colab's browser preview still need a run
in your own Colab session.

## Your traffic-video evaluation

No accuracy claim is made until you run and manually review your video.
Fill in the following after completing the notebook:

| Clip/source | Frames reviewed | Manual count | Predicted count | Absolute error |
|---|---:|---:|---:|---:|
| Your first clip | — | — | — | — |
| A different evaluation clip | — | — | — | — |

Use the same counting line, direction, vehicle classes, time interval, and
once-per-vehicle convention for manual and predicted counts. Absolute error is
`abs(predicted - manual)`. When manual count is positive, relative error is
`100 * abs(predicted - manual) / manual`. A zero manual count has undefined
relative error; report absolute error instead.

Record hardware, package versions, confidence, line position, video source,
lighting, and observed failures. Processing FPS in the summary includes model
initialization and output work, so it is not a pure inference benchmark.

## Limitations

- Fixed camera expected; camera movement invalidates a static counting line.
- Crowding, tiny vehicles, darkness, occlusion, and class confusion reduce reliability.
- Lost or changed IDs can create missed or duplicate counts.
- No lane segmentation, speed calibration, traffic-light state, or violations.
- Colab runtime files are temporary; copy completed runs to Drive.
- A successful local smoke test is not a verified run in your Colab account.
