# Input videos

Opening the notebook from GitHub does not share anyone's Google Drive: each
runner supplies a video from their own Drive and sets `VIDEO_PATH` accordingly.

In Google Drive, use `MyDrive/traffic-detection/data/videos/traffic.mp4`.
Any other path can be selected in the notebook. No dataset download is required
to use the pretrained model. Use a short fixed-camera clip with visible traffic.

For public data, see the [AI City dataset access page](https://www.aicitychallenge.org/ai-city-challenge-dataset-access/).
Select a vehicle-counting or tracking dataset, follow its terms, and document the
specific clip and source in `docs/results.md`. Video licenses are independent
of this repository's code license. Do not commit large videos to Git.
