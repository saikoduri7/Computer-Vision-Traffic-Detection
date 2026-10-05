import pytest

from traffic_detection.config import Config, load_config


@pytest.mark.parametrize("settings", [
    {"confidence": 0}, {"confidence": float("nan")}, {"confidence": 2},
    {"line_start": [1.1, .5]}, {"line_start": [.1, .6], "line_end": [.1, .6]},
    {"max_frames": 0}, {"image_size": 1.5}, {"max_gap_frames": -1},
    {"count_classes": [80]}, {"detect_classes": [2], "count_classes": [7]},
    {"line_margin_px": -1}, {"save_predictions": "false"},
])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        Config(**settings).validate()


def test_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.pixel_line(101, 101) == ((10, 60), (90, 60))


def test_unknown_fields(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("confidnce: 0.5")
    with pytest.raises(ValueError):
        load_config(path)
