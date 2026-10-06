import pytest

from traffic_detection.counter import LineCounter


def obj(x, y, track_id=1, class_id=2):
    return dict(track_id=track_id, class_id=class_id, bbox=[x - 2, y - 5, x + 2, y])


def counter():
    return LineCounter((10, 50), (90, 50), [1, 2, 3, 5, 7], margin=3, max_gap=3)


def test_counts_once_with_direction_and_class():
    c = counter()
    assert c.update([obj(50, 40)], 0, 0) == []
    events = c.update([obj(50, 60)], 1, .1)
    assert events[0]["direction"] == "negative_to_positive"
    assert events[0]["class_name"] == "car"
    assert c.update([obj(50, 40)], 2, .2) == []
    assert c.summary()["total"] == 1


def test_opposite_direction():
    c = counter()
    c.update([obj(50, 60)], 0, 0)
    assert c.update([obj(50, 40)], 1, .1)[0]["direction"] == "positive_to_negative"


def test_jitter_and_dead_band():
    c = counter()
    for index, y in enumerate([40, 49, 51, 48, 40]):
        assert c.update([obj(50, y)], index, index / 10) == []
    assert c.summary()["total"] == 0


def test_crossing_through_dead_band():
    c = counter()
    c.update([obj(50, 40)], 0, 0)
    c.update([obj(50, 50)], 1, .1)
    assert len(c.update([obj(50, 60)], 2, .2)) == 1


@pytest.mark.parametrize("x", [0, 100])
def test_outside_endpoints_does_not_count(x):
    c = counter()
    c.update([obj(x, 40)], 0, 0)
    assert c.update([obj(x, 60)], 1, .1) == []


def test_diagonal_line():
    c = LineCounter((10, 10), (90, 90), [2], margin=1)
    c.update([obj(50, 40)], 0, 0)
    assert len(c.update([obj(50, 60)], 1, .1)) == 1


def test_missing_id_and_people_are_not_counted():
    c = counter()
    c.update([obj(50, 40, None), obj(50, 40, 2, 0)], 0, 0)
    assert c.update([obj(50, 60, None), obj(50, 60, 2, 0)], 1, .1) == []


def test_gap_does_not_invent_crossing():
    c = counter()
    c.update([obj(50, 40)], 0, 0)
    assert c.update([obj(50, 60)], 5, .5) == []


def test_empty_frames_and_unique_ids():
    c = counter()
    c.update([obj(50, 40, 1), obj(60, 40, 2)], 0, 0)
    c.update([], 1, .1)
    assert len(c.update([obj(50, 60, 1), obj(60, 60, 2)], 2, .2)) == 2


def test_frames_must_increase():
    c = counter()
    c.update([], 0, 0)
    with pytest.raises(ValueError):
        c.update([], 0, 0)
