import pytest

from ok.ui.qt.video_template.recorder import VideoTemplateRecorder, mouse
from ok.ui.qt.video_template.store import VideoTemplateStore


def make_active_recorder(tmp_path, monkeypatch, times, points):
    recorder = VideoTemplateRecorder(store=VideoTemplateStore(root=tmp_path))
    recorder.is_recording = True
    recorder.started_at = times[0]
    time_values = iter(times[1:])
    point_values = iter(points)
    monkeypatch.setattr("ok.ui.qt.video_template.recorder.time.time", lambda: next(time_values))
    monkeypatch.setattr(recorder, "_relative_coords", lambda *_: next(point_values))
    return recorder


def test_window_focus_change_does_not_drop_in_flight_click(tmp_path, monkeypatch):
    recorder = make_active_recorder(
        tmp_path,
        monkeypatch,
        times=[100, 100.0, 100.2],
        points=[(0.4, 0.5), (0.4, 0.5)],
    )

    recorder._on_click(400, 500, mouse.Button.left, True)
    recorder._on_window(False)
    recorder._on_click(400, 500, mouse.Button.left, False)

    assert recorder.event_count() == 1
    assert recorder.events[0]["type"] == "click"
    assert recorder.events[0]["down_time"] == pytest.approx(0.2)


def test_release_without_down_is_recovered_as_click(tmp_path, monkeypatch):
    recorder = make_active_recorder(
        tmp_path,
        monkeypatch,
        times=[100, 100.3],
        points=[(0.3, 0.4)],
    )

    recorder._on_click(300, 400, mouse.Button.left, False)

    assert recorder.event_count() == 1
    assert recorder.events[0]["type"] == "click"
    assert recorder.events[0]["recovered"] == "missing_down"
    assert recorder.recovered_event_count == 1


def test_edge_points_are_clamped_instead_of_dropped(tmp_path, monkeypatch):
    recorder = make_active_recorder(
        tmp_path,
        monkeypatch,
        times=[100, 100.0, 100.1],
        points=[(-0.01, 1.02), (-0.015, 1.025)],
    )

    recorder._on_click(-10, 1020, mouse.Button.left, True)
    recorder._on_click(-15, 1025, mouse.Button.left, False)

    assert recorder.event_count() == 1
    assert recorder.events[0]["type"] == "click"
    assert recorder.events[0]["x"] == 0
    assert recorder.events[0]["y"] == 1


def test_far_outside_release_finishes_existing_down(tmp_path, monkeypatch):
    recorder = make_active_recorder(
        tmp_path,
        monkeypatch,
        times=[100, 100.0, 100.1],
        points=[(0.5, 0.5), (1.5, 1.5)],
    )

    recorder._on_click(500, 500, mouse.Button.left, True)
    recorder._on_click(1500, 1500, mouse.Button.left, False)

    assert recorder.event_count() == 1
    assert recorder.events[0]["type"] == "click"
    assert recorder.events[0]["x"] == 0.5
    assert recorder.events[0]["y"] == 0.5
    assert recorder.current_down == {}


def test_polled_input_transitions_record_click(tmp_path, monkeypatch):
    recorder = make_active_recorder(
        tmp_path,
        monkeypatch,
        times=[100, 100.0, 100.05, 100.1],
        points=[(0.5, 0.6), (0.5, 0.6), (0.5, 0.6)],
    )
    states = iter([
        {"left": True, "right": False, "middle": False},
        {"left": False, "right": False, "middle": False},
    ])
    monkeypatch.setattr("ok.ui.qt.video_template.recorder.win32gui.GetCursorPos", lambda: (500, 600))
    monkeypatch.setattr(recorder, "_read_polled_buttons", lambda: next(states))
    recorder.polled_buttons = {"left": False, "right": False, "middle": False}

    recorder._poll_input_once()
    recorder._poll_input_once()

    assert recorder.event_count() == 1
    assert recorder.events[0]["type"] == "click"
    assert recorder.events[0]["x"] == 0.5
    assert recorder.events[0]["y"] == 0.6
