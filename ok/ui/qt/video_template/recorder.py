import math
import os
import threading
import time

import cv2
import win32api
import win32con
import win32gui
from pynput import mouse

from ok import og
from ok.ui.qt.Communicate import communicate
from ok.ui.qt.video_template.store import VideoTemplateStore
from ok.util.logger import Logger


logger = Logger.get_logger(__name__)


class VideoTemplateRecorder:
    """Records clean capture frames plus click/drag facts into a package."""

    EDGE_TOLERANCE = 0.03
    CLICK_DISTANCE = 0.02
    CLICK_DURATION = 0.6
    MOVE_DISTANCE = 0.01
    INPUT_POLL_INTERVAL = 0.01

    def __init__(self, store=None, fps=10):
        self.store = store or VideoTemplateStore()
        self.fps = fps
        self.package = None
        self.is_recording = False
        self.events = []
        self.frame_count = 0
        self.started_at = 0
        self.mouse_listener = None
        self.record_thread = None
        self.input_thread = None
        self.writer = None
        self.target_hwnd = None
        self.current_down = {}
        self.polled_buttons = {}
        self.discarded_event_count = 0
        self.recovered_event_count = 0
        self._lock = threading.Lock()

    def start(self, preview_settings=None):
        if self.is_recording:
            return self.package
        capture_method = getattr(og.device_manager, "capture_method", None)
        if capture_method is None:
            raise RuntimeError("No capture method available. Please start capture first.")
        frame = capture_method.get_frame()
        if frame is None:
            raise RuntimeError("Failed to capture the first frame.")

        height, width = frame.shape[:2]
        self.package = self.store.create_package({
            "width": width,
            "height": height,
            "fps": self.fps,
            "preview_settings": preview_settings or {},
            "capture_method": str(getattr(getattr(og.device_manager, "config", {}), "get", lambda *_: "")("capture", "")),
        })
        self.events = []
        self.frame_count = 0
        self.started_at = time.time()
        self.is_recording = True
        self.target_hwnd = getattr(getattr(og.device_manager, "hwnd_window", None), "hwnd", None)
        self.current_down = {}
        self.polled_buttons = self._read_polled_buttons()
        self.discarded_event_count = 0
        self.recovered_event_count = 0

        communicate.window.connect(self._on_window)
        self.input_thread = threading.Thread(target=self._poll_input_loop, name="VideoTemplateInputRecorder", daemon=True)
        self.input_thread.start()
        self.record_thread = threading.Thread(target=self._record_loop, name="VideoTemplateRecorder", daemon=True)
        self.record_thread.start()
        return self.package

    def stop(self):
        if not self.is_recording:
            return self.package
        self.is_recording = False
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.input_thread:
            self.input_thread.join(timeout=1)
            self.input_thread = None
        try:
            communicate.window.disconnect(self._on_window)
        except Exception:
            pass
        if self.record_thread:
            self.record_thread.join(timeout=3)
            self.record_thread = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.current_down:
            self.current_down = {}

        duration = max(0, time.time() - self.started_at)
        with self._lock:
            events = list(self.events)
        self.package.save_events(events)
        self.package.update_meta(duration_seconds=duration, frame_count=self.frame_count,
                                 event_count=len(events), stopped_at=time.time(),
                                 discarded_event_count=self.discarded_event_count,
                                 recovered_event_count=self.recovered_event_count)
        return self.package

    def elapsed_seconds(self):
        if not self.started_at:
            return 0
        return max(0, time.time() - self.started_at)

    def event_count(self):
        with self._lock:
            return len(self.events)

    def _record_loop(self):
        capture_method = getattr(og.device_manager, "capture_method", None)
        frame_interval = 1 / max(1, self.fps)
        while self.is_recording:
            start = time.time()
            try:
                frame = capture_method.get_frame() if capture_method else None
                if frame is not None:
                    self._write_frame(frame)
            except Exception as e:
                logger.error(f"video template record frame failed: {e}")
            sleep_time = frame_interval - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _poll_input_loop(self):
        while self.is_recording:
            try:
                self._poll_input_once()
            except Exception as e:
                logger.error("video template poll input failed", e)
            time.sleep(self.INPUT_POLL_INTERVAL)

    def _poll_input_once(self):
        x, y = win32gui.GetCursorPos()
        current = self._read_polled_buttons()
        for button_text, pressed in current.items():
            was_pressed = self.polled_buttons.get(button_text, False)
            if pressed != was_pressed:
                self._handle_click(x, y, button_text, pressed)
        self.polled_buttons = current
        if any(current.values()):
            self._handle_move(x, y)

    @staticmethod
    def _read_polled_buttons():
        return {
            "left": bool(win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000),
            "right": bool(win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & 0x8000),
            "middle": bool(win32api.GetAsyncKeyState(win32con.VK_MBUTTON) & 0x8000),
        }

    def _write_frame(self, frame):
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.package.recording_path, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.package.recording_path}")
        self.writer.write(frame)
        self.frame_count += 1

    def _on_window(self, visible, *args):
        # The signal reflects foreground state. Recording must keep in-flight
        # mouse state while popups or overlays briefly steal focus.
        pass

    def _on_click(self, x, y, button, pressed):
        try:
            self._handle_click(x, y, button, pressed)
        except Exception as e:
            logger.error("video template record click failed", e)

    def _on_move(self, x, y):
        try:
            self._handle_move(x, y)
        except Exception as e:
            logger.error("video template record move failed", e)

    def _handle_click(self, x, y, button, pressed):
        if not self.is_recording:
            return
        rel_x, rel_y = self._relative_coords(x, y)
        point = self._capture_point(rel_x, rel_y)
        event_time = time.time() - self.started_at
        button_text = self._button_text(button)
        if point is None:
            if pressed:
                self._track_discarded_event()
                return
            down = self.current_down.pop(button_text, None)
            if not down:
                self._track_discarded_event()
                return
            last_point = (down.get("path") or [down])[-1]
            self._record_completed_pointer_event(down, event_time, last_point["x"], last_point["y"], button_text)
            return
        rel_x, rel_y = point
        if pressed:
            self.current_down[button_text] = {
                "time": event_time,
                "screen_x": x,
                "screen_y": y,
                "x": rel_x,
                "y": rel_y,
                "button": button_text,
                "path": [{"time": event_time, "x": rel_x, "y": rel_y}],
            }
            return

        down = self.current_down.pop(button_text, None)
        if not down:
            self._record_recovered_click(event_time, rel_x, rel_y, button_text)
            return
        self._record_completed_pointer_event(down, event_time, rel_x, rel_y, button_text)

    def _handle_move(self, x, y):
        if not self.is_recording or not self.current_down:
            return
        rel_x, rel_y = self._relative_coords(x, y)
        point = self._capture_point(rel_x, rel_y)
        if point is None:
            return
        rel_x, rel_y = point
        event_time = time.time() - self.started_at
        for down in self.current_down.values():
            path = down.setdefault("path", [])
            if path and math.hypot(rel_x - path[-1]["x"], rel_y - path[-1]["y"]) < self.MOVE_DISTANCE:
                continue
            path.append({"time": event_time, "x": rel_x, "y": rel_y})

    def _record_recovered_click(self, event_time, rel_x, rel_y, button_text):
        with self._lock:
            event_id = f"event_{len(self.events) + 1:04d}"
            self.events.append({
                "id": event_id,
                "type": "click",
                "time": event_time,
                "x": rel_x,
                "y": rel_y,
                "button": button_text,
                "down_time": 0,
                "recovered": "missing_down",
            })
            self.recovered_event_count += 1

    def _record_completed_pointer_event(self, down, event_time, rel_x, rel_y, button_text):
        distance = math.hypot(rel_x - down["x"], rel_y - down["y"])
        duration = max(0, event_time - down["time"])
        with self._lock:
            event_id = f"event_{len(self.events) + 1:04d}"
            if distance < self.CLICK_DISTANCE and duration < self.CLICK_DURATION:
                self.events.append({
                    "id": event_id,
                    "type": "click",
                    "time": down["time"],
                    "x": down["x"],
                    "y": down["y"],
                    "button": button_text,
                    "down_time": duration,
                })
            else:
                path = list(down.get("path") or [])
                if not path or path[-1].get("time") != event_time:
                    path.append({"time": event_time, "x": rel_x, "y": rel_y})
                self.events.append({
                    "id": event_id,
                    "type": "drag",
                    "time": down["time"],
                    "end_time": event_time,
                    "from_x": down["x"],
                    "from_y": down["y"],
                    "to_x": rel_x,
                    "to_y": rel_y,
                    "button": button_text,
                    "duration": duration,
                    "path": path,
                })

    def _track_discarded_event(self):
        with self._lock:
            self.discarded_event_count += 1

    def _relative_coords(self, x, y):
        try:
            hwnd_window = getattr(og.device_manager, "hwnd_window", None)
            if hwnd_window is not None and hwnd_window.hwnd == self.target_hwnd:
                rel_x = x - (hwnd_window.x + hwnd_window.real_x_offset)
                rel_y = y - (hwnd_window.y + hwnd_window.real_y_offset)
                width = hwnd_window.real_width or hwnd_window.width
                height = hwnd_window.real_height or hwnd_window.height
                return self._normalize(rel_x, rel_y, width, height)
            if self.target_hwnd:
                client_pos = win32gui.ClientToScreen(self.target_hwnd, (0, 0))
                left, top, right, bottom = win32gui.GetClientRect(self.target_hwnd)
                return self._normalize(x - client_pos[0], y - client_pos[1], right - left, bottom - top)
        except Exception:
            pass
        return self._normalize(x, y, getattr(og.device_manager, "width", 0), getattr(og.device_manager, "height", 0))

    @staticmethod
    def _normalize(x, y, width, height):
        if width > 0 and height > 0:
            return x / width, y / height
        return x, y

    @staticmethod
    def _inside_capture(x, y):
        return 0 <= x <= 1 and 0 <= y <= 1

    def _capture_point(self, x, y):
        if self._inside_capture(x, y):
            return x, y
        if -self.EDGE_TOLERANCE <= x <= 1 + self.EDGE_TOLERANCE and \
                -self.EDGE_TOLERANCE <= y <= 1 + self.EDGE_TOLERANCE:
            return min(1, max(0, x)), min(1, max(0, y))
        return None

    @staticmethod
    def _button_text(button):
        if isinstance(button, str):
            return button
        if button == mouse.Button.right:
            return "right"
        if button == mouse.Button.middle:
            return "middle"
        return "left"
