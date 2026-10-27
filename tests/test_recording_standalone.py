import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QPushButton

from ok import Config, og
from ok.core.template_store import CocoTemplateStore
from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage, VideoTemplateTab
from ok.ui.qt.video_template.ocr_processor import VideoTemplateOcrProcessor, confirm_frame
from ok.ui.qt.video_template.recorder import VideoTemplateRecorder
from ok.ui.qt.video_template.store import VideoTemplatePackage, VideoTemplateStore


def test_recording_editor_starts_without_flow_or_tree_modules(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Config, "config_folder", str(tmp_path / "configs"))
    monkeypatch.setattr(og, "config", {})
    monkeypatch.setattr(og, "app", SimpleNamespace(icon=QIcon()))
    monkeypatch.setattr(og, "device_manager", SimpleNamespace(capture_method=None))
    monkeypatch.setattr(og, "feature_set", None, raising=False)
    tab = VideoTemplateTab()
    try:
        tab.resize(1280, 800)
        tab.show()
        app.processEvents()
        assert tab.tabs.count() == 2
        assert tab.tabs.tabText(0) == "录屏"
        assert tab.edit_page.stack.count() == 2
        assert not tab.grab().isNull()
        assert all("加工为可视化流程" not in button.text() for button in tab.findChildren(QPushButton))
        assert not any("behavior_tree" in name or "visual_flow" in name for name in sys.modules)
    finally:
        for timer in tab.findChildren(QTimer):
            timer.stop()
        tab.close()
        tab.deleteLater()
        app.processEvents()


def test_recorded_video_keyframes_and_annotations_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coco_path = tmp_path / "模板" / "annotations.json"
    monkeypatch.setattr(og, "config", {"template_matching": {"coco_feature_json": str(coco_path)}})
    ocr_result = SimpleNamespace(
        boxes=[[[35, 20], [65, 20], [65, 40], [35, 40]]],
        txts=["录屏按钮"], scores=[0.99],
    )
    executor = SimpleNamespace(
        config={"ocr": {"default": {"lib": "rapidocr"}}},
        ocr_lib=lambda _name: lambda _image, **_kwargs: ocr_result,
        ocr_po_translation=None, text_fix={},
    )
    monkeypatch.setattr(og, "executor", executor)
    store = VideoTemplateStore(tmp_path / "recordings")
    recorder = VideoTemplateRecorder(store=store, fps=10)
    recorder.package = store.create_package({"width": 100, "height": 60, "fps": 10})
    recorder.is_recording = True
    recorder.started_at = time.time() - 0.6
    recorder.events = [{"id": "event_0001", "type": "click", "time": 0.2, "x": 0.5, "y": 0.5}]
    frame = np.zeros((60, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (35, 20), (65, 40), (240, 180, 70), thickness=-1)
    try:
        for _ in range(6):
            recorder._write_frame(frame)
    finally:
        recorder.stop()
    package = VideoTemplatePackage(recorder.package.folder)
    assert package.load_meta()["frame_count"] == 6
    assert package.load_events()[0]["type"] == "click"

    frames = VideoTemplateOcrProcessor(package).process_all_operation_frames()
    assert len(frames) == 2
    assert frames[0]["ocr_text"] == "录屏按钮"
    confirm_frame(package, frames[0]["id"], template_name="录屏按钮")
    manifest = package.export_confirmed()
    assert len(manifest["frames"]) == 1
    assert Path(package.folder, manifest["frames"][0]["exported_image_file"]).is_file()

    source_path = Path(package.folder, frames[0]["image_file"])
    exported_path, labels = EditVideoTemplatePage.write_annotations_to_template_coco(
        str(source_path), [{"label": "录屏按钮", "bbox": [35, 20, 30, 20]}], "recorded_frame.png",
    )
    coco = CocoTemplateStore(coco_path.parent, coco_name=coco_path.name).load()
    assert labels == ["录屏按钮"]
    assert Path(exported_path).is_file()
    assert coco["categories"][0]["name"] == "录屏按钮"
    assert coco["annotations"][0]["bbox"] == [35, 20, 30, 20]
    assert json.loads(coco_path.read_text(encoding="utf-8")) == coco
